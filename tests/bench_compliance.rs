//! Instruction-compliance bench: scores whether the solver obeys the quiz,
//! not just whether answers are right.
//!
//! Live test (needs `GEMINI_API_KEY`, skipped offline like the other live
//! tests). Runs `tests/fixtures/math_quiz.png` 3x at `--rpm 10` equivalent
//! and prints a per-question table plus per-run time and call counts.
//!
//! Targets: full compliance, solve under 60 s, at most 6 solve calls.
//! Nothing quiz-specific is hardcoded: sentence ranges and show-work rules
//! are read from each section's own instructions.

use google_ai_rs::Client;
use mirage::generation::quiz::solutions::{
    Answerable, LlmContext, count_sentences, is_plain_text, parse_sentence_range,
    requires_show_work,
};
use mirage::generation::quiz::{Quiz, QuizQuestion};
use std::path::PathBuf;
use std::time::Instant;

const RUNS: usize = 3;
const MAX_SOLVE_CALLS: u64 = 6;
const MAX_SOLVE_MS: u128 = 60_000;
const DEFAULT_MODEL: &str = "gemini-2.5-flash-lite";

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("math_quiz.png")
}

fn question_text(q: &QuizQuestion) -> &str {
    match q {
        QuizQuestion::MultipleChoice { question, .. }
        | QuizQuestion::TrueFalse { question }
        | QuizQuestion::ModifiedTrueFalse { question }
        | QuizQuestion::OtherQuestion { question } => question,
    }
}

fn type_tag(q: &QuizQuestion) -> &'static str {
    match q {
        QuizQuestion::MultipleChoice { .. } => "MC",
        QuizQuestion::TrueFalse { .. } => "TF",
        QuizQuestion::ModifiedTrueFalse { .. } => "MTF",
        QuizQuestion::OtherQuestion { .. } => "FR",
    }
}

/// Generic label-leak check: does stored text still start with a printed
/// `1.`/`2)`/`A.`/`b)` marker?
fn has_leading_label(s: &str) -> bool {
    let t = s.trim_start().trim_start_matches(['-', '*', '•']).trim_start();
    let digits = t.len() - t.trim_start_matches(|c: char| c.is_ascii_digit()).len();
    if digits > 0
        && let Some(rest) = t.get(digits..)
    {
        let mut chars = rest.chars();
        if matches!(chars.next(), Some('.') | Some(')'))
            && chars.as_str().starts_with(char::is_whitespace)
        {
            return true;
        }
    }
    let mut chars = t.chars();
    if let Some(c) = chars.next()
        && c.is_ascii_alphabetic()
    {
        let mut rest = chars.as_str().chars();
        if matches!(rest.next(), Some('.') | Some(')')) {
            let tail = rest.as_str();
            if !tail.is_empty() && tail.starts_with(char::is_whitespace) {
                return true;
            }
        }
    }
    false
}

#[tokio::test]
async fn bench_instruction_compliance() {
    let api_key = match std::env::var("GEMINI_API_KEY") {
        Ok(k) if !k.trim().is_empty() => k,
        _ => {
            eprintln!("skipping live bench: GEMINI_API_KEY not set");
            return;
        }
    };
    let models: Vec<String> = match std::env::var("GEMINI_MODEL") {
        Ok(m) if !m.trim().is_empty() => vec![m],
        _ => vec![DEFAULT_MODEL.to_string()],
    };
    let model = &models[0];
    let path = fixture_path();
    let client = Client::new(api_key).await.unwrap();

    println!("run | sec | q | type | ok | failed_checks | sec_ms | run_calls");
    let mut grand_compliant = 0usize;
    let mut grand_total = 0usize;

    for run in 1..=RUNS {
        let t0 = Instant::now();
        let quiz = Quiz::from_image(&client, model, &path)
            .await
            .expect("transcription failed");
        let transcribe_ms = t0.elapsed().as_millis();

        let context = LlmContext::new(client.clone(), model.clone(), 10, 6);
        let mut run_compliant = 0usize;
        let mut run_total = 0usize;
        let mut solve_ms_total = 0u128;

        for (si, section) in quiz.0.iter().enumerate() {
            let t1 = Instant::now();
            let answers = section.answer(&context).await.expect("solving failed");
            let sec_ms = t1.elapsed().as_millis();
            solve_ms_total += sec_ms;

            let range = section.instructions.as_deref().and_then(parse_sentence_range);
            let work = section
                .instructions
                .as_deref()
                .map(requires_show_work)
                .unwrap_or(false);

            for (qi, (question, answer)) in
                section.questions.iter().zip(answers.iter()).enumerate()
            {
                run_total += 1;
                let mut failed: Vec<&str> = Vec::new();

                // Transcript checks.
                if has_leading_label(question_text(question)) {
                    failed.push("label-leak");
                }
                if let QuizQuestion::MultipleChoice { options, .. } = question
                    && options.iter().any(|o| has_leading_label(o))
                {
                    failed.push("option-label-leak");
                }
                if let QuizQuestion::ModifiedTrueFalse { .. } = question
                    && !question_text(question).contains("**")
                {
                    failed.push("no-emphasis");
                }

                // Answer checks.
                let text = answer.text();
                if !answer.is_complete() {
                    failed.push("incomplete");
                }
                if !is_plain_text(&text) {
                    failed.push("not-plain-text");
                }
                if let (Some((lo, hi)), QuizQuestion::OtherQuestion { .. }) = (range, question) {
                    let n = count_sentences(&text);
                    if n < lo || n > hi {
                        failed.push("sentence-count");
                    }
                }
                if work
                    && matches!(question, QuizQuestion::OtherQuestion { .. })
                    && text.lines().filter(|l| !l.trim().is_empty()).count() < 2
                {
                    failed.push("no-work");
                }

                if failed.is_empty() {
                    run_compliant += 1;
                }
                println!(
                    "{run} | {} | {} | {} | {} | {} | {sec_ms} | {}",
                    si + 1,
                    qi + 1,
                    type_tag(question),
                    failed.is_empty(),
                    if failed.is_empty() {
                        "-".to_string()
                    } else {
                        failed.join(",")
                    },
                    context.calls(),
                );
            }
        }

        let calls = context.calls();
        let solve_ms = solve_ms_total;
        println!(
            "run {run} summary: compliance {run_compliant}/{run_total}, \
transcribe {transcribe_ms} ms, solve {solve_ms} ms, solve calls {calls}"
        );
        grand_compliant += run_compliant;
        grand_total += run_total;

        assert!(
            calls <= MAX_SOLVE_CALLS,
            "run {run}: {calls} solve calls, target is at most {MAX_SOLVE_CALLS}"
        );
        assert!(
            solve_ms < MAX_SOLVE_MS,
            "run {run}: solve took {solve_ms} ms, target is under {} ms",
            MAX_SOLVE_MS
        );
        assert!(
            run_total >= 18,
            "run {run}: transcribed {run_total} questions, fixture has 18"
        );
        if run_total != 18 {
            eprintln!("run {run}: note: transcribed {run_total} questions (fixture has 18)");
        }
        assert_eq!(
            run_compliant, run_total,
            "run {run}: only {run_compliant}/{run_total} compliant (see failed_checks above)"
        );
    }

    println!("bench total: compliance {grand_compliant}/{grand_total} over {RUNS} runs");
    eprintln!("tip: run with `cargo test -- --nocapture` to see the table");
}
