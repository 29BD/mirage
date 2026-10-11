use crate::generation::quiz::solutions::Answer;
use crate::generation::quiz::{Quiz, QuizQuestion};

/// Render a transcribed [`Quiz`] (plus optional answers) as CLI-friendly markdown.
///
/// Layout (an improvement on the original draft):
/// ```text
/// # Section 1
/// *section instructions, if any*
///
///   1) question text
///      a) first option
///      b) second option
///      > answer
/// ```
pub fn render_quiz(quiz: &Quiz, answers: Option<&[Vec<Answer>]>) -> String {
    let mut out = String::new();

    for (si, section) in quiz.0.iter().enumerate() {
        if si > 0 {
            out.push_str("\n---\n\n");
        }
        out.push_str(&format!("# Section {}\n", si + 1));
        if let Some(instructions) = section
            .instructions
            .as_ref()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
        {
            out.push_str(&format!("*{instructions}*\n"));
        }
        out.push('\n');

        for (qi, question) in section.questions.iter().enumerate() {
            match question {
                QuizQuestion::MultipleChoice { question, options } => {
                    out.push_str(&format!("  {}) {}\n", qi + 1, question.trim()));
                    for (oi, option) in options.iter().enumerate() {
                        let label = option_label(oi);
                        out.push_str(&format!("     {label}) {}\n", option.trim()));
                    }
                }
                QuizQuestion::TrueFalse { question } => {
                    out.push_str(&format!(
                        "  {}) {} _(True / False)_\n",
                        qi + 1,
                        question.trim()
                    ));
                }
                QuizQuestion::ModifiedTrueFalse { question } => {
                    out.push_str(&format!(
                        "  {}) {} _(True / False; correct the **bold** part)_\n",
                        qi + 1,
                        question.trim()
                    ));
                }
                QuizQuestion::OtherQuestion { question } => {
                    out.push_str(&format!("  {}) {}\n", qi + 1, question.trim()));
                }
            }

            if let Some(answer) = answers
                .and_then(|all| all.get(si))
                .and_then(|sec| sec.get(qi))
            {
                let text = answer.text();
                if text.is_empty() {
                    out.push_str("     > (no answer)\n");
                } else {
                    // Keep multi-line answers readable under one blockquote.
                    for (i, line) in text.lines().enumerate() {
                        let _ = i;
                        out.push_str(&format!("     > {}\n", line.trim()));
                    }
                }
            }
        }
    }

    out
}

pub(crate) fn option_label(index: usize) -> String {
    // a, b, ... z, aa, ab, ... for very long option lists.
    let mut n = index;
    let mut label = String::new();
    loop {
        let ch = (b'a' + (n % 26) as u8) as char;
        label.insert(0, ch);
        if n < 26 {
            break;
        }
        n = n / 26 - 1;
    }
    label
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::quiz::QuizSection;

    fn sample_quiz() -> Quiz {
        Quiz(vec![
            QuizSection {
                instructions: Some("Choose the best answer.".to_string()),
                questions: vec![
                    QuizQuestion::MultipleChoice {
                        question: "What is 2 + 2?".to_string(),
                        options: vec!["3".to_string(), "4".to_string()],
                    },
                    QuizQuestion::TrueFalse {
                        question: "The sky is blue.".to_string(),
                    },
                    QuizQuestion::ModifiedTrueFalse {
                        question: "The sky is **green**.".to_string(),
                    },
                ],
            },
            QuizSection {
                instructions: None,
                questions: vec![QuizQuestion::OtherQuestion {
                    question: "Explain gravity.".to_string(),
                }],
            },
        ])
    }

    #[test]
    fn renders_transcript_without_answers() {
        let rendered = render_quiz(&sample_quiz(), None);
        assert!(rendered.contains("# Section 1"));
        assert!(rendered.contains("# Section 2"));
        assert!(rendered.contains("*Choose the best answer.*"));
        assert!(rendered.contains("1) What is 2 + 2?"));
        assert!(rendered.contains("a) 3"));
        assert!(rendered.contains("b) 4"));
        assert!(rendered.contains("2) The sky is blue. _(True / False)_"));
        assert!(rendered.contains("3) The sky is **green**. _(True / False; correct the **bold** part)_"));
        assert!(rendered.contains("1) Explain gravity."));
        assert!(!rendered.contains('>'));
    }

    #[test]
    fn renders_answers_as_blockquotes() {
        let quiz = sample_quiz();
        let answers = vec![
            vec![
                Answer::Simple("4".to_string()),
                Answer::Simple("True".to_string()),
                Answer::ModifiedTrueFalse {
                    verdict: false,
                    correction: Some("blue".to_string()),
                },
            ],
            vec![Answer::Simple("Mass attracts mass.".to_string())],
        ];
        let rendered = render_quiz(&quiz, Some(&answers));
        assert!(rendered.contains("> 4"));
        assert!(rendered.contains("> True"));
        assert!(rendered.contains("> FALSE — blue"));
        assert!(rendered.contains("> Mass attracts mass."));
    }
}
