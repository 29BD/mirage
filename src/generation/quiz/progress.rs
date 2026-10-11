//! Progress events for quiz solving.
//!
//! [`QuizProgress`] describes the per-question lifecycle while a quiz is
//! being answered (`Stalled` → `Waiting` → `Generating` → `Done`/`Failed`).
//! Sinks implement [`ProgressSink`] (a sync `emit`, so it can fire from
//! anywhere in the pipeline); [`NullSink`] drops everything and
//! [`MpscSink`] forwards events over a tokio channel for CLI/TUI consumers.
//!
//! Indices are 0-based positions within the answered quiz: `section` counts
//! [`Quiz`](super::Quiz) sections, `question` counts questions within that
//! section.

use std::time::Duration;

/// Why a question is waiting instead of generating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitReason {
    /// Pacing between requests so a burst of calls stays under quota.
    Throttle,
    /// Server-reported rate limit; the request backs off and retries.
    RateLimit,
}

impl std::fmt::Display for WaitReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WaitReason::Throttle => write!(f, "throttle"),
            WaitReason::RateLimit => write!(f, "rate limit"),
        }
    }
}

/// Lifecycle event for one question while solving a quiz.
///
/// A section batch request covers every question in that section at once,
/// so a batch emits `Generating` (then `Done`) for each of its questions
/// around the single shared request; use `answer_per_question` for states
/// that track exactly one request each.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuizProgress {
    /// Queued, with no worker/slot claimed yet.
    QuestionStalled { section: usize, question: usize },
    /// Sleeping before (re)trying: pacing or rate-limit backoff.
    QuestionWaiting {
        section: usize,
        question: usize,
        reason: WaitReason,
        wait: Duration,
    },
    /// A request covering this question is in flight.
    QuestionGenerating { section: usize, question: usize },
    /// Answered (batch-validated or single).
    QuestionDone { section: usize, question: usize },
    /// Terminal failure; `error` is the one-line [`LlmError`](crate::LlmError)
    /// message. No further retries will be attempted.
    QuestionFailed {
        section: usize,
        question: usize,
        error: String,
    },
}

impl QuizProgress {
    /// 0-based section index within the quiz.
    pub fn section(&self) -> usize {
        match self {
            QuizProgress::QuestionStalled { section, .. }
            | QuizProgress::QuestionWaiting { section, .. }
            | QuizProgress::QuestionGenerating { section, .. }
            | QuizProgress::QuestionDone { section, .. }
            | QuizProgress::QuestionFailed { section, .. } => *section,
        }
    }

    /// 0-based question index within the section.
    pub fn question(&self) -> usize {
        match self {
            QuizProgress::QuestionStalled { question, .. }
            | QuizProgress::QuestionWaiting { question, .. }
            | QuizProgress::QuestionGenerating { question, .. }
            | QuizProgress::QuestionDone { question, .. }
            | QuizProgress::QuestionFailed { question, .. } => *question,
        }
    }
}

/// Sink for [`QuizProgress`] events.
///
/// Sync `emit` (never blocks the solve loop); `Send + Sync + Clone` so one
/// sink can be shared across the whole pipeline.
pub trait ProgressSink: Send + Sync + Clone {
    /// Record one event. Must not block or panic on a full/closed channel.
    fn emit(&self, event: QuizProgress);
}

/// Drops all events. The default sink: plain `answer()` calls stay silent.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullSink;

impl ProgressSink for NullSink {
    fn emit(&self, _event: QuizProgress) {}
}

/// Forwards events over a tokio unbounded channel (unbounded, so `emit`
/// stays sync and never applies backpressure to the solve loop).
#[derive(Debug, Clone)]
pub struct MpscSink {
    tx: tokio::sync::mpsc::UnboundedSender<QuizProgress>,
}

impl MpscSink {
    /// Fresh sink plus its receiving end.
    pub fn channel() -> (Self, tokio::sync::mpsc::UnboundedReceiver<QuizProgress>) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        (Self { tx }, rx)
    }

    /// Wrap an existing sender (e.g. a fan-out to several consumers).
    pub fn from_sender(tx: tokio::sync::mpsc::UnboundedSender<QuizProgress>) -> Self {
        Self { tx }
    }
}

impl ProgressSink for MpscSink {
    fn emit(&self, event: QuizProgress) {
        let _ = self.tx.send(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indices_round_trip() {
        let e = QuizProgress::QuestionWaiting {
            section: 2,
            question: 5,
            reason: WaitReason::RateLimit,
            wait: Duration::from_secs(1),
        };
        assert_eq!(e.section(), 2);
        assert_eq!(e.question(), 5);
        assert_eq!(format!("{}", WaitReason::Throttle), "throttle");
        assert_eq!(format!("{}", WaitReason::RateLimit), "rate limit");
    }

    #[test]
    fn null_sink_drops_events() {
        NullSink.emit(QuizProgress::QuestionDone {
            section: 0,
            question: 0,
        });
    }

    #[test]
    fn mpsc_sink_delivers_in_order() {
        let (sink, mut rx) = MpscSink::channel();
        sink.emit(QuizProgress::QuestionStalled {
            section: 0,
            question: 1,
        });
        sink.emit(QuizProgress::QuestionDone {
            section: 0,
            question: 1,
        });
        drop(sink);
        let mut got = Vec::new();
        while let Ok(e) = rx.try_recv() {
            got.push(e);
        }
        assert_eq!(
            got,
            vec![
                QuizProgress::QuestionStalled {
                    section: 0,
                    question: 1
                },
                QuizProgress::QuestionDone {
                    section: 0,
                    question: 1
                },
            ]
        );
    }
}
