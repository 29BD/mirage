use thiserror::Error;
use std::time::Duration;

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("generation request failed: {0}")]
    Generation(#[from] google_ai_rs::Error),
}

impl LlmError {
    /// Full text used for rate-limit detection (Display drops some context,
    /// so also scan Debug).
    fn haystack(&self) -> String {
        format!("{self}\n{self:?}")
    }

    pub fn is_rate_limited(&self) -> bool {
        let m = self.haystack().to_lowercase();
        m.contains("resourceexhausted")
            || m.contains("quota exceeded")
            || m.contains("rate limit")
            || m.contains("429")
    }

    /// Server-suggested wait ("Please retry in 36.01s"), clamped to 1s..=5min.
    pub fn retry_after(&self) -> Option<Duration> {
        retry_after_from_message(&self.haystack())
    }

    /// Short message for CLI output. The raw google error Display embeds the
    /// whole proto details blob (the giant `details: [...]` array), which is
    /// unreadable, so collapse rate-limit failures to one line.
    pub fn friendly(&self) -> String {
        if self.is_rate_limited() {
            match self.retry_after() {
                Some(d) => format!(
                    "Gemini rate limit hit (free tier ~15 req/min). Retry in ~{}s, or rerun with a lower --rpm.",
                    d.as_secs().max(1),
                ),
                None => "Gemini rate limit hit (free tier ~15 req/min). Wait a minute and retry, or rerun with a lower --rpm.".to_string(),
            }
        } else {
            // Still avoid the details blob: first line of the Display only.
            let full = self.haystack_line();
            full.lines().next().unwrap_or("generation request failed").to_string()
        }
    }

    fn haystack_line(&self) -> String {
        format!("{self}")
    }
}

/// Parses "retry in 36.01s" (case-insensitive) out of an API error message.
pub fn retry_after_from_message(msg: &str) -> Option<Duration> {
    let lower = msg.to_lowercase();
    let idx = lower.find("retry in ")?;
    let rest = lower[idx + "retry in ".len()..].trim_start();
    let num_end = rest.find('s').unwrap_or(rest.len());
    let secs: f64 = rest[..num_end].trim().parse().ok()?;
    if !secs.is_finite() {
        return None;
    }
    Some(Duration::from_secs_f64(secs.clamp(1.0, 300.0)))
}

/// Backoff when the server gives no explicit retry delay.
pub fn backoff_delay(attempt: u32) -> Duration {
    let secs = 5.0 * 2f64.powi(attempt.saturating_sub(1) as i32);
    Duration::from_secs_f64(secs.min(60.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_retry_in_seconds() {
        let d = retry_after_from_message("Please retry in 36.01429893s.").unwrap();
        assert!((d.as_secs_f64() - 36.01).abs() < 0.1);
    }

    #[test]
    fn parses_retry_in_whole_seconds() {
        assert_eq!(
            retry_after_from_message("retry in 5s").unwrap(),
            Duration::from_secs(5)
        );
    }

    #[test]
    fn clamps_extreme_delays() {
        assert_eq!(
            retry_after_from_message("retry in 0.01s").unwrap(),
            Duration::from_secs(1)
        );
        assert_eq!(
            retry_after_from_message("retry in 9999s").unwrap(),
            Duration::from_secs(300)
        );
    }

    #[test]
    fn no_hint_returns_none() {
        assert!(retry_after_from_message("boom").is_none());
    }

    #[test]
    fn backoff_grows_and_caps() {
        assert_eq!(backoff_delay(1), Duration::from_secs(5));
        assert_eq!(backoff_delay(2), Duration::from_secs(10));
        assert_eq!(backoff_delay(10), Duration::from_secs(60));
    }
}
