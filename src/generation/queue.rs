use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio::time::{Instant, sleep};

/// Paces outgoing LLM calls so a burst of per-question requests doesn't blow
/// through the free-tier quota (gemini-3.1-flash-lite: 15 req/min).
///
/// Sequential loop + this gate = at most `rpm` requests per rolling minute.
#[derive(Debug, Clone)]
pub struct Throttle {
    interval: Duration,
    last: Arc<Mutex<Option<Instant>>>,
}

impl Throttle {
    pub fn from_rpm(rpm: u32) -> Self {
        let rpm = rpm.clamp(1, 60);
        Self {
            interval: Duration::from_secs_f64(60.0 / f64::from(rpm)),
            last: Arc::new(Mutex::new(None)),
        }
    }

    #[allow(dead_code)]
    pub fn interval(&self) -> Duration {
        self.interval
    }

    /// Sleep until the next slot is due, then claim it.
    pub async fn wait(&self) {
        let due = {
            let guard = self.last.lock().await;
            guard.map(|last| last + self.interval)
        };
        if let Some(due) = due {
            let now = Instant::now();
            if due > now {
                let wait = due - now;
                tracing::debug!(
                    wait_secs = wait.as_secs_f64(),
                    interval_secs = self.interval.as_secs_f64(),
                    "throttling: waiting for next request slot"
                );
                sleep(wait).await;
                *self.last.lock().await = Some(due);
                return;
            }
            tracing::trace!("throttle slot already due, proceeding without sleep");
        } else {
            tracing::trace!("first throttle slot claimed immediately");
        }
        *self.last.lock().await = Some(Instant::now());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rpm_sets_interval() {
        assert_eq!(Throttle::from_rpm(15).interval(), Duration::from_secs(4));
        assert_eq!(Throttle::from_rpm(10).interval(), Duration::from_secs(6));
        assert_eq!(Throttle::from_rpm(60).interval(), Duration::from_secs(1));
    }

    #[test]
    fn rpm_clamps() {
        assert_eq!(Throttle::from_rpm(0).interval(), Duration::from_secs(60));
        assert_eq!(Throttle::from_rpm(600).interval(), Duration::from_secs(1));
    }

    #[tokio::test]
    async fn first_wait_is_immediate() {
        let t = Throttle::from_rpm(10);
        // Should return ~instantly, not sleep 6s.
        tokio::time::timeout(Duration::from_secs(2), t.wait())
            .await
            .expect("first wait should be immediate");
    }

    #[tokio::test]
    async fn second_wait_enforces_interval() {
        let t = Throttle::from_rpm(60); // 1s interval keeps the test fast
        t.wait().await;
        let start = Instant::now();
        t.wait().await;
        assert!(start.elapsed() >= Duration::from_millis(900));
    }
}
