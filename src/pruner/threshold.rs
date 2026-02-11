use super::Pruner;
use crate::sampler::CompletedTrial;

/// Prune trials whose intermediate values exceed fixed thresholds.
///
/// Useful for cutting off trials that are clearly diverging or stuck
/// at bad values early in training.
///
/// # Examples
///
/// ```
/// use optimizer::pruner::ThresholdPruner;
///
/// // Prune if the intermediate value exceeds 100.0 or falls below 0.0
/// let pruner = ThresholdPruner::new().upper(100.0).lower(0.0);
/// ```
pub struct ThresholdPruner {
    /// Prune if intermediate value is greater than this. `None` = no upper bound.
    upper: Option<f64>,
    /// Prune if intermediate value is less than this. `None` = no lower bound.
    lower: Option<f64>,
}

impl ThresholdPruner {
    /// Create a new `ThresholdPruner` with no thresholds set.
    ///
    /// By default, no pruning occurs. Use [`upper`](Self::upper) and
    /// [`lower`](Self::lower) to set bounds.
    #[must_use]
    pub fn new() -> Self {
        Self {
            upper: None,
            lower: None,
        }
    }

    /// Set the upper threshold. Trials with intermediate values above this
    /// will be pruned.
    #[must_use]
    pub fn upper(mut self, threshold: f64) -> Self {
        self.upper = Some(threshold);
        self
    }

    /// Set the lower threshold. Trials with intermediate values below this
    /// will be pruned.
    #[must_use]
    pub fn lower(mut self, threshold: f64) -> Self {
        self.lower = Some(threshold);
        self
    }
}

impl Default for ThresholdPruner {
    fn default() -> Self {
        Self::new()
    }
}

impl Pruner for ThresholdPruner {
    fn should_prune(
        &self,
        _trial_id: u64,
        _step: u64,
        intermediate_values: &[(u64, f64)],
        _completed_trials: &[CompletedTrial],
    ) -> bool {
        let Some(&(_, latest_value)) = intermediate_values.last() else {
            return false;
        };
        if let Some(upper) = self.upper
            && latest_value > upper
        {
            return true;
        }
        if let Some(lower) = self.lower
            && latest_value < lower
        {
            return true;
        }
        false
    }
}
