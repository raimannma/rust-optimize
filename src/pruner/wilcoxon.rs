//! Wilcoxon pruner — statistically rigorous pruning for noisy objectives.
//!
//! Uses the Wilcoxon signed-rank test to compare the current trial's
//! intermediate values against the best completed trial at matching steps.
//! The test accounts for the paired, step-aligned nature of the comparison
//! and only prunes when the difference is statistically significant.
//!
//! This is more principled than [`MedianPruner`](super::MedianPruner) for
//! noisy objectives because a single bad step won't trigger pruning — the
//! test considers the full distribution of paired differences.
//!
//! # When to use
//!
//! - When intermediate values have high variance (e.g., mini-batch loss,
//!   stochastic reward signals)
//! - When you want a statistical guarantee that pruned trials are truly worse
//! - When you have enough steps (at least 6) for a meaningful test
//!
//! For less noisy objectives, [`MedianPruner`](super::MedianPruner) is simpler
//! and often sufficient.
//!
//! # Configuration
//!
//! | Option | Default | Description |
//! |--------|---------|-------------|
//! | `p_value_threshold` | 0.05 | Significance level — lower is more conservative |
//! | `n_warmup_steps` | 0 | Skip pruning in the first N steps |
//! | `n_min_trials` | 1 | Require at least N completed trials before pruning |
//!
//! # Example
//!
//! ```
//! use optimizer::Direction;
//! use optimizer::pruner::WilcoxonPruner;
//!
//! let pruner = WilcoxonPruner::new(Direction::Minimize)
//!     .p_value_threshold(0.05)
//!     .n_warmup_steps(5)
//!     .n_min_trials(1);
//! ```

use core::cmp::Ordering;

use super::Pruner;
use crate::sampler::CompletedTrial;
use crate::types::{Direction, TrialState};

/// Prune trials using a Wilcoxon signed-rank test comparing intermediate
/// values against the best completed trial.
///
/// More principled than `MedianPruner` for noisy objectives — it accounts
/// for the paired nature of step-aligned comparisons and doesn't prune
/// on random fluctuations.
///
/// The test compares intermediate values at matching steps between the
/// current trial and the best completed trial. If the current trial is
/// statistically significantly worse (p < threshold), it is pruned.
///
/// # Examples
///
/// ```
/// use optimizer::Direction;
/// use optimizer::pruner::WilcoxonPruner;
///
/// let pruner = WilcoxonPruner::new(Direction::Minimize)
///     .p_value_threshold(0.05)
///     .n_warmup_steps(5)
///     .n_min_trials(1);
/// ```
pub struct WilcoxonPruner {
    /// Significance level (default 0.05). Lower = more conservative.
    p_value_threshold: f64,
    /// Don't prune in the first N steps (let the trial warm up).
    n_warmup_steps: u64,
    /// Require at least N completed trials before pruning.
    n_min_trials: usize,
    /// The optimization direction.
    direction: Direction,
}

impl WilcoxonPruner {
    /// Create a new `WilcoxonPruner` for the given optimization direction.
    ///
    /// By default, `p_value_threshold` is 0.05, `n_warmup_steps` is 0,
    /// and `n_min_trials` is 1.
    #[must_use]
    pub fn new(direction: Direction) -> Self {
        Self {
            p_value_threshold: 0.05,
            n_warmup_steps: 0,
            n_min_trials: 1,
            direction,
        }
    }

    /// Set the p-value threshold for significance.
    ///
    /// Must be in (0.0, 1.0). Lower values are more conservative (harder to prune).
    ///
    /// # Panics
    ///
    /// Panics if `p` is not in the open interval (0.0, 1.0).
    #[must_use]
    pub fn p_value_threshold(mut self, p: f64) -> Self {
        assert!(
            p > 0.0 && p < 1.0,
            "p_value_threshold must be in (0.0, 1.0)"
        );
        self.p_value_threshold = p;
        self
    }

    /// Set the number of warmup steps. No pruning occurs before this step.
    #[must_use]
    pub fn n_warmup_steps(mut self, n: u64) -> Self {
        self.n_warmup_steps = n;
        self
    }

    /// Set the minimum number of completed trials required before pruning.
    #[must_use]
    pub fn n_min_trials(mut self, n: usize) -> Self {
        self.n_min_trials = n;
        self
    }
}

impl Pruner for WilcoxonPruner {
    fn should_prune(
        &self,
        _trial_id: u64,
        step: u64,
        intermediate_values: &[(u64, f64)],
        completed_trials: &[CompletedTrial],
    ) -> bool {
        if step < self.n_warmup_steps {
            return false;
        }

        let completed: Vec<&CompletedTrial> = completed_trials
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .collect();

        if completed.len() < self.n_min_trials {
            return false;
        }

        // Find the best completed trial by final objective value.
        let best = match self.direction {
            Direction::Minimize => completed
                .iter()
                .min_by(|a, b| a.value.partial_cmp(&b.value).unwrap_or(Ordering::Equal)),
            Direction::Maximize => completed
                .iter()
                .max_by(|a, b| a.value.partial_cmp(&b.value).unwrap_or(Ordering::Equal)),
        };

        let Some(best) = best else {
            return false;
        };

        // Pair intermediate values at matching steps.
        let pairs: Vec<(f64, f64)> = intermediate_values
            .iter()
            .filter_map(|&(s, current_v)| {
                best.intermediate_values
                    .iter()
                    .find(|(bs, _)| *bs == s)
                    .map(|&(_, best_v)| (current_v, best_v))
            })
            .collect();

        // Need at least 6 pairs for a meaningful test.
        if pairs.len() < 6 {
            return false;
        }

        // Compute signed differences: current - best.
        // For minimization: positive diff means current is worse.
        // For maximization: negative diff means current is worse.
        let differences: Vec<f64> = pairs
            .iter()
            .map(|&(current, best_v)| current - best_v)
            .collect();

        // Run the Wilcoxon signed-rank test.
        let p_value = wilcoxon_signed_rank_test(&differences, self.direction);

        p_value < self.p_value_threshold
    }
}

/// Perform a one-sided Wilcoxon signed-rank test.
///
/// Tests whether the values tend to be worse than zero (positive for
/// minimization, negative for maximization).
///
/// Returns a p-value. Small p-values indicate the current trial is
/// significantly worse.
fn wilcoxon_signed_rank_test(differences: &[f64], direction: Direction) -> f64 {
    // 1. Remove zero differences.
    let nonzero: Vec<f64> = differences.iter().copied().filter(|d| *d != 0.0).collect();
    let n = nonzero.len();

    if n < 6 {
        return 1.0; // Not enough data
    }

    // 2. Rank by absolute value.
    let mut abs_ranked: Vec<(usize, f64, f64)> = nonzero
        .iter()
        .enumerate()
        .map(|(i, &d)| (i, d.abs(), d))
        .collect();
    abs_ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    // 3. Assign ranks with tie correction.
    let ranks = assign_ranks(&abs_ranked);

    // 4. Compute W+ (sum of ranks for positive differences) and
    //    W- (sum of ranks for negative differences).
    let mut w_plus = 0.0;
    let mut w_minus = 0.0;
    for (i, &(_, _, orig)) in abs_ranked.iter().enumerate() {
        if orig > 0.0 {
            w_plus += ranks[i];
        } else {
            w_minus += ranks[i];
        }
    }

    // For one-sided test:
    // - Minimization: we want to detect positive diffs (current worse).
    //   Large W+ means significantly worse. Test statistic = W-.
    // - Maximization: we want to detect negative diffs (current worse).
    //   Large W- means significantly worse. Test statistic = W+.
    let w = match direction {
        Direction::Minimize => w_minus,
        Direction::Maximize => w_plus,
    };

    // 5. Normal approximation for the p-value.
    #[allow(clippy::cast_precision_loss)]
    let n_f = n as f64;
    let mean = n_f * (n_f + 1.0) / 4.0;
    let variance = n_f * (n_f + 1.0) * (2.0 * n_f + 1.0) / 24.0;

    // Tie correction for variance.
    let tie_correction = compute_tie_correction(&ranks);
    let adjusted_variance = variance - tie_correction;

    if adjusted_variance <= 0.0 {
        return 1.0;
    }

    let std_dev = adjusted_variance.sqrt();

    // Continuity correction: shift W by 0.5 towards mean.
    let continuity = if w < mean { 0.5 } else { -0.5 };
    let z = (w + continuity - mean) / std_dev;

    // One-sided p-value (lower tail): probability that the test statistic
    // is this small or smaller under H0.
    normal_cdf(z)
}

/// Assign average ranks, handling ties.
fn assign_ranks(sorted: &[(usize, f64, f64)]) -> Vec<f64> {
    let n = sorted.len();
    let mut ranks = vec![0.0; n];
    let mut i = 0;

    while i < n {
        let mut j = i;
        // Find all items tied with sorted[i].
        while j < n
            && (sorted[j].1 - sorted[i].1).abs() < f64::EPSILON * sorted[i].1.max(1.0) * 100.0
        {
            j += 1;
        }
        // Average rank for the tie group. Ranks are 1-based.
        #[allow(clippy::cast_precision_loss)]
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for rank in ranks.iter_mut().take(j).skip(i) {
            *rank = avg_rank;
        }
        i = j;
    }

    ranks
}

/// Compute the tie correction term for the variance.
/// For each tie group of size t, subtract t^3 - t from the sum,
/// then divide by 48.
fn compute_tie_correction(ranks: &[f64]) -> f64 {
    let mut correction = 0.0;
    let mut i = 0;

    while i < ranks.len() {
        let mut j = i;
        while j < ranks.len() && (ranks[j] - ranks[i]).abs() < f64::EPSILON {
            j += 1;
        }
        #[allow(clippy::cast_precision_loss)]
        let t = (j - i) as f64;
        if t > 1.0 {
            correction += t * t * t - t;
        }
        i = j;
    }

    correction / 48.0
}

/// Standard normal CDF using an approximation (Abramowitz & Stegun).
fn normal_cdf(x: f64) -> f64 {
    // Use the complementary error function relationship:
    // Φ(x) = 0.5 * erfc(-x / √2)
    0.5 * erfc(-x / core::f64::consts::SQRT_2)
}

/// Complementary error function approximation.
/// Maximum error: 1.5 × 10⁻⁷ (Abramowitz & Stegun formula 7.1.26).
fn erfc(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let result = poly * (-x * x).exp();

    if x >= 0.0 { result } else { 2.0 - result }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn trial_with_values(
        id: u64,
        value: f64,
        intermediate_values: Vec<(u64, f64)>,
    ) -> CompletedTrial {
        CompletedTrial::with_intermediate_values(
            id,
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            value,
            intermediate_values,
            HashMap::new(),
        )
    }

    #[test]
    fn no_prune_during_warmup() {
        let pruner = WilcoxonPruner::new(Direction::Minimize).n_warmup_steps(10);
        let completed = vec![trial_with_values(
            0,
            0.1,
            (0..20).map(|s| (s, 0.1)).collect(),
        )];
        let current: Vec<(u64, f64)> = (0..8).map(|s| (s, 100.0)).collect();
        assert!(!pruner.should_prune(1, 7, &current, &completed));
    }

    #[test]
    fn no_prune_with_insufficient_trials() {
        let pruner = WilcoxonPruner::new(Direction::Minimize).n_min_trials(5);
        let completed = vec![trial_with_values(
            0,
            0.1,
            (0..20).map(|s| (s, 0.1)).collect(),
        )];
        let current: Vec<(u64, f64)> = (0..10).map(|s| (s, 100.0)).collect();
        assert!(!pruner.should_prune(1, 9, &current, &completed));
    }

    #[test]
    fn no_prune_with_fewer_than_6_pairs() {
        let pruner = WilcoxonPruner::new(Direction::Minimize);
        let completed = vec![trial_with_values(
            0,
            0.1,
            (0..5).map(|s| (s, 0.1)).collect(),
        )];
        // Only 5 matching steps
        let current: Vec<(u64, f64)> = (0..5).map(|s| (s, 100.0)).collect();
        assert!(!pruner.should_prune(1, 4, &current, &completed));
    }

    #[test]
    fn prune_when_consistently_worse_minimize() {
        let pruner = WilcoxonPruner::new(Direction::Minimize);

        // Best trial has low values.
        let best_values: Vec<(u64, f64)> = (0..20).map(|s| (s, 0.1)).collect();
        let completed = vec![trial_with_values(0, 0.1, best_values)];

        // Current trial is consistently much worse.
        let current: Vec<(u64, f64)> = (0..20).map(|s| (s, 10.0)).collect();
        assert!(pruner.should_prune(1, 19, &current, &completed));
    }

    #[test]
    fn prune_when_consistently_worse_maximize() {
        let pruner = WilcoxonPruner::new(Direction::Maximize);

        // Best trial has high values.
        let best_values: Vec<(u64, f64)> = (0..20).map(|s| (s, 10.0)).collect();
        let completed = vec![trial_with_values(0, 10.0, best_values)];

        // Current trial is consistently much worse.
        let current: Vec<(u64, f64)> = (0..20).map(|s| (s, 0.1)).collect();
        assert!(pruner.should_prune(1, 19, &current, &completed));
    }

    #[test]
    fn no_prune_when_statistically_similar() {
        let pruner = WilcoxonPruner::new(Direction::Minimize);

        // Best trial and current trial have very similar values with noise.
        let best_values: Vec<(u64, f64)> = (0..20_u64)
            .map(|s| {
                let noise = if s.is_multiple_of(2) { 0.01 } else { -0.01 };
                (s, 1.0 + noise)
            })
            .collect();
        let completed = vec![trial_with_values(0, 1.0, best_values)];

        // Current trial is similar — alternating above/below.
        let current: Vec<(u64, f64)> = (0..20_u64)
            .map(|s| {
                let noise = if s.is_multiple_of(2) { -0.01 } else { 0.01 };
                (s, 1.0 + noise)
            })
            .collect();
        assert!(!pruner.should_prune(1, 19, &current, &completed));
    }

    #[test]
    fn selects_best_trial_minimize() {
        let pruner = WilcoxonPruner::new(Direction::Minimize);

        // Two completed trials: trial 0 is better (lower).
        let completed = vec![
            trial_with_values(0, 0.1, (0..20).map(|s| (s, 0.1)).collect()),
            trial_with_values(1, 5.0, (0..20).map(|s| (s, 5.0)).collect()),
        ];

        // Current trial is worse than the best but similar to the second.
        let current: Vec<(u64, f64)> = (0..20).map(|s| (s, 5.0)).collect();
        assert!(pruner.should_prune(2, 19, &current, &completed));
    }

    #[test]
    fn selects_best_trial_maximize() {
        let pruner = WilcoxonPruner::new(Direction::Maximize);

        // Two completed trials: trial 1 is better (higher).
        let completed = vec![
            trial_with_values(0, 0.1, (0..20).map(|s| (s, 0.1)).collect()),
            trial_with_values(1, 10.0, (0..20).map(|s| (s, 10.0)).collect()),
        ];

        // Current trial is worse than the best.
        let current: Vec<(u64, f64)> = (0..20).map(|s| (s, 0.1)).collect();
        assert!(pruner.should_prune(2, 19, &current, &completed));
    }

    #[test]
    fn ignores_pruned_trials() {
        let pruner = WilcoxonPruner::new(Direction::Minimize);

        // Only a pruned trial — no complete trials.
        let mut trial = trial_with_values(0, 0.1, (0..20).map(|s| (s, 0.1)).collect());
        trial.state = TrialState::Pruned;
        let completed = vec![trial];

        let current: Vec<(u64, f64)> = (0..20).map(|s| (s, 100.0)).collect();
        assert!(!pruner.should_prune(1, 19, &current, &completed));
    }

    #[test]
    fn lower_p_value_is_more_conservative() {
        let strict = WilcoxonPruner::new(Direction::Minimize).p_value_threshold(0.001);
        let lenient = WilcoxonPruner::new(Direction::Minimize).p_value_threshold(0.1);

        let completed = vec![trial_with_values(
            0,
            0.1,
            (0..20).map(|s| (s, 0.1)).collect(),
        )];

        // Moderately worse — should pass lenient but maybe not strict.
        let current: Vec<(u64, f64)> = (0..20)
            .map(|s| if s < 15 { (s, 0.2) } else { (s, 0.15) })
            .collect();

        let lenient_prunes = lenient.should_prune(1, 19, &current, &completed);
        let strict_prunes = strict.should_prune(1, 19, &current, &completed);

        // A stricter threshold should never prune when a lenient one doesn't.
        if !lenient_prunes {
            assert!(!strict_prunes);
        }
    }

    #[test]
    #[should_panic(expected = "p_value_threshold must be in (0.0, 1.0)")]
    fn panics_on_zero_p_value() {
        let _ = WilcoxonPruner::new(Direction::Minimize).p_value_threshold(0.0);
    }

    #[test]
    #[should_panic(expected = "p_value_threshold must be in (0.0, 1.0)")]
    fn panics_on_one_p_value() {
        let _ = WilcoxonPruner::new(Direction::Minimize).p_value_threshold(1.0);
    }

    #[test]
    fn correct_signed_rank_statistic() {
        // Known example: differences [1, 2, 3, 4, 5, 6] (all positive).
        // Ranks: 1, 2, 3, 4, 5, 6. W+ = 21, W- = 0.
        // For minimization (testing if positive = worse), W- = 0.
        // This should give a very small p-value.
        let diffs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let p = wilcoxon_signed_rank_test(&diffs, Direction::Minimize);
        assert!(
            p < 0.05,
            "p-value {p} should be < 0.05 for all-positive diffs"
        );
    }

    #[test]
    fn symmetric_differences_not_significant() {
        // Balanced differences: half positive, half negative.
        let diffs = vec![1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0];
        let p = wilcoxon_signed_rank_test(&diffs, Direction::Minimize);
        assert!(p > 0.05, "p-value {p} should be > 0.05 for symmetric diffs");
    }

    #[test]
    fn normal_cdf_known_values() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!(normal_cdf(-10.0) < 1e-6);
        assert!((normal_cdf(10.0) - 1.0).abs() < 1e-6);
        assert!((normal_cdf(-1.96) - 0.025).abs() < 0.001);
    }

    #[test]
    fn no_intermediate_values() {
        let pruner = WilcoxonPruner::new(Direction::Minimize);
        let completed = vec![trial_with_values(
            0,
            0.1,
            (0..20).map(|s| (s, 0.1)).collect(),
        )];
        assert!(!pruner.should_prune(1, 0, &[], &completed));
    }

    #[test]
    fn no_completed_trials() {
        let pruner = WilcoxonPruner::new(Direction::Minimize);
        let current: Vec<(u64, f64)> = (0..20).map(|s| (s, 1.0)).collect();
        assert!(!pruner.should_prune(1, 19, &current, &[]));
    }
}
