//! Pruner trait and implementations for trial pruning.
//!
//! Pruners decide whether to stop (prune) a trial early based on its
//! intermediate values compared to other trials. This is useful for
//! discarding unpromising trials before they complete, saving compute.
//!
//! # How pruning works
//!
//! During optimization, each trial reports intermediate values at discrete
//! steps (e.g., validation loss after each training epoch). A pruner inspects
//! these values and compares them against completed trials to decide whether
//! the current trial should be stopped early.
//!
//! The typical flow is:
//!
//! 1. Call [`Trial::report`](crate::Trial::report) to record an intermediate value.
//! 2. Call [`Trial::should_prune`](crate::Trial::should_prune) to check the pruner's decision.
//! 3. If the pruner says prune, return [`TrialPruned`](crate::TrialPruned) from the objective.
//!
//! # Available pruners
//!
//! | Pruner | Algorithm | Best for |
//! |--------|-----------|----------|
//! | [`MedianPruner`] | Prune below median at each step | General-purpose default |
//! | [`PercentilePruner`] | Prune below configurable percentile | Tunable aggressiveness |
//! | [`ThresholdPruner`] | Prune outside fixed bounds | Known divergence limits |
//! | [`PatientPruner`] | Require N consecutive prune signals | Noisy intermediate values |
//! | [`SuccessiveHalvingPruner`] | Keep top 1/η fraction at each rung | Budget-aware pruning |
//! | [`HyperbandPruner`] | Multiple SHA brackets with different budgets | Robust to budget choice |
//! | [`WilcoxonPruner`] | Statistical signed-rank test vs. best trial | Rigorous noisy pruning |
//! | [`NopPruner`] | Never prune | Disabling pruning explicitly |
//!
//! # When to use pruning
//!
//! Pruning is most beneficial when:
//!
//! - The objective function has a natural notion of "steps" (e.g., training epochs)
//! - Early steps are informative about final performance
//! - Trials are expensive enough that stopping bad ones early saves significant time
//!
//! Start with [`MedianPruner`] for most use cases. Switch to [`WilcoxonPruner`]
//! if your intermediate values are noisy, or to [`HyperbandPruner`] if you want
//! automatic budget scheduling.

mod hyperband;
mod median;
mod nop;
mod patient;
pub(crate) mod percentile;
mod successive_halving;
mod threshold;
mod wilcoxon;

pub use hyperband::HyperbandPruner;
pub use median::MedianPruner;
pub use nop::NopPruner;
pub use patient::PatientPruner;
pub use percentile::PercentilePruner;
pub use successive_halving::SuccessiveHalvingPruner;
pub use threshold::ThresholdPruner;
pub use wilcoxon::WilcoxonPruner;

use crate::sampler::CompletedTrial;

/// Trait for pluggable trial pruning strategies.
///
/// Pruners are consulted after each intermediate value is reported to
/// decide whether the trial should be stopped early. The trait requires
/// `Send + Sync` to support concurrent and async optimization.
///
/// # Implementing a custom pruner
///
/// ```
/// use optimizer::pruner::Pruner;
/// use optimizer::sampler::CompletedTrial;
///
/// struct MyPruner {
///     threshold: f64,
/// }
///
/// impl Pruner for MyPruner {
///     fn should_prune(
///         &self,
///         _trial_id: u64,
///         _step: u64,
///         intermediate_values: &[(u64, f64)],
///         _completed_trials: &[CompletedTrial],
///     ) -> bool {
///         // Prune if the latest value exceeds the threshold
///         intermediate_values
///             .last()
///             .is_some_and(|&(_, v)| v > self.threshold)
///     }
/// }
/// ```
pub trait Pruner: Send + Sync {
    /// Decide whether to prune a trial at the given step.
    ///
    /// # Arguments
    ///
    /// * `trial_id` - The current trial's ID.
    /// * `step` - The step at which the intermediate value was reported.
    /// * `intermediate_values` - All `(step, value)` pairs reported so far for this trial.
    /// * `completed_trials` - History of all completed trials (for comparison).
    fn should_prune(
        &self,
        trial_id: u64,
        step: u64,
        intermediate_values: &[(u64, f64)],
        completed_trials: &[CompletedTrial],
    ) -> bool;
}
