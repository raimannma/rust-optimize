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
//!
//! # Stateful vs stateless pruners
//!
//! **Stateless** pruners make their decision purely from the arguments passed
//! to [`Pruner::should_prune`] — they hold no mutable per-trial state.
//! [`MedianPruner`], [`PercentilePruner`], [`ThresholdPruner`],
//! [`WilcoxonPruner`], and [`NopPruner`] are all stateless.
//!
//! **Stateful** pruners track information across calls. [`PatientPruner`]
//! uses `Mutex<HashMap<u64, u64>>` to count consecutive prune signals per
//! trial. [`HyperbandPruner`] uses `Mutex` and `AtomicU64` for bracket
//! assignment state. When writing a stateful pruner, wrap mutable state in a
//! `Mutex` and key it by `trial_id` to keep trials independent.
//!
//! # Cold start and warmup
//!
//! Two builder parameters control when pruning begins:
//!
//! - **`n_warmup_steps`** — skip pruning before step N *within a trial*,
//!   giving the objective time to stabilize.
//! - **`n_min_trials`** — require N completed trials before pruning any trial,
//!   ensuring a meaningful comparison baseline.
//!
//! See [`MedianPruner`] for the canonical implementation of both parameters.
//! Custom pruners should expose similar knobs when applicable.
//!
//! # Composing pruners
//!
//! [`PatientPruner`] demonstrates the decorator pattern: it wraps any
//! `Box<dyn Pruner>` and adds patience logic on top. Custom pruners can use
//! the same pattern to layer multiple pruning conditions — for example,
//! combining a statistical test with a hard threshold.
//!
//! # Thread safety
//!
//! The [`Pruner`] trait requires `Send + Sync`.
//! [`Study`](crate::Study) stores the pruner as `Arc<dyn Pruner>`, so
//! multiple threads may call [`Pruner::should_prune`] concurrently.
//!
//! - **Stateless pruners** satisfy `Send + Sync` automatically.
//! - **Stateful pruners** should use `std::sync::Mutex` or
//!   `parking_lot::Mutex` to protect mutable state, keyed by `trial_id`.
//!
//! # Testing custom pruners
//!
//! Recommended test categories:
//!
//! 1. **Never-prune baseline** — empty history and early steps should not
//!    prune.
//! 2. **Known-prune scenario** — a clearly worse trial should be pruned.
//! 3. **Known-keep scenario** — a well-performing trial should survive.
//! 4. **Warmup respected** — pruning must be suppressed during warmup steps
//!    and while the minimum trial count has not been reached.
//! 5. **Per-trial independence** — stateful pruners must not leak state
//!    between different `trial_id` values.

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
///
/// A stateful pruner that tracks per-trial state with a `Mutex`:
///
/// ```
/// use std::collections::HashMap;
/// use std::sync::Mutex;
/// use optimizer::pruner::Pruner;
/// use optimizer::sampler::CompletedTrial;
///
/// /// Prune after the value worsens for `max_stale` consecutive steps.
/// struct StalePruner {
///     max_stale: u64,
///     // Per-trial: (previous_value, consecutive_stale_count)
///     state: Mutex<HashMap<u64, (f64, u64)>>,
/// }
///
/// impl StalePruner {
///     fn new(max_stale: u64) -> Self {
///         Self { max_stale, state: Mutex::new(HashMap::new()) }
///     }
/// }
///
/// impl Pruner for StalePruner {
///     fn should_prune(
///         &self,
///         trial_id: u64,
///         _step: u64,
///         intermediate_values: &[(u64, f64)],
///         _completed_trials: &[CompletedTrial],
///     ) -> bool {
///         let Some(&(_, current)) = intermediate_values.last() else {
///             return false;
///         };
///         let mut state = self.state.lock().unwrap();
///         let entry = state.entry(trial_id).or_insert((current, 0));
///         if current >= entry.0 {
///             entry.1 += 1;
///         } else {
///             entry.1 = 0;
///         }
///         entry.0 = current;
///         entry.1 >= self.max_stale
///     }
/// }
/// ```
///
/// See the [module-level documentation](self) for a comprehensive guide
/// covering warmup, composition, thread safety, and testing.
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
