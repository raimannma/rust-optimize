//! No-op pruner — never prune any trial.
//!
//! This is the default pruner used when no pruner is configured on a
//! [`Study`](crate::Study). It unconditionally returns `false` for every
//! pruning decision, allowing all trials to run to completion.
//!
//! # When to use
//!
//! - When you want to explicitly disable pruning
//! - As a baseline to compare against other pruners
//! - Already used by default — you rarely need to configure this manually

use super::Pruner;
use crate::sampler::CompletedTrial;

/// A pruner that never prunes. This is the default when no pruner is configured.
pub struct NopPruner;

impl Pruner for NopPruner {
    fn should_prune(
        &self,
        _trial_id: u64,
        _step: u64,
        _intermediate_values: &[(u64, f64)],
        _completed_trials: &[CompletedTrial],
    ) -> bool {
        false
    }
}
