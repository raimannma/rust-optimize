use std::sync::Arc;

use parking_lot::RwLock;

use super::Storage;
use crate::sampler::CompletedTrial;

/// In-memory trial storage (the default).
///
/// This is a thin wrapper around `Arc<RwLock<Vec<CompletedTrial<V>>>>`.
pub struct MemoryStorage<V> {
    trials: Arc<RwLock<Vec<CompletedTrial<V>>>>,
}

impl<V> MemoryStorage<V> {
    /// Creates a new, empty in-memory store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            trials: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Creates an in-memory store pre-populated with `trials`.
    #[must_use]
    pub fn with_trials(trials: Vec<CompletedTrial<V>>) -> Self {
        Self {
            trials: Arc::new(RwLock::new(trials)),
        }
    }
}

impl<V> Default for MemoryStorage<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Send + Sync> Storage<V> for MemoryStorage<V> {
    fn push(&self, trial: CompletedTrial<V>) {
        self.trials.write().push(trial);
    }

    fn trials_arc(&self) -> &Arc<RwLock<Vec<CompletedTrial<V>>>> {
        &self.trials
    }
}
