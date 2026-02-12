//! Error types for the optimizer crate.
//!
//! All fallible operations in the crate return [`Result<T>`], which is an
//! alias for `core::result::Result<T, Error>`. The [`Error`] enum covers
//! parameter validation, sampling conflicts, pruning signals, and
//! feature-gated I/O errors.

/// Errors returned by optimizer operations.
///
/// Most variants are returned during parameter validation or trial
/// management. The [`TrialPruned`](Error::TrialPruned) variant has special
/// significance — it signals early stopping and is typically raised via
/// the [`TrialPruned`](super::TrialPruned) convenience type.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The lower bound exceeds the upper bound in a
    /// [`FloatParam`](crate::parameter::FloatParam) or
    /// [`IntParam`](crate::parameter::IntParam).
    #[error("invalid bounds: low ({low}) must be less than or equal to high ({high})")]
    InvalidBounds {
        /// The lower bound value.
        low: f64,
        /// The upper bound value.
        high: f64,
    },

    /// Log-scale is enabled but the lower bound is not positive (float) or
    /// is less than 1 (integer).
    #[error("invalid log bounds: low must be positive for log scale")]
    InvalidLogBounds,

    /// The step size provided to a parameter is not positive.
    #[error("invalid step: step must be positive")]
    InvalidStep,

    /// A [`CategoricalParam`](crate::parameter::CategoricalParam) was created
    /// with an empty choices vector.
    #[error("categorical choices cannot be empty")]
    EmptyChoices,

    /// The same [`ParamId`](crate::parameter::ParamId) was suggested twice
    /// with a different distribution configuration.
    #[error("parameter conflict for '{name}': {reason}")]
    ParameterConflict {
        /// The name of the conflicting parameter.
        name: String,
        /// The reason for the conflict.
        reason: String,
    },

    /// [`Study::best_trial`](crate::Study::best_trial) or similar was called
    /// before any trial completed successfully.
    #[error("no completed trials available")]
    NoCompletedTrials,

    /// The gamma value for TPE sampling is outside the open interval (0, 1).
    #[error("invalid gamma: {0} must be in (0.0, 1.0)")]
    InvalidGamma(f64),

    /// A KDE bandwidth value is not positive.
    #[error("invalid bandwidth: {0} must be positive")]
    InvalidBandwidth(f64),

    /// A kernel density estimator was constructed with no samples.
    #[error("KDE requires at least one sample")]
    EmptySamples,

    /// Multivariate KDE samples have zero dimensions.
    #[error("multivariate KDE samples must have at least one dimension")]
    ZeroDimensions,

    /// A sample in the multivariate KDE has a different number of dimensions
    /// than the first sample.
    #[error(
        "dimension mismatch: expected {expected} dimensions but sample {sample_index} has {got}"
    )]
    DimensionMismatch {
        /// The expected number of dimensions.
        expected: usize,
        /// The actual number of dimensions in the sample.
        got: usize,
        /// The index of the sample with mismatched dimensions.
        sample_index: usize,
    },

    /// The bandwidth vector length does not match the number of KDE dimensions.
    #[error("bandwidth dimension mismatch: expected {expected} bandwidths but got {got}")]
    BandwidthDimensionMismatch {
        /// The expected number of bandwidths.
        expected: usize,
        /// The actual number of bandwidths provided.
        got: usize,
    },

    /// The objective signalled that this trial should be pruned (stopped
    /// early). Typically raised via `Err(TrialPruned)?` inside the
    /// objective closure.
    #[error("trial was pruned")]
    TrialPruned,

    /// The multi-objective closure returned a different number of values
    /// than the number of directions configured on the study.
    #[error("objective dimension mismatch: expected {expected} values, got {got}")]
    ObjectiveDimensionMismatch {
        /// The expected number of objective values.
        expected: usize,
        /// The actual number of objective values returned.
        got: usize,
    },

    /// An internal invariant was violated. This indicates a bug in the
    /// library rather than a user error.
    #[error("internal error: {0}")]
    Internal(&'static str),

    /// An async worker task failed. Only available with the `async` feature.
    #[cfg(feature = "async")]
    #[error("async task error: {0}")]
    TaskError(String),

    /// A storage I/O operation failed. Only available with the `journal`
    /// feature.
    #[cfg(feature = "journal")]
    #[error("storage error: {0}")]
    Storage(String),
}

/// A convenience alias for `core::result::Result<T, Error>`.
pub type Result<T> = core::result::Result<T, Error>;

/// Convenience type for signalling a pruned trial from an objective function.
///
/// Implements `Into<Error>` so it can be used with `?` in objectives that
/// return `Result<V, Error>`.
///
/// # Examples
///
/// ```
/// use optimizer::{Error, TrialPruned};
///
/// fn objective_that_prunes() -> Result<f64, Error> {
///     // ... some computation ...
///     Err(TrialPruned)?
/// }
/// ```
#[derive(Debug)]
pub struct TrialPruned;

impl core::fmt::Display for TrialPruned {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "trial was pruned")
    }
}

impl From<TrialPruned> for Error {
    fn from(_: TrialPruned) -> Self {
        Error::TrialPruned
    }
}
