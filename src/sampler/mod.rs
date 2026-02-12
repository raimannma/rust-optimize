//! Sampler trait and implementations for parameter sampling.
//!
//! A sampler generates parameter values for each trial. It receives a
//! [`Distribution`] describing the parameter space, a monotonically increasing
//! `trial_id`, and the list of all [`CompletedTrial`]s so far, and returns a
//! [`ParamValue`] that matches the distribution variant.
//!
//! # Available samplers
//!
//! ## Single-objective
//!
//! | Sampler | Algorithm | Best for |
//! |---------|-----------|----------|
//! | [`RandomSampler`] | Uniform independent sampling | Baselines, startup phases |
//! | [`TpeSampler`] | Tree-Parzen Estimator | General-purpose Bayesian optimization |
//! | [`TpeSampler`] (multivariate) | Multivariate TPE with tree-structured Parzen | Correlated parameters |
//! | [`GridSampler`] | Exhaustive grid evaluation | Small discrete spaces |
//! | [`SobolSampler`]\* | Quasi-random Sobol sequences | Uniform coverage without model |
//! | [`CmaEsSampler`]\* | Covariance Matrix Adaptation | Continuous, non-separable problems |
//! | [`GpSampler`]\* | Gaussian Process with EI | Expensive, low-dimensional functions |
//! | [`DESampler`] | Differential Evolution | Population-based, multi-modal landscapes |
//! | [`BohbSampler`] | Bayesian Optimization + `HyperBand` | Combined sampling and pruning |
//!
//! \*Requires a feature flag (`sobol`, `cma-es`, or `gp`).
//!
//! ## Multi-objective
//!
//! | Sampler | Algorithm | Best for |
//! |---------|-----------|----------|
//! | [`Nsga2Sampler`] | NSGA-II | General multi-objective with 2-3 objectives |
//! | [`Nsga3Sampler`] | NSGA-III | Many-objective (4+ objectives) |
//! | [`MoeadSampler`] | MOEA/D with decomposition | Structured Pareto front exploration |
//! | [`MotpeSampler`] | Multi-objective TPE | Bayesian multi-objective |
//!
//! # Implementing a custom sampler
//!
//! Implement the [`Sampler`] trait with its single method:
//!
//! ```rust
//! use optimizer::sampler::{Sampler, CompletedTrial};
//! use optimizer::distribution::Distribution;
//! use optimizer::param::ParamValue;
//!
//! /// A sampler that always picks the midpoint of each distribution.
//! struct MidpointSampler;
//!
//! impl Sampler for MidpointSampler {
//!     fn sample(
//!         &self,
//!         distribution: &Distribution,
//!         _trial_id: u64,
//!         _history: &[CompletedTrial],
//!     ) -> ParamValue {
//!         match distribution {
//!             Distribution::Float(fd) => {
//!                 ParamValue::Float((fd.low + fd.high) / 2.0)
//!             }
//!             Distribution::Int(id) => {
//!                 ParamValue::Int((id.low + id.high) / 2)
//!             }
//!             Distribution::Categorical(cd) => {
//!                 ParamValue::Categorical(cd.n_choices / 2)
//!             }
//!         }
//!     }
//! }
//! ```
//!
//! The arguments to [`Sampler::sample`]:
//!
//! - **`distribution`** — a [`Distribution::Float`], [`Distribution::Int`], or
//!   [`Distribution::Categorical`] that describes the parameter bounds,
//!   log-scale flag, and optional step size.
//! - **`trial_id`** — a monotonically increasing identifier. Useful for
//!   deterministic RNG seeding (see [Stateless vs stateful samplers]).
//! - **`history`** — all completed trials so far. May be empty on the first
//!   trial. Model-based samplers use this to guide future sampling.
//! - **Return value** — the [`ParamValue`] variant *must* match the
//!   distribution variant (`Float` → `ParamValue::Float`, etc.).
//!
//! [Stateless vs stateful samplers]: #stateless-vs-stateful-samplers
//!
//! # Stateless vs stateful samplers
//!
//! **Stateless** samplers derive all randomness from a deterministic function
//! of `seed + trial_id + distribution`. They use an [`AtomicU64`] call-sequence
//! counter to disambiguate multiple calls within the same trial, but need no
//! `Mutex`. See [`RandomSampler`] and [`TpeSampler`] for this pattern.
//!
//! **Stateful** samplers maintain mutable state (e.g. a population pool)
//! across calls. Wrap mutable state in `parking_lot::Mutex<State>` and lock
//! for the duration of [`Sampler::sample`]. See [`DESampler`] and
//! [`GridSampler`] for this pattern.
//!
//! [`AtomicU64`]: core::sync::atomic::AtomicU64
//!
//! # Cold start handling
//!
//! Model-based samplers need completed trials before their surrogate model is
//! useful. The standard pattern is to check `history.len() < n_startup_trials`
//! and fall back to random sampling during the startup phase. Expose this as a
//! builder parameter so users can tune the trade-off between exploration and
//! exploitation. See [`TpeSampler`] for a reference implementation.
//!
//! # Reading trial history
//!
//! The `history` slice contains only completed trials (never pending ones).
//! Common operations:
//!
//! - **Extract a parameter value:**
//!   `trial.params.get(&param_id)` returns `Option<&ParamValue>`.
//! - **Find the best trial:**
//!   `history.iter().min_by(|a, b| a.value.partial_cmp(&b.value).unwrap())`.
//! - **Filter by state:**
//!   `history.iter().filter(|t| t.state == TrialState::Complete)`.
//! - **Check feasibility:**
//!   `trial.is_feasible()` returns `true` when all constraints are ≤ 0.
//!
//! # Thread safety
//!
//! The [`Sampler`] trait requires `Send + Sync`. [`Study`](crate::Study) stores
//! the sampler as `Arc<dyn Sampler>`, so multiple threads may call
//! [`Sampler::sample`] concurrently.
//!
//! - **Stateless:** `AtomicU64` counters satisfy `Send + Sync` without locking.
//! - **Stateful:** use `parking_lot::Mutex` (the crate convention) or
//!   `std::sync::Mutex` to protect mutable state.
//!
//! # Testing custom samplers
//!
//! Recommended test categories:
//!
//! 1. **Bounds compliance** — sample many values and assert they fall within
//!    the distribution range.
//! 2. **Step / log-scale correctness** — verify that discretized and
//!    log-scaled distributions produce valid values.
//! 3. **Reproducibility** — the same seed must produce the same output.
//! 4. **History sensitivity** — model-based samplers should produce different
//!    (better) samples as history grows.
//! 5. **Empty history** — `sample()` must not panic when `history` is empty.
//!
//! # Using a custom sampler with Study
//!
//! ```rust
//! use optimizer::{Direction, Study};
//! use optimizer::sampler::{Sampler, CompletedTrial};
//! use optimizer::distribution::Distribution;
//! use optimizer::param::ParamValue;
//!
//! struct MySampler;
//! impl Sampler for MySampler {
//!     fn sample(
//!         &self,
//!         distribution: &Distribution,
//!         _trial_id: u64,
//!         _history: &[CompletedTrial],
//!     ) -> ParamValue {
//!         match distribution {
//!             Distribution::Float(fd) => ParamValue::Float(fd.low),
//!             Distribution::Int(id) => ParamValue::Int(id.low),
//!             Distribution::Categorical(_) => ParamValue::Categorical(0),
//!         }
//!     }
//! }
//!
//! let study: Study<f64> = Study::with_sampler(Direction::Minimize, MySampler);
//! ```
//!
//! The sampler is wrapped in `Arc<dyn Sampler>` internally.
//!
//! # Reference implementations
//!
//! - [`RandomSampler`] — simplest sampler; stateless, ignores history.
//! - [`TpeSampler`] — model-based with cold start fallback.
//! - [`DESampler`] — stateful, population-based.
//! - [`GridSampler`] — deterministic, exhaustive search.

pub mod bohb;
#[cfg(feature = "cma-es")]
pub mod cma_es;
pub(crate) mod common;
pub mod de;
pub(crate) mod genetic;
#[cfg(feature = "gp")]
pub mod gp;
pub mod grid;
pub mod moead;
pub mod motpe;
pub mod nsga2;
pub mod nsga3;
pub mod random;
#[cfg(feature = "sobol")]
pub mod sobol;
pub mod tpe;

use std::collections::HashMap;

pub use bohb::BohbSampler;
#[cfg(feature = "cma-es")]
pub use cma_es::CmaEsSampler;
pub use de::{DESampler, DEStrategy};
#[cfg(feature = "gp")]
pub use gp::GpSampler;
pub use grid::GridSampler;
pub use moead::{Decomposition, MoeadSampler};
pub use motpe::MotpeSampler;
pub use nsga2::Nsga2Sampler;
pub use nsga3::Nsga3Sampler;
pub use random::RandomSampler;
#[cfg(feature = "sobol")]
pub use sobol::SobolSampler;
pub use tpe::TpeSampler;

use crate::distribution::Distribution;
use crate::param::ParamValue;
use crate::parameter::{ParamId, Parameter};
use crate::trial::AttrValue;
use crate::types::TrialState;

/// A completed trial with its parameters, distributions, and objective value.
///
/// This struct stores the results of a completed trial, including all sampled
/// parameter values, their distributions, and the objective value returned
/// by the objective function.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CompletedTrial<V = f64> {
    /// The unique identifier for this trial.
    pub id: u64,
    /// The sampled parameter values, keyed by parameter id.
    pub params: HashMap<ParamId, ParamValue>,
    /// The parameter distributions used, keyed by parameter id.
    pub distributions: HashMap<ParamId, Distribution>,
    /// Human-readable labels for parameters, keyed by parameter id.
    pub param_labels: HashMap<ParamId, String>,
    /// The objective value returned by the objective function.
    pub value: V,
    /// Intermediate objective values reported during the trial.
    pub intermediate_values: Vec<(u64, f64)>,
    /// The state of the trial (Complete, Pruned, or Failed).
    pub state: TrialState,
    /// User-defined attributes stored during the trial.
    pub user_attrs: HashMap<String, AttrValue>,
    /// Constraint values for this trial (<=0.0 means feasible).
    #[cfg_attr(feature = "serde", serde(default))]
    pub constraints: Vec<f64>,
}

impl<V> CompletedTrial<V> {
    /// Creates a new completed trial.
    pub fn new(
        id: u64,
        params: HashMap<ParamId, ParamValue>,
        distributions: HashMap<ParamId, Distribution>,
        param_labels: HashMap<ParamId, String>,
        value: V,
    ) -> Self {
        Self {
            id,
            params,
            distributions,
            param_labels,
            value,
            intermediate_values: Vec::new(),
            state: TrialState::Complete,
            user_attrs: HashMap::new(),
            constraints: Vec::new(),
        }
    }

    /// Creates a new completed trial with intermediate values and user attributes.
    pub fn with_intermediate_values(
        id: u64,
        params: HashMap<ParamId, ParamValue>,
        distributions: HashMap<ParamId, Distribution>,
        param_labels: HashMap<ParamId, String>,
        value: V,
        intermediate_values: Vec<(u64, f64)>,
        user_attrs: HashMap<String, AttrValue>,
    ) -> Self {
        Self {
            id,
            params,
            distributions,
            param_labels,
            value,
            intermediate_values,
            state: TrialState::Complete,
            user_attrs,
            constraints: Vec::new(),
        }
    }

    /// Returns the typed value for the given parameter.
    ///
    /// Looks up the parameter by its unique id and casts the stored
    /// [`ParamValue`] to the parameter's typed value.
    ///
    /// Returns `None` if the parameter was not used in this trial or if
    /// the stored value is incompatible with the parameter type (e.g., a
    /// `Float` value stored for an `IntParam`).
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::parameter::{FloatParam, Parameter};
    /// use optimizer::{Direction, Study};
    ///
    /// let study: Study<f64> = Study::new(Direction::Minimize);
    /// let x = FloatParam::new(-10.0, 10.0);
    ///
    /// study
    ///     .optimize(5, |trial: &mut optimizer::Trial| {
    ///         let val = x.suggest(trial)?;
    ///         Ok::<_, optimizer::Error>(val * val)
    ///     })
    ///     .unwrap();
    ///
    /// let best = study.best_trial().unwrap();
    /// let x_val: f64 = best.get(&x).unwrap();
    /// assert!((-10.0..=10.0).contains(&x_val));
    /// ```
    pub fn get<P: Parameter>(&self, param: &P) -> Option<P::Value> {
        self.params
            .get(&param.id())
            .and_then(|v| param.cast_param_value(v).ok())
    }

    /// Returns `true` if all constraints are satisfied (values <= 0.0).
    ///
    /// A trial with no constraints is considered feasible.
    #[must_use]
    pub fn is_feasible(&self) -> bool {
        self.constraints.iter().all(|&c| c <= 0.0)
    }

    /// Gets a user attribute by key.
    #[must_use]
    pub fn user_attr(&self, key: &str) -> Option<&AttrValue> {
        self.user_attrs.get(key)
    }

    /// Returns all user attributes.
    #[must_use]
    pub fn user_attrs(&self) -> &HashMap<String, AttrValue> {
        &self.user_attrs
    }

    /// Validates that all floating-point fields are finite (not NaN or
    /// Infinity).
    ///
    /// Checks distribution bounds, parameter values, constraints, and
    /// intermediate values.  Returns a description of the first invalid
    /// field found, or `Ok(())` if everything is valid.
    ///
    /// # Errors
    ///
    /// Returns a `String` describing the first non-finite value found.
    pub fn validate(&self) -> core::result::Result<(), String> {
        for (id, dist) in &self.distributions {
            if let Distribution::Float(fd) = dist {
                if !fd.low.is_finite() {
                    return Err(format!(
                        "trial {}: float distribution for param {id} has non-finite low bound ({})",
                        self.id, fd.low
                    ));
                }
                if !fd.high.is_finite() {
                    return Err(format!(
                        "trial {}: float distribution for param {id} has non-finite high bound ({})",
                        self.id, fd.high
                    ));
                }
                if let Some(step) = fd.step
                    && !step.is_finite()
                {
                    return Err(format!(
                        "trial {}: float distribution for param {id} has non-finite step ({step})",
                        self.id
                    ));
                }
            }
        }

        for (id, pv) in &self.params {
            if let ParamValue::Float(v) = pv
                && !v.is_finite()
            {
                return Err(format!(
                    "trial {}: param {id} has non-finite float value ({v})",
                    self.id
                ));
            }
        }

        for (i, &c) in self.constraints.iter().enumerate() {
            if !c.is_finite() {
                return Err(format!(
                    "trial {}: constraint[{i}] is non-finite ({c})",
                    self.id
                ));
            }
        }

        for &(step, v) in &self.intermediate_values {
            if !v.is_finite() {
                return Err(format!(
                    "trial {}: intermediate value at step {step} is non-finite ({v})",
                    self.id
                ));
            }
        }

        Ok(())
    }
}

/// A pending (running) trial with its parameters and distributions, but no objective value yet.
///
/// This struct represents a trial that has been started and has sampled parameters,
/// but is still running and hasn't returned an objective value. It is used with the
/// constant liar strategy for parallel optimization.
#[derive(Clone, Debug)]
pub struct PendingTrial {
    /// The unique identifier for this trial.
    pub id: u64,
    /// The sampled parameter values, keyed by parameter id.
    pub params: HashMap<ParamId, ParamValue>,
    /// The parameter distributions used, keyed by parameter id.
    pub distributions: HashMap<ParamId, Distribution>,
    /// Human-readable labels for parameters, keyed by parameter id.
    pub param_labels: HashMap<ParamId, String>,
}

impl PendingTrial {
    /// Creates a new pending trial.
    #[must_use]
    pub fn new(
        id: u64,
        params: HashMap<ParamId, ParamValue>,
        distributions: HashMap<ParamId, Distribution>,
        param_labels: HashMap<ParamId, String>,
    ) -> Self {
        Self {
            id,
            params,
            distributions,
            param_labels,
        }
    }
}

/// Trait for pluggable parameter sampling strategies.
///
/// Samplers are responsible for generating parameter values based on
/// the distribution and historical trial data. The trait requires
/// `Send + Sync` to support concurrent and async optimization.
///
/// # Implementing a custom sampler
///
/// ```
/// use optimizer::sampler::{Sampler, CompletedTrial};
/// use optimizer::distribution::Distribution;
/// use optimizer::param::ParamValue;
///
/// struct NoisySampler {
///     noise_scale: f64,
///     seed: u64,
/// }
///
/// impl Sampler for NoisySampler {
///     fn sample(
///         &self,
///         distribution: &Distribution,
///         trial_id: u64,
///         history: &[CompletedTrial],
///     ) -> ParamValue {
///         // Find the best value seen so far, or fall back to the midpoint
///         match distribution {
///             Distribution::Float(fd) => {
///                 let center = if history.is_empty() {
///                     (fd.low + fd.high) / 2.0
///                 } else {
///                     history.iter()
///                         .filter_map(|t| t.params.values().next())
///                         .filter_map(|v| if let ParamValue::Float(f) = v { Some(*f) } else { None })
///                         .next()
///                         .unwrap_or((fd.low + fd.high) / 2.0)
///                 };
///                 let noise = (trial_id as f64 * 0.1).sin() * self.noise_scale;
///                 ParamValue::Float(center + noise)
///             }
///             Distribution::Int(id) => ParamValue::Int((id.low + id.high) / 2),
///             Distribution::Categorical(cd) => ParamValue::Categorical(trial_id as usize % cd.n_choices),
///         }
///     }
/// }
/// ```
///
/// See the [module-level documentation](self) for a comprehensive guide
/// covering cold start handling, thread safety patterns, and testing.
pub trait Sampler: Send + Sync {
    /// Samples a parameter value from the given distribution.
    ///
    /// # Arguments
    ///
    /// * `distribution` - The parameter distribution to sample from.
    /// * `trial_id` - The unique ID of the trial being sampled for.
    /// * `history` - Historical completed trials for informed sampling.
    ///
    /// # Returns
    ///
    /// A `ParamValue` sampled from the distribution.
    fn sample(
        &self,
        distribution: &Distribution,
        trial_id: u64,
        history: &[CompletedTrial],
    ) -> ParamValue;
}
