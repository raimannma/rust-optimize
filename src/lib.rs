#![forbid(unsafe_code)]
#![deny(clippy::all)]
#![deny(unreachable_pub)]
#![deny(clippy::correctness)]
#![deny(clippy::suspicious)]
#![deny(clippy::style)]
#![deny(clippy::complexity)]
#![deny(clippy::perf)]
#![deny(clippy::pedantic)]
#![deny(clippy::std_instead_of_core)]

//! Bayesian and population-based optimization library with an Optuna-like API
//! for hyperparameter tuning and black-box optimization. It ships 12 samplers
//! (from random search to CMA-ES and NSGA-III), 8 pruners, async/parallel
//! evaluation, and optional journal-based persistence — all with zero required
//! feature flags for the common case.
//!
//! # Getting Started
//!
//! Minimize a function in five lines — no feature flags needed:
//!
//! ```
//! use optimizer::prelude::*;
//!
//! let study: Study<f64> = Study::new(Direction::Minimize);
//! let x = FloatParam::new(-10.0, 10.0).name("x");
//!
//! study
//!     .optimize(50, |trial| {
//!         let v = x.suggest(trial)?;
//!         Ok::<_, Error>((v - 3.0).powi(2))
//!     })
//!     .unwrap();
//!
//! let best = study.best_trial().unwrap();
//! println!("x = {:.4}, f(x) = {:.4}", best.get(&x).unwrap(), best.value);
//! ```
//!
//! # Core Concepts
//!
//! | Type | Role |
//! |------|------|
//! | [`Study`] | Drive an optimization loop: create trials, record results, track the best. |
//! | [`Trial`] | A single evaluation of the objective function, carrying suggested parameter values. |
//! | [`Parameter`] | Define the search space — [`FloatParam`], [`IntParam`], [`CategoricalParam`], [`BoolParam`], [`EnumParam`]. |
//! | [`Sampler`](sampler::Sampler) | Strategy for choosing the next point to evaluate (TPE, CMA-ES, random, etc.). |
//! | [`Direction`] | Whether the study minimizes or maximizes the objective value. |
//!
//! # Sampler Guide
//!
//! ## Single-objective samplers
//!
//! | Sampler | Algorithm | Best for | Feature flag |
//! |---------|-----------|----------|--------------|
//! | [`RandomSampler`] | Uniform random | Baselines, high-dimensional | — |
//! | [`TpeSampler`] | Tree-Parzen Estimator | General-purpose Bayesian | — |
//! | [`GridSearchSampler`] | Exhaustive grid | Small, discrete spaces | — |
//! | [`SobolSampler`] | Sobol quasi-random sequence | Space-filling, low dimensions | `sobol` |
//! | [`CmaEsSampler`] | CMA-ES | Continuous, moderate dimensions | `cma-es` |
//! | [`GpSampler`] | Gaussian Process + EI | Expensive objectives, few trials | `gp` |
//! | [`DifferentialEvolutionSampler`] | Differential Evolution | Non-convex, population-based | — |
//! | [`BohbSampler`] | BOHB (TPE + `HyperBand`) | Budget-aware early stopping | — |
//!
//! ## Multi-objective samplers
//!
//! | Sampler | Algorithm | Best for | Feature flag |
//! |---------|-----------|----------|--------------|
//! | [`Nsga2Sampler`] | NSGA-II | 2-3 objectives | — |
//! | [`Nsga3Sampler`] | NSGA-III (reference-point) | 3+ objectives | — |
//! | [`MoeadSampler`] | MOEA/D (decomposition) | Many objectives, structured fronts | — |
//! | [`MotpeSampler`] | Multi-Objective TPE | Bayesian multi-objective | — |
//!
//! # Feature Flags
//!
//! | Flag | What it enables | Default |
//! |------|----------------|---------|
//! | `async` | Async/parallel optimization via tokio ([`Study::optimize_async`], [`Study::optimize_parallel`]) | off |
//! | `derive` | `#[derive(Categorical)]` for enum parameters | off |
//! | `serde` | `Serialize`/`Deserialize` on public types, [`Study::save`]/[`Study::load`] | off |
//! | `journal` | [`JournalStorage`] — JSONL persistence with file locking (enables `serde`) | off |
//! | `sobol` | [`SobolSampler`] — quasi-random low-discrepancy sequences | off |
//! | `cma-es` | [`CmaEsSampler`] — Covariance Matrix Adaptation Evolution Strategy | off |
//! | `gp` | [`GpSampler`] — Gaussian Process surrogate with Expected Improvement | off |
//! | `tracing` | Structured log events via [`tracing`](https://docs.rs/tracing) at key optimization points | off |

/// Emit a `tracing::info!` event when the `tracing` feature is enabled.
/// No-op otherwise.
#[cfg(feature = "tracing")]
macro_rules! trace_info {
    ($($arg:tt)*) => { tracing::info!($($arg)*) };
}

#[cfg(not(feature = "tracing"))]
macro_rules! trace_info {
    ($($arg:tt)*) => {};
}

/// Emit a `tracing::debug!` event when the `tracing` feature is enabled.
/// No-op otherwise.
#[cfg(feature = "tracing")]
macro_rules! trace_debug {
    ($($arg:tt)*) => { tracing::debug!($($arg)*) };
}

#[cfg(not(feature = "tracing"))]
macro_rules! trace_debug {
    ($($arg:tt)*) => {};
}

mod distribution;
mod error;
mod fanova;
mod importance;
mod kde;
pub mod multi_objective;
mod param;
pub mod parameter;
pub mod pareto;
pub mod pruner;
mod rng_util;
pub mod sampler;
pub mod storage;
mod study;
mod trial;
mod types;
mod visualization;

pub use error::{Error, Result, TrialPruned};
pub use fanova::{FanovaConfig, FanovaResult};
pub use multi_objective::{MultiObjectiveSampler, MultiObjectiveStudy, MultiObjectiveTrial};
#[cfg(feature = "derive")]
pub use optimizer_derive::Categorical;
pub use param::ParamValue;
pub use parameter::{
    BoolParam, Categorical, CategoricalParam, EnumParam, FloatParam, IntParam, ParamId, Parameter,
};
pub use pruner::{
    HyperbandPruner, MedianPruner, NopPruner, PatientPruner, PercentilePruner, Pruner,
    SuccessiveHalvingPruner, ThresholdPruner, WilcoxonPruner,
};
pub use sampler::CompletedTrial;
pub use sampler::bohb::BohbSampler;
#[cfg(feature = "cma-es")]
pub use sampler::cma_es::CmaEsSampler;
pub use sampler::differential_evolution::{
    DifferentialEvolutionSampler, DifferentialEvolutionStrategy,
};
#[cfg(feature = "gp")]
pub use sampler::gp::GpSampler;
pub use sampler::grid::GridSearchSampler;
pub use sampler::moead::{Decomposition, MoeadSampler};
pub use sampler::motpe::MotpeSampler;
pub use sampler::nsga2::Nsga2Sampler;
pub use sampler::nsga3::Nsga3Sampler;
pub use sampler::random::RandomSampler;
#[cfg(feature = "sobol")]
pub use sampler::sobol::SobolSampler;
pub use sampler::tpe::TpeSampler;
#[cfg(feature = "journal")]
pub use storage::JournalStorage;
pub use storage::{MemoryStorage, Storage};
#[cfg(feature = "serde")]
pub use study::StudySnapshot;
pub use study::{Study, StudyBuilder};
pub use trial::{AttrValue, Trial};
pub use types::{Direction, TrialState};
pub use visualization::generate_html_report;

/// Convenient wildcard import for the most common types.
///
/// ```
/// use optimizer::prelude::*;
/// ```
pub mod prelude {
    #[cfg(feature = "derive")]
    pub use optimizer_derive::Categorical as DeriveCategory;

    pub use crate::error::{Error, Result, TrialPruned};
    pub use crate::fanova::{FanovaConfig, FanovaResult};
    pub use crate::multi_objective::{MultiObjectiveStudy, MultiObjectiveTrial};
    pub use crate::param::ParamValue;
    pub use crate::parameter::{
        BoolParam, Categorical, CategoricalParam, EnumParam, FloatParam, IntParam, Parameter,
    };
    pub use crate::pruner::{
        HyperbandPruner, MedianPruner, NopPruner, PatientPruner, PercentilePruner, Pruner,
        SuccessiveHalvingPruner, ThresholdPruner,
    };
    pub use crate::sampler::CompletedTrial;
    pub use crate::sampler::bohb::BohbSampler;
    #[cfg(feature = "cma-es")]
    pub use crate::sampler::cma_es::CmaEsSampler;
    pub use crate::sampler::differential_evolution::{
        DifferentialEvolutionSampler, DifferentialEvolutionStrategy,
    };
    #[cfg(feature = "gp")]
    pub use crate::sampler::gp::GpSampler;
    pub use crate::sampler::grid::GridSearchSampler;
    pub use crate::sampler::moead::{Decomposition, MoeadSampler};
    pub use crate::sampler::motpe::MotpeSampler;
    pub use crate::sampler::nsga2::Nsga2Sampler;
    pub use crate::sampler::nsga3::Nsga3Sampler;
    pub use crate::sampler::random::RandomSampler;
    #[cfg(feature = "sobol")]
    pub use crate::sampler::sobol::SobolSampler;
    pub use crate::sampler::tpe::TpeSampler;
    #[cfg(feature = "journal")]
    pub use crate::storage::JournalStorage;
    pub use crate::storage::{MemoryStorage, Storage};
    #[cfg(feature = "serde")]
    pub use crate::study::StudySnapshot;
    pub use crate::study::{Study, StudyBuilder};
    pub use crate::trial::{AttrValue, Trial};
    pub use crate::types::Direction;
    pub use crate::visualization::generate_html_report;
}
