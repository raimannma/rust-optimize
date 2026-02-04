//! Tree-Parzen Estimator (TPE) sampler implementation and utilities.
//!
//! This module provides TPE-based sampling for Bayesian optimization,
//! including support for intersection search space calculation.

mod gamma;
mod multivariate;
mod sampler;
pub mod search_space;

pub use gamma::{FixedGamma, GammaStrategy, HyperoptGamma, LinearGamma, SqrtGamma};
pub use multivariate::{
    ConstantLiarStrategy, MultivariateTpeSampler, MultivariateTpeSamplerBuilder,
};
pub use sampler::{TpeSampler, TpeSamplerBuilder};
pub use search_space::{GroupDecomposedSearchSpace, IntersectionSearchSpace};
