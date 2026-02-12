//! Raw parameter value storage.
//!
//! [`ParamValue`] is the type-erased representation of a sampled parameter.
//! Users rarely construct `ParamValue` directly — the
//! [`Parameter::suggest`](crate::parameter::Parameter::suggest) method returns
//! the already-typed value (e.g., `f64` for [`FloatParam`](crate::parameter::FloatParam)).
//!
//! `ParamValue` is useful when inspecting raw trial data via
//! [`Trial::params`](crate::Trial::params) or
//! [`CompletedTrial::params`](crate::sampler::CompletedTrial).

/// A type-erased sampled parameter value.
///
/// Stores float, integer, or categorical (index) values uniformly.
/// For categorical parameters the `Categorical` variant stores the
/// zero-based index into the choices array, not the choice itself.
///
/// # Display
///
/// `ParamValue` implements [`Display`](core::fmt::Display): floats and
/// integers print their numeric value, and categoricals print `category(i)`.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ParamValue {
    /// A floating-point parameter value (from [`FloatParam`](crate::parameter::FloatParam)).
    Float(f64),
    /// An integer parameter value (from [`IntParam`](crate::parameter::IntParam)).
    Int(i64),
    /// A categorical index into the choices array (from
    /// [`CategoricalParam`](crate::parameter::CategoricalParam),
    /// [`BoolParam`](crate::parameter::BoolParam), or
    /// [`EnumParam`](crate::parameter::EnumParam)).
    Categorical(usize),
}

impl core::fmt::Display for ParamValue {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Float(v) => write!(f, "{v}"),
            Self::Int(v) => write!(f, "{v}"),
            Self::Categorical(v) => write!(f, "category({v})"),
        }
    }
}
