//! Parameter value storage types.

/// Represents a sampled parameter value.
///
/// This enum stores different parameter value types uniformly.
/// For categorical parameters, the `Categorical` variant stores
/// the index into the choices array.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ParamValue {
    /// A floating-point parameter value.
    Float(f64),
    /// An integer parameter value.
    Int(i64),
    /// A categorical parameter value, stored as an index into the choices array.
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
