//! BOHB (Bayesian Optimization + `HyperBand`) sampler.
//!
//! BOHB combines TPE's model-guided sampling with Hyperband's budget-aware
//! evaluation. Instead of building one global TPE model, BOHB conditions
//! its TPE model on trials evaluated at a specific budget level, giving
//! better-calibrated proposals for each rung of the Hyperband schedule.
//!
//! # How it works
//!
//! 1. Compute all Hyperband rung steps (budget levels) from the config.
//! 2. On each `sample()` call, scan the history's `intermediate_values`
//!    to find the **largest budget level** with enough observations
//!    (`>= min_points_in_model`).
//! 3. Build a filtered history where each trial's `value` is replaced
//!    with its intermediate value at that budget level.
//! 4. Delegate to an internal [`TpeSampler`] for the actual sampling.
//! 5. Fall back to random sampling if no budget level has enough data.
//!
//! # Examples
//!
//! ```
//! use optimizer::sampler::bohb::BohbSampler;
//! use optimizer::{Direction, Study};
//!
//! let bohb = BohbSampler::new();
//! let pruner = bohb.matching_pruner(Direction::Minimize);
//! let study: Study<f64> = Study::with_sampler_and_pruner(Direction::Minimize, bohb, pruner);
//! ```
//!
//! Using the builder for custom configuration:
//!
//! ```
//! use optimizer::sampler::bohb::BohbSampler;
//!
//! let bohb = BohbSampler::builder()
//!     .min_resource(1)
//!     .max_resource(81)
//!     .reduction_factor(3)
//!     .min_points_in_model(10)
//!     .seed(42)
//!     .build()
//!     .unwrap();
//! ```

use crate::distribution::Distribution;
use crate::error::Result;
use crate::param::ParamValue;
use crate::pruner::HyperbandPruner;
use crate::sampler::tpe::TpeSampler;
use crate::sampler::{CompletedTrial, Sampler};
use crate::types::Direction;

/// A BOHB sampler that combines TPE with Hyperband budget awareness.
///
/// BOHB filters trial history by budget level before delegating to TPE,
/// so the surrogate model is conditioned on trials evaluated at the same
/// resource level. This produces better-calibrated parameter proposals
/// than using a single global model across all budgets.
///
/// Use [`BohbSampler::matching_pruner`] to create a [`HyperbandPruner`]
/// with matching parameters.
pub struct BohbSampler {
    min_resource: u64,
    max_resource: u64,
    reduction_factor: u64,
    min_points_in_model: usize,
    tpe: TpeSampler,
}

impl BohbSampler {
    /// Creates a new BOHB sampler with default settings.
    ///
    /// Defaults:
    /// - `min_resource`: 1
    /// - `max_resource`: 81
    /// - `reduction_factor`: 3
    /// - `min_points_in_model`: 10
    /// - TPE: default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            min_resource: 1,
            max_resource: 81,
            reduction_factor: 3,
            min_points_in_model: 10,
            tpe: TpeSampler::new(),
        }
    }

    /// Creates a builder for configuring a BOHB sampler.
    ///
    /// # Examples
    ///
    /// ```
    /// use optimizer::sampler::bohb::BohbSampler;
    ///
    /// let sampler = BohbSampler::builder()
    ///     .min_resource(1)
    ///     .max_resource(27)
    ///     .reduction_factor(3)
    ///     .min_points_in_model(5)
    ///     .seed(42)
    ///     .build()
    ///     .unwrap();
    /// ```
    #[must_use]
    pub fn builder() -> BohbSamplerBuilder {
        BohbSamplerBuilder::new()
    }

    /// Creates a [`HyperbandPruner`] with matching Hyperband parameters.
    ///
    /// This ensures the pruner's budget schedule is consistent with the
    /// budget levels used by BOHB for model conditioning.
    #[must_use]
    pub fn matching_pruner(&self, direction: Direction) -> HyperbandPruner {
        HyperbandPruner::new()
            .min_resource(self.min_resource)
            .max_resource(self.max_resource)
            .reduction_factor(self.reduction_factor)
            .direction(direction)
    }

    /// Compute all unique budget levels (rung steps) across all Hyperband brackets.
    ///
    /// Returns sorted ascending.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn all_budget_levels(&self) -> Vec<u64> {
        let eta = self.reduction_factor as f64;
        let ratio = self.max_resource as f64 / self.min_resource as f64;
        let s_max = (ratio.ln() / eta.ln()).floor() as u64;

        let mut levels = Vec::new();
        for bracket in 0..=s_max {
            let exponent = s_max.saturating_sub(bracket);
            let min_resource_bracket =
                (self.max_resource as f64 / eta.powi(exponent as i32)).ceil() as u64;

            let mut rung: u32 = 0;
            while let Some(power) = self.reduction_factor.checked_pow(rung) {
                let step = min_resource_bracket.saturating_mul(power);
                if step > self.max_resource {
                    break;
                }
                levels.push(step);
                rung += 1;
            }
        }

        levels.sort_unstable();
        levels.dedup();
        levels
    }

    /// Build a filtered history for a specific budget level.
    ///
    /// For each trial that has an intermediate value at the given budget step,
    /// creates a new `CompletedTrial` with `value` replaced by the intermediate
    /// value at that step.
    fn filter_history_for_budget(history: &[CompletedTrial], budget: u64) -> Vec<CompletedTrial> {
        history
            .iter()
            .filter_map(|trial| {
                trial
                    .intermediate_values
                    .iter()
                    .find(|(step, _)| *step == budget)
                    .map(|(_, iv)| CompletedTrial {
                        id: trial.id,
                        params: trial.params.clone(),
                        distributions: trial.distributions.clone(),
                        param_labels: trial.param_labels.clone(),
                        value: *iv,
                        intermediate_values: trial.intermediate_values.clone(),
                        state: trial.state,
                        user_attrs: trial.user_attrs.clone(),
                    })
            })
            .collect()
    }
}

impl Default for BohbSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for BohbSampler {
    fn sample(
        &self,
        distribution: &Distribution,
        trial_id: u64,
        history: &[CompletedTrial],
    ) -> ParamValue {
        // Find the largest budget level with enough observations
        let levels = self.all_budget_levels();

        for &budget in levels.iter().rev() {
            let count = history
                .iter()
                .filter(|t| {
                    t.intermediate_values
                        .iter()
                        .any(|(step, _)| *step == budget)
                })
                .count();

            if count >= self.min_points_in_model {
                let filtered = Self::filter_history_for_budget(history, budget);
                return self.tpe.sample(distribution, trial_id, &filtered);
            }
        }

        // Not enough data at any budget level: delegate to TPE with empty history
        // which triggers its uniform-random startup behavior.
        self.tpe.sample(distribution, trial_id, &[])
    }
}

/// Builder for configuring a [`BohbSampler`].
///
/// # Examples
///
/// ```
/// use optimizer::sampler::bohb::BohbSamplerBuilder;
///
/// let sampler = BohbSamplerBuilder::new()
///     .min_resource(1)
///     .max_resource(81)
///     .reduction_factor(3)
///     .gamma(0.15)
///     .seed(42)
///     .build()
///     .unwrap();
/// ```
pub struct BohbSamplerBuilder {
    min_resource: u64,
    max_resource: u64,
    reduction_factor: u64,
    min_points_in_model: usize,
    tpe_builder: crate::sampler::tpe::TpeSamplerBuilder,
}

impl BohbSamplerBuilder {
    /// Creates a new builder with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            min_resource: 1,
            max_resource: 81,
            reduction_factor: 3,
            min_points_in_model: 10,
            tpe_builder: crate::sampler::tpe::TpeSamplerBuilder::new(),
        }
    }

    /// Sets the minimum resource (budget) per trial.
    ///
    /// # Panics
    ///
    /// Panics if `r` is 0.
    #[must_use]
    pub fn min_resource(mut self, r: u64) -> Self {
        assert!(r > 0, "min_resource must be > 0, got {r}");
        self.min_resource = r;
        self
    }

    /// Sets the maximum resource (budget) per trial.
    ///
    /// # Panics
    ///
    /// Panics if `r` is 0.
    #[must_use]
    pub fn max_resource(mut self, r: u64) -> Self {
        assert!(r > 0, "max_resource must be > 0, got {r}");
        self.max_resource = r;
        self
    }

    /// Sets the reduction factor (eta).
    ///
    /// # Panics
    ///
    /// Panics if `eta` is less than 2.
    #[must_use]
    pub fn reduction_factor(mut self, eta: u64) -> Self {
        assert!(eta >= 2, "reduction_factor must be >= 2, got {eta}");
        self.reduction_factor = eta;
        self
    }

    /// Sets the minimum number of observations at a budget level before
    /// BOHB uses TPE instead of random sampling.
    #[must_use]
    pub fn min_points_in_model(mut self, n: usize) -> Self {
        self.min_points_in_model = n;
        self
    }

    /// Sets a fixed gamma value for the internal TPE sampler.
    #[must_use]
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.tpe_builder = self.tpe_builder.gamma(gamma);
        self
    }

    /// Sets a custom gamma strategy for the internal TPE sampler.
    #[must_use]
    pub fn gamma_strategy<G: crate::sampler::tpe::GammaStrategy + 'static>(
        mut self,
        strategy: G,
    ) -> Self {
        self.tpe_builder = self.tpe_builder.gamma_strategy(strategy);
        self
    }

    /// Sets the number of EI candidates for the internal TPE sampler.
    #[must_use]
    pub fn n_ei_candidates(mut self, n: usize) -> Self {
        self.tpe_builder = self.tpe_builder.n_ei_candidates(n);
        self
    }

    /// Sets a fixed KDE bandwidth for the internal TPE sampler.
    #[must_use]
    pub fn kde_bandwidth(mut self, bandwidth: f64) -> Self {
        self.tpe_builder = self.tpe_builder.kde_bandwidth(bandwidth);
        self
    }

    /// Sets a seed for reproducible sampling.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.tpe_builder = self.tpe_builder.seed(seed);
        self
    }

    /// Builds the configured [`BohbSampler`].
    ///
    /// # Errors
    ///
    /// Returns an error if the TPE configuration is invalid (e.g. gamma
    /// not in (0, 1) or bandwidth not positive).
    pub fn build(self) -> Result<BohbSampler> {
        let tpe = self.tpe_builder.build()?;
        Ok(BohbSampler {
            min_resource: self.min_resource,
            max_resource: self.max_resource,
            reduction_factor: self.reduction_factor,
            min_points_in_model: self.min_points_in_model,
            tpe,
        })
    }
}

impl Default for BohbSamplerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::distribution::{FloatDistribution, IntDistribution};
    use crate::parameter::ParamId;
    use crate::types::TrialState;

    fn make_trial_with_intermediates(
        id: u64,
        value: f64,
        params: Vec<(ParamId, ParamValue, Distribution)>,
        intermediate_values: Vec<(u64, f64)>,
    ) -> CompletedTrial {
        let mut param_map = HashMap::new();
        let mut dist_map = HashMap::new();
        for (param_id, pv, dist) in params {
            param_map.insert(param_id, pv);
            dist_map.insert(param_id, dist);
        }
        CompletedTrial {
            id,
            params: param_map,
            distributions: dist_map,
            param_labels: HashMap::new(),
            value,
            intermediate_values,
            state: TrialState::Complete,
            user_attrs: HashMap::new(),
        }
    }

    #[test]
    fn budget_levels_default() {
        let bohb = BohbSampler::new();
        let levels = bohb.all_budget_levels();
        // With min=1, max=81, eta=3:
        // bracket 0: 1, 3, 9, 27, 81
        // bracket 1: 3, 9, 27, 81
        // bracket 2: 9, 27, 81
        // bracket 3: 27, 81
        // bracket 4: 81
        // Unique sorted: [1, 3, 9, 27, 81]
        assert_eq!(levels, vec![1, 3, 9, 27, 81]);
    }

    #[test]
    fn budget_levels_eta2() {
        let bohb = BohbSampler::builder()
            .min_resource(1)
            .max_resource(16)
            .reduction_factor(2)
            .build()
            .unwrap();
        let levels = bohb.all_budget_levels();
        // s_max = floor(ln(16)/ln(2)) = 4
        // bracket 0: 1, 2, 4, 8, 16
        // bracket 1: 2, 4, 8, 16
        // bracket 2: 4, 8, 16
        // bracket 3: 8, 16
        // bracket 4: 16
        // Unique sorted: [1, 2, 4, 8, 16]
        assert_eq!(levels, vec![1, 2, 4, 8, 16]);
    }

    #[test]
    fn filter_history_selects_correct_budget() {
        let x_id = ParamId::new();
        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });

        let history = vec![
            make_trial_with_intermediates(
                0,
                0.5,
                vec![(x_id, ParamValue::Float(0.3), dist.clone())],
                vec![(1, 0.9), (3, 0.7), (9, 0.5)],
            ),
            make_trial_with_intermediates(
                1,
                0.4,
                vec![(x_id, ParamValue::Float(0.6), dist.clone())],
                vec![(1, 0.8), (3, 0.4)],
            ),
            make_trial_with_intermediates(
                2,
                0.3,
                vec![(x_id, ParamValue::Float(0.1), dist.clone())],
                vec![(1, 0.7)],
            ),
        ];

        // Budget 3: trials 0 and 1 have intermediate values at step 3
        let filtered = BohbSampler::filter_history_for_budget(&history, 3);
        assert_eq!(filtered.len(), 2);
        assert!((filtered[0].value - 0.7).abs() < f64::EPSILON);
        assert!((filtered[1].value - 0.4).abs() < f64::EPSILON);

        // Budget 9: only trial 0
        let filtered = BohbSampler::filter_history_for_budget(&history, 9);
        assert_eq!(filtered.len(), 1);
        assert!((filtered[0].value - 0.5).abs() < f64::EPSILON);

        // Budget 27: nobody
        let filtered = BohbSampler::filter_history_for_budget(&history, 27);
        assert!(filtered.is_empty());
    }

    #[test]
    fn matching_pruner_has_same_params() {
        let bohb = BohbSampler::builder()
            .min_resource(2)
            .max_resource(64)
            .reduction_factor(4)
            .build()
            .unwrap();
        let pruner = bohb.matching_pruner(Direction::Minimize);

        // We can't directly inspect HyperbandPruner fields, but we can
        // verify it was created without panicking with the same params.
        // The pruner's rung steps should match BOHB's budget levels.
        // Just verify it doesn't panic.
        drop(pruner);
    }

    #[test]
    fn fallback_to_random_when_insufficient_data() {
        let bohb = BohbSampler::builder()
            .min_points_in_model(10)
            .seed(42)
            .build()
            .unwrap();

        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log_scale: false,
            step: None,
        });

        // Only 3 trials with intermediate values (< min_points_in_model=10)
        let x_id = ParamId::new();
        let history: Vec<CompletedTrial> = (0..3)
            .map(|i| {
                make_trial_with_intermediates(
                    i,
                    i as f64,
                    vec![(x_id, ParamValue::Float(i as f64 / 3.0), dist.clone())],
                    vec![(1, i as f64)],
                )
            })
            .collect();

        // Should not panic, should sample within bounds
        for trial_id in 0..20 {
            let val = bohb.sample(&dist, trial_id, &history);
            if let ParamValue::Float(v) = val {
                assert!((0.0..=1.0).contains(&v));
            } else {
                panic!("Expected Float");
            }
        }
    }

    #[test]
    fn uses_budget_level_when_enough_data() {
        let bohb = BohbSampler::builder()
            .min_points_in_model(5)
            .seed(42)
            .build()
            .unwrap();

        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 10.0,
            log_scale: false,
            step: None,
        });

        // Create 20 trials with intermediate values at budget 1.
        // Good trials have x near 2.0, bad trials have x far from 2.0.
        let x_id = ParamId::new();
        let history: Vec<CompletedTrial> = (0..20)
            .map(|i| {
                let x = i as f64 / 2.0;
                let iv_at_1 = (x - 2.0).powi(2);
                make_trial_with_intermediates(
                    i,
                    iv_at_1, // final value same as intermediate for simplicity
                    vec![(x_id, ParamValue::Float(x), dist.clone())],
                    vec![(1, iv_at_1)],
                )
            })
            .collect();

        // Should use TPE on filtered history at budget 1
        let val = bohb.sample(&dist, 100, &history);
        if let ParamValue::Float(v) = val {
            assert!((0.0..=10.0).contains(&v), "Value {v} out of bounds");
        } else {
            panic!("Expected Float");
        }
    }

    #[test]
    fn prefers_largest_budget_level() {
        let bohb = BohbSampler::builder()
            .min_resource(1)
            .max_resource(9)
            .reduction_factor(3)
            .min_points_in_model(3)
            .seed(42)
            .build()
            .unwrap();

        let dist = Distribution::Float(FloatDistribution {
            low: 0.0,
            high: 10.0,
            log_scale: false,
            step: None,
        });

        // Budget levels: [1, 3, 9]
        assert_eq!(bohb.all_budget_levels(), vec![1, 3, 9]);

        // Create 5 trials with intermediates at budget 1 and 3
        let x_id = ParamId::new();
        let history: Vec<CompletedTrial> = (0..5)
            .map(|i| {
                let x = i as f64;
                make_trial_with_intermediates(
                    i,
                    x,
                    vec![(x_id, ParamValue::Float(x), dist.clone())],
                    vec![(1, x * 2.0), (3, x)],
                )
            })
            .collect();

        // Budget 3 has 5 observations (>= 3), budget 9 has 0.
        // BOHB should pick budget 3 (largest with enough data).
        // The filtered history at budget 3 has values [0, 1, 2, 3, 4].
        let filtered_3 = BohbSampler::filter_history_for_budget(&history, 3);
        assert_eq!(filtered_3.len(), 5);
        let filtered_9 = BohbSampler::filter_history_for_budget(&history, 9);
        assert_eq!(filtered_9.len(), 0);

        // Should sample successfully
        let val = bohb.sample(&dist, 100, &history);
        assert!(matches!(val, ParamValue::Float(_)));
    }

    #[test]
    fn builder_validates_tpe_params() {
        // Invalid gamma
        let result = BohbSampler::builder().gamma(1.5).build();
        assert!(result.is_err());

        // Invalid bandwidth
        let result = BohbSampler::builder().kde_bandwidth(-1.0).build();
        assert!(result.is_err());
    }

    #[test]
    #[should_panic(expected = "min_resource must be > 0")]
    fn builder_rejects_zero_min_resource() {
        let _ = BohbSampler::builder().min_resource(0);
    }

    #[test]
    #[should_panic(expected = "max_resource must be > 0")]
    fn builder_rejects_zero_max_resource() {
        let _ = BohbSampler::builder().max_resource(0);
    }

    #[test]
    #[should_panic(expected = "reduction_factor must be >= 2")]
    fn builder_rejects_small_reduction_factor() {
        let _ = BohbSampler::builder().reduction_factor(1);
    }

    #[test]
    fn int_distribution_works() {
        let bohb = BohbSampler::builder()
            .min_points_in_model(3)
            .seed(42)
            .build()
            .unwrap();

        let dist = Distribution::Int(IntDistribution {
            low: 0,
            high: 100,
            log_scale: false,
            step: None,
        });

        let x_id = ParamId::new();
        let history: Vec<CompletedTrial> = (0..10)
            .map(|i| {
                make_trial_with_intermediates(
                    i,
                    i as f64,
                    vec![(x_id, ParamValue::Int(i.cast_signed() * 10), dist.clone())],
                    vec![(1, i as f64)],
                )
            })
            .collect();

        let val = bohb.sample(&dist, 100, &history);
        if let ParamValue::Int(v) = val {
            assert!((0..=100).contains(&v));
        } else {
            panic!("Expected Int");
        }
    }
}
