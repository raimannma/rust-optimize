use crate::distribution::Distribution;

/// Generate a random `f64` in the range `[low, high)`.
#[inline]
pub(crate) fn f64_range(rng: &mut fastrand::Rng, low: f64, high: f64) -> f64 {
    low + rng.f64() * (high - low)
}

/// Combine a base seed, trial id, and distribution fingerprint into a
/// deterministic per-call seed using `MurmurHash3`'s 64-bit finalizer.
#[inline]
pub(crate) fn mix_seed(base: u64, trial_id: u64, dist_fingerprint: u64) -> u64 {
    let mut h = base
        .wrapping_mul(0xff51_afd7_ed55_8ccd)
        .wrapping_add(trial_id)
        .wrapping_mul(0xc4ce_b9fe_1a85_ec53)
        .wrapping_add(dist_fingerprint);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^= h >> 33;
    h
}

/// Stable `u64` fingerprint for a [`Distribution`], using variant tags and
/// `f64::to_bits()` for float fields so that distinct distributions within
/// the same trial produce different RNG streams.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) fn distribution_fingerprint(distribution: &Distribution) -> u64 {
    match distribution {
        Distribution::Float(d) => {
            let mut h: u64 = 1;
            h = h.wrapping_mul(31).wrapping_add(d.low.to_bits());
            h = h.wrapping_mul(31).wrapping_add(d.high.to_bits());
            h = h.wrapping_mul(31).wrapping_add(u64::from(d.log_scale));
            if let Some(step) = d.step {
                h = h.wrapping_mul(31).wrapping_add(step.to_bits());
            }
            h
        }
        Distribution::Int(d) => {
            let mut h: u64 = 2;
            h = h.wrapping_mul(31).wrapping_add(d.low as u64);
            h = h.wrapping_mul(31).wrapping_add(d.high as u64);
            h = h.wrapping_mul(31).wrapping_add(u64::from(d.log_scale));
            if let Some(step) = d.step {
                h = h.wrapping_mul(31).wrapping_add(step as u64);
            }
            h
        }
        Distribution::Categorical(d) => {
            let mut h: u64 = 3;
            h = h.wrapping_mul(31).wrapping_add(d.n_choices as u64);
            h
        }
    }
}
