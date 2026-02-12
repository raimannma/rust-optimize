//! Parameter importance via Spearman rank correlation.
//!
//! Compute the absolute Spearman rank correlation between each parameter
//! and the objective value to estimate which parameters most influence
//! the outcome. This is a lightweight, non-parametric alternative to
//! [`fANOVA`](crate::fanova) that works well for monotonic relationships.
//!
//! # How it works
//!
//! 1. Rank parameter values and objective values independently
//! 2. Compute the Pearson correlation on the ranks (= Spearman ρ)
//! 3. Take the absolute value (direction of correlation is not relevant
//!    for importance)
//!
//! # When to use
//!
//! - **Quick importance check**: call
//!   [`Study::param_importance()`](crate::Study::param_importance) after
//!   optimization for a fast, interpretable ranking
//! - **Monotonic relationships**: Spearman captures monotonic (not just
//!   linear) correlations but may miss non-monotonic effects or interactions
//! - For interaction detection or non-linear importance, use
//!   [`fANOVA`](crate::fanova) instead

/// Assign average ranks to a slice of `f64` values (handles ties).
#[allow(clippy::cast_precision_loss, clippy::float_cmp)]
pub(crate) fn rank(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        // Find the run of tied values.
        let mut j = i + 1;
        while j < n && indexed[j].1 == indexed[i].1 {
            j += 1;
        }
        // Average rank for the tie group (1-based ranks).
        let avg = (i + 1..=j).sum::<usize>() as f64 / (j - i) as f64;
        for item in &indexed[i..j] {
            ranks[item.0] = avg;
        }
        i = j;
    }
    ranks
}

/// Pearson correlation coefficient on two equal-length slices.
#[allow(clippy::cast_precision_loss)]
fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom == 0.0 { 0.0 } else { cov / denom }
}

/// Spearman rank correlation (Pearson on ranks).
pub(crate) fn spearman(x: &[f64], y: &[f64]) -> f64 {
    pearson(&rank(x), &rank(y))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_no_ties() {
        let ranks = rank(&[30.0, 10.0, 20.0]);
        assert_eq!(ranks, vec![3.0, 1.0, 2.0]);
    }

    #[test]
    fn rank_with_ties() {
        let ranks = rank(&[10.0, 20.0, 20.0, 30.0]);
        assert_eq!(ranks, vec![1.0, 2.5, 2.5, 4.0]);
    }

    #[test]
    fn perfect_positive_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = spearman(&x, &y);
        assert!((r - 1.0).abs() < 1e-10);
    }

    #[test]
    fn perfect_negative_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let r = spearman(&x, &y);
        assert!((r + 1.0).abs() < 1e-10);
    }

    #[test]
    fn zero_variance_returns_zero() {
        let x = vec![5.0, 5.0, 5.0];
        let y = vec![1.0, 2.0, 3.0];
        assert!(spearman(&x, &y).abs() < f64::EPSILON);
    }
}
