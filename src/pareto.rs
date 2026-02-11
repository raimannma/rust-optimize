//! Pareto dominance utilities for multi-objective optimization.
//!
//! Provides fast non-dominated sorting (Deb et al., 2002) and crowding
//! distance computation used by both `MultiObjectiveStudy::pareto_front()`
//! and `Nsga2Sampler`.

use crate::types::Direction;

/// Returns `true` if solution `a` Pareto-dominates solution `b`.
///
/// A solution dominates another if it is at least as good in all objectives
/// and strictly better in at least one, respecting the given directions.
#[allow(clippy::module_name_repetitions)]
pub(crate) fn dominates(a: &[f64], b: &[f64], directions: &[Direction]) -> bool {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), directions.len());

    let mut strictly_better = false;
    for ((&av, &bv), dir) in a.iter().zip(b.iter()).zip(directions.iter()) {
        let better = match dir {
            Direction::Minimize => av < bv,
            Direction::Maximize => av > bv,
        };
        let worse = match dir {
            Direction::Minimize => av > bv,
            Direction::Maximize => av < bv,
        };
        if worse {
            return false;
        }
        if better {
            strictly_better = true;
        }
    }
    strictly_better
}

/// Constrained dominance: feasible beats infeasible, among infeasible
/// prefer lower total constraint violation, among feasible use Pareto dominance.
pub(crate) fn constrained_dominates(
    a_values: &[f64],
    b_values: &[f64],
    a_constraints: &[f64],
    b_constraints: &[f64],
    directions: &[Direction],
) -> bool {
    let a_feasible = a_constraints.iter().all(|&c| c <= 0.0);
    let b_feasible = b_constraints.iter().all(|&c| c <= 0.0);

    match (a_feasible, b_feasible) {
        (true, false) => true,
        (false, true) => false,
        (false, false) => {
            let a_violation: f64 = a_constraints.iter().map(|c| c.max(0.0)).sum();
            let b_violation: f64 = b_constraints.iter().map(|c| c.max(0.0)).sum();
            a_violation < b_violation
        }
        (true, true) => dominates(a_values, b_values, directions),
    }
}

/// Fast non-dominated sorting (Deb et al., 2002).
///
/// Returns `Vec<Vec<usize>>` where `fronts[0]` is the Pareto front,
/// each inner vec contains indices into `values`.
///
/// Complexity: O(M * N^2) where M = objectives, N = solutions.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn fast_non_dominated_sort(
    values: &[Vec<f64>],
    directions: &[Direction],
) -> Vec<Vec<usize>> {
    fast_non_dominated_sort_constrained(values, directions, &[])
}

/// Fast non-dominated sorting with constraint support.
///
/// `constraints` is either empty (no constraints) or has the same length
/// as `values`, where each entry is the constraint vector for that solution.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn fast_non_dominated_sort_constrained(
    values: &[Vec<f64>],
    directions: &[Direction],
    constraints: &[Vec<f64>],
) -> Vec<Vec<usize>> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }

    let has_constraints = !constraints.is_empty();
    let empty_constraints: Vec<f64> = Vec::new();

    // S_p: set of solutions dominated by p
    let mut dominated_by: Vec<Vec<usize>> = vec![Vec::new(); n];
    // n_p: domination count for p
    let mut domination_count: Vec<usize> = vec![0; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let (a_c, b_c) = if has_constraints {
                (&constraints[i], &constraints[j])
            } else {
                (&empty_constraints, &empty_constraints)
            };

            let i_dom_j = if has_constraints {
                constrained_dominates(&values[i], &values[j], a_c, b_c, directions)
            } else {
                dominates(&values[i], &values[j], directions)
            };
            let j_dom_i = if has_constraints {
                constrained_dominates(&values[j], &values[i], b_c, a_c, directions)
            } else {
                dominates(&values[j], &values[i], directions)
            };

            if i_dom_j {
                dominated_by[i].push(j);
                domination_count[j] += 1;
            } else if j_dom_i {
                dominated_by[j].push(i);
                domination_count[i] += 1;
            }
        }
    }

    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut current_front: Vec<usize> = (0..n).filter(|&i| domination_count[i] == 0).collect();

    while !current_front.is_empty() {
        let mut next_front: Vec<usize> = Vec::new();
        for &p in &current_front {
            for &q in &dominated_by[p] {
                domination_count[q] -= 1;
                if domination_count[q] == 0 {
                    next_front.push(q);
                }
            }
        }
        fronts.push(current_front);
        current_front = next_front;
    }

    fronts
}

/// Crowding distance for one front.
///
/// Boundary solutions get `f64::INFINITY`. Returns one distance value per
/// solution in the front, in the same order as `front_indices`.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn crowding_distance(front_indices: &[usize], values: &[Vec<f64>]) -> Vec<f64> {
    let n = front_indices.len();
    if n <= 2 {
        return vec![f64::INFINITY; n];
    }

    let m = values[front_indices[0]].len(); // number of objectives
    let mut distances = vec![0.0_f64; n];

    // Helper to look up objective value for a front member.
    let val = |front_pos: usize, obj: usize| -> f64 { values[front_indices[front_pos]][obj] };

    for obj in 0..m {
        // Sort front positions by this objective
        let mut sorted: Vec<usize> = (0..n).collect();
        sorted.sort_by(|&a, &b| {
            val(a, obj)
                .partial_cmp(&val(b, obj))
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        // Boundary solutions get infinity
        distances[sorted[0]] = f64::INFINITY;
        distances[sorted[n - 1]] = f64::INFINITY;

        let range = val(sorted[n - 1], obj) - val(sorted[0], obj);
        if range > 0.0 {
            for i in 1..(n - 1) {
                distances[sorted[i]] += (val(sorted[i + 1], obj) - val(sorted[i - 1], obj)) / range;
            }
        }
    }

    distances
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dominates_basic() {
        let dirs = [Direction::Minimize, Direction::Minimize];
        assert!(dominates(&[1.0, 1.0], &[2.0, 2.0], &dirs));
        assert!(!dominates(&[2.0, 2.0], &[1.0, 1.0], &dirs));
        // Equal does not dominate
        assert!(!dominates(&[1.0, 1.0], &[1.0, 1.0], &dirs));
    }

    #[test]
    fn test_dominates_incomparable() {
        let dirs = [Direction::Minimize, Direction::Minimize];
        assert!(!dominates(&[1.0, 3.0], &[3.0, 1.0], &dirs));
        assert!(!dominates(&[3.0, 1.0], &[1.0, 3.0], &dirs));
    }

    #[test]
    fn test_dominates_maximize() {
        let dirs = [Direction::Maximize, Direction::Minimize];
        // a = (5, 1) vs b = (3, 2): a is better in both
        assert!(dominates(&[5.0, 1.0], &[3.0, 2.0], &dirs));
        assert!(!dominates(&[3.0, 2.0], &[5.0, 1.0], &dirs));
    }

    #[test]
    fn test_nds_known() {
        let values = vec![
            vec![1.0, 5.0], // front 0
            vec![5.0, 1.0], // front 0
            vec![3.0, 3.0], // front 0 (non-dominated)
            vec![4.0, 4.0], // front 1 (dominated by #2)
            vec![6.0, 6.0], // front 2
        ];
        let dirs = [Direction::Minimize, Direction::Minimize];
        let fronts = fast_non_dominated_sort(&values, &dirs);

        assert_eq!(fronts.len(), 3);
        let mut f0 = fronts[0].clone();
        f0.sort_unstable();
        assert_eq!(f0, vec![0, 1, 2]);
        assert_eq!(fronts[1], vec![3]);
        assert_eq!(fronts[2], vec![4]);
    }

    #[test]
    fn test_crowding_boundaries() {
        let values = vec![vec![1.0, 5.0], vec![3.0, 3.0], vec![5.0, 1.0]];
        let front = vec![0, 1, 2];
        let cd = crowding_distance(&front, &values);
        assert!(cd[0].is_infinite());
        assert!(cd[2].is_infinite());
        assert!(cd[1].is_finite());
        assert!(cd[1] > 0.0);
    }
}
