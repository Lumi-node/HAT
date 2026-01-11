//! # Merge
//!
//! Trait and implementations for composing multiple points into one.
//!
//! This is one of the five primitives of ARMS:
//! `Merge: fn(points) -> point` - Compose together
//!
//! Merge is used for hierarchical composition:
//! - Chunks → Document
//! - Documents → Session
//! - Sessions → Domain
//!
//! Merge functions are pluggable - use whichever fits your use case.

use super::Point;

/// Trait for merging multiple points into one
///
/// Used for hierarchical composition and aggregation.
pub trait Merge: Send + Sync {
    /// Merge multiple points into a single point
    ///
    /// All points must have the same dimensionality.
    /// The slice must not be empty.
    fn merge(&self, points: &[Point]) -> Point;

    /// Name of this merge function (for debugging/config)
    fn name(&self) -> &'static str;
}

// ============================================================================
// IMPLEMENTATIONS
// ============================================================================

/// Mean (average) of all points
///
/// The centroid of the input points.
/// Good default for most hierarchical composition.
#[derive(Clone, Copy, Debug, Default)]
pub struct Mean;

impl Merge for Mean {
    fn merge(&self, points: &[Point]) -> Point {
        assert!(!points.is_empty(), "Cannot merge empty slice");

        let dims = points[0].dimensionality();
        let n = points.len() as f32;

        let mut result = vec![0.0; dims];
        for p in points {
            assert_eq!(
                p.dimensionality(),
                dims,
                "All points must have same dimensionality"
            );
            for (r, d) in result.iter_mut().zip(p.dims()) {
                *r += d / n;
            }
        }

        Point::new(result)
    }

    fn name(&self) -> &'static str {
        "mean"
    }
}

/// Weighted mean of points
///
/// Each point contributes proportionally to its weight.
/// Useful for recency weighting, importance weighting, etc.
#[derive(Clone, Debug)]
pub struct WeightedMean {
    weights: Vec<f32>,
}

impl WeightedMean {
    /// Create a new weighted mean with given weights
    ///
    /// Weights will be normalized (divided by sum) during merge.
    pub fn new(weights: Vec<f32>) -> Self {
        Self { weights }
    }

    /// Create with uniform weights (equivalent to Mean)
    pub fn uniform(n: usize) -> Self {
        Self {
            weights: vec![1.0; n],
        }
    }

    /// Create with recency weighting (more recent = higher weight)
    ///
    /// `decay` should be in (0, 1). Smaller = faster decay.
    /// First point is oldest, last is most recent.
    pub fn recency(n: usize, decay: f32) -> Self {
        let weights: Vec<f32> = (0..n).map(|i| decay.powi((n - 1 - i) as i32)).collect();
        Self { weights }
    }
}

impl Merge for WeightedMean {
    fn merge(&self, points: &[Point]) -> Point {
        assert!(!points.is_empty(), "Cannot merge empty slice");
        assert_eq!(
            points.len(),
            self.weights.len(),
            "Number of points must match number of weights"
        );

        let dims = points[0].dimensionality();
        let total_weight: f32 = self.weights.iter().sum();

        let mut result = vec![0.0; dims];
        for (p, &w) in points.iter().zip(&self.weights) {
            assert_eq!(
                p.dimensionality(),
                dims,
                "All points must have same dimensionality"
            );
            let normalized_w = w / total_weight;
            for (r, d) in result.iter_mut().zip(p.dims()) {
                *r += d * normalized_w;
            }
        }

        Point::new(result)
    }

    fn name(&self) -> &'static str {
        "weighted_mean"
    }
}

/// Max pooling across points
///
/// Takes the maximum value of each dimension across all points.
/// Preserves the strongest activations.
#[derive(Clone, Copy, Debug, Default)]
pub struct MaxPool;

impl Merge for MaxPool {
    fn merge(&self, points: &[Point]) -> Point {
        assert!(!points.is_empty(), "Cannot merge empty slice");

        let dims = points[0].dimensionality();
        let mut result = points[0].dims().to_vec();

        for p in &points[1..] {
            assert_eq!(
                p.dimensionality(),
                dims,
                "All points must have same dimensionality"
            );
            for (r, d) in result.iter_mut().zip(p.dims()) {
                *r = r.max(*d);
            }
        }

        Point::new(result)
    }

    fn name(&self) -> &'static str {
        "max_pool"
    }
}

/// Min pooling across points
///
/// Takes the minimum value of each dimension across all points.
#[derive(Clone, Copy, Debug, Default)]
pub struct MinPool;

impl Merge for MinPool {
    fn merge(&self, points: &[Point]) -> Point {
        assert!(!points.is_empty(), "Cannot merge empty slice");

        let dims = points[0].dimensionality();
        let mut result = points[0].dims().to_vec();

        for p in &points[1..] {
            assert_eq!(
                p.dimensionality(),
                dims,
                "All points must have same dimensionality"
            );
            for (r, d) in result.iter_mut().zip(p.dims()) {
                *r = r.min(*d);
            }
        }

        Point::new(result)
    }

    fn name(&self) -> &'static str {
        "min_pool"
    }
}

/// Sum of all points (no averaging)
///
/// Simple additive composition.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sum;

impl Merge for Sum {
    fn merge(&self, points: &[Point]) -> Point {
        assert!(!points.is_empty(), "Cannot merge empty slice");

        let dims = points[0].dimensionality();
        let mut result = vec![0.0; dims];

        for p in points {
            assert_eq!(
                p.dimensionality(),
                dims,
                "All points must have same dimensionality"
            );
            for (r, d) in result.iter_mut().zip(p.dims()) {
                *r += d;
            }
        }

        Point::new(result)
    }

    fn name(&self) -> &'static str {
        "sum"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_single() {
        let points = vec![Point::new(vec![1.0, 2.0, 3.0])];
        let merged = Mean.merge(&points);
        assert_eq!(merged.dims(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_mean_multiple() {
        let points = vec![
            Point::new(vec![1.0, 2.0]),
            Point::new(vec![3.0, 4.0]),
        ];
        let merged = Mean.merge(&points);
        assert_eq!(merged.dims(), &[2.0, 3.0]);
    }

    #[test]
    fn test_weighted_mean() {
        let points = vec![
            Point::new(vec![0.0, 0.0]),
            Point::new(vec![10.0, 10.0]),
        ];
        // Weight second point 3x more than first
        let merger = WeightedMean::new(vec![1.0, 3.0]);
        let merged = merger.merge(&points);
        // (0*0.25 + 10*0.75, 0*0.25 + 10*0.75) = (7.5, 7.5)
        assert!((merged.dims()[0] - 7.5).abs() < 0.0001);
        assert!((merged.dims()[1] - 7.5).abs() < 0.0001);
    }

    #[test]
    fn test_weighted_mean_recency() {
        let merger = WeightedMean::recency(3, 0.5);
        // decay = 0.5, n = 3
        // weights: [0.5^2, 0.5^1, 0.5^0] = [0.25, 0.5, 1.0]
        assert_eq!(merger.weights.len(), 3);
        assert!((merger.weights[0] - 0.25).abs() < 0.0001);
        assert!((merger.weights[1] - 0.5).abs() < 0.0001);
        assert!((merger.weights[2] - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_max_pool() {
        let points = vec![
            Point::new(vec![1.0, 5.0, 2.0]),
            Point::new(vec![3.0, 2.0, 4.0]),
            Point::new(vec![2.0, 3.0, 1.0]),
        ];
        let merged = MaxPool.merge(&points);
        assert_eq!(merged.dims(), &[3.0, 5.0, 4.0]);
    }

    #[test]
    fn test_min_pool() {
        let points = vec![
            Point::new(vec![1.0, 5.0, 2.0]),
            Point::new(vec![3.0, 2.0, 4.0]),
            Point::new(vec![2.0, 3.0, 1.0]),
        ];
        let merged = MinPool.merge(&points);
        assert_eq!(merged.dims(), &[1.0, 2.0, 1.0]);
    }

    #[test]
    fn test_sum() {
        let points = vec![
            Point::new(vec![1.0, 2.0]),
            Point::new(vec![3.0, 4.0]),
        ];
        let merged = Sum.merge(&points);
        assert_eq!(merged.dims(), &[4.0, 6.0]);
    }

    #[test]
    fn test_merge_names() {
        assert_eq!(Mean.name(), "mean");
        assert_eq!(MaxPool.name(), "max_pool");
        assert_eq!(MinPool.name(), "min_pool");
        assert_eq!(Sum.name(), "sum");
    }

    #[test]
    #[should_panic(expected = "Cannot merge empty")]
    fn test_merge_empty_panics() {
        let points: Vec<Point> = vec![];
        Mean.merge(&points);
    }

    #[test]
    #[should_panic(expected = "same dimensionality")]
    fn test_merge_dimension_mismatch_panics() {
        let points = vec![
            Point::new(vec![1.0, 2.0]),
            Point::new(vec![1.0, 2.0, 3.0]),
        ];
        Mean.merge(&points);
    }
}
