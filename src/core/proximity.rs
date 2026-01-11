//! # Proximity
//!
//! Trait and implementations for measuring how related two points are.
//!
//! This is one of the five primitives of ARMS:
//! `Proximity: fn(a, b) -> f32` - How related?
//!
//! Proximity functions are pluggable - use whichever fits your use case.

use super::Point;

/// Trait for measuring proximity between points
///
/// Higher values typically mean more similar/related.
/// The exact semantics depend on the implementation.
pub trait Proximity: Send + Sync {
    /// Compute proximity between two points
    ///
    /// Both points must have the same dimensionality.
    fn proximity(&self, a: &Point, b: &Point) -> f32;

    /// Name of this proximity function (for debugging/config)
    fn name(&self) -> &'static str;
}

// ============================================================================
// IMPLEMENTATIONS
// ============================================================================

/// Cosine similarity
///
/// Measures the cosine of the angle between two vectors.
/// Returns a value in [-1, 1] where 1 means identical direction.
///
/// Best for: Normalized vectors, semantic similarity.
#[derive(Clone, Copy, Debug, Default)]
pub struct Cosine;

impl Proximity for Cosine {
    fn proximity(&self, a: &Point, b: &Point) -> f32 {
        assert_eq!(
            a.dimensionality(),
            b.dimensionality(),
            "Points must have same dimensionality"
        );

        let dot: f32 = a
            .dims()
            .iter()
            .zip(b.dims().iter())
            .map(|(x, y)| x * y)
            .sum();

        let mag_a = a.magnitude();
        let mag_b = b.magnitude();

        if mag_a == 0.0 || mag_b == 0.0 {
            return 0.0;
        }

        dot / (mag_a * mag_b)
    }

    fn name(&self) -> &'static str {
        "cosine"
    }
}

/// Euclidean distance
///
/// The straight-line distance between two points.
/// Returns a value in [0, ∞) where 0 means identical.
///
/// Note: This returns DISTANCE, not similarity.
/// Lower values = more similar.
#[derive(Clone, Copy, Debug, Default)]
pub struct Euclidean;

impl Proximity for Euclidean {
    fn proximity(&self, a: &Point, b: &Point) -> f32 {
        assert_eq!(
            a.dimensionality(),
            b.dimensionality(),
            "Points must have same dimensionality"
        );

        let dist_sq: f32 = a
            .dims()
            .iter()
            .zip(b.dims().iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum();

        dist_sq.sqrt()
    }

    fn name(&self) -> &'static str {
        "euclidean"
    }
}

/// Squared Euclidean distance
///
/// Same ordering as Euclidean but faster (no sqrt).
/// Use when you only need to compare distances, not absolute values.
#[derive(Clone, Copy, Debug, Default)]
pub struct EuclideanSquared;

impl Proximity for EuclideanSquared {
    fn proximity(&self, a: &Point, b: &Point) -> f32 {
        assert_eq!(
            a.dimensionality(),
            b.dimensionality(),
            "Points must have same dimensionality"
        );

        a.dims()
            .iter()
            .zip(b.dims().iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum()
    }

    fn name(&self) -> &'static str {
        "euclidean_squared"
    }
}

/// Dot product
///
/// The raw dot product without normalization.
/// Returns a value that depends on magnitudes.
///
/// Best for: When magnitude matters, not just direction.
#[derive(Clone, Copy, Debug, Default)]
pub struct DotProduct;

impl Proximity for DotProduct {
    fn proximity(&self, a: &Point, b: &Point) -> f32 {
        assert_eq!(
            a.dimensionality(),
            b.dimensionality(),
            "Points must have same dimensionality"
        );

        a.dims()
            .iter()
            .zip(b.dims().iter())
            .map(|(x, y)| x * y)
            .sum()
    }

    fn name(&self) -> &'static str {
        "dot_product"
    }
}

/// Manhattan (L1) distance
///
/// Sum of absolute differences along each dimension.
/// Returns a value in [0, ∞) where 0 means identical.
#[derive(Clone, Copy, Debug, Default)]
pub struct Manhattan;

impl Proximity for Manhattan {
    fn proximity(&self, a: &Point, b: &Point) -> f32 {
        assert_eq!(
            a.dimensionality(),
            b.dimensionality(),
            "Points must have same dimensionality"
        );

        a.dims()
            .iter()
            .zip(b.dims().iter())
            .map(|(x, y)| (x - y).abs())
            .sum()
    }

    fn name(&self) -> &'static str {
        "manhattan"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_identical() {
        let a = Point::new(vec![1.0, 0.0, 0.0]);
        let b = Point::new(vec![1.0, 0.0, 0.0]);
        let cos = Cosine.proximity(&a, &b);
        assert!((cos - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_opposite() {
        let a = Point::new(vec![1.0, 0.0, 0.0]);
        let b = Point::new(vec![-1.0, 0.0, 0.0]);
        let cos = Cosine.proximity(&a, &b);
        assert!((cos - (-1.0)).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = Point::new(vec![1.0, 0.0, 0.0]);
        let b = Point::new(vec![0.0, 1.0, 0.0]);
        let cos = Cosine.proximity(&a, &b);
        assert!(cos.abs() < 0.0001);
    }

    #[test]
    fn test_euclidean() {
        let a = Point::new(vec![0.0, 0.0]);
        let b = Point::new(vec![3.0, 4.0]);
        let dist = Euclidean.proximity(&a, &b);
        assert!((dist - 5.0).abs() < 0.0001);
    }

    #[test]
    fn test_euclidean_squared() {
        let a = Point::new(vec![0.0, 0.0]);
        let b = Point::new(vec![3.0, 4.0]);
        let dist_sq = EuclideanSquared.proximity(&a, &b);
        assert!((dist_sq - 25.0).abs() < 0.0001);
    }

    #[test]
    fn test_dot_product() {
        let a = Point::new(vec![1.0, 2.0, 3.0]);
        let b = Point::new(vec![4.0, 5.0, 6.0]);
        let dot = DotProduct.proximity(&a, &b);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!((dot - 32.0).abs() < 0.0001);
    }

    #[test]
    fn test_manhattan() {
        let a = Point::new(vec![0.0, 0.0]);
        let b = Point::new(vec![3.0, 4.0]);
        let dist = Manhattan.proximity(&a, &b);
        assert!((dist - 7.0).abs() < 0.0001);
    }

    #[test]
    fn test_proximity_names() {
        assert_eq!(Cosine.name(), "cosine");
        assert_eq!(Euclidean.name(), "euclidean");
        assert_eq!(DotProduct.name(), "dot_product");
        assert_eq!(Manhattan.name(), "manhattan");
    }

    #[test]
    #[should_panic(expected = "same dimensionality")]
    fn test_dimension_mismatch_panics() {
        let a = Point::new(vec![1.0, 2.0]);
        let b = Point::new(vec![1.0, 2.0, 3.0]);
        Cosine.proximity(&a, &b);
    }
}
