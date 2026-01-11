//! # Point
//!
//! A position in dimensional space. The fundamental primitive.
//!
//! Dimensionality is NOT fixed - configure it for your model.
//! 768-dim, 1024-dim, 4096-dim, or any size you need.
//!
//! The point IS the thought's position.
//! The position IS its relationship to all other thoughts.

/// A point in dimensional space
#[derive(Clone, Debug, PartialEq)]
pub struct Point {
    dims: Vec<f32>,
}

impl Point {
    /// Create a new point from a vector of dimensions
    ///
    /// # Example
    /// ```
    /// use arms::Point;
    /// let p = Point::new(vec![1.0, 2.0, 3.0]);
    /// assert_eq!(p.dimensionality(), 3);
    /// ```
    pub fn new(dims: Vec<f32>) -> Self {
        Self { dims }
    }

    /// Create an origin point (all zeros) of given dimensionality
    ///
    /// # Example
    /// ```
    /// use arms::Point;
    /// let origin = Point::origin(768);
    /// assert_eq!(origin.dimensionality(), 768);
    /// assert!(origin.dims().iter().all(|&x| x == 0.0));
    /// ```
    pub fn origin(dims: usize) -> Self {
        Self {
            dims: vec![0.0; dims],
        }
    }

    /// Get the dimensionality of this point
    pub fn dimensionality(&self) -> usize {
        self.dims.len()
    }

    /// Access the dimensions as a slice
    pub fn dims(&self) -> &[f32] {
        &self.dims
    }

    /// Mutable access to dimensions
    pub fn dims_mut(&mut self) -> &mut [f32] {
        &mut self.dims
    }

    /// Calculate the magnitude (L2 norm) of this point
    ///
    /// # Example
    /// ```
    /// use arms::Point;
    /// let p = Point::new(vec![3.0, 4.0]);
    /// assert!((p.magnitude() - 5.0).abs() < 0.0001);
    /// ```
    pub fn magnitude(&self) -> f32 {
        self.dims.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Check if this point is normalized (magnitude ≈ 1.0)
    pub fn is_normalized(&self) -> bool {
        let mag = self.magnitude();
        (mag - 1.0).abs() < 0.001
    }

    /// Return a normalized copy of this point
    ///
    /// If magnitude is zero, returns a clone of self.
    ///
    /// # Example
    /// ```
    /// use arms::Point;
    /// let p = Point::new(vec![3.0, 4.0]);
    /// let normalized = p.normalize();
    /// assert!(normalized.is_normalized());
    /// ```
    pub fn normalize(&self) -> Self {
        let mag = self.magnitude();
        if mag == 0.0 {
            return self.clone();
        }
        Self {
            dims: self.dims.iter().map(|x| x / mag).collect(),
        }
    }

    /// Add another point to this one (element-wise)
    pub fn add(&self, other: &Point) -> Self {
        assert_eq!(
            self.dimensionality(),
            other.dimensionality(),
            "Points must have same dimensionality"
        );
        Self {
            dims: self
                .dims
                .iter()
                .zip(other.dims.iter())
                .map(|(a, b)| a + b)
                .collect(),
        }
    }

    /// Scale this point by a scalar
    pub fn scale(&self, scalar: f32) -> Self {
        Self {
            dims: self.dims.iter().map(|x| x * scalar).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_point() {
        let p = Point::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(p.dimensionality(), 3);
        assert_eq!(p.dims(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_origin() {
        let origin = Point::origin(768);
        assert_eq!(origin.dimensionality(), 768);
        assert!(origin.dims().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_magnitude() {
        let p = Point::new(vec![3.0, 4.0]);
        assert!((p.magnitude() - 5.0).abs() < 0.0001);
    }

    #[test]
    fn test_normalize() {
        let p = Point::new(vec![3.0, 4.0]);
        let normalized = p.normalize();
        assert!(normalized.is_normalized());
        assert!((normalized.dims()[0] - 0.6).abs() < 0.0001);
        assert!((normalized.dims()[1] - 0.8).abs() < 0.0001);
    }

    #[test]
    fn test_normalize_zero() {
        let p = Point::origin(3);
        let normalized = p.normalize();
        assert_eq!(normalized.dims(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_add() {
        let a = Point::new(vec![1.0, 2.0]);
        let b = Point::new(vec![3.0, 4.0]);
        let c = a.add(&b);
        assert_eq!(c.dims(), &[4.0, 6.0]);
    }

    #[test]
    fn test_scale() {
        let p = Point::new(vec![1.0, 2.0]);
        let scaled = p.scale(2.0);
        assert_eq!(scaled.dims(), &[2.0, 4.0]);
    }

    #[test]
    #[should_panic(expected = "same dimensionality")]
    fn test_add_different_dims_panics() {
        let a = Point::new(vec![1.0, 2.0]);
        let b = Point::new(vec![1.0, 2.0, 3.0]);
        let _ = a.add(&b);
    }
}
