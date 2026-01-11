//! # Core Domain
//!
//! Pure math, no I/O. The foundation of ARMS.
//!
//! This module contains the fundamental types and operations:
//! - `Point` - A position in dimensional space
//! - `Id` - Unique identifier for placed points
//! - `Blob` - Raw payload data
//! - `Proximity` - Trait for measuring relatedness
//! - `Merge` - Trait for composing points
//!
//! ## Design Principles
//!
//! - All functions are pure (deterministic, no side effects)
//! - No I/O operations
//! - No external dependencies beyond std
//! - Fully testable in isolation

mod point;
mod id;
mod blob;
pub mod proximity;
pub mod merge;
pub mod config;

// Re-exports
pub use point::Point;
pub use id::Id;
pub use blob::Blob;

/// A point that has been placed in the space
#[derive(Clone)]
pub struct PlacedPoint {
    /// Unique identifier
    pub id: Id,
    /// Position in dimensional space
    pub point: Point,
    /// Attached payload
    pub blob: Blob,
}

impl PlacedPoint {
    /// Create a new placed point
    pub fn new(id: Id, point: Point, blob: Blob) -> Self {
        Self { id, point, blob }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_placed_point_creation() {
        let id = Id::now();
        let point = Point::new(vec![1.0, 2.0, 3.0]);
        let blob = Blob::new(vec![1, 2, 3]);

        let placed = PlacedPoint::new(id, point.clone(), blob);

        assert_eq!(placed.point.dimensionality(), 3);
        assert_eq!(placed.blob.size(), 3);
    }
}
