//! # Place Port
//!
//! Trait for placing points in the space.
//!
//! This is one of the five primitives of ARMS:
//! `Place: fn(point, data) -> id` - Exist in space
//!
//! Implemented by storage adapters (Memory, NVMe, etc.)

use crate::core::{Blob, Id, PlacedPoint, Point};

/// Result type for place operations
pub type PlaceResult<T> = Result<T, PlaceError>;

/// Errors that can occur during place operations
#[derive(Debug, Clone, PartialEq)]
pub enum PlaceError {
    /// The point has wrong dimensionality for this space
    DimensionalityMismatch { expected: usize, got: usize },

    /// Storage capacity exceeded
    CapacityExceeded,

    /// Point with this ID already exists
    DuplicateId(Id),

    /// Storage backend error
    StorageError(String),
}

impl std::fmt::Display for PlaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlaceError::DimensionalityMismatch { expected, got } => {
                write!(f, "Dimensionality mismatch: expected {}, got {}", expected, got)
            }
            PlaceError::CapacityExceeded => write!(f, "Storage capacity exceeded"),
            PlaceError::DuplicateId(id) => write!(f, "Duplicate ID: {}", id),
            PlaceError::StorageError(msg) => write!(f, "Storage error: {}", msg),
        }
    }
}

impl std::error::Error for PlaceError {}

/// Trait for placing points in the space
///
/// Storage adapters implement this trait.
pub trait Place: Send + Sync {
    /// Place a point with its payload in the space
    ///
    /// Returns the ID assigned to the placed point.
    fn place(&mut self, point: Point, blob: Blob) -> PlaceResult<Id>;

    /// Place a point with a specific ID
    ///
    /// Use when you need deterministic IDs (e.g., replication, testing).
    fn place_with_id(&mut self, id: Id, point: Point, blob: Blob) -> PlaceResult<()>;

    /// Remove a point from the space
    ///
    /// Returns the removed point if it existed.
    fn remove(&mut self, id: Id) -> Option<PlacedPoint>;

    /// Get a placed point by ID
    ///
    /// Returns None if not found.
    fn get(&self, id: Id) -> Option<&PlacedPoint>;

    /// Check if a point exists
    fn contains(&self, id: Id) -> bool {
        self.get(id).is_some()
    }

    /// Get the number of placed points
    fn len(&self) -> usize;

    /// Check if the space is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate over all placed points
    fn iter(&self) -> Box<dyn Iterator<Item = &PlacedPoint> + '_>;

    /// Get current storage size in bytes
    fn size_bytes(&self) -> usize;

    /// Clear all points
    fn clear(&mut self);
}
