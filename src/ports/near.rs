//! # Near Port
//!
//! Trait for finding related points.
//!
//! This is one of the five primitives of ARMS:
//! `Near: fn(point, k) -> ids` - What's related?
//!
//! Implemented by index adapters (Flat, HNSW, etc.)

use crate::core::{Id, Point};

/// Result type for near operations
pub type NearResult<T> = Result<T, NearError>;

/// A search result with ID and distance/similarity score
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    /// The ID of the found point
    pub id: Id,

    /// Distance or similarity score
    /// Interpretation depends on the proximity function used.
    pub score: f32,
}

impl SearchResult {
    pub fn new(id: Id, score: f32) -> Self {
        Self { id, score }
    }
}

/// Errors that can occur during near operations
#[derive(Debug, Clone, PartialEq)]
pub enum NearError {
    /// The query point has wrong dimensionality
    DimensionalityMismatch { expected: usize, got: usize },

    /// Index is not built/ready
    IndexNotReady,

    /// Index backend error
    IndexError(String),
}

impl std::fmt::Display for NearError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NearError::DimensionalityMismatch { expected, got } => {
                write!(f, "Dimensionality mismatch: expected {}, got {}", expected, got)
            }
            NearError::IndexNotReady => write!(f, "Index not ready"),
            NearError::IndexError(msg) => write!(f, "Index error: {}", msg),
        }
    }
}

impl std::error::Error for NearError {}

/// Trait for finding related points
///
/// Index adapters implement this trait.
pub trait Near: Send + Sync {
    /// Find k nearest points to query
    ///
    /// Returns results sorted by relevance (most relevant first).
    fn near(&self, query: &Point, k: usize) -> NearResult<Vec<SearchResult>>;

    /// Find all points within a distance/similarity threshold
    ///
    /// For distance metrics (Euclidean), finds points with distance < threshold.
    /// For similarity metrics (Cosine), finds points with similarity > threshold.
    fn within(&self, query: &Point, threshold: f32) -> NearResult<Vec<SearchResult>>;

    /// Add a point to the index
    ///
    /// Call this after placing a point in storage.
    fn add(&mut self, id: Id, point: &Point) -> NearResult<()>;

    /// Remove a point from the index
    fn remove(&mut self, id: Id) -> NearResult<()>;

    /// Rebuild the index (if needed for performance)
    fn rebuild(&mut self) -> NearResult<()>;

    /// Check if the index is ready for queries
    fn is_ready(&self) -> bool;

    /// Get the number of indexed points
    fn len(&self) -> usize;

    /// Check if the index is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
