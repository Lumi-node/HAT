//! # Arms Engine
//!
//! The main ARMS orchestrator.
//!
//! This struct wires together:
//! - Storage (Place port)
//! - Index (Near port)
//! - Configuration
//!
//! And exposes a unified API for storing and retrieving points.

use crate::core::{Blob, Id, PlacedPoint, Point};
use crate::core::config::ArmsConfig;
use crate::ports::{Near, NearResult, Place, PlaceResult, SearchResult};
use crate::adapters::storage::MemoryStorage;
use crate::adapters::index::FlatIndex;

/// The main ARMS engine
///
/// Orchestrates storage and indexing with a unified API.
pub struct Arms {
    /// Configuration
    config: ArmsConfig,

    /// Storage backend (Place port)
    storage: Box<dyn Place>,

    /// Index backend (Near port)
    index: Box<dyn Near>,
}

impl Arms {
    /// Create a new ARMS instance with default adapters
    ///
    /// Uses MemoryStorage and FlatIndex.
    /// For production, use `Arms::with_adapters` with appropriate backends.
    pub fn new(config: ArmsConfig) -> Self {
        let storage = Box::new(MemoryStorage::new(config.dimensionality));
        let index = Box::new(FlatIndex::new(
            config.dimensionality,
            config.proximity.clone(),
            true, // Assuming cosine-like similarity by default
        ));

        Self {
            config,
            storage,
            index,
        }
    }

    /// Create with custom adapters
    pub fn with_adapters(
        config: ArmsConfig,
        storage: Box<dyn Place>,
        index: Box<dyn Near>,
    ) -> Self {
        Self {
            config,
            storage,
            index,
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &ArmsConfig {
        &self.config
    }

    /// Get the dimensionality of this space
    pub fn dimensionality(&self) -> usize {
        self.config.dimensionality
    }

    // ========================================================================
    // PLACE OPERATIONS
    // ========================================================================

    /// Place a point in the space
    ///
    /// The point will be normalized if configured to do so.
    /// Returns the assigned ID.
    pub fn place(&mut self, point: Point, blob: Blob) -> PlaceResult<Id> {
        // Normalize if configured
        let point = if self.config.normalize_on_insert {
            point.normalize()
        } else {
            point
        };

        // Store in storage
        let id = self.storage.place(point.clone(), blob)?;

        // Add to index
        if let Err(e) = self.index.add(id, &point) {
            // Rollback storage if index fails
            self.storage.remove(id);
            return Err(crate::ports::PlaceError::StorageError(format!(
                "Index error: {:?}",
                e
            )));
        }

        Ok(id)
    }

    /// Place multiple points at once
    pub fn place_batch(&mut self, items: Vec<(Point, Blob)>) -> Vec<PlaceResult<Id>> {
        items
            .into_iter()
            .map(|(point, blob)| self.place(point, blob))
            .collect()
    }

    /// Remove a point from the space
    pub fn remove(&mut self, id: Id) -> Option<PlacedPoint> {
        // Remove from index first
        let _ = self.index.remove(id);

        // Then from storage
        self.storage.remove(id)
    }

    /// Get a point by ID
    pub fn get(&self, id: Id) -> Option<&PlacedPoint> {
        self.storage.get(id)
    }

    /// Check if a point exists
    pub fn contains(&self, id: Id) -> bool {
        self.storage.contains(id)
    }

    /// Get the number of stored points
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Check if the space is empty
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Clear all points
    pub fn clear(&mut self) {
        self.storage.clear();
        let _ = self.index.rebuild(); // Reset index
    }

    // ========================================================================
    // NEAR OPERATIONS
    // ========================================================================

    /// Find k nearest points to query
    pub fn near(&self, query: &Point, k: usize) -> NearResult<Vec<SearchResult>> {
        // Normalize query if configured
        let query = if self.config.normalize_on_insert {
            query.normalize()
        } else {
            query.clone()
        };

        self.index.near(&query, k)
    }

    /// Find all points within threshold
    pub fn within(&self, query: &Point, threshold: f32) -> NearResult<Vec<SearchResult>> {
        let query = if self.config.normalize_on_insert {
            query.normalize()
        } else {
            query.clone()
        };

        self.index.within(&query, threshold)
    }

    /// Find and retrieve k nearest points (with full data)
    pub fn near_with_data(&self, query: &Point, k: usize) -> NearResult<Vec<(&PlacedPoint, f32)>> {
        let results = self.near(query, k)?;

        Ok(results
            .into_iter()
            .filter_map(|r| self.storage.get(r.id).map(|p| (p, r.score)))
            .collect())
    }

    // ========================================================================
    // MERGE OPERATIONS
    // ========================================================================

    /// Merge multiple points into one using the configured merge function
    pub fn merge(&self, points: &[Point]) -> Point {
        self.config.merge.merge(points)
    }

    /// Compute proximity between two points
    pub fn proximity(&self, a: &Point, b: &Point) -> f32 {
        self.config.proximity.proximity(a, b)
    }

    // ========================================================================
    // STATS
    // ========================================================================

    /// Get storage size in bytes
    pub fn size_bytes(&self) -> usize {
        self.storage.size_bytes()
    }

    /// Get index stats
    pub fn index_len(&self) -> usize {
        self.index.len()
    }

    /// Check if index is ready
    pub fn is_ready(&self) -> bool {
        self.index.is_ready()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_arms() -> Arms {
        Arms::new(ArmsConfig::new(3))
    }

    #[test]
    fn test_arms_place_and_get() {
        let mut arms = create_test_arms();

        let point = Point::new(vec![1.0, 0.0, 0.0]);
        let blob = Blob::from_str("test data");

        let id = arms.place(point, blob).unwrap();

        let retrieved = arms.get(id).unwrap();
        assert_eq!(retrieved.blob.as_str(), Some("test data"));
    }

    #[test]
    fn test_arms_near() {
        let mut arms = create_test_arms();

        // Add some points
        arms.place(Point::new(vec![1.0, 0.0, 0.0]), Blob::from_str("x")).unwrap();
        arms.place(Point::new(vec![0.0, 1.0, 0.0]), Blob::from_str("y")).unwrap();
        arms.place(Point::new(vec![0.0, 0.0, 1.0]), Blob::from_str("z")).unwrap();

        // Query
        let query = Point::new(vec![1.0, 0.0, 0.0]);
        let results = arms.near(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        // First result should have highest similarity
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn test_arms_near_with_data() {
        let mut arms = create_test_arms();

        arms.place(Point::new(vec![1.0, 0.0, 0.0]), Blob::from_str("x")).unwrap();
        arms.place(Point::new(vec![0.0, 1.0, 0.0]), Blob::from_str("y")).unwrap();

        let query = Point::new(vec![1.0, 0.0, 0.0]);
        let results = arms.near_with_data(&query, 1).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.blob.as_str(), Some("x"));
    }

    #[test]
    fn test_arms_remove() {
        let mut arms = create_test_arms();

        let id = arms.place(Point::new(vec![1.0, 0.0, 0.0]), Blob::empty()).unwrap();

        assert!(arms.contains(id));
        assert_eq!(arms.len(), 1);

        arms.remove(id);

        assert!(!arms.contains(id));
        assert_eq!(arms.len(), 0);
    }

    #[test]
    fn test_arms_merge() {
        let arms = create_test_arms();

        let points = vec![
            Point::new(vec![1.0, 0.0, 0.0]),
            Point::new(vec![0.0, 1.0, 0.0]),
        ];

        let merged = arms.merge(&points);

        // Mean of [1,0,0] and [0,1,0] = [0.5, 0.5, 0]
        assert!((merged.dims()[0] - 0.5).abs() < 0.0001);
        assert!((merged.dims()[1] - 0.5).abs() < 0.0001);
        assert!((merged.dims()[2] - 0.0).abs() < 0.0001);
    }

    #[test]
    fn test_arms_clear() {
        let mut arms = create_test_arms();

        for i in 0..10 {
            arms.place(Point::new(vec![i as f32, 0.0, 0.0]), Blob::empty()).unwrap();
        }

        assert_eq!(arms.len(), 10);

        arms.clear();

        assert_eq!(arms.len(), 0);
        assert!(arms.is_empty());
    }

    #[test]
    fn test_arms_normalizes_on_insert() {
        let mut arms = create_test_arms();

        // Insert a non-normalized point
        let point = Point::new(vec![3.0, 4.0, 0.0]); // magnitude = 5
        let id = arms.place(point, Blob::empty()).unwrap();

        let retrieved = arms.get(id).unwrap();

        // Should be normalized
        assert!(retrieved.point.is_normalized());
    }
}
