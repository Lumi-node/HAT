//! # Flat Index Adapter
//!
//! Brute force nearest neighbor search.
//! Compares query against ALL points - O(n) per query.
//!
//! Good for:
//! - Testing
//! - Small datasets (< 10,000 points)
//! - When exact results are required
//!
//! Not good for:
//! - Large datasets (use HNSW instead)

use std::collections::HashMap;
use std::sync::Arc;

use crate::core::{Id, Point};
use crate::core::proximity::Proximity;
use crate::ports::{Near, NearError, NearResult, SearchResult};

/// Brute force index - searches all points
pub struct FlatIndex {
    /// Stored points (ID -> Point)
    points: HashMap<Id, Point>,

    /// Expected dimensionality
    dimensionality: usize,

    /// Proximity function to use
    proximity: Arc<dyn Proximity>,

    /// Whether higher proximity = more similar
    /// true for cosine/dot product, false for euclidean
    higher_is_better: bool,
}

impl FlatIndex {
    /// Create a new flat index
    ///
    /// `higher_is_better` indicates whether higher proximity scores mean more similar.
    /// - `true` for Cosine, DotProduct
    /// - `false` for Euclidean, Manhattan
    pub fn new(
        dimensionality: usize,
        proximity: Arc<dyn Proximity>,
        higher_is_better: bool,
    ) -> Self {
        Self {
            points: HashMap::new(),
            dimensionality,
            proximity,
            higher_is_better,
        }
    }

    /// Create with cosine similarity (higher = better)
    pub fn cosine(dimensionality: usize) -> Self {
        use crate::core::proximity::Cosine;
        Self::new(dimensionality, Arc::new(Cosine), true)
    }

    /// Create with euclidean distance (lower = better)
    pub fn euclidean(dimensionality: usize) -> Self {
        use crate::core::proximity::Euclidean;
        Self::new(dimensionality, Arc::new(Euclidean), false)
    }

    /// Sort results by relevance
    fn sort_results(&self, results: &mut Vec<SearchResult>) {
        if self.higher_is_better {
            // Higher score = more relevant, sort descending
            results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        } else {
            // Lower score = more relevant, sort ascending
            results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        }
    }
}

impl Near for FlatIndex {
    fn near(&self, query: &Point, k: usize) -> NearResult<Vec<SearchResult>> {
        // Check dimensionality
        if query.dimensionality() != self.dimensionality {
            return Err(NearError::DimensionalityMismatch {
                expected: self.dimensionality,
                got: query.dimensionality(),
            });
        }

        // Compute proximity to all points
        let mut results: Vec<SearchResult> = self
            .points
            .iter()
            .map(|(id, point)| {
                let score = self.proximity.proximity(query, point);
                SearchResult::new(*id, score)
            })
            .collect();

        // Sort by relevance
        self.sort_results(&mut results);

        // Take top k
        results.truncate(k);

        Ok(results)
    }

    fn within(&self, query: &Point, threshold: f32) -> NearResult<Vec<SearchResult>> {
        // Check dimensionality
        if query.dimensionality() != self.dimensionality {
            return Err(NearError::DimensionalityMismatch {
                expected: self.dimensionality,
                got: query.dimensionality(),
            });
        }

        // Find all points within threshold
        let mut results: Vec<SearchResult> = self
            .points
            .iter()
            .filter_map(|(id, point)| {
                let score = self.proximity.proximity(query, point);
                let within = if self.higher_is_better {
                    score >= threshold
                } else {
                    score <= threshold
                };
                if within {
                    Some(SearchResult::new(*id, score))
                } else {
                    None
                }
            })
            .collect();

        // Sort by relevance
        self.sort_results(&mut results);

        Ok(results)
    }

    fn add(&mut self, id: Id, point: &Point) -> NearResult<()> {
        if point.dimensionality() != self.dimensionality {
            return Err(NearError::DimensionalityMismatch {
                expected: self.dimensionality,
                got: point.dimensionality(),
            });
        }

        self.points.insert(id, point.clone());
        Ok(())
    }

    fn remove(&mut self, id: Id) -> NearResult<()> {
        self.points.remove(&id);
        Ok(())
    }

    fn rebuild(&mut self) -> NearResult<()> {
        // Flat index doesn't need rebuilding
        Ok(())
    }

    fn is_ready(&self) -> bool {
        true // Always ready
    }

    fn len(&self) -> usize {
        self.points.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_index() -> FlatIndex {
        let mut index = FlatIndex::cosine(3);

        // Add some test points
        let points = vec![
            (Id::from_bytes([1; 16]), Point::new(vec![1.0, 0.0, 0.0])),
            (Id::from_bytes([2; 16]), Point::new(vec![0.0, 1.0, 0.0])),
            (Id::from_bytes([3; 16]), Point::new(vec![0.0, 0.0, 1.0])),
            (Id::from_bytes([4; 16]), Point::new(vec![0.7, 0.7, 0.0]).normalize()),
        ];

        for (id, point) in points {
            index.add(id, &point).unwrap();
        }

        index
    }

    #[test]
    fn test_flat_index_near() {
        let index = setup_index();

        // Query for points near [1, 0, 0]
        let query = Point::new(vec![1.0, 0.0, 0.0]);
        let results = index.near(&query, 2).unwrap();

        assert_eq!(results.len(), 2);

        // First result should be [1, 0, 0] with cosine = 1.0
        assert_eq!(results[0].id, Id::from_bytes([1; 16]));
        assert!((results[0].score - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_flat_index_within_cosine() {
        let index = setup_index();

        // Find all points with cosine > 0.5 to [1, 0, 0]
        let query = Point::new(vec![1.0, 0.0, 0.0]);
        let results = index.within(&query, 0.5).unwrap();

        // Should find [1,0,0] (cosine=1.0) and [0.7,0.7,0] (cosine≈0.707)
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_flat_index_euclidean() {
        let mut index = FlatIndex::euclidean(2);

        index.add(Id::from_bytes([1; 16]), &Point::new(vec![0.0, 0.0])).unwrap();
        index.add(Id::from_bytes([2; 16]), &Point::new(vec![1.0, 0.0])).unwrap();
        index.add(Id::from_bytes([3; 16]), &Point::new(vec![5.0, 0.0])).unwrap();

        let query = Point::new(vec![0.0, 0.0]);
        let results = index.near(&query, 2).unwrap();

        // Nearest should be [0,0] with distance 0
        assert_eq!(results[0].id, Id::from_bytes([1; 16]));
        assert!((results[0].score - 0.0).abs() < 0.0001);

        // Second nearest should be [1,0] with distance 1
        assert_eq!(results[1].id, Id::from_bytes([2; 16]));
        assert!((results[1].score - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_flat_index_add_remove() {
        let mut index = FlatIndex::cosine(3);

        let id = Id::from_bytes([1; 16]);
        let point = Point::new(vec![1.0, 0.0, 0.0]);

        index.add(id, &point).unwrap();
        assert_eq!(index.len(), 1);

        index.remove(id).unwrap();
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_flat_index_dimensionality_check() {
        let mut index = FlatIndex::cosine(3);

        let wrong_dims = Point::new(vec![1.0, 0.0]); // 2 dims
        let result = index.add(Id::now(), &wrong_dims);

        match result {
            Err(NearError::DimensionalityMismatch { expected, got }) => {
                assert_eq!(expected, 3);
                assert_eq!(got, 2);
            }
            _ => panic!("Expected DimensionalityMismatch error"),
        }
    }

    #[test]
    fn test_flat_index_ready() {
        let index = FlatIndex::cosine(3);
        assert!(index.is_ready());
    }
}
