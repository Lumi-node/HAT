//! # Memory Storage Adapter
//!
//! In-memory storage using HashMap.
//! Fast, but volatile (data lost on shutdown).
//!
//! Good for:
//! - Testing
//! - Hot tier storage
//! - Small datasets

use std::collections::HashMap;

use crate::core::{Blob, Id, PlacedPoint, Point};
use crate::ports::{Place, PlaceError, PlaceResult};

/// In-memory storage adapter
pub struct MemoryStorage {
    /// The stored points
    points: HashMap<Id, PlacedPoint>,

    /// Expected dimensionality
    dimensionality: usize,

    /// Maximum capacity in bytes (0 = unlimited)
    capacity: usize,

    /// Current size in bytes
    current_size: usize,
}

impl MemoryStorage {
    /// Create a new memory storage with specified dimensionality
    pub fn new(dimensionality: usize) -> Self {
        Self {
            points: HashMap::new(),
            dimensionality,
            capacity: 0,
            current_size: 0,
        }
    }

    /// Create with a capacity limit
    pub fn with_capacity(dimensionality: usize, capacity: usize) -> Self {
        Self {
            points: HashMap::new(),
            dimensionality,
            capacity,
            current_size: 0,
        }
    }

    /// Calculate size of a placed point in bytes
    fn point_size(point: &PlacedPoint) -> usize {
        // Id: 16 bytes
        // Point: dims.len() * 4 bytes (f32)
        // Blob: data.len() bytes
        // Overhead: ~48 bytes for struct padding and HashMap entry
        16 + (point.point.dimensionality() * 4) + point.blob.size() + 48
    }
}

impl Place for MemoryStorage {
    fn place(&mut self, point: Point, blob: Blob) -> PlaceResult<Id> {
        // Check dimensionality
        if point.dimensionality() != self.dimensionality {
            return Err(PlaceError::DimensionalityMismatch {
                expected: self.dimensionality,
                got: point.dimensionality(),
            });
        }

        let id = Id::now();
        let placed = PlacedPoint::new(id, point, blob);

        // Check capacity
        let size = Self::point_size(&placed);
        if self.capacity > 0 && self.current_size + size > self.capacity {
            return Err(PlaceError::CapacityExceeded);
        }

        self.current_size += size;
        self.points.insert(id, placed);

        Ok(id)
    }

    fn place_with_id(&mut self, id: Id, point: Point, blob: Blob) -> PlaceResult<()> {
        // Check dimensionality
        if point.dimensionality() != self.dimensionality {
            return Err(PlaceError::DimensionalityMismatch {
                expected: self.dimensionality,
                got: point.dimensionality(),
            });
        }

        // Check for duplicates
        if self.points.contains_key(&id) {
            return Err(PlaceError::DuplicateId(id));
        }

        let placed = PlacedPoint::new(id, point, blob);

        // Check capacity
        let size = Self::point_size(&placed);
        if self.capacity > 0 && self.current_size + size > self.capacity {
            return Err(PlaceError::CapacityExceeded);
        }

        self.current_size += size;
        self.points.insert(id, placed);

        Ok(())
    }

    fn remove(&mut self, id: Id) -> Option<PlacedPoint> {
        if let Some(placed) = self.points.remove(&id) {
            self.current_size -= Self::point_size(&placed);
            Some(placed)
        } else {
            None
        }
    }

    fn get(&self, id: Id) -> Option<&PlacedPoint> {
        self.points.get(&id)
    }

    fn len(&self) -> usize {
        self.points.len()
    }

    fn iter(&self) -> Box<dyn Iterator<Item = &PlacedPoint> + '_> {
        Box::new(self.points.values())
    }

    fn size_bytes(&self) -> usize {
        self.current_size
    }

    fn clear(&mut self) {
        self.points.clear();
        self.current_size = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_storage_place() {
        let mut storage = MemoryStorage::new(3);

        let point = Point::new(vec![1.0, 2.0, 3.0]);
        let blob = Blob::from_str("test");

        let id = storage.place(point, blob).unwrap();

        assert_eq!(storage.len(), 1);
        assert!(storage.contains(id));
    }

    #[test]
    fn test_memory_storage_get() {
        let mut storage = MemoryStorage::new(3);

        let point = Point::new(vec![1.0, 2.0, 3.0]);
        let blob = Blob::from_str("hello");

        let id = storage.place(point, blob).unwrap();

        let retrieved = storage.get(id).unwrap();
        assert_eq!(retrieved.blob.as_str(), Some("hello"));
    }

    #[test]
    fn test_memory_storage_remove() {
        let mut storage = MemoryStorage::new(3);

        let point = Point::new(vec![1.0, 2.0, 3.0]);
        let id = storage.place(point, Blob::empty()).unwrap();

        assert_eq!(storage.len(), 1);

        let removed = storage.remove(id);
        assert!(removed.is_some());
        assert_eq!(storage.len(), 0);
        assert!(!storage.contains(id));
    }

    #[test]
    fn test_memory_storage_dimensionality_check() {
        let mut storage = MemoryStorage::new(3);

        let wrong_dims = Point::new(vec![1.0, 2.0]); // 2 dims, expected 3

        let result = storage.place(wrong_dims, Blob::empty());

        match result {
            Err(PlaceError::DimensionalityMismatch { expected, got }) => {
                assert_eq!(expected, 3);
                assert_eq!(got, 2);
            }
            _ => panic!("Expected DimensionalityMismatch error"),
        }
    }

    #[test]
    fn test_memory_storage_capacity() {
        // Small capacity - enough for one point but not two
        // Point size: 16 (id) + 12 (3 f32s) + 10 (blob) + 48 (overhead) = 86 bytes
        let mut storage = MemoryStorage::with_capacity(3, 150);

        let point = Point::new(vec![1.0, 2.0, 3.0]);
        let blob = Blob::new(vec![0u8; 10]); // Small blob

        // First one should succeed
        storage.place(point.clone(), blob.clone()).unwrap();

        // Second should fail due to capacity
        let result = storage.place(point, blob);
        assert!(matches!(result, Err(PlaceError::CapacityExceeded)));
    }

    #[test]
    fn test_memory_storage_clear() {
        let mut storage = MemoryStorage::new(3);

        for i in 0..10 {
            let point = Point::new(vec![i as f32, 0.0, 0.0]);
            storage.place(point, Blob::empty()).unwrap();
        }

        assert_eq!(storage.len(), 10);
        assert!(storage.size_bytes() > 0);

        storage.clear();

        assert_eq!(storage.len(), 0);
        assert_eq!(storage.size_bytes(), 0);
    }

    #[test]
    fn test_memory_storage_iter() {
        let mut storage = MemoryStorage::new(2);

        storage.place(Point::new(vec![1.0, 0.0]), Blob::empty()).unwrap();
        storage.place(Point::new(vec![0.0, 1.0]), Blob::empty()).unwrap();

        let points: Vec<_> = storage.iter().collect();
        assert_eq!(points.len(), 2);
    }
}
