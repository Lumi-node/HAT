//! # HAT Persistence Layer
//!
//! Serialization and deserialization for HAT indexes.
//!
//! ## Format
//!
//! The HAT persistence format is a simple binary format:
//!
//! ```text
//! [Header: 32 bytes]
//!   - Magic: "HAT\0" (4 bytes)
//!   - Version: u32 (4 bytes)
//!   - Dimensionality: u32 (4 bytes)
//!   - Container count: u64 (8 bytes)
//!   - Root ID: 16 bytes (or zeros if none)
//!   - Reserved: 0 bytes (for future use)
//!
//! [Containers: variable]
//!   For each container:
//!     - ID: 16 bytes
//!     - Level: u8 (0=Root, 1=Session, 2=Document, 3=Chunk)
//!     - Timestamp: u64 (8 bytes)
//!     - Child count: u32 (4 bytes)
//!     - Child IDs: child_count * 16 bytes
//!     - Descendant count: u64 (8 bytes)
//!     - Centroid: dimensionality * 4 bytes (f32s)
//!     - Has accumulated sum: u8 (0 or 1)
//!     - Accumulated sum: dimensionality * 4 bytes (if has_accumulated_sum)
//!
//! [Active State: 32 bytes]
//!   - Active session ID: 16 bytes (or zeros)
//!   - Active document ID: 16 bytes (or zeros)
//!
//! [Learnable Router Weights: variable, optional]
//!   - Has weights: u8 (0 or 1)
//!   - If has weights: dimensionality * 4 bytes (f32s)
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! // Save
//! let bytes = hat.to_bytes()?;
//! std::fs::write("index.hat", bytes)?;
//!
//! // Load
//! let bytes = std::fs::read("index.hat")?;
//! let hat = HatIndex::from_bytes(&bytes)?;
//! ```

use crate::core::{Id, Point};
use std::io::{self, Read, Write, Cursor};

/// Magic bytes for HAT file format
const MAGIC: &[u8; 4] = b"HAT\0";

/// Current format version
const VERSION: u32 = 1;

/// Error type for persistence operations
#[derive(Debug)]
pub enum PersistError {
    /// Invalid magic bytes
    InvalidMagic,
    /// Unsupported version
    UnsupportedVersion(u32),
    /// IO error
    Io(io::Error),
    /// Data corruption
    Corrupted(String),
    /// Dimension mismatch
    DimensionMismatch { expected: usize, found: usize },
}

impl std::fmt::Display for PersistError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PersistError::InvalidMagic => write!(f, "Invalid HAT file magic bytes"),
            PersistError::UnsupportedVersion(v) => write!(f, "Unsupported HAT version: {}", v),
            PersistError::Io(e) => write!(f, "IO error: {}", e),
            PersistError::Corrupted(msg) => write!(f, "Data corruption: {}", msg),
            PersistError::DimensionMismatch { expected, found } => {
                write!(f, "Dimension mismatch: expected {}, found {}", expected, found)
            }
        }
    }
}

impl std::error::Error for PersistError {}

impl From<io::Error> for PersistError {
    fn from(e: io::Error) -> Self {
        PersistError::Io(e)
    }
}

/// Container level as u8
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LevelByte {
    Root = 0,
    Session = 1,
    Document = 2,
    Chunk = 3,
}

impl LevelByte {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(LevelByte::Root),
            1 => Some(LevelByte::Session),
            2 => Some(LevelByte::Document),
            3 => Some(LevelByte::Chunk),
            _ => None,
        }
    }
}

/// Serialized container data
#[derive(Debug, Clone)]
pub struct SerializedContainer {
    pub id: Id,
    pub level: LevelByte,
    pub timestamp: u64,
    pub children: Vec<Id>,
    pub descendant_count: u64,
    pub centroid: Vec<f32>,
    pub accumulated_sum: Option<Vec<f32>>,
}

/// Serialized HAT index
#[derive(Debug, Clone)]
pub struct SerializedHat {
    pub version: u32,
    pub dimensionality: u32,
    pub root_id: Option<Id>,
    pub containers: Vec<SerializedContainer>,
    pub active_session: Option<Id>,
    pub active_document: Option<Id>,
    pub router_weights: Option<Vec<f32>>,
}

impl SerializedHat {
    /// Serialize to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>, PersistError> {
        let mut buf = Vec::new();

        // Header
        buf.write_all(MAGIC)?;
        buf.write_all(&self.version.to_le_bytes())?;
        buf.write_all(&self.dimensionality.to_le_bytes())?;
        buf.write_all(&(self.containers.len() as u64).to_le_bytes())?;

        // Root ID
        if let Some(id) = &self.root_id {
            buf.write_all(id.as_bytes())?;
        } else {
            buf.write_all(&[0u8; 16])?;
        }

        // Containers
        for container in &self.containers {
            // ID
            buf.write_all(container.id.as_bytes())?;

            // Level
            buf.write_all(&[container.level as u8])?;

            // Timestamp
            buf.write_all(&container.timestamp.to_le_bytes())?;

            // Children
            buf.write_all(&(container.children.len() as u32).to_le_bytes())?;
            for child_id in &container.children {
                buf.write_all(child_id.as_bytes())?;
            }

            // Descendant count
            buf.write_all(&container.descendant_count.to_le_bytes())?;

            // Centroid
            for &v in &container.centroid {
                buf.write_all(&v.to_le_bytes())?;
            }

            // Accumulated sum
            if let Some(sum) = &container.accumulated_sum {
                buf.write_all(&[1u8])?;
                for &v in sum {
                    buf.write_all(&v.to_le_bytes())?;
                }
            } else {
                buf.write_all(&[0u8])?;
            }
        }

        // Active state
        if let Some(id) = &self.active_session {
            buf.write_all(id.as_bytes())?;
        } else {
            buf.write_all(&[0u8; 16])?;
        }

        if let Some(id) = &self.active_document {
            buf.write_all(id.as_bytes())?;
        } else {
            buf.write_all(&[0u8; 16])?;
        }

        // Router weights
        if let Some(weights) = &self.router_weights {
            buf.write_all(&[1u8])?;
            for &w in weights {
                buf.write_all(&w.to_le_bytes())?;
            }
        } else {
            buf.write_all(&[0u8])?;
        }

        Ok(buf)
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self, PersistError> {
        let mut cursor = Cursor::new(data);

        // Read header
        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(PersistError::InvalidMagic);
        }

        let mut version_bytes = [0u8; 4];
        cursor.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);
        if version != VERSION {
            return Err(PersistError::UnsupportedVersion(version));
        }

        let mut dims_bytes = [0u8; 4];
        cursor.read_exact(&mut dims_bytes)?;
        let dimensionality = u32::from_le_bytes(dims_bytes);

        let mut count_bytes = [0u8; 8];
        cursor.read_exact(&mut count_bytes)?;
        let container_count = u64::from_le_bytes(count_bytes);

        let mut root_bytes = [0u8; 16];
        cursor.read_exact(&mut root_bytes)?;
        let root_id = if root_bytes == [0u8; 16] {
            None
        } else {
            Some(Id::from_bytes(root_bytes))
        };

        // Read containers
        let mut containers = Vec::with_capacity(container_count as usize);
        for _ in 0..container_count {
            // ID
            let mut id_bytes = [0u8; 16];
            cursor.read_exact(&mut id_bytes)?;
            let id = Id::from_bytes(id_bytes);

            // Level
            let mut level_byte = [0u8; 1];
            cursor.read_exact(&mut level_byte)?;
            let level = LevelByte::from_u8(level_byte[0])
                .ok_or_else(|| PersistError::Corrupted(format!("Invalid level: {}", level_byte[0])))?;

            // Timestamp
            let mut ts_bytes = [0u8; 8];
            cursor.read_exact(&mut ts_bytes)?;
            let timestamp = u64::from_le_bytes(ts_bytes);

            // Children
            let mut child_count_bytes = [0u8; 4];
            cursor.read_exact(&mut child_count_bytes)?;
            let child_count = u32::from_le_bytes(child_count_bytes) as usize;

            let mut children = Vec::with_capacity(child_count);
            for _ in 0..child_count {
                let mut child_bytes = [0u8; 16];
                cursor.read_exact(&mut child_bytes)?;
                children.push(Id::from_bytes(child_bytes));
            }

            // Descendant count
            let mut desc_bytes = [0u8; 8];
            cursor.read_exact(&mut desc_bytes)?;
            let descendant_count = u64::from_le_bytes(desc_bytes);

            // Centroid
            let mut centroid = Vec::with_capacity(dimensionality as usize);
            for _ in 0..dimensionality {
                let mut v_bytes = [0u8; 4];
                cursor.read_exact(&mut v_bytes)?;
                centroid.push(f32::from_le_bytes(v_bytes));
            }

            // Accumulated sum
            let mut has_sum = [0u8; 1];
            cursor.read_exact(&mut has_sum)?;
            let accumulated_sum = if has_sum[0] == 1 {
                let mut sum = Vec::with_capacity(dimensionality as usize);
                for _ in 0..dimensionality {
                    let mut v_bytes = [0u8; 4];
                    cursor.read_exact(&mut v_bytes)?;
                    sum.push(f32::from_le_bytes(v_bytes));
                }
                Some(sum)
            } else {
                None
            };

            containers.push(SerializedContainer {
                id,
                level,
                timestamp,
                children,
                descendant_count,
                centroid,
                accumulated_sum,
            });
        }

        // Active state
        let mut active_session_bytes = [0u8; 16];
        cursor.read_exact(&mut active_session_bytes)?;
        let active_session = if active_session_bytes == [0u8; 16] {
            None
        } else {
            Some(Id::from_bytes(active_session_bytes))
        };

        let mut active_document_bytes = [0u8; 16];
        cursor.read_exact(&mut active_document_bytes)?;
        let active_document = if active_document_bytes == [0u8; 16] {
            None
        } else {
            Some(Id::from_bytes(active_document_bytes))
        };

        // Router weights (optional - may not be present in older files)
        let router_weights = if cursor.position() < data.len() as u64 {
            let mut has_weights = [0u8; 1];
            cursor.read_exact(&mut has_weights)?;
            if has_weights[0] == 1 {
                let mut weights = Vec::with_capacity(dimensionality as usize);
                for _ in 0..dimensionality {
                    let mut w_bytes = [0u8; 4];
                    cursor.read_exact(&mut w_bytes)?;
                    weights.push(f32::from_le_bytes(w_bytes));
                }
                Some(weights)
            } else {
                None
            }
        } else {
            None
        };

        Ok(SerializedHat {
            version,
            dimensionality,
            root_id,
            containers,
            active_session,
            active_document,
            router_weights,
        })
    }
}

/// Helper to read ID from Option
fn id_to_bytes(id: &Option<Id>) -> [u8; 16] {
    match id {
        Some(id) => *id.as_bytes(),
        None => [0u8; 16],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialized_hat_roundtrip() {
        let original = SerializedHat {
            version: VERSION,
            dimensionality: 128,
            root_id: Some(Id::now()),
            containers: vec![
                SerializedContainer {
                    id: Id::now(),
                    level: LevelByte::Root,
                    timestamp: 1234567890,
                    children: vec![Id::now(), Id::now()],
                    descendant_count: 10,
                    centroid: vec![0.1; 128],
                    accumulated_sum: None,
                },
                SerializedContainer {
                    id: Id::now(),
                    level: LevelByte::Chunk,
                    timestamp: 1234567891,
                    children: vec![],
                    descendant_count: 1,
                    centroid: vec![0.5; 128],
                    accumulated_sum: Some(vec![0.5; 128]),
                },
            ],
            active_session: Some(Id::now()),
            active_document: None,
            router_weights: Some(vec![1.0; 128]),
        };

        let bytes = original.to_bytes().unwrap();
        let restored = SerializedHat::from_bytes(&bytes).unwrap();

        assert_eq!(restored.version, original.version);
        assert_eq!(restored.dimensionality, original.dimensionality);
        assert_eq!(restored.containers.len(), original.containers.len());
        assert!(restored.router_weights.is_some());
    }

    #[test]
    fn test_invalid_magic() {
        let bad_data = b"BAD\0rest of data...";
        let result = SerializedHat::from_bytes(bad_data);
        assert!(matches!(result, Err(PersistError::InvalidMagic)));
    }

    #[test]
    fn test_level_byte_conversion() {
        assert_eq!(LevelByte::from_u8(0), Some(LevelByte::Root));
        assert_eq!(LevelByte::from_u8(1), Some(LevelByte::Session));
        assert_eq!(LevelByte::from_u8(2), Some(LevelByte::Document));
        assert_eq!(LevelByte::from_u8(3), Some(LevelByte::Chunk));
        assert_eq!(LevelByte::from_u8(4), None);
    }
}
