//! # Id
//!
//! Unique identifier for placed points.
//!
//! Format: 128 bits = [timestamp_ms:48][counter:16][random:64]
//! - Timestamp provides natural temporal ordering
//! - Counter prevents collisions within same millisecond
//! - Random portion adds uniqueness
//! - Sortable by time when compared
//! - No external dependencies (not UUID, just bytes)

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Global counter for uniqueness within same millisecond
static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Unique identifier for a placed point
///
/// 128 bits, timestamp-prefixed for natural time ordering.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct Id([u8; 16]);

impl Id {
    /// Generate a new Id for the current moment
    ///
    /// Uses current timestamp + counter + random bytes for uniqueness.
    pub fn now() -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // Atomically increment counter for uniqueness
        let counter = COUNTER.fetch_add(1, Ordering::Relaxed);

        let mut bytes = [0u8; 16];

        // First 6 bytes: timestamp (48 bits)
        bytes[0] = (timestamp >> 40) as u8;
        bytes[1] = (timestamp >> 32) as u8;
        bytes[2] = (timestamp >> 24) as u8;
        bytes[3] = (timestamp >> 16) as u8;
        bytes[4] = (timestamp >> 8) as u8;
        bytes[5] = timestamp as u8;

        // Next 2 bytes: counter (16 bits) - ensures uniqueness within millisecond
        bytes[6] = (counter >> 8) as u8;
        bytes[7] = counter as u8;

        // Remaining 8 bytes: pseudo-random based on timestamp and counter
        let random_seed = timestamp
            .wrapping_mul(6364136223846793005)
            .wrapping_add(counter);
        bytes[8] = (random_seed >> 56) as u8;
        bytes[9] = (random_seed >> 48) as u8;
        bytes[10] = (random_seed >> 40) as u8;
        bytes[11] = (random_seed >> 32) as u8;
        bytes[12] = (random_seed >> 24) as u8;
        bytes[13] = (random_seed >> 16) as u8;
        bytes[14] = (random_seed >> 8) as u8;
        bytes[15] = random_seed as u8;

        Self(bytes)
    }

    /// Create an Id from raw bytes
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    /// Get the raw bytes
    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    /// Extract the timestamp component (milliseconds since epoch)
    pub fn timestamp_ms(&self) -> u64 {
        ((self.0[0] as u64) << 40)
            | ((self.0[1] as u64) << 32)
            | ((self.0[2] as u64) << 24)
            | ((self.0[3] as u64) << 16)
            | ((self.0[4] as u64) << 8)
            | (self.0[5] as u64)
    }

    /// Create a nil/zero Id (useful for testing)
    pub fn nil() -> Self {
        Self([0u8; 16])
    }

    /// Check if this is a nil Id
    pub fn is_nil(&self) -> bool {
        self.0 == [0u8; 16]
    }
}

impl std::fmt::Display for Id {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Display as hex string
        for byte in &self.0 {
            write!(f, "{:02x}", byte)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_id_creation() {
        let id = Id::now();
        assert!(!id.is_nil());
    }

    #[test]
    fn test_id_timestamp() {
        let before = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let id = Id::now();

        let after = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let ts = id.timestamp_ms();
        assert!(ts >= before);
        assert!(ts <= after);
    }

    #[test]
    fn test_id_ordering() {
        let id1 = Id::now();
        thread::sleep(Duration::from_millis(2));
        let id2 = Id::now();

        // id2 should be greater (later timestamp)
        assert!(id2 > id1);
    }

    #[test]
    fn test_id_from_bytes() {
        let bytes = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let id = Id::from_bytes(bytes);
        assert_eq!(id.as_bytes(), &bytes);
    }

    #[test]
    fn test_id_nil() {
        let nil = Id::nil();
        assert!(nil.is_nil());
        assert_eq!(nil.timestamp_ms(), 0);
    }

    #[test]
    fn test_id_display() {
        let id = Id::from_bytes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        let display = format!("{}", id);
        assert_eq!(display, "000102030405060708090a0b0c0d0e0f");
    }
}
