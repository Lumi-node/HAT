//! # Configuration
//!
//! ARMS configuration - define your space.
//!
//! Everything is configurable, not hardcoded:
//! - Dimensionality
//! - Proximity function
//! - Merge function
//! - Tier settings
//!
//! "If we say it's a rock now, in 2 years it can never be carved into a wheel."

use super::proximity::{Cosine, Proximity};
use super::merge::{Mean, Merge};
use std::sync::Arc;

/// Main ARMS configuration
///
/// Defines the dimensional space and default operations.
#[derive(Clone)]
pub struct ArmsConfig {
    /// Dimensionality of the space
    ///
    /// Set this to match your model's hidden size.
    /// Examples: 768 (BERT), 1024 (GPT-2 medium), 4096 (large models)
    pub dimensionality: usize,

    /// Proximity function for similarity calculations
    pub proximity: Arc<dyn Proximity>,

    /// Merge function for hierarchical composition
    pub merge: Arc<dyn Merge>,

    /// Whether to normalize points on insertion
    pub normalize_on_insert: bool,

    /// Tier configuration
    pub tiers: TierConfig,
}

impl ArmsConfig {
    /// Create a new configuration with specified dimensionality
    ///
    /// Uses default proximity (Cosine) and merge (Mean) functions.
    pub fn new(dimensionality: usize) -> Self {
        Self {
            dimensionality,
            proximity: Arc::new(Cosine),
            merge: Arc::new(Mean),
            normalize_on_insert: true,
            tiers: TierConfig::default(),
        }
    }

    /// Set a custom proximity function
    pub fn with_proximity<P: Proximity + 'static>(mut self, proximity: P) -> Self {
        self.proximity = Arc::new(proximity);
        self
    }

    /// Set a custom merge function
    pub fn with_merge<M: Merge + 'static>(mut self, merge: M) -> Self {
        self.merge = Arc::new(merge);
        self
    }

    /// Set normalization behavior
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize_on_insert = normalize;
        self
    }

    /// Set tier configuration
    pub fn with_tiers(mut self, tiers: TierConfig) -> Self {
        self.tiers = tiers;
        self
    }
}

impl Default for ArmsConfig {
    /// Default configuration: 768 dimensions, cosine proximity, mean merge
    fn default() -> Self {
        Self::new(768)
    }
}

/// Tier configuration for storage management
#[derive(Clone, Debug)]
pub struct TierConfig {
    /// Hot tier (RAM) capacity in bytes
    pub hot_capacity: usize,

    /// Warm tier (NVMe) capacity in bytes
    pub warm_capacity: usize,

    /// Number of accesses before promoting to hotter tier
    pub promote_after_accesses: u32,

    /// Milliseconds since last access before evicting to colder tier
    pub evict_after_ms: u64,
}

impl TierConfig {
    /// Create a new tier configuration
    pub fn new(hot_capacity: usize, warm_capacity: usize) -> Self {
        Self {
            hot_capacity,
            warm_capacity,
            promote_after_accesses: 3,
            evict_after_ms: 3600 * 1000, // 1 hour
        }
    }

    /// Tiny config for testing
    pub fn tiny() -> Self {
        Self {
            hot_capacity: 1024 * 1024,        // 1 MB
            warm_capacity: 10 * 1024 * 1024,  // 10 MB
            promote_after_accesses: 2,
            evict_after_ms: 60 * 1000, // 1 minute
        }
    }
}

impl Default for TierConfig {
    fn default() -> Self {
        Self {
            hot_capacity: 1024 * 1024 * 1024,       // 1 GB
            warm_capacity: 100 * 1024 * 1024 * 1024, // 100 GB
            promote_after_accesses: 3,
            evict_after_ms: 3600 * 1000, // 1 hour
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::proximity::Euclidean;
    use crate::core::merge::MaxPool;

    #[test]
    fn test_default_config() {
        let config = ArmsConfig::default();
        assert_eq!(config.dimensionality, 768);
        assert!(config.normalize_on_insert);
        assert_eq!(config.proximity.name(), "cosine");
        assert_eq!(config.merge.name(), "mean");
    }

    #[test]
    fn test_custom_config() {
        let config = ArmsConfig::new(4096)
            .with_proximity(Euclidean)
            .with_merge(MaxPool)
            .with_normalize(false);

        assert_eq!(config.dimensionality, 4096);
        assert!(!config.normalize_on_insert);
        assert_eq!(config.proximity.name(), "euclidean");
        assert_eq!(config.merge.name(), "max_pool");
    }

    #[test]
    fn test_tier_config() {
        let tiers = TierConfig::new(1024, 2048);
        assert_eq!(tiers.hot_capacity, 1024);
        assert_eq!(tiers.warm_capacity, 2048);
    }

    #[test]
    fn test_tier_tiny() {
        let tiers = TierConfig::tiny();
        assert_eq!(tiers.hot_capacity, 1024 * 1024);
        assert_eq!(tiers.evict_after_ms, 60 * 1000);
    }
}
