//! # Latency Port
//!
//! Trait for runtime latency measurement and adaptation.
//!
//! This enables the model to know its actual retrieval constraints:
//! - How fast is the hot tier right now?
//! - How much budget do I have for retrieval?
//! - Should I use fewer, faster retrievals or more, slower ones?

use std::time::Duration;

/// Storage tier levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tier {
    /// RAM storage - fastest
    Hot,
    /// NVMe storage - fast
    Warm,
    /// Archive storage - slow
    Cold,
}

impl Tier {
    /// Get expected latency range for this tier
    pub fn expected_latency(&self) -> (Duration, Duration) {
        match self {
            Tier::Hot => (Duration::from_micros(1), Duration::from_millis(1)),
            Tier::Warm => (Duration::from_millis(1), Duration::from_millis(10)),
            Tier::Cold => (Duration::from_millis(10), Duration::from_millis(100)),
        }
    }
}

/// Latency measurement result
#[derive(Debug, Clone)]
pub struct LatencyMeasurement {
    /// The tier that was measured
    pub tier: Tier,

    /// Measured latency for a single operation
    pub latency: Duration,

    /// Throughput (operations per second) if measured
    pub throughput_ops: Option<f64>,

    /// Timestamp of measurement
    pub measured_at: std::time::Instant,
}

/// Budget allocation for retrieval operations
#[derive(Debug, Clone)]
pub struct LatencyBudget {
    /// Total time budget for this retrieval batch
    pub total: Duration,

    /// Maximum time per individual retrieval
    pub per_operation: Duration,

    /// Maximum number of operations in this budget
    pub max_operations: usize,
}

impl Default for LatencyBudget {
    fn default() -> Self {
        Self {
            total: Duration::from_millis(50),
            per_operation: Duration::from_millis(5),
            max_operations: 10,
        }
    }
}

/// Tier statistics
#[derive(Debug, Clone)]
pub struct TierStats {
    /// The tier
    pub tier: Tier,

    /// Number of points in this tier
    pub count: usize,

    /// Total size in bytes
    pub size_bytes: usize,

    /// Capacity in bytes
    pub capacity_bytes: usize,

    /// Usage ratio (0.0 to 1.0)
    pub usage_ratio: f32,
}

/// Trait for latency measurement and adaptation
///
/// System adapters implement this trait.
pub trait Latency: Send + Sync {
    /// Probe a tier to measure current latency
    ///
    /// Performs a small test operation to measure actual latency.
    fn probe(&mut self, tier: Tier) -> LatencyMeasurement;

    /// Get the current latency budget
    fn budget(&self) -> LatencyBudget;

    /// Set a new latency budget
    fn set_budget(&mut self, budget: LatencyBudget);

    /// Get available capacity in a tier
    fn available_capacity(&self, tier: Tier) -> usize;

    /// Recommend which tier to use for an access pattern
    ///
    /// `expected_accesses` is the expected number of accesses for this data.
    fn recommend_tier(&self, expected_accesses: u32) -> Tier;

    /// Get statistics for a tier
    fn tier_stats(&self, tier: Tier) -> TierStats;

    /// Get statistics for all tiers
    fn all_stats(&self) -> Vec<TierStats> {
        vec![
            self.tier_stats(Tier::Hot),
            self.tier_stats(Tier::Warm),
            self.tier_stats(Tier::Cold),
        ]
    }
}
