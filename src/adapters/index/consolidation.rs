//! # Consolidation Phases for HAT
//!
//! Background maintenance operations inspired by memory consolidation in the brain.
//! Like sleep stages (REM/NREM), HAT needs periodic "offline" maintenance to:
//!
//! 1. **Recompute Centroids**: Incremental updates accumulate drift - recompute from scratch
//! 2. **Rebalance Tree**: Merge underpopulated containers, split overpopulated ones
//! 3. **Prune Stale Branches**: Remove containers with no descendants
//! 4. **Optimize Layout**: Reorder children for better cache locality
//!
//! ## Design Philosophy
//!
//! Consolidation is designed to be:
//! - **Non-blocking**: Can run incrementally, yielding to queries
//! - **Resumable**: Can pause and resume without data loss
//! - **Observable**: Reports progress and metrics for benchmarking
//!
//! ## Consolidation Levels
//!
//! Like sleep stages, different consolidation depths:
//!
//! - **Light** (α): Recompute centroids only (~NREM Stage 1)
//! - **Medium** (β): + Rebalance tree structure (~NREM Stage 2-3)
//! - **Deep** (δ): + Optimize layout, prune stale (~NREM Stage 4 / SWS)
//! - **Full** (θ): Complete rebuild from scratch (~REM)

use std::collections::{HashMap, HashSet, VecDeque};

use crate::core::{Id, Point};

/// Consolidation level - determines how deep the maintenance goes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsolidationLevel {
    /// Light: Recompute centroids only
    /// Fast, minimal disruption, good for frequent runs
    Light,

    /// Medium: Recompute centroids + rebalance tree
    /// Moderate time, restructures containers
    Medium,

    /// Deep: Full maintenance including layout optimization
    /// Longer time, comprehensive cleanup
    Deep,

    /// Full: Complete rebuild from leaf nodes
    /// Longest time, guarantees optimal structure
    Full,
}

impl Default for ConsolidationLevel {
    fn default() -> Self {
        ConsolidationLevel::Medium
    }
}

/// Configuration for consolidation operations
#[derive(Debug, Clone)]
pub struct ConsolidationConfig {
    /// Target level of consolidation
    pub level: ConsolidationLevel,

    /// Maximum containers to process per tick (for incremental consolidation)
    pub batch_size: usize,

    /// Minimum children before considering merge
    pub merge_threshold: usize,

    /// Maximum children before considering split
    pub split_threshold: usize,

    /// Maximum centroid drift (L2) before triggering recompute
    /// 0.0 = always recompute, higher values = more lenient
    pub drift_threshold: f32,

    /// Whether to collect detailed metrics
    pub collect_metrics: bool,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            level: ConsolidationLevel::Medium,
            batch_size: 100,
            merge_threshold: 3,
            split_threshold: 100,
            drift_threshold: 0.01,
            collect_metrics: true,
        }
    }
}

impl ConsolidationConfig {
    pub fn light() -> Self {
        Self {
            level: ConsolidationLevel::Light,
            ..Default::default()
        }
    }

    pub fn medium() -> Self {
        Self {
            level: ConsolidationLevel::Medium,
            ..Default::default()
        }
    }

    pub fn deep() -> Self {
        Self {
            level: ConsolidationLevel::Deep,
            ..Default::default()
        }
    }

    pub fn full() -> Self {
        Self {
            level: ConsolidationLevel::Full,
            ..Default::default()
        }
    }

    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }
}

/// Current state of consolidation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsolidationPhase {
    /// Not currently consolidating
    Idle,

    /// Phase 1: Collecting all leaf points
    CollectingLeaves,

    /// Phase 2: Recomputing centroids bottom-up
    RecomputingCentroids,

    /// Phase 3: Identifying containers to merge/split
    AnalyzingStructure,

    /// Phase 4: Performing merges
    Merging,

    /// Phase 5: Performing splits
    Splitting,

    /// Phase 6: Pruning empty containers
    Pruning,

    /// Phase 7: Optimizing layout
    OptimizingLayout,

    /// Consolidation complete
    Complete,
}

/// Metrics collected during consolidation
#[derive(Debug, Clone, Default)]
pub struct ConsolidationMetrics {
    /// Total containers processed
    pub containers_processed: usize,

    /// Centroids recomputed
    pub centroids_recomputed: usize,

    /// Average centroid drift (L2 norm of delta)
    pub avg_centroid_drift: f32,

    /// Maximum centroid drift observed
    pub max_centroid_drift: f32,

    /// Number of containers merged
    pub containers_merged: usize,

    /// Number of containers split
    pub containers_split: usize,

    /// Number of empty containers pruned
    pub containers_pruned: usize,

    /// Time spent in each phase (microseconds)
    pub phase_times_us: HashMap<String, u64>,

    /// Total consolidation time (microseconds)
    pub total_time_us: u64,

    /// Number of ticks (for incremental consolidation)
    pub ticks: usize,
}

/// Progress report for observable consolidation
#[derive(Debug, Clone)]
pub struct ConsolidationProgress {
    /// Current phase
    pub phase: ConsolidationPhase,

    /// Percentage complete (0.0 - 1.0)
    pub progress: f32,

    /// Containers remaining in current phase
    pub remaining: usize,

    /// Running metrics
    pub metrics: ConsolidationMetrics,
}

/// Internal state for resumable consolidation
#[derive(Debug)]
pub struct ConsolidationState {
    /// Configuration
    pub config: ConsolidationConfig,

    /// Current phase
    pub phase: ConsolidationPhase,

    /// Collected metrics
    pub metrics: ConsolidationMetrics,

    /// Queue of containers to process in current phase
    pub work_queue: VecDeque<Id>,

    /// Set of containers already processed
    pub processed: HashSet<Id>,

    /// Accumulated centroid drifts for averaging
    centroid_drifts: Vec<f32>,

    /// Containers identified for merging (pairs)
    merge_candidates: Vec<(Id, Id)>,

    /// Containers identified for splitting
    split_candidates: Vec<Id>,

    /// Phase start timestamp (for timing)
    phase_start_us: u64,

    /// Consolidation start timestamp
    start_us: u64,
}

impl ConsolidationState {
    /// Create a new consolidation state
    pub fn new(config: ConsolidationConfig) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        Self {
            config,
            phase: ConsolidationPhase::Idle,
            metrics: ConsolidationMetrics::default(),
            work_queue: VecDeque::new(),
            processed: HashSet::new(),
            centroid_drifts: Vec::new(),
            merge_candidates: Vec::new(),
            split_candidates: Vec::new(),
            phase_start_us: now,
            start_us: now,
        }
    }

    /// Start consolidation
    pub fn start(&mut self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        self.start_us = now;
        self.phase_start_us = now;
        self.phase = ConsolidationPhase::CollectingLeaves;
        self.metrics = ConsolidationMetrics::default();
        self.work_queue.clear();
        self.processed.clear();
        self.centroid_drifts.clear();
        self.merge_candidates.clear();
        self.split_candidates.clear();
    }

    /// Transition to next phase
    pub fn next_phase(&mut self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        // Record time for previous phase
        let phase_time = now - self.phase_start_us;
        let phase_name = format!("{:?}", self.phase);
        self.metrics.phase_times_us.insert(phase_name, phase_time);

        // Compute average drift if we have samples
        if !self.centroid_drifts.is_empty() {
            self.metrics.avg_centroid_drift =
                self.centroid_drifts.iter().sum::<f32>() / self.centroid_drifts.len() as f32;
        }

        // Determine next phase based on level
        self.phase = match (self.phase, self.config.level) {
            (ConsolidationPhase::Idle, _) => ConsolidationPhase::CollectingLeaves,

            (ConsolidationPhase::CollectingLeaves, _) => ConsolidationPhase::RecomputingCentroids,

            (ConsolidationPhase::RecomputingCentroids, ConsolidationLevel::Light) => {
                ConsolidationPhase::Complete
            }
            (ConsolidationPhase::RecomputingCentroids, _) => {
                ConsolidationPhase::AnalyzingStructure
            }

            (ConsolidationPhase::AnalyzingStructure, _) => ConsolidationPhase::Merging,

            (ConsolidationPhase::Merging, _) => ConsolidationPhase::Splitting,

            (ConsolidationPhase::Splitting, ConsolidationLevel::Medium) => {
                ConsolidationPhase::Complete
            }
            (ConsolidationPhase::Splitting, _) => ConsolidationPhase::Pruning,

            (ConsolidationPhase::Pruning, _) => ConsolidationPhase::OptimizingLayout,

            (ConsolidationPhase::OptimizingLayout, _) => ConsolidationPhase::Complete,

            (ConsolidationPhase::Complete, _) => ConsolidationPhase::Complete,
        };

        // Reset for new phase
        self.phase_start_us = now;
        self.work_queue.clear();
        self.processed.clear();

        // Record total time if complete
        if self.phase == ConsolidationPhase::Complete {
            self.metrics.total_time_us = now - self.start_us;
        }
    }

    /// Record a centroid drift
    pub fn record_drift(&mut self, drift: f32) {
        self.centroid_drifts.push(drift);
        if drift > self.metrics.max_centroid_drift {
            self.metrics.max_centroid_drift = drift;
        }
    }

    /// Add merge candidate pair
    pub fn add_merge_candidate(&mut self, a: Id, b: Id) {
        self.merge_candidates.push((a, b));
    }

    /// Add split candidate
    pub fn add_split_candidate(&mut self, id: Id) {
        self.split_candidates.push(id);
    }

    /// Get next merge candidate pair
    pub fn next_merge(&mut self) -> Option<(Id, Id)> {
        self.merge_candidates.pop()
    }

    /// Get next split candidate
    pub fn next_split(&mut self) -> Option<Id> {
        self.split_candidates.pop()
    }

    /// Check if there are pending merge candidates
    pub fn has_merges(&self) -> bool {
        !self.merge_candidates.is_empty()
    }

    /// Check if there are pending split candidates
    pub fn has_splits(&self) -> bool {
        !self.split_candidates.is_empty()
    }

    /// Check if consolidation is complete
    pub fn is_complete(&self) -> bool {
        self.phase == ConsolidationPhase::Complete
    }

    /// Get progress report
    pub fn progress(&self) -> ConsolidationProgress {
        let remaining = self.work_queue.len();
        let total = remaining + self.processed.len();
        let progress = if total > 0 {
            self.processed.len() as f32 / total as f32
        } else {
            1.0
        };

        ConsolidationProgress {
            phase: self.phase,
            progress,
            remaining,
            metrics: self.metrics.clone(),
        }
    }
}

/// Result of a single consolidation tick
#[derive(Debug)]
pub enum ConsolidationTickResult {
    /// Still working, more ticks needed
    Continue(ConsolidationProgress),

    /// Consolidation complete
    Complete(ConsolidationMetrics),
}

/// Trait for types that support consolidation
pub trait Consolidate {
    /// Begin consolidation with given config
    fn begin_consolidation(&mut self, config: ConsolidationConfig);

    /// Execute one tick of consolidation
    /// Returns Continue if more work remains, Complete when done
    fn consolidation_tick(&mut self) -> ConsolidationTickResult;

    /// Run consolidation to completion (blocking)
    fn consolidate(&mut self, config: ConsolidationConfig) -> ConsolidationMetrics {
        self.begin_consolidation(config);
        loop {
            match self.consolidation_tick() {
                ConsolidationTickResult::Continue(_) => continue,
                ConsolidationTickResult::Complete(metrics) => return metrics,
            }
        }
    }

    /// Check if consolidation is in progress
    fn is_consolidating(&self) -> bool;

    /// Get current consolidation progress
    fn consolidation_progress(&self) -> Option<ConsolidationProgress>;

    /// Cancel ongoing consolidation
    fn cancel_consolidation(&mut self);
}

/// Helper for computing exact centroids from a set of points
pub fn compute_exact_centroid(points: &[Point]) -> Option<Point> {
    if points.is_empty() {
        return None;
    }

    let dims = points[0].dimensionality();
    let mut sum = vec![0.0f32; dims];

    for point in points {
        for (i, &val) in point.dims().iter().enumerate() {
            sum[i] += val;
        }
    }

    let n = points.len() as f32;
    let mean: Vec<f32> = sum.iter().map(|s| s / n).collect();

    Some(Point::new(mean).normalize())
}

/// Helper to measure centroid drift
pub fn centroid_drift(old: &Point, new: &Point) -> f32 {
    old.dims()
        .iter()
        .zip(new.dims().iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consolidation_config_levels() {
        let light = ConsolidationConfig::light();
        assert_eq!(light.level, ConsolidationLevel::Light);

        let medium = ConsolidationConfig::medium();
        assert_eq!(medium.level, ConsolidationLevel::Medium);

        let deep = ConsolidationConfig::deep();
        assert_eq!(deep.level, ConsolidationLevel::Deep);

        let full = ConsolidationConfig::full();
        assert_eq!(full.level, ConsolidationLevel::Full);
    }

    #[test]
    fn test_consolidation_state_phases() {
        let config = ConsolidationConfig::light();
        let mut state = ConsolidationState::new(config);

        assert_eq!(state.phase, ConsolidationPhase::Idle);

        state.start();
        assert_eq!(state.phase, ConsolidationPhase::CollectingLeaves);

        state.next_phase();
        assert_eq!(state.phase, ConsolidationPhase::RecomputingCentroids);

        // Light level skips to complete after centroids
        state.next_phase();
        assert_eq!(state.phase, ConsolidationPhase::Complete);
        assert!(state.is_complete());
    }

    #[test]
    fn test_consolidation_state_medium_phases() {
        let config = ConsolidationConfig::medium();
        let mut state = ConsolidationState::new(config);

        state.start();
        assert_eq!(state.phase, ConsolidationPhase::CollectingLeaves);

        state.next_phase();
        assert_eq!(state.phase, ConsolidationPhase::RecomputingCentroids);

        state.next_phase();
        assert_eq!(state.phase, ConsolidationPhase::AnalyzingStructure);

        state.next_phase();
        assert_eq!(state.phase, ConsolidationPhase::Merging);

        state.next_phase();
        assert_eq!(state.phase, ConsolidationPhase::Splitting);

        // Medium level completes after splitting
        state.next_phase();
        assert_eq!(state.phase, ConsolidationPhase::Complete);
    }

    #[test]
    fn test_centroid_computation() {
        let points = vec![
            Point::new(vec![1.0, 0.0, 0.0]),
            Point::new(vec![0.0, 1.0, 0.0]),
            Point::new(vec![0.0, 0.0, 1.0]),
        ];

        let centroid = compute_exact_centroid(&points).unwrap();

        // Should be normalized mean
        let expected_unnorm = (1.0f32 / 3.0).sqrt();
        for dim in centroid.dims() {
            assert!((dim - expected_unnorm).abs() < 0.01);
        }
    }

    #[test]
    fn test_centroid_drift() {
        let old = Point::new(vec![1.0, 0.0, 0.0]);
        let new = Point::new(vec![0.9, 0.1, 0.0]).normalize();

        let drift = centroid_drift(&old, &new);
        assert!(drift > 0.0);
        assert!(drift < 1.0);
    }

    #[test]
    fn test_drift_recording() {
        let config = ConsolidationConfig::default();
        let mut state = ConsolidationState::new(config);

        state.record_drift(0.05);
        state.record_drift(0.10);
        state.record_drift(0.02);

        assert_eq!(state.metrics.max_centroid_drift, 0.10);
        assert_eq!(state.centroid_drifts.len(), 3);
    }
}
