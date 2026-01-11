//! # Index Adapters
//!
//! Implementations of the Near port for different index backends.
//!
//! Available adapters:
//! - `FlatIndex` - Brute force search (exact, O(n) per query)
//! - `HatIndex` - Hierarchical Attention Tree (approximate, O(log n) per query)
//!
//! Consolidation support:
//! - `Consolidate` trait for background maintenance operations
//! - `ConsolidationConfig` to configure maintenance behavior
//!
//! Subspace support:
//! - `Subspace` representation for containers capturing variance/spread
//! - `SubspaceConfig` for configuring subspace-aware routing
//!
//! Learnable routing:
//! - `LearnableRouter` for adapting routing weights from feedback
//! - `LearnableRoutingConfig` for configuring online learning

mod flat;
mod hat;
mod consolidation;
mod subspace;
mod learnable_routing;
mod persistence;

pub use flat::FlatIndex;
pub use hat::{HatIndex, HatConfig, CentroidMethod, ContainerLevel, SessionSummary, DocumentSummary, HatStats};
pub use consolidation::{
    Consolidate, ConsolidationConfig, ConsolidationLevel, ConsolidationPhase,
    ConsolidationState, ConsolidationMetrics, ConsolidationProgress, ConsolidationTickResult,
    compute_exact_centroid, centroid_drift,
};
pub use subspace::{
    Subspace, SubspaceConfig, subspace_similarity, combined_subspace_similarity,
    query_subspace_alignment, subspace_spread, subspace_isotropy,
};
pub use learnable_routing::{
    LearnableRouter, LearnableRoutingConfig, RoutingFeedback, RouterStats,
    compute_routing_score,
};
pub use persistence::{
    PersistError, SerializedHat, SerializedContainer, LevelByte,
};
