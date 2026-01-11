//! # Ports
//!
//! Trait definitions for adapters. Contracts only, no implementations.
//!
//! This is the hexagonal architecture boundary:
//! - Ports define WHAT operations are needed
//! - Adapters define HOW they're implemented
//!
//! The CORE doesn't know about adapters.
//! Adapters implement these port traits.

mod place;
mod near;
mod latency;

// Re-export traits
pub use place::Place;
pub use near::Near;
pub use latency::Latency;

// Re-export types from place
pub use place::{PlaceError, PlaceResult};

// Re-export types from near
pub use near::{NearError, NearResult, SearchResult};

// Re-export types from latency
pub use latency::{Tier, LatencyBudget, LatencyMeasurement, TierStats};
