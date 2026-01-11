//! # ARMS - Attention Reasoning Memory Store
//!
//! > "The hippocampus of artificial minds"
//!
//! ARMS is a spatial memory fabric for AI models. It stores computed attention
//! states at their native dimensional coordinates, enabling instant retrieval
//! by proximity rather than traditional indexing.
//!
//! ## Philosophy
//!
//! - **Position IS relationship** - No foreign keys, proximity defines connection
//! - **Configurable, not hardcoded** - Dimensionality, proximity functions, all flexible
//! - **Generators over assets** - Algorithms, not rigid structures
//! - **Pure core, swappable adapters** - Hexagonal architecture
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                         ARMS                                 │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                              │
//! │  CORE (pure math, no I/O)                                   │
//! │    Point, Id, Blob, Proximity, Merge                        │
//! │                                                              │
//! │  PORTS (trait contracts)                                     │
//! │    Place, Near, Latency                                     │
//! │                                                              │
//! │  ADAPTERS (swappable implementations)                       │
//! │    Storage: Memory, NVMe                                    │
//! │    Index: Flat, HNSW                                        │
//! │    API: Python bindings                                      │
//! │                                                              │
//! │  ENGINE (orchestration)                                      │
//! │    Arms - the main entry point                              │
//! │                                                              │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use arms::{Arms, ArmsConfig, Point};
//!
//! // Create ARMS with default config (768 dimensions)
//! let mut arms = Arms::new(ArmsConfig::default());
//!
//! // Place a point in the space
//! let point = Point::new(vec![0.1; 768]);
//! let id = arms.place(point, b"my data".to_vec());
//!
//! // Find nearby points
//! let query = Point::new(vec![0.1; 768]);
//! let neighbors = arms.near(&query, 5);
//! ```

// ============================================================================
// MODULES
// ============================================================================

/// Core domain - pure math, no I/O
/// Contains: Point, Id, Blob, Proximity trait, Merge trait
pub mod core;

/// Port definitions - trait contracts for adapters
/// Contains: Place trait, Near trait, Latency trait
pub mod ports;

/// Adapter implementations - swappable components
/// Contains: storage, index, python submodules
pub mod adapters;

/// Engine - orchestration layer
/// Contains: Arms main struct
pub mod engine;

// ============================================================================
// PYTHON BINDINGS (when enabled)
// ============================================================================

#[cfg(feature = "python")]
pub use adapters::python::*;

// ============================================================================
// RE-EXPORTS (public API)
// ============================================================================

// Core types
pub use crate::core::{Point, Id, Blob, PlacedPoint};
pub use crate::core::proximity::{Proximity, Cosine, Euclidean, DotProduct};
pub use crate::core::merge::{Merge, Mean, WeightedMean, MaxPool};
pub use crate::core::config::ArmsConfig;

// Port traits
pub use crate::ports::{Place, Near, Latency};

// Engine
pub use crate::engine::Arms;

// ============================================================================
// CRATE-LEVEL DOCUMENTATION
// ============================================================================

/// The five primitives of ARMS:
///
/// 1. **Point**: `Vec<f32>` - Any dimensionality
/// 2. **Proximity**: `fn(a, b) -> f32` - How related?
/// 3. **Merge**: `fn(points) -> point` - Compose together
/// 4. **Place**: `fn(point, data) -> id` - Exist in space
/// 5. **Near**: `fn(point, k) -> ids` - What's related?
///
/// Everything else is configuration or adapters.
#[doc(hidden)]
pub const _PRIMITIVES: () = ();
