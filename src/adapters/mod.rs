//! # Adapters
//!
//! Swappable implementations of port traits.
//!
//! This is where the hexagonal architecture meets reality:
//! - Storage adapters: Memory, NVMe
//! - Index adapters: Flat (brute force), HNSW (approximate)
//! - Attention state serialization
//! - Python bindings (when enabled)
//!
//! Each adapter implements one or more port traits.
//! Adapters can be swapped without changing core logic.

pub mod storage;
pub mod index;
pub mod attention;

#[cfg(feature = "python")]
pub mod python;
