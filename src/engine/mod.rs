//! # Engine
//!
//! The orchestration layer that wires everything together.
//!
//! This is where:
//! - Configuration is applied
//! - Adapters are connected to ports
//! - The unified ARMS interface is exposed

mod arms;

pub use arms::Arms;
