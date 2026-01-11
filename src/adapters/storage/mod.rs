//! # Storage Adapters
//!
//! Implementations of the Place port for different storage backends.
//!
//! Available adapters:
//! - `MemoryStorage` - In-memory HashMap (fast, volatile)
//! - `NvmeStorage` - Memory-mapped NVMe (persistent, large) [TODO]

mod memory;

pub use memory::MemoryStorage;

// TODO: Add NVMe adapter
// mod nvme;
// pub use nvme::NvmeStorage;
