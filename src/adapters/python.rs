//! # Python Bindings
//!
//! PyO3 bindings for ARMS-HAT, enabling Python integration with LLMs.
//!
//! ## Python API
//!
//! ```python
//! from arms_hat import HatIndex, SearchResult
//!
//! # Create index for OpenAI embeddings (1536 dims)
//! index = HatIndex.cosine(1536)
//!
//! # Add embeddings
//! id = index.add([0.1, 0.2, ...])  # Auto-generates ID
//! index.add_with_id("custom_id", [0.1, 0.2, ...])  # Custom ID
//!
//! # Query
//! results = index.near([0.1, 0.2, ...], k=10)
//! for result in results:
//!     print(f"{result.id}: {result.score}")
//!
//! # Session management
//! index.new_session()
//! index.new_document()
//!
//! # Persistence
//! index.save("memory.hat")
//! loaded = HatIndex.load("memory.hat")
//! ```

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyIOError};

use crate::core::{Id, Point};
use crate::adapters::index::{HatIndex as RustHatIndex, HatConfig, ConsolidationConfig, Consolidate};
use crate::ports::Near;

/// Python wrapper for search results
#[pyclass(name = "SearchResult")]
#[derive(Clone)]
pub struct PySearchResult {
    /// The ID as a hex string
    #[pyo3(get)]
    pub id: String,

    /// The similarity/distance score
    #[pyo3(get)]
    pub score: f32,
}

#[pymethods]
impl PySearchResult {
    fn __repr__(&self) -> String {
        format!("SearchResult(id='{}', score={:.4})", self.id, self.score)
    }

    fn __str__(&self) -> String {
        format!("{}: {:.4}", self.id, self.score)
    }
}

/// Python wrapper for HAT index configuration
#[pyclass(name = "HatConfig")]
#[derive(Clone)]
pub struct PyHatConfig {
    inner: HatConfig,
}

#[pymethods]
impl PyHatConfig {
    #[new]
    fn new() -> Self {
        Self { inner: HatConfig::default() }
    }

    /// Set beam width for search (default: 3)
    fn with_beam_width(mut slf: PyRefMut<'_, Self>, width: usize) -> PyRefMut<'_, Self> {
        slf.inner.beam_width = width;
        slf
    }

    /// Set temporal weight (0.0 = pure semantic, 1.0 = pure temporal)
    fn with_temporal_weight(mut slf: PyRefMut<'_, Self>, weight: f32) -> PyRefMut<'_, Self> {
        slf.inner.temporal_weight = weight;
        slf
    }

    /// Set propagation threshold for sparse updates
    fn with_propagation_threshold(mut slf: PyRefMut<'_, Self>, threshold: f32) -> PyRefMut<'_, Self> {
        slf.inner.propagation_threshold = threshold;
        slf
    }

    fn __repr__(&self) -> String {
        format!(
            "HatConfig(beam_width={}, temporal_weight={:.2}, propagation_threshold={:.3})",
            self.inner.beam_width, self.inner.temporal_weight, self.inner.propagation_threshold
        )
    }
}

/// Session summary for coarse-grained retrieval
#[pyclass(name = "SessionSummary")]
#[derive(Clone)]
pub struct PySessionSummary {
    #[pyo3(get)]
    pub id: String,

    #[pyo3(get)]
    pub score: f32,

    #[pyo3(get)]
    pub chunk_count: usize,

    #[pyo3(get)]
    pub timestamp_ms: u64,
}

#[pymethods]
impl PySessionSummary {
    fn __repr__(&self) -> String {
        format!(
            "SessionSummary(id='{}', score={:.4}, chunks={})",
            self.id, self.score, self.chunk_count
        )
    }
}

/// Document summary for mid-level retrieval
#[pyclass(name = "DocumentSummary")]
#[derive(Clone)]
pub struct PyDocumentSummary {
    #[pyo3(get)]
    pub id: String,

    #[pyo3(get)]
    pub score: f32,

    #[pyo3(get)]
    pub chunk_count: usize,
}

#[pymethods]
impl PyDocumentSummary {
    fn __repr__(&self) -> String {
        format!(
            "DocumentSummary(id='{}', score={:.4}, chunks={})",
            self.id, self.score, self.chunk_count
        )
    }
}

/// Index statistics
#[pyclass(name = "HatStats")]
#[derive(Clone)]
pub struct PyHatStats {
    #[pyo3(get)]
    pub global_count: usize,

    #[pyo3(get)]
    pub session_count: usize,

    #[pyo3(get)]
    pub document_count: usize,

    #[pyo3(get)]
    pub chunk_count: usize,
}

#[pymethods]
impl PyHatStats {
    /// Total number of indexed points
    #[getter]
    fn total_points(&self) -> usize {
        self.chunk_count
    }

    fn __repr__(&self) -> String {
        format!(
            "HatStats(points={}, sessions={}, documents={}, chunks={})",
            self.chunk_count, self.session_count, self.document_count, self.chunk_count
        )
    }
}

/// Hierarchical Attention Tree Index
///
/// A semantic memory index optimized for conversation history retrieval.
/// Uses hierarchical structure (session -> document -> chunk) to enable
/// O(log n) queries while maintaining high recall.
#[pyclass(name = "HatIndex")]
pub struct PyHatIndex {
    inner: RustHatIndex,
}

#[pymethods]
impl PyHatIndex {
    /// Create a new HAT index with cosine similarity
    ///
    /// Args:
    ///     dimensionality: Number of embedding dimensions (e.g., 1536 for OpenAI)
    #[staticmethod]
    fn cosine(dimensionality: usize) -> Self {
        Self {
            inner: RustHatIndex::cosine(dimensionality),
        }
    }

    /// Create a new HAT index with custom configuration
    ///
    /// Args:
    ///     dimensionality: Number of embedding dimensions
    ///     config: HatConfig instance
    #[staticmethod]
    fn with_config(dimensionality: usize, config: &PyHatConfig) -> Self {
        Self {
            inner: RustHatIndex::cosine(dimensionality).with_config(config.inner.clone()),
        }
    }

    /// Add an embedding to the index
    ///
    /// Args:
    ///     embedding: List of floats (must match dimensionality)
    ///
    /// Returns:
    ///     str: The generated ID as a hex string
    fn add(&mut self, embedding: Vec<f32>) -> PyResult<String> {
        let point = Point::new(embedding);
        let id = Id::now();

        self.inner.add(id, &point)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

        Ok(format!("{}", id))
    }

    /// Add an embedding with a custom ID
    ///
    /// Args:
    ///     id_hex: 32-character hex string for the ID
    ///     embedding: List of floats (must match dimensionality)
    fn add_with_id(&mut self, id_hex: &str, embedding: Vec<f32>) -> PyResult<()> {
        let id = parse_id_hex(id_hex)?;
        let point = Point::new(embedding);

        self.inner.add(id, &point)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

        Ok(())
    }

    /// Find k nearest neighbors to a query embedding
    ///
    /// Args:
    ///     query: Query embedding (list of floats)
    ///     k: Number of results to return
    ///
    /// Returns:
    ///     List[SearchResult]: Results sorted by relevance (best first)
    fn near(&self, query: Vec<f32>, k: usize) -> PyResult<Vec<PySearchResult>> {
        let point = Point::new(query);

        let results = self.inner.near(&point, k)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

        Ok(results.into_iter().map(|r| PySearchResult {
            id: format!("{}", r.id),
            score: r.score,
        }).collect())
    }

    /// Start a new session (conversation boundary)
    ///
    /// Call this when starting a new conversation or context.
    fn new_session(&mut self) {
        self.inner.new_session();
    }

    /// Start a new document within the current session
    ///
    /// Call this for logical groupings within a conversation
    /// (e.g., topic change, user turn).
    fn new_document(&mut self) {
        self.inner.new_document();
    }

    /// Get index statistics
    fn stats(&self) -> PyHatStats {
        let s = self.inner.stats();
        PyHatStats {
            global_count: s.global_count,
            session_count: s.session_count,
            document_count: s.document_count,
            chunk_count: s.chunk_count,
        }
    }

    /// Get the number of indexed points
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Check if the index is empty
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Remove a point by ID
    ///
    /// Args:
    ///     id_hex: 32-character hex string for the ID
    fn remove(&mut self, id_hex: &str) -> PyResult<()> {
        let id = parse_id_hex(id_hex)?;

        self.inner.remove(id)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

        Ok(())
    }

    /// Find similar sessions (coarse-grained search)
    ///
    /// Args:
    ///     query: Query embedding
    ///     k: Number of sessions to return
    ///
    /// Returns:
    ///     List[SessionSummary]: Most relevant sessions
    fn near_sessions(&self, query: Vec<f32>, k: usize) -> PyResult<Vec<PySessionSummary>> {
        let point = Point::new(query);

        let results = self.inner.near_sessions(&point, k)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

        Ok(results.into_iter().map(|s| PySessionSummary {
            id: format!("{}", s.id),
            score: s.score,
            chunk_count: s.chunk_count,
            timestamp_ms: s.timestamp,
        }).collect())
    }

    /// Find similar documents within a session
    ///
    /// Args:
    ///     session_id: Session ID (hex string)
    ///     query: Query embedding
    ///     k: Number of documents to return
    ///
    /// Returns:
    ///     List[DocumentSummary]: Most relevant documents in the session
    fn near_documents(&self, session_id: &str, query: Vec<f32>, k: usize) -> PyResult<Vec<PyDocumentSummary>> {
        let sid = parse_id_hex(session_id)?;
        let point = Point::new(query);

        let results = self.inner.near_documents(sid, &point, k)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

        Ok(results.into_iter().map(|d| PyDocumentSummary {
            id: format!("{}", d.id),
            score: d.score,
            chunk_count: d.chunk_count,
        }).collect())
    }

    /// Find chunks within a specific document
    ///
    /// Args:
    ///     doc_id: Document ID (hex string)
    ///     query: Query embedding
    ///     k: Number of results to return
    ///
    /// Returns:
    ///     List[SearchResult]: Most relevant chunks in the document
    fn near_in_document(&self, doc_id: &str, query: Vec<f32>, k: usize) -> PyResult<Vec<PySearchResult>> {
        let did = parse_id_hex(doc_id)?;
        let point = Point::new(query);

        let results = self.inner.near_in_document(did, &point, k)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

        Ok(results.into_iter().map(|r| PySearchResult {
            id: format!("{}", r.id),
            score: r.score,
        }).collect())
    }

    /// Run light consolidation (background maintenance)
    ///
    /// This optimizes the index structure. Call periodically
    /// (e.g., after every 100 inserts).
    fn consolidate(&mut self) {
        self.inner.consolidate(ConsolidationConfig::light());
    }

    /// Run full consolidation (more thorough optimization)
    fn consolidate_full(&mut self) {
        self.inner.consolidate(ConsolidationConfig::full());
    }

    /// Save the index to a file
    ///
    /// Args:
    ///     path: File path to save to
    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save_to_file(std::path::Path::new(path))
            .map_err(|e| PyIOError::new_err(format!("{}", e)))
    }

    /// Load an index from a file
    ///
    /// Args:
    ///     path: File path to load from
    ///
    /// Returns:
    ///     HatIndex: The loaded index
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = RustHatIndex::load_from_file(std::path::Path::new(path))
            .map_err(|e| PyIOError::new_err(format!("{}", e)))?;

        Ok(Self { inner })
    }

    /// Serialize the index to bytes
    ///
    /// Returns:
    ///     bytes: Serialized index data
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let data = self.inner.to_bytes()
            .map_err(|e| PyIOError::new_err(format!("{}", e)))?;
        Ok(pyo3::types::PyBytes::new_bound(py, &data))
    }

    /// Load an index from bytes
    ///
    /// Args:
    ///     data: Serialized index data
    ///
    /// Returns:
    ///     HatIndex: The loaded index
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        let inner = RustHatIndex::from_bytes(data)
            .map_err(|e| PyIOError::new_err(format!("{}", e)))?;

        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        let stats = self.inner.stats();
        format!(
            "HatIndex(points={}, sessions={})",
            stats.chunk_count, stats.session_count
        )
    }
}

/// Parse a hex string to an Id
fn parse_id_hex(hex: &str) -> PyResult<Id> {
    if hex.len() != 32 {
        return Err(PyValueError::new_err(
            format!("ID must be 32 hex characters, got {}", hex.len())
        ));
    }

    let mut bytes = [0u8; 16];
    for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
        let high = hex_char_to_nibble(chunk[0])?;
        let low = hex_char_to_nibble(chunk[1])?;
        bytes[i] = (high << 4) | low;
    }

    Ok(Id::from_bytes(bytes))
}

fn hex_char_to_nibble(c: u8) -> PyResult<u8> {
    match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'a'..=b'f' => Ok(c - b'a' + 10),
        b'A'..=b'F' => Ok(c - b'A' + 10),
        _ => Err(PyValueError::new_err(format!("Invalid hex character: {}", c as char))),
    }
}

/// ARMS-HAT Python module
#[pymodule]
fn arms_hat(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHatIndex>()?;
    m.add_class::<PyHatConfig>()?;
    m.add_class::<PySearchResult>()?;
    m.add_class::<PySessionSummary>()?;
    m.add_class::<PyDocumentSummary>()?;
    m.add_class::<PyHatStats>()?;

    // Add module docstring
    m.add("__doc__", "ARMS-HAT: Hierarchical Attention Tree for AI memory retrieval")?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
