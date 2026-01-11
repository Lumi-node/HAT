//! # HAT Index Adapter
//!
//! Hierarchical Attention Tree - a novel index structure for AI memory.
//! Exploits known semantic hierarchy and temporal locality.
//!
//! Key insight: Unlike HNSW which learns topology from data,
//! HAT uses KNOWN hierarchy (session → document → chunk).
//!
//! Query complexity: O(log n) via tree descent
//! Insert complexity: O(log n) with incremental centroid updates

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::core::{Id, Point};
use crate::core::proximity::Proximity;
use crate::core::merge::Merge;
use crate::ports::{Near, NearError, NearResult, SearchResult};

use super::consolidation::{
    Consolidate, ConsolidationConfig, ConsolidationPhase, ConsolidationState,
    ConsolidationMetrics, ConsolidationProgress, ConsolidationTickResult,
    compute_exact_centroid, centroid_drift,
};

/// Centroid computation method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CentroidMethod {
    /// Euclidean mean + renormalize (fast but geometrically imprecise)
    Euclidean,
    /// Fréchet mean on hypersphere (manifold-aware, more accurate)
    Frechet,
}

impl Default for CentroidMethod {
    fn default() -> Self {
        CentroidMethod::Euclidean
    }
}

/// HAT configuration parameters
#[derive(Debug, Clone)]
pub struct HatConfig {
    /// Maximum children per container before splitting
    pub max_children: usize,

    /// Minimum children to maintain (for merging)
    pub min_children: usize,

    /// Number of branches to explore at each level (beam width)
    pub beam_width: usize,

    /// Weight for temporal proximity in scoring (0.0 = pure semantic)
    pub temporal_weight: f32,

    /// Time decay factor (higher = faster decay)
    pub time_decay: f32,

    /// Threshold for sparse centroid propagation (0.0 = always propagate)
    /// Only propagate to parent if centroid change magnitude exceeds this
    pub propagation_threshold: f32,

    /// Method for computing centroids
    pub centroid_method: CentroidMethod,

    /// Number of iterations for Fréchet mean computation
    pub frechet_iterations: usize,

    /// Enable subspace-aware routing (default: false for backward compatibility)
    pub subspace_enabled: bool,

    /// Configuration for subspace representation
    pub subspace_config: super::subspace::SubspaceConfig,

    /// Enable learnable routing (default: false for backward compatibility)
    pub learnable_routing_enabled: bool,

    /// Configuration for learnable routing
    pub learnable_routing_config: super::learnable_routing::LearnableRoutingConfig,
}

impl Default for HatConfig {
    fn default() -> Self {
        Self {
            max_children: 50,
            min_children: 5,
            beam_width: 3,
            temporal_weight: 0.0, // Start with pure semantic
            time_decay: 0.001,
            propagation_threshold: 0.0, // Default: always propagate (backward compatible)
            centroid_method: CentroidMethod::Euclidean, // Default: backward compatible
            frechet_iterations: 5, // Enough for convergence on hypersphere
            subspace_enabled: false, // Default: disabled for backward compatibility
            subspace_config: super::subspace::SubspaceConfig::default(),
            learnable_routing_enabled: false, // Default: disabled for backward compatibility
            learnable_routing_config: super::learnable_routing::LearnableRoutingConfig::default(),
        }
    }
}

impl HatConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_beam_width(mut self, width: usize) -> Self {
        self.beam_width = width;
        self
    }

    pub fn with_temporal_weight(mut self, weight: f32) -> Self {
        self.temporal_weight = weight;
        self
    }

    pub fn with_propagation_threshold(mut self, threshold: f32) -> Self {
        self.propagation_threshold = threshold;
        self
    }

    pub fn with_centroid_method(mut self, method: CentroidMethod) -> Self {
        self.centroid_method = method;
        self
    }

    pub fn with_frechet_iterations(mut self, iterations: usize) -> Self {
        self.frechet_iterations = iterations;
        self
    }

    pub fn with_subspace_enabled(mut self, enabled: bool) -> Self {
        self.subspace_enabled = enabled;
        self
    }

    pub fn with_subspace_config(mut self, config: super::subspace::SubspaceConfig) -> Self {
        self.subspace_config = config;
        self.subspace_enabled = true;  // Automatically enable when config is provided
        self
    }

    pub fn with_learnable_routing_enabled(mut self, enabled: bool) -> Self {
        self.learnable_routing_enabled = enabled;
        self
    }

    pub fn with_learnable_routing_config(mut self, config: super::learnable_routing::LearnableRoutingConfig) -> Self {
        self.learnable_routing_config = config;
        self.learnable_routing_enabled = true;  // Automatically enable when config is provided
        self
    }
}

/// Level in the hierarchy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerLevel {
    /// Root level - single global container
    Global,
    /// Session level - conversation/context boundaries
    Session,
    /// Document level - logical groupings within session
    Document,
    /// Chunk level - leaf nodes, actual attention states
    Chunk,
}

impl ContainerLevel {
    fn child_level(&self) -> Option<ContainerLevel> {
        match self {
            ContainerLevel::Global => Some(ContainerLevel::Session),
            ContainerLevel::Session => Some(ContainerLevel::Document),
            ContainerLevel::Document => Some(ContainerLevel::Chunk),
            ContainerLevel::Chunk => None,
        }
    }

    fn depth(&self) -> usize {
        match self {
            ContainerLevel::Global => 0,
            ContainerLevel::Session => 1,
            ContainerLevel::Document => 2,
            ContainerLevel::Chunk => 3,
        }
    }
}

/// Summary of a session for coarse queries (multi-resolution API)
#[derive(Debug, Clone)]
pub struct SessionSummary {
    /// Session ID
    pub id: Id,

    /// Similarity score to query
    pub score: f32,

    /// Number of chunks in this session
    pub chunk_count: usize,

    /// Session timestamp
    pub timestamp: u64,
}

/// Summary of a document for coarse queries
#[derive(Debug, Clone)]
pub struct DocumentSummary {
    /// Document ID
    pub id: Id,

    /// Similarity score to query
    pub score: f32,

    /// Number of chunks in this document
    pub chunk_count: usize,

    /// Document timestamp
    pub timestamp: u64,
}

/// A container in the HAT hierarchy
#[derive(Debug, Clone)]
struct Container {
    /// Unique identifier
    id: Id,

    /// Level in hierarchy
    level: ContainerLevel,

    /// Centroid (mean of children)
    centroid: Point,

    /// Creation timestamp (ms since epoch)
    timestamp: u64,

    /// Child container IDs (empty for chunks)
    children: Vec<Id>,

    /// Number of descendant chunks (for weighted centroid updates)
    descendant_count: usize,

    /// Accumulated sum of all descendant points (for Euclidean centroid)
    /// Stored as unnormalized to enable incremental updates
    accumulated_sum: Option<Point>,

    /// Subspace representation (optional, for non-chunk containers)
    /// Captures variance/spread of points within the container
    subspace: Option<super::subspace::Subspace>,
}

impl Container {
    fn new(id: Id, level: ContainerLevel, centroid: Point) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // For chunks, the accumulated sum is the point itself
        let accumulated_sum = if level == ContainerLevel::Chunk {
            Some(centroid.clone())
        } else {
            None
        };

        // Initialize subspace for non-chunk containers
        let subspace = if level != ContainerLevel::Chunk {
            Some(super::subspace::Subspace::new(centroid.dimensionality()))
        } else {
            None
        };

        Self {
            id,
            level,
            centroid,
            timestamp,
            children: Vec::new(),
            descendant_count: if level == ContainerLevel::Chunk { 1 } else { 0 },
            accumulated_sum,
            subspace,
        }
    }

    fn is_leaf(&self) -> bool {
        self.level == ContainerLevel::Chunk
    }
}

/// Hierarchical Attention Tree Index
pub struct HatIndex {
    /// All containers (including root, sessions, documents, chunks)
    containers: HashMap<Id, Container>,

    /// Root container ID
    root_id: Option<Id>,

    /// Current active session (where new documents go)
    active_session: Option<Id>,

    /// Current active document (where new chunks go)
    active_document: Option<Id>,

    /// Expected dimensionality
    dimensionality: usize,

    /// Proximity function
    proximity: Arc<dyn Proximity>,

    /// Merge function (for centroids)
    merge: Arc<dyn Merge>,

    /// Whether higher proximity = more similar
    higher_is_better: bool,

    /// Configuration
    config: HatConfig,

    /// Consolidation state (None if not consolidating)
    consolidation_state: Option<ConsolidationState>,

    /// Cache of child points during consolidation
    consolidation_points_cache: HashMap<Id, Vec<Point>>,

    /// Learnable router for adaptive routing weights
    learnable_router: Option<super::learnable_routing::LearnableRouter>,
}

impl HatIndex {
    /// Create a new HAT index with cosine similarity
    pub fn cosine(dimensionality: usize) -> Self {
        use crate::core::proximity::Cosine;
        use crate::core::merge::Mean;
        Self::new(
            dimensionality,
            Arc::new(Cosine),
            Arc::new(Mean),
            true,
            HatConfig::default(),
        )
    }

    /// Create with custom config
    pub fn with_config(mut self, config: HatConfig) -> Self {
        // Initialize learnable router if enabled
        if config.learnable_routing_enabled {
            self.learnable_router = Some(super::learnable_routing::LearnableRouter::new(
                self.dimensionality,
                config.learnable_routing_config.clone(),
            ));
        }
        self.config = config;
        self
    }

    /// Create with custom proximity and merge functions
    pub fn new(
        dimensionality: usize,
        proximity: Arc<dyn Proximity>,
        merge: Arc<dyn Merge>,
        higher_is_better: bool,
        config: HatConfig,
    ) -> Self {
        // Initialize learnable router if enabled
        let learnable_router = if config.learnable_routing_enabled {
            Some(super::learnable_routing::LearnableRouter::new(
                dimensionality,
                config.learnable_routing_config.clone(),
            ))
        } else {
            None
        };

        Self {
            containers: HashMap::new(),
            root_id: None,
            active_session: None,
            active_document: None,
            dimensionality,
            proximity,
            merge,
            higher_is_better,
            config,
            consolidation_state: None,
            consolidation_points_cache: HashMap::new(),
            learnable_router,
        }
    }

    /// Compute distance (lower = more similar)
    fn distance(&self, a: &Point, b: &Point) -> f32 {
        let prox = self.proximity.proximity(a, b);
        if self.higher_is_better {
            1.0 - prox
        } else {
            prox
        }
    }

    /// Compute temporal distance (normalized to 0-1)
    fn temporal_distance(&self, t1: u64, t2: u64) -> f32 {
        let diff = (t1 as i64 - t2 as i64).unsigned_abs() as f64;
        // Exponential decay: e^(-λ * diff)
        // diff is in milliseconds, normalize to hours
        let hours = diff / (1000.0 * 60.0 * 60.0);
        (1.0 - (-self.config.time_decay as f64 * hours).exp()) as f32
    }

    /// Combined distance with temporal component, optional subspace, and learnable routing
    fn combined_distance(&self, query: &Point, query_time: u64, container: &Container) -> f32 {
        // Compute semantic distance
        let semantic = if self.config.learnable_routing_enabled {
            // Use learnable routing weights
            if let Some(ref router) = self.learnable_router {
                // weighted_similarity returns similarity (higher = better)
                // convert to distance (lower = better)
                let sim = router.weighted_similarity(query, &container.centroid);
                1.0 - sim
            } else {
                self.distance(query, &container.centroid)
            }
        } else if self.config.subspace_enabled && !container.is_leaf() {
            // Use subspace-aware similarity if available
            if let Some(ref subspace) = container.subspace {
                // combined_subspace_similarity returns similarity (higher = better)
                // convert to distance (lower = better)
                let sim = super::subspace::combined_subspace_similarity(
                    query, subspace, &self.config.subspace_config
                );
                1.0 - sim
            } else {
                self.distance(query, &container.centroid)
            }
        } else {
            self.distance(query, &container.centroid)
        };

        let temporal = self.temporal_distance(query_time, container.timestamp);

        // Weighted combination
        let w = self.config.temporal_weight;
        semantic * (1.0 - w) + temporal * w
    }

    /// Ensure root exists
    fn ensure_root(&mut self) {
        if self.root_id.is_none() {
            let root = Container::new(
                Id::now(),
                ContainerLevel::Global,
                Point::origin(self.dimensionality),
            );
            let root_id = root.id;
            self.containers.insert(root_id, root);
            self.root_id = Some(root_id);
        }
    }

    /// Ensure active session exists
    fn ensure_session(&mut self) {
        self.ensure_root();

        if self.active_session.is_none() {
            let session = Container::new(
                Id::now(),
                ContainerLevel::Session,
                Point::origin(self.dimensionality),
            );
            let session_id = session.id;
            self.containers.insert(session_id, session);

            // Add to root's children
            if let Some(root_id) = self.root_id {
                if let Some(root) = self.containers.get_mut(&root_id) {
                    root.children.push(session_id);
                }
            }

            self.active_session = Some(session_id);
        }
    }

    /// Ensure active document exists
    fn ensure_document(&mut self) {
        self.ensure_session();

        if self.active_document.is_none() {
            let document = Container::new(
                Id::now(),
                ContainerLevel::Document,
                Point::origin(self.dimensionality),
            );
            let doc_id = document.id;
            self.containers.insert(doc_id, document);

            // Add to session's children
            if let Some(session_id) = self.active_session {
                if let Some(session) = self.containers.get_mut(&session_id) {
                    session.children.push(doc_id);
                }
            }

            self.active_document = Some(doc_id);
        }
    }

    /// Start a new session (call this to create session boundaries)
    pub fn new_session(&mut self) {
        self.active_session = None;
        self.active_document = None;
    }

    /// Start a new document within current session
    pub fn new_document(&mut self) {
        self.active_document = None;
    }

    /// Compute Fréchet mean on the unit hypersphere using iterative algorithm
    /// This finds the point that minimizes sum of squared geodesic distances
    fn compute_frechet_mean(&self, points: &[Point], initial: &Point) -> Point {
        let mut mean = initial.clone();
        let iterations = self.config.frechet_iterations;

        for _ in 0..iterations {
            // Compute weighted tangent vectors (log map)
            let mut tangent_sum = vec![0.0f32; mean.dimensionality()];

            for point in points {
                // Log map: project point onto tangent space at mean
                // For unit sphere: log_p(q) = θ * (q - (q·p)p) / ||q - (q·p)p||
                // where θ = arccos(p·q)
                let dot: f32 = mean.dims().iter()
                    .zip(point.dims().iter())
                    .map(|(a, b)| a * b)
                    .sum();

                // Clamp dot product to valid range for arccos
                let dot_clamped = dot.clamp(-1.0, 1.0);
                let theta = dot_clamped.acos();

                if theta.abs() < 1e-8 {
                    // Points are identical, tangent vector is zero
                    continue;
                }

                // Direction in tangent space
                let mut direction: Vec<f32> = point.dims().iter()
                    .zip(mean.dims().iter())
                    .map(|(q, p)| q - dot * p)
                    .collect();

                // Normalize direction
                let dir_norm: f32 = direction.iter().map(|x| x * x).sum::<f32>().sqrt();
                if dir_norm < 1e-8 {
                    continue;
                }

                for (i, d) in direction.iter_mut().enumerate() {
                    tangent_sum[i] += theta * (*d / dir_norm);
                }
            }

            // Average tangent vector
            let n = points.len() as f32;
            for t in tangent_sum.iter_mut() {
                *t /= n;
            }

            // Compute tangent vector magnitude
            let tangent_norm: f32 = tangent_sum.iter().map(|x| x * x).sum::<f32>().sqrt();

            if tangent_norm < 1e-8 {
                // Converged
                break;
            }

            // Exp map: move along geodesic from mean in tangent direction
            // For unit sphere: exp_p(v) = cos(||v||)p + sin(||v||)(v/||v||)
            let cos_t = tangent_norm.cos();
            let sin_t = tangent_norm.sin();

            let new_dims: Vec<f32> = mean.dims().iter()
                .zip(tangent_sum.iter())
                .map(|(p, v)| cos_t * p + sin_t * (v / tangent_norm))
                .collect();

            mean = Point::new(new_dims);
        }

        // Ensure result is normalized (on the unit sphere)
        mean.normalize()
    }

    /// Update centroid incrementally when adding a child
    /// Returns the magnitude of the change (for sparse propagation)
    fn update_centroid(&mut self, container_id: Id, new_point: &Point) -> f32 {
        let method = self.config.centroid_method;

        // First, extract what we need from the container
        let (old_centroid, n, accumulated_sum) = {
            if let Some(container) = self.containers.get(&container_id) {
                (
                    container.centroid.clone(),
                    container.descendant_count as f32,
                    container.accumulated_sum.clone(),
                )
            } else {
                return 0.0;
            }
        };

        // Handle first child case
        if n == 0.0 {
            if let Some(container) = self.containers.get_mut(&container_id) {
                container.centroid = new_point.clone();
                container.accumulated_sum = Some(new_point.clone());
                container.descendant_count += 1;
            }
            return f32::MAX; // Always propagate first point
        }

        // Compute new centroid based on method
        let (new_centroid, new_sum) = match method {
            CentroidMethod::Euclidean => {
                // Incremental Euclidean mean using accumulated sum
                let new_sum = if let Some(ref sum) = accumulated_sum {
                    sum.dims().iter()
                        .zip(new_point.dims().iter())
                        .map(|(s, p)| s + p)
                        .collect::<Vec<f32>>()
                } else {
                    new_point.dims().to_vec()
                };

                // Compute centroid as normalized mean
                let count = n + 1.0;
                let mean_dims: Vec<f32> = new_sum.iter().map(|s| s / count).collect();
                let centroid = Point::new(mean_dims).normalize();
                (centroid, Point::new(new_sum))
            }
            CentroidMethod::Frechet => {
                // Update accumulated sum
                let new_sum = if let Some(ref sum) = accumulated_sum {
                    sum.dims().iter()
                        .zip(new_point.dims().iter())
                        .map(|(s, p)| s + p)
                        .collect::<Vec<f32>>()
                } else {
                    new_point.dims().to_vec()
                };

                // For incremental Fréchet, use geodesic interpolation
                let new_count = n + 1.0;
                let weight = 1.0 / new_count;
                let centroid = Self::geodesic_interpolate_static(&old_centroid, new_point, weight);
                (centroid, Point::new(new_sum))
            }
        };

        // Now update the container
        let subspace_enabled = self.config.subspace_enabled;
        if let Some(container) = self.containers.get_mut(&container_id) {
            container.centroid = new_centroid.clone();
            container.accumulated_sum = Some(new_sum);
            container.descendant_count += 1;

            // Update subspace if enabled, incremental covariance is on, and not a chunk
            // When incremental_covariance is false (default), we skip the expensive
            // O(d²) outer product accumulation per insert, deferring to consolidation.
            if subspace_enabled
                && self.config.subspace_config.incremental_covariance
                && container.level != ContainerLevel::Chunk
            {
                if let Some(ref mut subspace) = container.subspace {
                    subspace.add_point(new_point);
                    // Principal directions recomputed during consolidation
                }
            }
        }

        // Calculate change magnitude (L2 norm of delta)
        let delta: f32 = old_centroid.dims()
            .iter()
            .zip(new_centroid.dims().iter())
            .map(|(old, new)| (new - old).powi(2))
            .sum::<f32>()
            .sqrt();

        delta
    }

    /// Static version of geodesic interpolation (no self reference needed)
    fn geodesic_interpolate_static(a: &Point, b: &Point, t: f32) -> Point {
        // Compute dot product
        let dot: f32 = a.dims().iter()
            .zip(b.dims().iter())
            .map(|(x, y)| x * y)
            .sum();

        // Clamp to valid range
        let dot_clamped = dot.clamp(-0.9999, 0.9999);
        let theta = dot_clamped.acos();

        if theta.abs() < 1e-8 {
            // Points are nearly identical
            return a.clone();
        }

        // Slerp formula: (sin((1-t)θ)/sin(θ)) * a + (sin(tθ)/sin(θ)) * b
        let sin_theta = theta.sin();
        let weight_a = ((1.0 - t) * theta).sin() / sin_theta;
        let weight_b = (t * theta).sin() / sin_theta;

        let result_dims: Vec<f32> = a.dims().iter()
            .zip(b.dims().iter())
            .map(|(x, y)| weight_a * x + weight_b * y)
            .collect();

        Point::new(result_dims).normalize()
    }

    /// Geodesic interpolation on the unit hypersphere (slerp)
    /// Returns a point t fraction of the way from a to b along the great circle
    fn geodesic_interpolate(&self, a: &Point, b: &Point, t: f32) -> Point {
        // Compute dot product
        let dot: f32 = a.dims().iter()
            .zip(b.dims().iter())
            .map(|(x, y)| x * y)
            .sum();

        // Clamp to valid range
        let dot_clamped = dot.clamp(-0.9999, 0.9999);
        let theta = dot_clamped.acos();

        if theta.abs() < 1e-8 {
            // Points are nearly identical
            return a.clone();
        }

        // Slerp formula: (sin((1-t)θ)/sin(θ)) * a + (sin(tθ)/sin(θ)) * b
        let sin_theta = theta.sin();
        let weight_a = ((1.0 - t) * theta).sin() / sin_theta;
        let weight_b = (t * theta).sin() / sin_theta;

        let result_dims: Vec<f32> = a.dims().iter()
            .zip(b.dims().iter())
            .map(|(x, y)| weight_a * x + weight_b * y)
            .collect();

        Point::new(result_dims).normalize()
    }

    /// Sparse propagation: only update parent if change exceeds threshold
    fn propagate_centroid_update(
        &mut self,
        container_id: Id,
        new_point: &Point,
        ancestors: &[Id],
    ) {
        let threshold = self.config.propagation_threshold;
        let mut delta = self.update_centroid(container_id, new_point);

        // Propagate up the tree if delta exceeds threshold
        for ancestor_id in ancestors {
            if delta < threshold {
                break; // Stop propagation - change too small
            }
            delta = self.update_centroid(*ancestor_id, new_point);
        }
    }

    /// Search the tree from a starting container
    fn search_tree(
        &self,
        query: &Point,
        query_time: u64,
        start_id: Id,
        k: usize,
    ) -> Vec<(Id, f32)> {
        let mut results: Vec<(Id, f32)> = Vec::new();

        // Adaptive beam width based on k
        let beam_width = self.config.beam_width.max(k);

        // BFS with beam search
        let mut current_level = vec![start_id];

        while !current_level.is_empty() {
            let mut next_level: Vec<(Id, f32)> = Vec::new();

            for container_id in &current_level {
                if let Some(container) = self.containers.get(container_id) {
                    if container.is_leaf() {
                        // Leaf node - add to results
                        let dist = self.combined_distance(query, query_time, container);
                        results.push((*container_id, dist));
                    } else {
                        // Internal node - score children and add to next level
                        for child_id in &container.children {
                            if let Some(child) = self.containers.get(child_id) {
                                let dist = self.combined_distance(query, query_time, child);
                                next_level.push((*child_id, dist));
                            }
                        }
                    }
                }
            }

            if next_level.is_empty() {
                break;
            }

            // Sort by distance and take beam_width best
            next_level.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            current_level = next_level
                .into_iter()
                .take(beam_width)
                .map(|(id, _)| id)
                .collect();
        }

        // Sort results and return top k
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        results
    }

    // =========================================================================
    // Multi-Resolution Query API (inspired by VAR next-scale prediction)
    // =========================================================================

    /// Coarse query: Get session summaries without descending to chunks
    /// Use this for fast "is there relevant memory?" checks
    pub fn near_sessions(&self, query: &Point, k: usize) -> NearResult<Vec<SessionSummary>> {
        if query.dimensionality() != self.dimensionality {
            return Err(NearError::DimensionalityMismatch {
                expected: self.dimensionality,
                got: query.dimensionality(),
            });
        }

        let root_id = match self.root_id {
            Some(id) => id,
            None => return Ok(vec![]),
        };

        let query_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // Get root's children (sessions)
        let root = match self.containers.get(&root_id) {
            Some(r) => r,
            None => return Ok(vec![]),
        };

        let mut sessions: Vec<SessionSummary> = root.children
            .iter()
            .filter_map(|session_id| {
                let session = self.containers.get(session_id)?;
                if session.level != ContainerLevel::Session {
                    return None;
                }
                let dist = self.combined_distance(query, query_time, session);
                let score = if self.higher_is_better { 1.0 - dist } else { dist };

                Some(SessionSummary {
                    id: *session_id,
                    score,
                    chunk_count: session.descendant_count,
                    timestamp: session.timestamp,
                })
            })
            .collect();

        // Sort by score (higher is better)
        sessions.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        sessions.truncate(k);

        Ok(sessions)
    }

    /// Refine within a specific session: Get document summaries
    pub fn near_documents(&self, session_id: Id, query: &Point, k: usize) -> NearResult<Vec<DocumentSummary>> {
        if query.dimensionality() != self.dimensionality {
            return Err(NearError::DimensionalityMismatch {
                expected: self.dimensionality,
                got: query.dimensionality(),
            });
        }

        let query_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let session = match self.containers.get(&session_id) {
            Some(s) => s,
            None => return Ok(vec![]),
        };

        let mut documents: Vec<DocumentSummary> = session.children
            .iter()
            .filter_map(|doc_id| {
                let doc = self.containers.get(doc_id)?;
                if doc.level != ContainerLevel::Document {
                    return None;
                }
                let dist = self.combined_distance(query, query_time, doc);
                let score = if self.higher_is_better { 1.0 - dist } else { dist };

                Some(DocumentSummary {
                    id: *doc_id,
                    score,
                    chunk_count: doc.descendant_count,
                    timestamp: doc.timestamp,
                })
            })
            .collect();

        documents.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        documents.truncate(k);

        Ok(documents)
    }

    /// Refine within a specific document: Get chunk results
    pub fn near_in_document(&self, doc_id: Id, query: &Point, k: usize) -> NearResult<Vec<SearchResult>> {
        if query.dimensionality() != self.dimensionality {
            return Err(NearError::DimensionalityMismatch {
                expected: self.dimensionality,
                got: query.dimensionality(),
            });
        }

        let query_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let doc = match self.containers.get(&doc_id) {
            Some(d) => d,
            None => return Ok(vec![]),
        };

        let mut chunks: Vec<SearchResult> = doc.children
            .iter()
            .filter_map(|chunk_id| {
                let chunk = self.containers.get(chunk_id)?;
                if chunk.level != ContainerLevel::Chunk {
                    return None;
                }
                let dist = self.combined_distance(query, query_time, chunk);
                let score = if self.higher_is_better { 1.0 - dist } else { dist };

                Some(SearchResult::new(*chunk_id, score))
            })
            .collect();

        chunks.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        chunks.truncate(k);

        Ok(chunks)
    }

    /// Get statistics about the tree structure
    pub fn stats(&self) -> HatStats {
        let mut stats = HatStats::default();

        for container in self.containers.values() {
            match container.level {
                ContainerLevel::Global => stats.global_count += 1,
                ContainerLevel::Session => stats.session_count += 1,
                ContainerLevel::Document => stats.document_count += 1,
                ContainerLevel::Chunk => stats.chunk_count += 1,
            }
        }

        stats
    }

    // =========================================================================
    // Learnable Routing API
    // =========================================================================

    /// Record positive feedback for a query result (successful retrieval)
    ///
    /// Call this when a retrieved result was useful/relevant.
    /// The router learns to route similar queries to similar containers.
    pub fn record_retrieval_success(&mut self, query: &Point, result_id: Id) {
        if let Some(ref mut router) = self.learnable_router {
            // Find the container for this result and record feedback for each level
            if let Some(container) = self.containers.get(&result_id) {
                router.record_success(query, &container.centroid, container.level.depth());
            }
        }
    }

    /// Record negative feedback for a query result (unsuccessful retrieval)
    ///
    /// Call this when a retrieved result was not useful/relevant.
    pub fn record_retrieval_failure(&mut self, query: &Point, result_id: Id) {
        if let Some(ref mut router) = self.learnable_router {
            if let Some(container) = self.containers.get(&result_id) {
                router.record_failure(query, &container.centroid, container.level.depth());
            }
        }
    }

    /// Record implicit feedback with a relevance score (0.0 = irrelevant, 1.0 = highly relevant)
    ///
    /// Use this for continuous feedback signals like click-through rate, dwell time, etc.
    pub fn record_implicit_feedback(&mut self, query: &Point, result_id: Id, relevance: f32) {
        if let Some(ref mut router) = self.learnable_router {
            if let Some(container) = self.containers.get(&result_id) {
                router.record_implicit(query, &container.centroid, container.level.depth(), relevance);
            }
        }
    }

    /// Get learnable router statistics (if enabled)
    pub fn router_stats(&self) -> Option<super::learnable_routing::RouterStats> {
        self.learnable_router.as_ref().map(|r| r.stats())
    }

    /// Get current routing weights (if learnable routing is enabled)
    pub fn routing_weights(&self) -> Option<&[f32]> {
        self.learnable_router.as_ref().map(|r| r.weights())
    }

    /// Reset learnable routing weights to uniform
    pub fn reset_routing_weights(&mut self) {
        if let Some(ref mut router) = self.learnable_router {
            router.reset_weights();
        }
    }

    /// Check if learnable routing is enabled
    pub fn is_learnable_routing_enabled(&self) -> bool {
        self.learnable_router.is_some()
    }
}

/// Statistics about the HAT tree structure
#[derive(Debug, Clone, Default)]
pub struct HatStats {
    pub global_count: usize,
    pub session_count: usize,
    pub document_count: usize,
    pub chunk_count: usize,
}

impl Near for HatIndex {
    fn near(&self, query: &Point, k: usize) -> NearResult<Vec<SearchResult>> {
        // Check dimensionality
        if query.dimensionality() != self.dimensionality {
            return Err(NearError::DimensionalityMismatch {
                expected: self.dimensionality,
                got: query.dimensionality(),
            });
        }

        // Handle empty index
        let root_id = match self.root_id {
            Some(id) => id,
            None => return Ok(vec![]),
        };

        // Current time for temporal scoring
        let query_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // Search tree
        let results = self.search_tree(query, query_time, root_id, k);

        // Convert to SearchResult
        let search_results: Vec<SearchResult> = results
            .into_iter()
            .map(|(id, dist)| {
                let score = if self.higher_is_better {
                    1.0 - dist
                } else {
                    dist
                };
                SearchResult::new(id, score)
            })
            .collect();

        Ok(search_results)
    }

    fn within(&self, query: &Point, threshold: f32) -> NearResult<Vec<SearchResult>> {
        // Check dimensionality
        if query.dimensionality() != self.dimensionality {
            return Err(NearError::DimensionalityMismatch {
                expected: self.dimensionality,
                got: query.dimensionality(),
            });
        }

        // Use near with all points, then filter
        let all_results = self.near(query, self.containers.len())?;

        let filtered: Vec<SearchResult> = all_results
            .into_iter()
            .filter(|r| {
                if self.higher_is_better {
                    r.score >= threshold
                } else {
                    r.score <= threshold
                }
            })
            .collect();

        Ok(filtered)
    }

    fn add(&mut self, id: Id, point: &Point) -> NearResult<()> {
        // Check dimensionality
        if point.dimensionality() != self.dimensionality {
            return Err(NearError::DimensionalityMismatch {
                expected: self.dimensionality,
                got: point.dimensionality(),
            });
        }

        // Ensure hierarchy exists
        self.ensure_document();

        // Create chunk container
        let chunk = Container::new(id, ContainerLevel::Chunk, point.clone());
        self.containers.insert(id, chunk);

        // Add to document's children
        if let Some(doc_id) = self.active_document {
            if let Some(doc) = self.containers.get_mut(&doc_id) {
                doc.children.push(id);
            }

            // Build ancestor chain for sparse propagation
            let mut ancestors = Vec::new();
            if let Some(session_id) = self.active_session {
                ancestors.push(session_id);
                if let Some(root_id) = self.root_id {
                    ancestors.push(root_id);
                }
            }

            // Sparse propagation: only update ancestors if change is significant
            self.propagate_centroid_update(doc_id, point, &ancestors);
        }

        // Check if document needs splitting
        if let Some(doc_id) = self.active_document {
            if let Some(doc) = self.containers.get(&doc_id) {
                if doc.children.len() >= self.config.max_children {
                    // Start a new document
                    self.new_document();
                }
            }
        }

        // Check if session needs splitting
        if let Some(session_id) = self.active_session {
            if let Some(session) = self.containers.get(&session_id) {
                if session.children.len() >= self.config.max_children {
                    // Start a new session
                    self.new_session();
                }
            }
        }

        Ok(())
    }

    fn remove(&mut self, id: Id) -> NearResult<()> {
        // Remove the chunk
        self.containers.remove(&id);

        // Note: We don't update centroids on remove for simplicity
        // A production implementation would need to handle this

        Ok(())
    }

    fn rebuild(&mut self) -> NearResult<()> {
        // Recalculate all centroids from scratch
        // For now, this is a no-op since we maintain incrementally
        Ok(())
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn len(&self) -> usize {
        // Count only chunk-level containers
        self.containers.values()
            .filter(|c| c.level == ContainerLevel::Chunk)
            .count()
    }
}

// =============================================================================
// Consolidation Implementation
// =============================================================================

impl HatIndex {
    /// Collect all leaf points for a container (recursively)
    fn collect_leaf_points(&self, container_id: Id) -> Vec<Point> {
        let container = match self.containers.get(&container_id) {
            Some(c) => c,
            None => return vec![],
        };

        if container.is_leaf() {
            return vec![container.centroid.clone()];
        }

        let mut points = Vec::new();
        for child_id in &container.children {
            points.extend(self.collect_leaf_points(*child_id));
        }
        points
    }

    /// Get all container IDs at a given level
    fn containers_at_level(&self, level: ContainerLevel) -> Vec<Id> {
        self.containers
            .iter()
            .filter(|(_, c)| c.level == level)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Recompute a container's centroid from its descendants
    fn recompute_centroid(&mut self, container_id: Id) -> Option<f32> {
        // First collect the points (need to release borrow)
        let points = self.collect_leaf_points(container_id);

        if points.is_empty() {
            return None;
        }

        let new_centroid = match compute_exact_centroid(&points) {
            Some(c) => c,
            None => return None,
        };

        // Get subspace config for recomputation
        let subspace_enabled = self.config.subspace_enabled;
        let subspace_rank = self.config.subspace_config.rank;

        // Now update the container
        let drift = if let Some(container) = self.containers.get_mut(&container_id) {
            let old_centroid = container.centroid.clone();
            let drift = centroid_drift(&old_centroid, &new_centroid);
            container.centroid = new_centroid;
            container.descendant_count = points.len();

            // Update accumulated sum
            let sum: Vec<f32> = points.iter()
                .fold(vec![0.0f32; self.dimensionality], |mut acc, p| {
                    for (i, &v) in p.dims().iter().enumerate() {
                        acc[i] += v;
                    }
                    acc
                });
            container.accumulated_sum = Some(Point::new(sum));

            // Recompute subspace during consolidation if enabled
            if subspace_enabled && container.level != ContainerLevel::Chunk {
                let mut subspace = super::subspace::Subspace::new(self.dimensionality);
                for point in &points {
                    subspace.add_point(point);
                }
                subspace.recompute_subspace(subspace_rank);
                container.subspace = Some(subspace);
            }

            Some(drift)
        } else {
            None
        };

        drift
    }

    /// Check if a container should be merged (too few children)
    fn should_merge(&self, container_id: Id, threshold: usize) -> bool {
        if let Some(container) = self.containers.get(&container_id) {
            // Don't merge chunks, root, or sessions (for now)
            if container.level == ContainerLevel::Chunk ||
               container.level == ContainerLevel::Global ||
               container.level == ContainerLevel::Session {
                return false;
            }
            container.children.len() < threshold
        } else {
            false
        }
    }

    /// Check if a container should be split (too many children)
    fn should_split(&self, container_id: Id, threshold: usize) -> bool {
        if let Some(container) = self.containers.get(&container_id) {
            // Don't split chunks
            if container.level == ContainerLevel::Chunk {
                return false;
            }
            container.children.len() > threshold
        } else {
            false
        }
    }

    /// Find a sibling container to merge with
    fn find_merge_sibling(&self, container_id: Id) -> Option<Id> {
        // Find parent
        let parent_id = self.containers.iter()
            .find(|(_, c)| c.children.contains(&container_id))
            .map(|(id, _)| *id)?;

        let parent = self.containers.get(&parent_id)?;

        // Find smallest sibling
        let mut smallest: Option<(Id, usize)> = None;
        for child_id in &parent.children {
            if *child_id == container_id {
                continue;
            }
            if let Some(child) = self.containers.get(child_id) {
                let size = child.children.len();
                if smallest.is_none() || size < smallest.unwrap().1 {
                    smallest = Some((*child_id, size));
                }
            }
        }

        smallest.map(|(id, _)| id)
    }

    /// Merge container B into container A
    fn merge_containers(&mut self, a_id: Id, b_id: Id) {
        // Get children from B
        let b_children: Vec<Id> = if let Some(b) = self.containers.get(&b_id) {
            b.children.clone()
        } else {
            return;
        };

        // Add children to A
        if let Some(a) = self.containers.get_mut(&a_id) {
            a.children.extend(b_children);
        }

        // Remove B from its parent's children
        let parent_id = self.containers.iter()
            .find(|(_, c)| c.children.contains(&b_id))
            .map(|(id, _)| *id);

        if let Some(pid) = parent_id {
            if let Some(parent) = self.containers.get_mut(&pid) {
                parent.children.retain(|id| *id != b_id);
            }
        }

        // Remove B
        self.containers.remove(&b_id);

        // Recompute A's centroid
        self.recompute_centroid(a_id);
    }

    /// Split a container into two
    fn split_container(&mut self, container_id: Id) -> Option<Id> {
        // Get container info
        let (level, children, parent_id) = {
            let container = self.containers.get(&container_id)?;
            let parent_id = self.containers.iter()
                .find(|(_, c)| c.children.contains(&container_id))
                .map(|(id, _)| *id);
            (container.level, container.children.clone(), parent_id)
        };

        if children.len() < 2 {
            return None;
        }

        // Simple split: divide children in half
        let mid = children.len() / 2;
        let (keep, move_to_new) = children.split_at(mid);

        // Create new container
        let new_id = Id::now();
        let new_container = Container::new(
            new_id,
            level,
            Point::origin(self.dimensionality),
        );
        self.containers.insert(new_id, new_container);

        // Update original container
        if let Some(container) = self.containers.get_mut(&container_id) {
            container.children = keep.to_vec();
        }

        // Set new container's children
        if let Some(new_container) = self.containers.get_mut(&new_id) {
            new_container.children = move_to_new.to_vec();
        }

        // Add new container to parent
        if let Some(pid) = parent_id {
            if let Some(parent) = self.containers.get_mut(&pid) {
                parent.children.push(new_id);
            }
        }

        // Recompute centroids
        self.recompute_centroid(container_id);
        self.recompute_centroid(new_id);

        Some(new_id)
    }

    /// Remove containers with no children (except chunks)
    fn prune_empty(&mut self) -> usize {
        let mut pruned = 0;

        loop {
            let empty_ids: Vec<Id> = self.containers
                .iter()
                .filter(|(_, c)| {
                    c.level != ContainerLevel::Chunk &&
                    c.level != ContainerLevel::Global &&
                    c.children.is_empty()
                })
                .map(|(id, _)| *id)
                .collect();

            if empty_ids.is_empty() {
                break;
            }

            for id in empty_ids {
                // Remove from parent's children
                let parent_id = self.containers.iter()
                    .find(|(_, c)| c.children.contains(&id))
                    .map(|(pid, _)| *pid);

                if let Some(pid) = parent_id {
                    if let Some(parent) = self.containers.get_mut(&pid) {
                        parent.children.retain(|cid| *cid != id);
                    }
                }

                self.containers.remove(&id);
                pruned += 1;
            }
        }

        pruned
    }
}

impl Consolidate for HatIndex {
    fn begin_consolidation(&mut self, config: ConsolidationConfig) {
        let mut state = ConsolidationState::new(config);
        state.start();

        // Initialize work queue with all containers for leaf collection
        let all_ids: VecDeque<Id> = self.containers.keys().copied().collect();
        state.work_queue = all_ids;

        self.consolidation_state = Some(state);
        self.consolidation_points_cache.clear();
    }

    fn consolidation_tick(&mut self) -> ConsolidationTickResult {
        // Take ownership of state to avoid borrow issues
        let mut state = match self.consolidation_state.take() {
            Some(s) => s,
            None => {
                return ConsolidationTickResult::Complete(ConsolidationMetrics::default());
            }
        };

        let batch_size = state.config.batch_size;

        match state.phase {
            ConsolidationPhase::Idle => {
                state.start();
            }

            ConsolidationPhase::CollectingLeaves => {
                state.next_phase();

                // Populate work queue with non-chunk containers (bottom-up)
                let docs = self.containers_at_level(ContainerLevel::Document);
                let sessions = self.containers_at_level(ContainerLevel::Session);
                let globals = self.containers_at_level(ContainerLevel::Global);

                state.work_queue.clear();
                state.work_queue.extend(docs);
                state.work_queue.extend(sessions);
                state.work_queue.extend(globals);
            }

            ConsolidationPhase::RecomputingCentroids => {
                let mut processed = 0;
                let mut to_recompute = Vec::new();

                while processed < batch_size {
                    match state.work_queue.pop_front() {
                        Some(id) => {
                            to_recompute.push(id);
                            state.processed.insert(id);
                            processed += 1;
                        }
                        None => break,
                    };
                }

                // Now recompute without holding state borrow
                for container_id in to_recompute {
                    if let Some(drift) = self.recompute_centroid(container_id) {
                        state.record_drift(drift);
                        state.metrics.centroids_recomputed += 1;
                    }
                    state.metrics.containers_processed += 1;
                }

                if state.work_queue.is_empty() {
                    state.next_phase();

                    if state.phase == ConsolidationPhase::AnalyzingStructure {
                        let docs = self.containers_at_level(ContainerLevel::Document);
                        state.work_queue.extend(docs);
                    }
                }
            }

            ConsolidationPhase::AnalyzingStructure => {
                let merge_threshold = state.config.merge_threshold;
                let split_threshold = state.config.split_threshold;
                let mut processed = 0;
                let mut to_analyze = Vec::new();

                while processed < batch_size {
                    match state.work_queue.pop_front() {
                        Some(id) => {
                            to_analyze.push(id);
                            state.processed.insert(id);
                            processed += 1;
                        }
                        None => break,
                    };
                }

                // Analyze without holding state borrow
                for container_id in to_analyze {
                    if self.should_merge(container_id, merge_threshold) {
                        if let Some(sibling) = self.find_merge_sibling(container_id) {
                            state.add_merge_candidate(container_id, sibling);
                        }
                    } else if self.should_split(container_id, split_threshold) {
                        state.add_split_candidate(container_id);
                    }
                }

                if state.work_queue.is_empty() {
                    state.next_phase();
                }
            }

            ConsolidationPhase::Merging => {
                let mut processed = 0;
                let mut to_merge = Vec::new();

                while processed < batch_size {
                    match state.next_merge() {
                        Some(pair) => {
                            to_merge.push(pair);
                            processed += 1;
                        }
                        None => break,
                    };
                }

                for (a, b) in to_merge {
                    self.merge_containers(a, b);
                    state.metrics.containers_merged += 1;
                }

                if !state.has_merges() {
                    state.next_phase();
                }
            }

            ConsolidationPhase::Splitting => {
                let mut processed = 0;
                let mut to_split = Vec::new();

                while processed < batch_size {
                    match state.next_split() {
                        Some(id) => {
                            to_split.push(id);
                            processed += 1;
                        }
                        None => break,
                    };
                }

                for container_id in to_split {
                    if self.split_container(container_id).is_some() {
                        state.metrics.containers_split += 1;
                    }
                }

                if !state.has_splits() {
                    state.next_phase();
                }
            }

            ConsolidationPhase::Pruning => {
                let pruned = self.prune_empty();
                state.metrics.containers_pruned = pruned;
                state.next_phase();
            }

            ConsolidationPhase::OptimizingLayout => {
                for container in self.containers.values_mut() {
                    if container.children.len() > 1 {
                        // Placeholder for future optimization
                    }
                }
                state.next_phase();
            }

            ConsolidationPhase::Complete => {
                // Already complete
            }
        }

        state.metrics.ticks += 1;

        if state.is_complete() {
            let metrics = state.metrics.clone();
            self.consolidation_points_cache.clear();
            ConsolidationTickResult::Complete(metrics)
        } else {
            let progress = state.progress();
            self.consolidation_state = Some(state);
            ConsolidationTickResult::Continue(progress)
        }
    }

    fn is_consolidating(&self) -> bool {
        self.consolidation_state.is_some()
    }

    fn consolidation_progress(&self) -> Option<ConsolidationProgress> {
        self.consolidation_state.as_ref().map(|s| s.progress())
    }

    fn cancel_consolidation(&mut self) {
        self.consolidation_state = None;
        self.consolidation_points_cache.clear();
    }
}

// =============================================================================
// Persistence Implementation
// =============================================================================

impl HatIndex {
    /// Serialize the index to bytes
    ///
    /// # Example
    /// ```rust,ignore
    /// let bytes = hat.to_bytes()?;
    /// std::fs::write("index.hat", bytes)?;
    /// ```
    pub fn to_bytes(&self) -> Result<Vec<u8>, super::persistence::PersistError> {
        use super::persistence::{SerializedHat, SerializedContainer, LevelByte};

        let containers: Vec<SerializedContainer> = self.containers.iter()
            .map(|(_, c)| {
                let level = match c.level {
                    ContainerLevel::Global => LevelByte::Root,
                    ContainerLevel::Session => LevelByte::Session,
                    ContainerLevel::Document => LevelByte::Document,
                    ContainerLevel::Chunk => LevelByte::Chunk,
                };

                SerializedContainer {
                    id: c.id,
                    level,
                    timestamp: c.timestamp,
                    children: c.children.clone(),
                    descendant_count: c.descendant_count as u64,
                    centroid: c.centroid.dims().to_vec(),
                    accumulated_sum: c.accumulated_sum.as_ref().map(|p| p.dims().to_vec()),
                }
            })
            .collect();

        let router_weights = self.learnable_router.as_ref()
            .map(|r| r.weights().to_vec());

        let serialized = SerializedHat {
            version: 1,
            dimensionality: self.dimensionality as u32,
            root_id: self.root_id,
            containers,
            active_session: self.active_session,
            active_document: self.active_document,
            router_weights,
        };

        serialized.to_bytes()
    }

    /// Deserialize an index from bytes
    ///
    /// # Example
    /// ```rust,ignore
    /// let bytes = std::fs::read("index.hat")?;
    /// let hat = HatIndex::from_bytes(&bytes)?;
    /// ```
    pub fn from_bytes(data: &[u8]) -> Result<Self, super::persistence::PersistError> {
        use super::persistence::{SerializedHat, LevelByte, PersistError};
        use crate::core::proximity::Cosine;
        use crate::core::merge::Mean;

        let serialized = SerializedHat::from_bytes(data)?;
        let dimensionality = serialized.dimensionality as usize;

        // Create a new index with default settings
        let mut index = Self::new(
            dimensionality,
            Arc::new(Cosine),
            Arc::new(Mean),
            true,
            HatConfig::default(),
        );

        // Restore containers
        for sc in serialized.containers {
            let level = match sc.level {
                LevelByte::Root => ContainerLevel::Global,
                LevelByte::Session => ContainerLevel::Session,
                LevelByte::Document => ContainerLevel::Document,
                LevelByte::Chunk => ContainerLevel::Chunk,
            };

            // Verify dimension
            if sc.centroid.len() != dimensionality {
                return Err(PersistError::DimensionMismatch {
                    expected: dimensionality,
                    found: sc.centroid.len(),
                });
            }

            let centroid = Point::new(sc.centroid);
            let accumulated_sum = sc.accumulated_sum.map(Point::new);

            let container = Container {
                id: sc.id,
                level,
                centroid,
                timestamp: sc.timestamp,
                children: sc.children,
                descendant_count: sc.descendant_count as usize,
                accumulated_sum,
                subspace: if level != ContainerLevel::Chunk {
                    Some(super::subspace::Subspace::new(dimensionality))
                } else {
                    None
                },
            };

            index.containers.insert(sc.id, container);
        }

        // Restore state
        index.root_id = serialized.root_id;
        index.active_session = serialized.active_session;
        index.active_document = serialized.active_document;

        // Restore router weights if present
        if let Some(weights) = serialized.router_weights {
            let mut router = super::learnable_routing::LearnableRouter::default_for_dims(dimensionality);
            let weight_bytes: Vec<u8> = weights.iter()
                .flat_map(|w| w.to_le_bytes())
                .collect();
            router.deserialize_weights(&weight_bytes)
                .map_err(|e| PersistError::Corrupted(e.to_string()))?;
            index.learnable_router = Some(router);
        }

        Ok(index)
    }

    /// Save the index to a file
    pub fn save_to_file(&self, path: &std::path::Path) -> Result<(), super::persistence::PersistError> {
        let bytes = self.to_bytes()?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Load an index from a file
    pub fn load_from_file(path: &std::path::Path) -> Result<Self, super::persistence::PersistError> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hat_add() {
        let mut index = HatIndex::cosine(3);

        let id = Id::now();
        let point = Point::new(vec![1.0, 0.0, 0.0]);

        index.add(id, &point).unwrap();

        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_hat_near() {
        let mut index = HatIndex::cosine(3);

        // Add some points
        let points = vec![
            Point::new(vec![1.0, 0.0, 0.0]),
            Point::new(vec![0.0, 1.0, 0.0]),
            Point::new(vec![0.0, 0.0, 1.0]),
            Point::new(vec![0.7, 0.7, 0.0]).normalize(),
        ];

        for point in &points {
            index.add(Id::now(), point).unwrap();
        }

        // Query near [1, 0, 0]
        let query = Point::new(vec![1.0, 0.0, 0.0]);
        let results = index.near(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        // First result should have high similarity (close to 1.0)
        assert!(results[0].score > 0.5);
    }

    #[test]
    fn test_hat_sessions() {
        let mut index = HatIndex::cosine(3);

        // Add points to first session
        for i in 0..5 {
            let point = Point::new(vec![1.0, i as f32 * 0.1, 0.0]).normalize();
            index.add(Id::now(), &point).unwrap();
        }

        // Start new session
        index.new_session();

        // Add points to second session
        for i in 0..5 {
            let point = Point::new(vec![0.0, 1.0, i as f32 * 0.1]).normalize();
            index.add(Id::now(), &point).unwrap();
        }

        assert_eq!(index.len(), 10);

        // Query should find both sessions
        let query = Point::new(vec![0.5, 0.5, 0.0]).normalize();
        let results = index.near(&query, 5).unwrap();

        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_hat_hierarchy_structure() {
        let mut index = HatIndex::cosine(3);

        // Add some points
        for _ in 0..10 {
            let point = Point::new(vec![1.0, 0.0, 0.0]);
            index.add(Id::now(), &point).unwrap();
        }

        // Should have: 1 root + 1 session + 1 document + 10 chunks = 13 containers
        assert!(index.containers.len() >= 13);

        // Check that root exists
        assert!(index.root_id.is_some());
    }

    #[test]
    fn test_hat_empty() {
        let index = HatIndex::cosine(3);

        let query = Point::new(vec![1.0, 0.0, 0.0]);
        let results = index.near(&query, 5).unwrap();

        assert!(results.is_empty());
    }

    #[test]
    fn test_hat_dimensionality_check() {
        let mut index = HatIndex::cosine(3);

        let wrong_dims = Point::new(vec![1.0, 0.0]); // 2 dims
        let result = index.add(Id::now(), &wrong_dims);

        match result {
            Err(NearError::DimensionalityMismatch { expected, got }) => {
                assert_eq!(expected, 3);
                assert_eq!(got, 2);
            }
            _ => panic!("Expected DimensionalityMismatch error"),
        }
    }

    #[test]
    fn test_hat_scale() {
        let mut index = HatIndex::cosine(128);

        // Add 1000 points
        for i in 0..1000 {
            let mut dims = vec![0.0f32; 128];
            dims[i % 128] = 1.0;
            let point = Point::new(dims).normalize();
            index.add(Id::now(), &point).unwrap();
        }

        assert_eq!(index.len(), 1000);

        // Query should work
        let query = Point::new(vec![1.0; 128]).normalize();
        let results = index.near(&query, 10).unwrap();

        assert_eq!(results.len(), 10);
    }
}
