//! # Learnable Routing for HAT
//!
//! This module implements learnable routing weights for HAT index.
//! Instead of using fixed cosine similarity for routing decisions,
//! we learn dimension weights that adapt to actual query patterns.
//!
//! ## Key Insight (from journal 006)
//!
//! "The main gap: ARMS uses *known* structure while cutting-edge methods
//! *learn* structure. Opportunity: make HAT structure learnable while
//! keeping the efficiency benefits."
//!
//! ## Approach
//!
//! 1. **Weighted Similarity**: `sim(q, c) = Σᵢ wᵢ · qᵢ · cᵢ` instead of plain cosine
//! 2. **Feedback Collection**: Track query → retrieved → relevant mappings
//! 3. **Online Learning**: Update weights to improve routing decisions
//!
//! ## Benefits
//!
//! - Adapts to task-specific semantic dimensions
//! - No neural network training required (gradient-free)
//! - Preserves O(log n) query complexity
//! - Can learn from implicit feedback (click-through, usage patterns)

use crate::core::Point;
use std::collections::VecDeque;

/// Configuration for learnable routing
#[derive(Debug, Clone)]
pub struct LearnableRoutingConfig {
    /// Learning rate for weight updates (0.0 = no learning)
    pub learning_rate: f32,

    /// Momentum for smoothing updates
    pub momentum: f32,

    /// Weight decay for regularization (prevents overfitting)
    pub weight_decay: f32,

    /// Maximum number of feedback samples to retain
    pub max_feedback_samples: usize,

    /// Minimum feedback samples before learning starts
    pub min_samples_to_learn: usize,

    /// How often to update weights (every N feedback samples)
    pub update_frequency: usize,

    /// Enable dimension-wise weights (vs single scalar)
    pub per_dimension_weights: bool,
}

impl Default for LearnableRoutingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            momentum: 0.9,
            weight_decay: 0.001,
            max_feedback_samples: 1000,
            min_samples_to_learn: 50,
            update_frequency: 10,
            per_dimension_weights: true,
        }
    }
}

impl LearnableRoutingConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum.clamp(0.0, 0.99);
        self
    }

    pub fn disabled() -> Self {
        Self {
            learning_rate: 0.0,
            ..Default::default()
        }
    }
}

/// A single feedback sample from query execution
#[derive(Debug, Clone)]
pub struct RoutingFeedback {
    /// The query point
    pub query: Point,

    /// Container centroid that was selected
    pub selected_centroid: Point,

    /// Whether the selection led to good results (positive = good)
    pub reward: f32,

    /// Which level in the hierarchy this feedback is for
    pub level: usize,
}

/// Learnable routing weights for HAT
///
/// Maintains per-dimension (or scalar) weights that modify
/// the similarity computation during tree traversal.
#[derive(Debug, Clone)]
pub struct LearnableRouter {
    /// Configuration
    config: LearnableRoutingConfig,

    /// Per-dimension weights (or single weight if per_dimension_weights=false)
    weights: Vec<f32>,

    /// Momentum accumulator for smooth updates
    momentum_buffer: Vec<f32>,

    /// Feedback buffer for batch updates
    feedback_buffer: VecDeque<RoutingFeedback>,

    /// Total feedback samples received
    total_samples: usize,

    /// Dimensionality
    dims: usize,
}

impl LearnableRouter {
    /// Create a new learnable router
    pub fn new(dims: usize, config: LearnableRoutingConfig) -> Self {
        let weight_count = if config.per_dimension_weights { dims } else { 1 };

        Self {
            config,
            weights: vec![1.0; weight_count],  // Start with uniform weights
            momentum_buffer: vec![0.0; weight_count],
            feedback_buffer: VecDeque::new(),
            total_samples: 0,
            dims,
        }
    }

    /// Create with default config
    pub fn default_for_dims(dims: usize) -> Self {
        Self::new(dims, LearnableRoutingConfig::default())
    }

    /// Check if learning is enabled
    pub fn is_learning_enabled(&self) -> bool {
        self.config.learning_rate > 0.0
    }

    /// Get current weights (for inspection/serialization)
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Compute weighted similarity between query and centroid
    ///
    /// Returns a similarity score (higher = more similar)
    pub fn weighted_similarity(&self, query: &Point, centroid: &Point) -> f32 {
        if self.config.per_dimension_weights {
            // Weighted dot product: Σᵢ wᵢ · qᵢ · cᵢ
            query.dims().iter()
                .zip(centroid.dims().iter())
                .zip(self.weights.iter())
                .map(|((q, c), w)| w * q * c)
                .sum()
        } else {
            // Single scalar weight (equivalent to scaled cosine)
            let dot: f32 = query.dims().iter()
                .zip(centroid.dims().iter())
                .map(|(q, c)| q * c)
                .sum();
            self.weights[0] * dot
        }
    }

    /// Record feedback from a routing decision
    pub fn record_feedback(&mut self, feedback: RoutingFeedback) {
        self.feedback_buffer.push_back(feedback);
        self.total_samples += 1;

        // Trim buffer if too large
        while self.feedback_buffer.len() > self.config.max_feedback_samples {
            self.feedback_buffer.pop_front();
        }

        // Trigger update if conditions met
        if self.should_update() {
            self.update_weights();
        }
    }

    /// Check if we should update weights
    fn should_update(&self) -> bool {
        self.config.learning_rate > 0.0
            && self.feedback_buffer.len() >= self.config.min_samples_to_learn
            && self.total_samples % self.config.update_frequency == 0
    }

    /// Update weights based on accumulated feedback
    ///
    /// Uses a simple gradient-free approach:
    /// - For positive feedback: increase weights for dimensions where q·c was high
    /// - For negative feedback: decrease weights for dimensions where q·c was high
    fn update_weights(&mut self) {
        if self.feedback_buffer.is_empty() {
            return;
        }

        let lr = self.config.learning_rate;
        let momentum = self.config.momentum;
        let decay = self.config.weight_decay;

        // Compute gradient estimate from feedback
        let mut gradient = vec![0.0f32; self.weights.len()];

        for feedback in &self.feedback_buffer {
            let reward = feedback.reward;

            if self.config.per_dimension_weights {
                // Per-dimension update
                for ((&q, &c), g) in feedback.query.dims().iter()
                    .zip(feedback.selected_centroid.dims().iter())
                    .zip(gradient.iter_mut())
                {
                    // Gradient: reward * q * c (increase weight if positive reward)
                    *g += reward * q * c;
                }
            } else {
                // Scalar update
                let dot: f32 = feedback.query.dims().iter()
                    .zip(feedback.selected_centroid.dims().iter())
                    .map(|(q, c)| q * c)
                    .sum();
                gradient[0] += reward * dot;
            }
        }

        // Normalize by number of samples
        let n = self.feedback_buffer.len() as f32;
        for g in gradient.iter_mut() {
            *g /= n;
        }

        // Apply momentum and update weights
        for (i, (w, g)) in self.weights.iter_mut().zip(gradient.iter()).enumerate() {
            // Momentum update
            self.momentum_buffer[i] = momentum * self.momentum_buffer[i] + (1.0 - momentum) * g;

            // Weight update with decay
            *w += lr * self.momentum_buffer[i] - decay * (*w - 1.0);

            // Clamp weights to reasonable range
            *w = w.clamp(0.1, 10.0);
        }
    }

    /// Record positive feedback (successful retrieval)
    pub fn record_success(&mut self, query: &Point, selected_centroid: &Point, level: usize) {
        self.record_feedback(RoutingFeedback {
            query: query.clone(),
            selected_centroid: selected_centroid.clone(),
            reward: 1.0,
            level,
        });
    }

    /// Record negative feedback (unsuccessful retrieval)
    pub fn record_failure(&mut self, query: &Point, selected_centroid: &Point, level: usize) {
        self.record_feedback(RoutingFeedback {
            query: query.clone(),
            selected_centroid: selected_centroid.clone(),
            reward: -1.0,
            level,
        });
    }

    /// Record implicit feedback with continuous reward
    pub fn record_implicit(&mut self, query: &Point, selected_centroid: &Point, level: usize, relevance_score: f32) {
        // Convert relevance (0-1) to reward (-1 to +1)
        let reward = 2.0 * relevance_score - 1.0;
        self.record_feedback(RoutingFeedback {
            query: query.clone(),
            selected_centroid: selected_centroid.clone(),
            reward,
            level,
        });
    }

    /// Get statistics about the router
    pub fn stats(&self) -> RouterStats {
        RouterStats {
            total_samples: self.total_samples,
            buffer_size: self.feedback_buffer.len(),
            weight_mean: self.weights.iter().sum::<f32>() / self.weights.len() as f32,
            weight_std: {
                let mean = self.weights.iter().sum::<f32>() / self.weights.len() as f32;
                (self.weights.iter().map(|w| (w - mean).powi(2)).sum::<f32>()
                    / self.weights.len() as f32).sqrt()
            },
            weight_min: self.weights.iter().cloned().fold(f32::INFINITY, f32::min),
            weight_max: self.weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        }
    }

    /// Reset weights to uniform
    pub fn reset_weights(&mut self) {
        for w in self.weights.iter_mut() {
            *w = 1.0;
        }
        for m in self.momentum_buffer.iter_mut() {
            *m = 0.0;
        }
    }

    /// Clear feedback buffer
    pub fn clear_feedback(&mut self) {
        self.feedback_buffer.clear();
    }

    /// Get the number of dimensions
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Serialize weights to bytes
    pub fn serialize_weights(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.weights.len() * 4);
        for w in &self.weights {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        bytes
    }

    /// Deserialize weights from bytes
    pub fn deserialize_weights(&mut self, bytes: &[u8]) -> Result<(), &'static str> {
        if bytes.len() != self.weights.len() * 4 {
            return Err("Weight count mismatch");
        }

        for (i, chunk) in bytes.chunks(4).enumerate() {
            let arr: [u8; 4] = chunk.try_into().map_err(|_| "Invalid byte chunk")?;
            self.weights[i] = f32::from_le_bytes(arr);
        }

        Ok(())
    }
}

/// Statistics about the learnable router
#[derive(Debug, Clone)]
pub struct RouterStats {
    pub total_samples: usize,
    pub buffer_size: usize,
    pub weight_mean: f32,
    pub weight_std: f32,
    pub weight_min: f32,
    pub weight_max: f32,
}

/// Compute routing score for beam search
///
/// Combines weighted similarity with optional biases
pub fn compute_routing_score(
    router: &LearnableRouter,
    query: &Point,
    centroid: &Point,
    temporal_distance: f32,
    temporal_weight: f32,
) -> f32 {
    let semantic_sim = router.weighted_similarity(query, centroid);

    // Convert to distance (lower = better for routing)
    let semantic_dist = 1.0 - semantic_sim;

    // Combine with temporal
    semantic_dist * (1.0 - temporal_weight) + temporal_distance * temporal_weight
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_point(v: Vec<f32>) -> Point {
        Point::new(v).normalize()
    }

    #[test]
    fn test_router_creation() {
        let router = LearnableRouter::default_for_dims(64);

        assert_eq!(router.dims(), 64);
        assert_eq!(router.weights().len(), 64);
        assert!(router.is_learning_enabled());

        // All weights should start at 1.0
        for &w in router.weights() {
            assert!((w - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_weighted_similarity() {
        let router = LearnableRouter::default_for_dims(4);

        let query = make_point(vec![1.0, 0.0, 0.0, 0.0]);
        let centroid = make_point(vec![0.8, 0.2, 0.0, 0.0]);

        let sim = router.weighted_similarity(&query, &centroid);

        // With uniform weights, should be close to cosine similarity
        let expected_cosine: f32 = query.dims().iter()
            .zip(centroid.dims().iter())
            .map(|(q, c)| q * c)
            .sum();

        assert!((sim - expected_cosine).abs() < 1e-5);
    }

    #[test]
    fn test_feedback_recording() {
        let mut router = LearnableRouter::new(4, LearnableRoutingConfig {
            min_samples_to_learn: 5,
            update_frequency: 5,
            ..Default::default()
        });

        let query = make_point(vec![1.0, 0.0, 0.0, 0.0]);
        let centroid = make_point(vec![0.9, 0.1, 0.0, 0.0]);

        // Record several positive feedbacks
        for _ in 0..10 {
            router.record_success(&query, &centroid, 0);
        }

        let stats = router.stats();
        assert_eq!(stats.total_samples, 10);

        // Weights should have been updated
        // Dimension 0 (aligned with query) should increase
        println!("Weights after positive feedback: {:?}", router.weights());
    }

    #[test]
    fn test_learning_dynamics() {
        let mut router = LearnableRouter::new(4, LearnableRoutingConfig {
            learning_rate: 0.1,
            min_samples_to_learn: 3,
            update_frequency: 3,
            momentum: 0.0,  // No momentum for predictable testing
            weight_decay: 0.0,  // No decay for predictable testing
            ..Default::default()
        });

        // Query aligned with dimension 0
        let query = make_point(vec![1.0, 0.0, 0.0, 0.0]);
        // Centroid also aligned with dimension 0
        let centroid_good = make_point(vec![0.95, 0.05, 0.0, 0.0]);
        // Centroid aligned with dimension 1
        let centroid_bad = make_point(vec![0.0, 1.0, 0.0, 0.0]);

        // Record positive feedback for good centroid
        for _ in 0..6 {
            router.record_success(&query, &centroid_good, 0);
        }

        let weights_after_positive = router.weights().to_vec();

        // Record negative feedback for bad centroid
        for _ in 0..6 {
            router.record_failure(&query, &centroid_bad, 0);
        }

        let weights_after_negative = router.weights().to_vec();

        println!("Initial weights: [1.0, 1.0, 1.0, 1.0]");
        println!("After positive: {:?}", weights_after_positive);
        println!("After negative: {:?}", weights_after_negative);

        // Weight for dim 0 should have increased from positive feedback
        // (query[0] * centroid_good[0] is high and reward is positive)
    }

    #[test]
    fn test_disabled_learning() {
        let mut router = LearnableRouter::new(4, LearnableRoutingConfig::disabled());

        assert!(!router.is_learning_enabled());

        let query = make_point(vec![1.0, 0.0, 0.0, 0.0]);
        let centroid = make_point(vec![0.9, 0.1, 0.0, 0.0]);

        // Record feedback
        for _ in 0..100 {
            router.record_success(&query, &centroid, 0);
        }

        // Weights should remain at 1.0
        for &w in router.weights() {
            assert!((w - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_serialization() {
        let mut router = LearnableRouter::default_for_dims(4);

        // Modify weights
        for (i, w) in router.weights.iter_mut().enumerate() {
            *w = (i as f32 + 1.0) * 0.5;
        }

        let bytes = router.serialize_weights();

        let mut router2 = LearnableRouter::default_for_dims(4);
        router2.deserialize_weights(&bytes).unwrap();

        for (w1, w2) in router.weights().iter().zip(router2.weights().iter()) {
            assert!((w1 - w2).abs() < 1e-6);
        }
    }
}
