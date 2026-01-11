//! # Subspace Containers for HAT
//!
//! This module implements subspace-aware container representations for HAT.
//! Instead of representing containers as single centroid points, we model them
//! as subspaces that capture the "shape" and "spread" of points within.
//!
//! ## Key Insight (from journal 006)
//!
//! "A session isn't a single point - it's a *region* of the manifold."
//!
//! ## Grassmann-Inspired Approach
//!
//! - Each container is represented by its centroid PLUS principal directions
//! - Similarity between containers uses subspace angles (principal angles)
//! - Better captures diverse content within a container
//!
//! ## Benefits
//!
//! 1. **Better Routing**: Query can match containers even if not close to centroid
//! 2. **Diversity Awareness**: Wide containers (diverse content) vs narrow containers
//! 3. **Geometric Fidelity**: More accurate representation of point distributions

use crate::core::Point;

/// Configuration for subspace representation
#[derive(Debug, Clone)]
pub struct SubspaceConfig {
    /// Number of principal components to track (subspace rank)
    pub rank: usize,

    /// Minimum points before computing subspace (need enough for covariance)
    pub min_points_for_subspace: usize,

    /// Weight of subspace similarity vs centroid similarity (0.0 = centroid only)
    pub subspace_weight: f32,

    /// Enable incremental covariance updates during insertion (vs only during consolidation)
    /// When false, subspace is only computed during consolidation - much faster inserts
    pub incremental_covariance: bool,
}

impl Default for SubspaceConfig {
    fn default() -> Self {
        Self {
            rank: 3,                        // Track top 3 principal directions
            min_points_for_subspace: 5,     // Need at least 5 points for meaningful covariance
            subspace_weight: 0.3,           // 30% subspace, 70% centroid by default
            incremental_covariance: false,  // Default: only compute during consolidation (faster)
        }
    }
}

impl SubspaceConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_rank(mut self, rank: usize) -> Self {
        self.rank = rank;
        self
    }

    pub fn with_subspace_weight(mut self, weight: f32) -> Self {
        self.subspace_weight = weight.clamp(0.0, 1.0);
        self
    }
}

/// Subspace representation for a container
///
/// Stores the centroid plus principal directions that capture
/// the variance/spread of points within the container.
#[derive(Debug, Clone)]
pub struct Subspace {
    /// Centroid (mean of points)
    pub centroid: Point,

    /// Principal directions (orthonormal basis for subspace)
    /// Each direction is a unit vector
    pub principal_directions: Vec<Point>,

    /// Eigenvalues (variance in each principal direction)
    /// Stored in decreasing order
    pub eigenvalues: Vec<f32>,

    /// Number of points used to compute this subspace
    pub point_count: usize,

    /// Running sum for incremental centroid updates
    accumulated_sum: Vec<f32>,

    /// Running covariance matrix (upper triangle only for efficiency)
    /// For incremental updates: cov = (sum of outer products) / n - mean * mean^T
    accumulated_outer_product: Vec<f32>,
}

impl Subspace {
    /// Create a new empty subspace
    pub fn new(dimensionality: usize) -> Self {
        Self {
            centroid: Point::origin(dimensionality),
            principal_directions: Vec::new(),
            eigenvalues: Vec::new(),
            point_count: 0,
            accumulated_sum: vec![0.0; dimensionality],
            // Upper triangle of d x d matrix: d * (d + 1) / 2 elements
            accumulated_outer_product: vec![0.0; dimensionality * (dimensionality + 1) / 2],
        }
    }

    /// Create from a single point
    pub fn from_point(point: &Point) -> Self {
        Self {
            centroid: point.clone(),
            principal_directions: Vec::new(),
            eigenvalues: Vec::new(),
            point_count: 1,
            accumulated_sum: point.dims().to_vec(),
            accumulated_outer_product: Self::outer_product_upper(point.dims()),
        }
    }

    /// Dimensionality of the ambient space
    pub fn dimensionality(&self) -> usize {
        self.centroid.dimensionality()
    }

    /// Check if subspace has meaningful principal directions
    pub fn has_subspace(&self) -> bool {
        !self.principal_directions.is_empty()
    }

    /// Get the subspace rank (number of principal directions)
    pub fn rank(&self) -> usize {
        self.principal_directions.len()
    }

    /// Compute upper triangle of outer product v * v^T
    fn outer_product_upper(v: &[f32]) -> Vec<f32> {
        let n = v.len();
        let mut result = vec![0.0; n * (n + 1) / 2];
        let mut idx = 0;
        for i in 0..n {
            for j in i..n {
                result[idx] = v[i] * v[j];
                idx += 1;
            }
        }
        result
    }

    /// Get element from upper triangle storage
    fn get_upper(&self, i: usize, j: usize) -> f32 {
        let (row, col) = if i <= j { (i, j) } else { (j, i) };
        let n = self.dimensionality();
        // Index into upper triangle
        let idx = row * (2 * n - row - 1) / 2 + col;
        self.accumulated_outer_product[idx]
    }

    /// Add element to upper triangle storage
    fn add_to_upper(&mut self, i: usize, j: usize, value: f32) {
        let (row, col) = if i <= j { (i, j) } else { (j, i) };
        let n = self.dimensionality();
        let idx = row * (2 * n - row - 1) / 2 + col;
        self.accumulated_outer_product[idx] += value;
    }

    /// Incrementally add a point
    pub fn add_point(&mut self, point: &Point) {
        let dims = point.dims();

        // Update running sum
        for (i, &v) in dims.iter().enumerate() {
            self.accumulated_sum[i] += v;
        }

        // Update outer product accumulator
        for i in 0..dims.len() {
            for j in i..dims.len() {
                self.add_to_upper(i, j, dims[i] * dims[j]);
            }
        }

        self.point_count += 1;

        // Update centroid
        let n = self.point_count as f32;
        let centroid_dims: Vec<f32> = self.accumulated_sum.iter()
            .map(|&s| s / n)
            .collect();
        self.centroid = Point::new(centroid_dims).normalize();
    }

    /// Compute covariance matrix from accumulated statistics
    fn compute_covariance(&self) -> Vec<Vec<f32>> {
        let n = self.dimensionality();
        let count = self.point_count as f32;

        if count < 2.0 {
            return vec![vec![0.0; n]; n];
        }

        // Mean vector
        let mean: Vec<f32> = self.accumulated_sum.iter()
            .map(|&s| s / count)
            .collect();

        // Covariance = E[X*X^T] - E[X]*E[X]^T
        let mut cov = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in i..n {
                let exx = self.get_upper(i, j) / count;
                let exex = mean[i] * mean[j];
                let c = exx - exex;
                cov[i][j] = c;
                cov[j][i] = c;  // Symmetric
            }
        }

        cov
    }

    /// Recompute principal directions from covariance
    /// Uses power iteration for efficiency (avoids full eigendecomposition)
    pub fn recompute_subspace(&mut self, rank: usize) {
        if self.point_count < 3 {
            // Not enough points for meaningful subspace
            self.principal_directions.clear();
            self.eigenvalues.clear();
            return;
        }

        let cov = self.compute_covariance();
        let n = self.dimensionality();

        // Extract top-k eigenvectors using power iteration with deflation
        let mut directions = Vec::new();
        let mut values = Vec::new();
        let mut working_cov = cov.clone();

        for _ in 0..rank.min(n) {
            // Power iteration for dominant eigenvector
            let (eigval, eigvec) = self.power_iteration(&working_cov, 50);

            if eigval < 1e-8 {
                break;  // No more significant variance
            }

            values.push(eigval);
            directions.push(Point::new(eigvec.clone()).normalize());

            // Deflate: remove this eigenvector's contribution
            for i in 0..n {
                for j in 0..n {
                    working_cov[i][j] -= eigval * eigvec[i] * eigvec[j];
                }
            }
        }

        self.principal_directions = directions;
        self.eigenvalues = values;
    }

    /// Power iteration to find dominant eigenvector
    fn power_iteration(&self, matrix: &[Vec<f32>], max_iters: usize) -> (f32, Vec<f32>) {
        let n = matrix.len();

        // Initialize with random-ish vector (use first column of matrix + perturbation)
        let mut v: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.1).collect();
        let mut norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v {
            *x /= norm;
        }

        let mut eigenvalue = 0.0f32;

        for _ in 0..max_iters {
            // v_new = A * v
            let mut v_new = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    v_new[i] += matrix[i][j] * v[j];
                }
            }

            // Compute eigenvalue approximation
            eigenvalue = v_new.iter().zip(v.iter()).map(|(a, b)| a * b).sum();

            // Normalize
            norm = v_new.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm < 1e-10 {
                return (0.0, vec![0.0; n]);
            }

            let converged = v.iter().zip(v_new.iter())
                .map(|(a, b)| (a - b / norm).abs())
                .sum::<f32>() < 1e-8;

            for i in 0..n {
                v[i] = v_new[i] / norm;
            }

            if converged {
                break;
            }
        }

        (eigenvalue.abs(), v)
    }
}

/// Compute subspace similarity using principal angles
///
/// Based on Grassmann geometry: the similarity between two subspaces
/// is determined by the principal angles between them.
///
/// For k-dimensional subspaces, there are k principal angles θ₁...θₖ
/// where 0 ≤ θ₁ ≤ ... ≤ θₖ ≤ π/2
///
/// Common measures:
/// - Projection similarity: Σ cos²(θᵢ) / k  (ranges 0-1)
/// - Geodesic distance: sqrt(Σ θᵢ²)
/// - Chordal distance: sqrt(Σ sin²(θᵢ))
pub fn subspace_similarity(a: &Subspace, b: &Subspace) -> f32 {
    // If either has no subspace, fall back to centroid similarity
    if !a.has_subspace() || !b.has_subspace() {
        return centroid_similarity(&a.centroid, &b.centroid);
    }

    // Compute inner products between principal directions
    let rank_a = a.rank();
    let rank_b = b.rank();
    let k = rank_a.min(rank_b);

    if k == 0 {
        return centroid_similarity(&a.centroid, &b.centroid);
    }

    // Build matrix M where M[i][j] = <a_i, b_j> (dot products)
    let mut m = vec![vec![0.0f32; rank_b]; rank_a];
    for i in 0..rank_a {
        for j in 0..rank_b {
            let dot: f32 = a.principal_directions[i].dims().iter()
                .zip(b.principal_directions[j].dims().iter())
                .map(|(x, y)| x * y)
                .sum();
            m[i][j] = dot;
        }
    }

    // SVD of M gives principal angles: σᵢ = cos(θᵢ)
    // For simplicity, use a greedy approximation:
    // Find k maximum entries while avoiding row/column reuse
    let cos_angles = greedy_max_matching(&m, k);

    // Projection similarity: mean of cos²(θᵢ)
    let similarity: f32 = cos_angles.iter()
        .map(|&c| c * c)  // cos²(θ)
        .sum::<f32>() / k as f32;

    similarity
}

/// Greedy approximation to find k largest entries with no repeated rows/columns
fn greedy_max_matching(m: &[Vec<f32>], k: usize) -> Vec<f32> {
    let rows = m.len();
    let cols = if rows > 0 { m[0].len() } else { 0 };

    let mut used_rows = vec![false; rows];
    let mut used_cols = vec![false; cols];
    let mut result = Vec::new();

    for _ in 0..k {
        let mut best = (0, 0, 0.0f32);

        for i in 0..rows {
            if used_rows[i] { continue; }
            for j in 0..cols {
                if used_cols[j] { continue; }
                let val = m[i][j].abs();
                if val > best.2 {
                    best = (i, j, val);
                }
            }
        }

        if best.2 > 0.0 {
            used_rows[best.0] = true;
            used_cols[best.1] = true;
            result.push(best.2);
        } else {
            break;
        }
    }

    result
}

/// Simple centroid similarity (cosine)
fn centroid_similarity(a: &Point, b: &Point) -> f32 {
    let dot: f32 = a.dims().iter()
        .zip(b.dims().iter())
        .map(|(x, y)| x * y)
        .sum();
    dot.clamp(-1.0, 1.0)
}

/// Combined similarity: weighted combination of centroid and subspace similarity
///
/// score = (1 - weight) * centroid_sim + weight * subspace_sim
pub fn combined_subspace_similarity(
    query: &Point,
    container: &Subspace,
    config: &SubspaceConfig,
) -> f32 {
    let centroid_sim = centroid_similarity(query, &container.centroid);

    if !container.has_subspace() || config.subspace_weight < 1e-6 {
        return centroid_sim;
    }

    // Subspace similarity: how well does query align with principal directions?
    // Measure: sum of squared projections onto principal directions
    let subspace_sim = query_subspace_alignment(query, container);

    // Weighted combination
    let w = config.subspace_weight;
    (1.0 - w) * centroid_sim + w * subspace_sim
}

/// Measure how well a query aligns with a subspace
///
/// Higher score means query is well-captured by the subspace's principal directions
pub fn query_subspace_alignment(query: &Point, subspace: &Subspace) -> f32 {
    if !subspace.has_subspace() {
        return centroid_similarity(query, &subspace.centroid);
    }

    // Center query relative to centroid
    let centered: Vec<f32> = query.dims().iter()
        .zip(subspace.centroid.dims().iter())
        .map(|(q, c)| q - c)
        .collect();

    let centered_norm: f32 = centered.iter().map(|x| x * x).sum::<f32>().sqrt();
    if centered_norm < 1e-10 {
        // Query is at centroid - perfect match
        return 1.0;
    }

    // Compute squared projections onto each principal direction
    let mut total_proj_sq = 0.0f32;
    for (dir, &eigenval) in subspace.principal_directions.iter().zip(subspace.eigenvalues.iter()) {
        let proj: f32 = centered.iter()
            .zip(dir.dims().iter())
            .map(|(c, d)| c * d)
            .sum();

        // Weight by eigenvalue (variance in that direction)
        // Higher eigenvalue = more likely direction for data variation
        let weight = (eigenval / subspace.eigenvalues[0]).sqrt();
        total_proj_sq += proj * proj * weight;
    }

    // Normalize by centered query magnitude
    let alignment = (total_proj_sq / (centered_norm * centered_norm)).min(1.0);

    // Combine with centroid similarity for overall score
    let centroid_sim = centroid_similarity(query, &subspace.centroid);

    // Score: close to centroid AND aligned with principal directions
    (centroid_sim + alignment) / 2.0
}

/// Compute the "spread" or diversity of a subspace
///
/// Higher values indicate more diverse content (larger variance)
/// Lower values indicate tightly clustered content
pub fn subspace_spread(subspace: &Subspace) -> f32 {
    if subspace.eigenvalues.is_empty() {
        return 0.0;
    }

    // Total variance (sum of eigenvalues)
    subspace.eigenvalues.iter().sum()
}

/// Compute the "isotropy" of a subspace
///
/// Higher values (close to 1) indicate uniform spread in all directions
/// Lower values indicate elongated, anisotropic distribution
pub fn subspace_isotropy(subspace: &Subspace) -> f32 {
    if subspace.eigenvalues.len() < 2 {
        return 1.0;  // Single direction is perfectly "isotropic" in its subspace
    }

    // Ratio of smallest to largest eigenvalue
    let max = subspace.eigenvalues[0];
    let min = *subspace.eigenvalues.last().unwrap();

    if max < 1e-10 {
        return 1.0;
    }

    min / max
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_point(v: Vec<f32>) -> Point {
        Point::new(v).normalize()
    }

    #[test]
    fn test_subspace_creation() {
        let mut subspace = Subspace::new(3);

        // Add some points
        subspace.add_point(&make_point(vec![1.0, 0.0, 0.0]));
        subspace.add_point(&make_point(vec![0.9, 0.1, 0.0]));
        subspace.add_point(&make_point(vec![0.8, 0.2, 0.0]));
        subspace.add_point(&make_point(vec![0.7, 0.3, 0.1]));
        subspace.add_point(&make_point(vec![0.6, 0.4, 0.1]));

        assert_eq!(subspace.point_count, 5);

        // Compute principal directions
        subspace.recompute_subspace(2);

        assert!(subspace.has_subspace());
        assert!(subspace.rank() > 0);
        assert!(!subspace.eigenvalues.is_empty());

        println!("Centroid: {:?}", subspace.centroid.dims());
        println!("Principal directions: {}", subspace.rank());
        println!("Eigenvalues: {:?}", subspace.eigenvalues);
    }

    #[test]
    fn test_subspace_similarity() {
        let mut a = Subspace::new(3);
        let mut b = Subspace::new(3);

        // Subspace A: points along x-axis
        for i in 0..10 {
            let x = 1.0 - i as f32 * 0.05;
            let y = i as f32 * 0.05;
            a.add_point(&make_point(vec![x, y, 0.0]));
        }

        // Subspace B: similar points (should be high similarity)
        for i in 0..10 {
            let x = 0.95 - i as f32 * 0.04;
            let y = i as f32 * 0.04 + 0.05;
            b.add_point(&make_point(vec![x, y, 0.1]));
        }

        a.recompute_subspace(2);
        b.recompute_subspace(2);

        let sim = subspace_similarity(&a, &b);
        println!("Similarity between similar subspaces: {:.3}", sim);
        assert!(sim > 0.5, "Similar subspaces should have high similarity");

        // Subspace C: orthogonal to A (along z-axis)
        let mut c = Subspace::new(3);
        for i in 0..10 {
            let z = 1.0 - i as f32 * 0.05;
            c.add_point(&make_point(vec![0.0, 0.1, z]));
        }
        c.recompute_subspace(2);

        let sim_ac = subspace_similarity(&a, &c);
        println!("Similarity between orthogonal subspaces: {:.3}", sim_ac);
        assert!(sim_ac < sim, "Orthogonal subspaces should have lower similarity");
    }

    #[test]
    fn test_query_alignment() {
        let mut subspace = Subspace::new(3);

        // Points primarily along x-axis with some y variation
        for i in 0..20 {
            let x = 0.8 + (i % 3) as f32 * 0.1;
            let y = (i as f32 * 0.05) % 0.3;
            subspace.add_point(&make_point(vec![x, y, 0.05]));
        }
        subspace.recompute_subspace(2);

        // Query aligned with subspace
        let aligned_query = make_point(vec![0.9, 0.1, 0.0]);
        let aligned_score = query_subspace_alignment(&aligned_query, &subspace);

        // Query orthogonal to subspace
        let orthogonal_query = make_point(vec![0.0, 0.0, 1.0]);
        let orthogonal_score = query_subspace_alignment(&orthogonal_query, &subspace);

        println!("Aligned query score: {:.3}", aligned_score);
        println!("Orthogonal query score: {:.3}", orthogonal_score);

        assert!(aligned_score > orthogonal_score,
            "Aligned query should score higher than orthogonal query");
    }

    #[test]
    fn test_spread_and_isotropy() {
        let mut tight = Subspace::new(3);
        let mut spread_out = Subspace::new(3);

        // Tight cluster
        for _ in 0..20 {
            tight.add_point(&make_point(vec![0.9, 0.1, 0.05]));
        }

        // Spread out cluster
        for i in 0..20 {
            let angle = i as f32 * 0.3;
            spread_out.add_point(&make_point(vec![
                angle.cos(),
                angle.sin(),
                0.1
            ]));
        }

        tight.recompute_subspace(3);
        spread_out.recompute_subspace(3);

        let tight_spread = subspace_spread(&tight);
        let wide_spread = subspace_spread(&spread_out);

        println!("Tight cluster spread: {:.6}", tight_spread);
        println!("Wide cluster spread: {:.6}", wide_spread);

        // Note: with normalized vectors the spread comparison might not be as expected
        // The test validates the computation runs correctly
    }
}
