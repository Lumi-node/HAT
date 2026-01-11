#!/bin/bash
#
# HAT Benchmark Reproducibility Suite
# ===================================
#
# This script runs all benchmarks from the HAT paper and generates
# a comprehensive results report.
#
# Usage:
#   ./run_all_benchmarks.sh [--quick]
#
# Options:
#   --quick    Run abbreviated benchmarks (faster, less thorough)
#
# Requirements:
#   - Rust toolchain (cargo)
#   - Python 3.8+ with venv
#   - ~2GB free disk space
#   - ~10 minutes for full suite, ~2 minutes for quick

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$SCRIPT_DIR/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="$RESULTS_DIR/benchmark_results_$TIMESTAMP.txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
QUICK_MODE=false
if [[ "$1" == "--quick" ]]; then
    QUICK_MODE=true
    echo -e "${YELLOW}Running in quick mode (abbreviated benchmarks)${NC}"
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "========================================================================"
echo "  HAT Benchmark Reproducibility Suite"
echo "  $(date)"
echo "========================================================================"
echo ""
echo "Project directory: $PROJECT_DIR"
echo "Results will be saved to: $RESULTS_FILE"
echo ""

# Initialize results file
cat > "$RESULTS_FILE" << EOF
HAT Benchmark Results
=====================
Date: $(date)
Host: $(hostname)
Rust: $(rustc --version)
Quick mode: $QUICK_MODE

EOF

cd "$PROJECT_DIR"

# Function to run a test and capture results
run_benchmark() {
    local name="$1"
    local test_name="$2"

    echo -e "${BLUE}[$name]${NC} Running..."
    echo "" >> "$RESULTS_FILE"
    echo "=== $name ===" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"

    if cargo test --test "$test_name" -- --nocapture 2>&1 | tee -a "$RESULTS_FILE"; then
        echo -e "${GREEN}[$name]${NC} PASSED"
    else
        echo -e "${RED}[$name]${NC} FAILED"
        echo "FAILED" >> "$RESULTS_FILE"
    fi
    echo ""
}

echo "========================================================================"
echo "  Phase 1: Building Project"
echo "========================================================================"

echo "Building release version..."
cargo build --release 2>&1 | tail -5

echo "Building test suite..."
cargo build --tests 2>&1 | tail -5

echo ""
echo "========================================================================"
echo "  Phase 2: Running Core Benchmarks"
echo "========================================================================"

# Phase 3.1: HAT vs HNSW
echo ""
echo "--- Phase 3.1: HAT vs HNSW Comparative Benchmark ---"
run_benchmark "HAT vs HNSW" "phase31_hat_vs_hnsw"

# Phase 3.2: Real Embeddings
echo ""
echo "--- Phase 3.2: Real Embedding Dimensions ---"
run_benchmark "Real Embeddings" "phase32_real_embeddings"

# Phase 3.3: Persistence
echo ""
echo "--- Phase 3.3: Persistence Layer ---"
run_benchmark "Persistence" "phase33_persistence"

# Phase 4.2: Attention State
echo ""
echo "--- Phase 4.2: Attention State Format ---"
run_benchmark "Attention State" "phase42_attention_state"

echo ""
echo "========================================================================"
echo "  Phase 3: Python Integration Tests"
echo "========================================================================"

# Check for Python venv
VENV_DIR="/tmp/arms-hat-bench-venv"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# Install dependencies
echo "Installing Python dependencies..."
pip install -q maturin pytest 2>/dev/null || true

# Build Python extension
echo "Building Python extension..."
maturin develop --features python 2>&1 | tail -3

# Run Python tests
echo ""
echo "--- Python Binding Tests ---"
echo "" >> "$RESULTS_FILE"
echo "=== Python Binding Tests ===" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

if python -m pytest "$PROJECT_DIR/python/tests/" -v 2>&1 | tee -a "$RESULTS_FILE"; then
    echo -e "${GREEN}[Python Tests]${NC} PASSED"
else
    echo -e "${RED}[Python Tests]${NC} FAILED"
fi

echo ""
echo "========================================================================"
echo "  Phase 4: End-to-End Demo"
echo "========================================================================"

echo "" >> "$RESULTS_FILE"
echo "=== End-to-End Demo ===" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Check for sentence-transformers
if pip show sentence-transformers >/dev/null 2>&1; then
    echo "Running end-to-end demo with real embeddings..."
    python "$PROJECT_DIR/examples/demo_hat_memory.py" 2>&1 | tee -a "$RESULTS_FILE"
else
    echo "Installing sentence-transformers for full demo..."
    pip install -q sentence-transformers 2>/dev/null || true

    if pip show sentence-transformers >/dev/null 2>&1; then
        python "$PROJECT_DIR/examples/demo_hat_memory.py" 2>&1 | tee -a "$RESULTS_FILE"
    else
        echo "Running demo with pseudo-embeddings (sentence-transformers not available)..."
        python "$PROJECT_DIR/examples/demo_hat_memory.py" 2>&1 | tee -a "$RESULTS_FILE"
    fi
fi

deactivate

echo ""
echo "========================================================================"
echo "  Summary"
echo "========================================================================"

# Extract key metrics from results
echo "" >> "$RESULTS_FILE"
echo "=== Summary ===" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Count passed tests
RUST_PASSED=$(grep -c "test .* ok" "$RESULTS_FILE" 2>/dev/null || echo "0")
PYTHON_PASSED=$(grep -c "PASSED" "$RESULTS_FILE" 2>/dev/null || echo "0")

echo "Results saved to: $RESULTS_FILE"
echo ""
echo "Key Results:"
echo "  - Rust tests passed: ~$RUST_PASSED"
echo "  - Python tests passed: ~$PYTHON_PASSED"
echo ""

# Extract recall metrics if available
if grep -q "HAT enables 100% recall" "$RESULTS_FILE"; then
    echo -e "${GREEN}Core claim validated: 100% recall achieved${NC}"
fi

if grep -q "Average retrieval latency" "$RESULTS_FILE"; then
    LATENCY=$(grep "Average retrieval latency" "$RESULTS_FILE" | tail -1 | grep -oE '[0-9]+\.[0-9]+ms')
    echo "  - Retrieval latency: $LATENCY"
fi

echo ""
echo "========================================================================"
echo "  Benchmark Complete"
echo "========================================================================"
echo ""
echo "Full results: $RESULTS_FILE"
echo ""
