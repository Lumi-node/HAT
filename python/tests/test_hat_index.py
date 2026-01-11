"""Tests for ARMS-HAT Python bindings."""

import pytest
import tempfile
import os


def test_import():
    """Test that the module can be imported."""
    from arms_hat import HatIndex, HatConfig, SearchResult


def test_create_index():
    """Test index creation."""
    from arms_hat import HatIndex

    index = HatIndex.cosine(128)
    assert len(index) == 0
    assert index.is_empty()


def test_add_and_query():
    """Test adding points and querying."""
    from arms_hat import HatIndex

    dims = 64
    index = HatIndex.cosine(dims)

    # Add some points
    ids = []
    for i in range(10):
        embedding = [0.0] * dims
        embedding[i % dims] = 1.0
        embedding[(i + 1) % dims] = 0.5
        id_ = index.add(embedding)
        ids.append(id_)
        assert len(id_) == 32  # Hex ID

    assert len(index) == 10
    assert not index.is_empty()

    # Query
    query = [0.0] * dims
    query[0] = 1.0
    query[1] = 0.5

    results = index.near(query, k=5)
    assert len(results) == 5

    # First result should be the closest match
    assert results[0].id == ids[0]
    assert results[0].score > 0.9  # High cosine similarity


def test_sessions():
    """Test session management."""
    from arms_hat import HatIndex

    index = HatIndex.cosine(32)

    # Add points to first session
    for i in range(5):
        index.add([float(i % 32 == j) for j in range(32)])

    # Start new session
    index.new_session()

    # Add points to second session
    for i in range(5):
        index.add([float((i + 10) % 32 == j) for j in range(32)])

    stats = index.stats()
    assert stats.session_count >= 1  # At least one session
    assert stats.chunk_count == 10


def test_documents():
    """Test document management within sessions."""
    from arms_hat import HatIndex

    index = HatIndex.cosine(32)

    # Add points to first document
    for i in range(3):
        index.add([1.0 if j == i else 0.0 for j in range(32)])

    # Start new document
    index.new_document()

    # Add points to second document
    for i in range(3):
        index.add([1.0 if j == i + 10 else 0.0 for j in range(32)])

    stats = index.stats()
    assert stats.document_count >= 1
    assert stats.chunk_count == 6


def test_persistence_bytes():
    """Test serialization to/from bytes."""
    from arms_hat import HatIndex

    dims = 64
    index = HatIndex.cosine(dims)

    # Add points
    ids = []
    for i in range(20):
        embedding = [0.1] * dims
        embedding[i % dims] = 1.0
        ids.append(index.add(embedding))

    # Serialize
    data = index.to_bytes()
    assert len(data) > 0

    # Deserialize
    loaded = HatIndex.from_bytes(data)
    assert len(loaded) == len(index)

    # Query should give same results
    query = [0.1] * dims
    query[0] = 1.0

    original_results = index.near(query, k=5)
    loaded_results = loaded.near(query, k=5)

    assert len(original_results) == len(loaded_results)
    assert original_results[0].id == loaded_results[0].id


def test_persistence_file():
    """Test save/load to file."""
    from arms_hat import HatIndex

    dims = 64
    index = HatIndex.cosine(dims)

    # Add points
    for i in range(10):
        embedding = [0.1] * dims
        embedding[i % dims] = 1.0
        index.add(embedding)

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".hat", delete=False) as f:
        path = f.name

    try:
        index.save(path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

        # Load
        loaded = HatIndex.load(path)
        assert len(loaded) == len(index)

    finally:
        os.unlink(path)


def test_config():
    """Test custom configuration."""
    from arms_hat import HatIndex, HatConfig

    config = HatConfig()
    # Chain configuration
    config = config.with_beam_width(5)
    config = config.with_temporal_weight(0.1)

    index = HatIndex.with_config(128, config)
    assert len(index) == 0


def test_remove():
    """Test point removal."""
    from arms_hat import HatIndex

    index = HatIndex.cosine(32)

    id1 = index.add([1.0] + [0.0] * 31)
    id2 = index.add([0.0, 1.0] + [0.0] * 30)

    assert len(index) == 2

    index.remove(id1)
    assert len(index) == 1

    # Query should only find id2
    results = index.near([0.0, 1.0] + [0.0] * 30, k=5)
    assert len(results) == 1
    assert results[0].id == id2


def test_consolidate():
    """Test consolidation."""
    from arms_hat import HatIndex

    index = HatIndex.cosine(32)

    # Add many points
    for i in range(100):
        embedding = [0.0] * 32
        embedding[i % 32] = 1.0
        index.add(embedding)

    # Consolidate should not error
    index.consolidate()
    index.consolidate_full()

    assert len(index) == 100


def test_stats():
    """Test stats retrieval."""
    from arms_hat import HatIndex

    index = HatIndex.cosine(64)

    for i in range(10):
        index.add([float(i % 64 == j) for j in range(64)])

    stats = index.stats()
    assert stats.chunk_count == 10
    assert stats.total_points == 10


def test_repr():
    """Test string representations."""
    from arms_hat import HatIndex, HatConfig, SearchResult

    index = HatIndex.cosine(64)
    repr_str = repr(index)
    assert "HatIndex" in repr_str

    config = HatConfig()
    repr_str = repr(config)
    assert "HatConfig" in repr_str


def test_near_sessions():
    """Test coarse-grained session search."""
    from arms_hat import HatIndex

    index = HatIndex.cosine(32)

    # Session 1: points along dimension 0
    for i in range(5):
        embedding = [0.0] * 32
        embedding[0] = 1.0
        embedding[i + 1] = 0.3
        index.add(embedding)

    index.new_session()

    # Session 2: points along dimension 10
    for i in range(5):
        embedding = [0.0] * 32
        embedding[10] = 1.0
        embedding[i + 11] = 0.3
        index.add(embedding)

    # Query similar to session 1
    query = [0.0] * 32
    query[0] = 1.0

    sessions = index.near_sessions(query, k=2)
    assert len(sessions) >= 1

    # First session should be more relevant
    if len(sessions) > 1:
        assert sessions[0].score >= sessions[1].score


def test_high_dimensions():
    """Test with OpenAI embedding dimensions."""
    from arms_hat import HatIndex

    dims = 1536  # OpenAI ada-002 dimensions
    index = HatIndex.cosine(dims)

    # Add some high-dimensional points
    for i in range(10):
        embedding = [(j * i * 0.01) % 1.0 for j in range(dims)]
        index.add(embedding)

    assert len(index) == 10

    # Query
    query = [0.5] * dims
    results = index.near(query, k=5)
    assert len(results) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
