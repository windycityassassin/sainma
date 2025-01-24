import pytest
import numpy as np
from typing import Dict, List
from sainma.search.query_processor import QueryProcessor
from sainma.search.scene_indexer import SceneIndexer
from sainma.search.search_engine import SearchEngine, SearchResult

@pytest.fixture
def query_processor():
    """Create a query processor instance."""
    return QueryProcessor()

@pytest.fixture
def scene_indexer(test_data_dir):
    """Create a scene indexer instance."""
    return SceneIndexer(index_dir=str(test_data_dir))

@pytest.fixture
def search_engine(test_data_dir):
    """Create a search engine instance."""
    return SearchEngine(
        use_gpu=False,
        min_scene_length=0.5,
        scene_threshold=0.3,
        clip_output_dir=str(test_data_dir)
    )

def test_query_processor_basic(query_processor):
    """Test basic query processing."""
    # Test action query
    query = "Show me the action scene where Iron Man fights Thanos"
    result = query_processor.process_query(query)
    
    assert result['intent'] == 'action_sequence'
    assert 'Iron Man' in result['characters']
    assert 'Thanos' in result['characters']
    assert 'action' in result['scene_types']
    assert 'fight' in result['actions']
    
    # Test emotional query
    query = "Find a sad scene with Peter Parker"
    result = query_processor.process_query(query)
    
    assert 'Peter Parker' in result['characters']
    assert result['emotion'] == 'sad'
    
    # Test temporal query
    query = "Show what happens after the explosion"
    result = query_processor.process_query(query)
    
    assert result['temporal_context']['relation'] == 'after'
    assert 'explosion' in result['temporal_context']['reference']

def test_query_processor_embeddings(query_processor):
    """Test query embedding generation."""
    queries = [
        "Show me a fight scene",
        "Find a romantic moment",
        "Show a car chase"
    ]
    
    for query in queries:
        result = query_processor.process_query(query)
        assert 'embedding' in result
        assert isinstance(result['embedding'], np.ndarray)
        assert result['embedding'].shape[-1] == 768  # BERT embedding size

def test_scene_indexer_metadata(scene_indexer, sample_video_path):
    """Test scene indexer metadata extraction."""
    metadata = scene_indexer.get_video_metadata(sample_video_path)
    
    assert metadata.fps == 30.0
    assert metadata.frame_count == 300  # 10 seconds * 30 fps
    assert metadata.width == 640
    assert metadata.height == 480
    assert metadata.duration == 10.0

def test_scene_indexer_features(scene_indexer, sample_video_path):
    """Test scene feature extraction."""
    features = scene_indexer.extract_features(sample_video_path)
    
    assert 'visual_features' in features
    assert 'audio_features' in features
    assert 'text_features' in features
    
    visual = features['visual_features']
    assert 'motion_density' in visual
    assert 'edge_density' in visual
    assert 'face_count' in visual

def test_search_engine_basic(search_engine, sample_video_path):
    """Test basic search functionality."""
    # Simple character query
    results = search_engine.search(
        "Show scenes with a person",
        movie_paths=[sample_video_path]
    )
    
    assert isinstance(results, List)
    assert all(isinstance(r, SearchResult) for r in results)
    
    # Check result properties
    for result in results:
        assert result.movie_id == sample_video_path
        assert isinstance(result.scenes, List)
        assert isinstance(result.total_score, float)
        assert isinstance(result.relevance_scores, Dict)
        assert 0 <= result.total_score <= 1.0

def test_search_engine_filters(search_engine, sample_video_path):
    """Test search with filters."""
    # Test duration filter
    results = search_engine.search(
        "Show any scene",
        movie_paths=[sample_video_path],
        filters={'min_duration': 2.0}
    )
    
    assert all(
        scene.end_time - scene.start_time >= 2.0
        for result in results
        for scene in result.scenes
    )
    
    # Test character filter
    results = search_engine.search(
        "Show any scene",
        movie_paths=[sample_video_path],
        filters={'min_faces': 1}
    )
    
    assert all(
        scene.visual_features['face_count'] >= 1
        for result in results
        for scene in result.scenes
    )

def test_search_engine_ranking(search_engine, sample_video_path):
    """Test search result ranking."""
    # Search with multiple components
    query = "Show an action scene with multiple people fighting"
    results = search_engine.search(
        query,
        movie_paths=[sample_video_path]
    )
    
    # Check ranking order
    if len(results) > 1:
        scores = [r.total_score for r in results]
        assert scores == sorted(scores, reverse=True)
    
    # Check score components
    for result in results:
        scores = result.relevance_scores
        assert 'semantic' in scores
        assert 'scene_type' in scores
        assert 'character' in scores
        
        # Verify score weights
        assert 0 <= scores['semantic'] <= 0.3
        assert 0 <= scores['scene_type'] <= 0.2
        assert 0 <= scores['character'] <= 0.2

def test_search_engine_clip_generation(
    search_engine,
    sample_video_path,
    test_data_dir
):
    """Test clip generation in search results."""
    results = search_engine.search(
        "Show me an action scene",
        movie_paths=[sample_video_path],
        generate_clips=True
    )
    
    for result in results:
        if result.clip:
            assert result.clip.path.startswith(str(test_data_dir))
            assert result.clip.duration <= 30.0  # Max duration
            assert result.clip.duration >= 1.0   # Min duration
