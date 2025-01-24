"""Search engine module."""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from sainma.clips.scene_detector import SceneDetector
from sainma.clips.clip_generator import ClipGenerator, ClipConfig
from sainma.search.query_processor import QueryProcessor
from sainma.search.scene_indexer import SceneIndexer
from sainma.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class SearchResult:
    """Represents a search result."""
    scene_id: str
    movie_id: str
    start_time: float
    end_time: float
    score: float
    clip_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class SearchEngine:
    """Search engine for finding movie scenes."""
    
    def __init__(
        self,
        use_gpu: bool = True,
        min_scene_length: float = 1.0,
        scene_threshold: float = 0.5,
        clip_output_dir: str = "clips"
    ):
        """Initialize the search engine."""
        self.use_gpu = use_gpu
        self.min_scene_length = min_scene_length
        self.scene_threshold = scene_threshold
        self.clip_output_dir = Path(clip_output_dir)
        self.clip_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.query_processor = QueryProcessor()
        self.scene_detector = SceneDetector(use_gpu=use_gpu)
        self.scene_indexer = SceneIndexer()
        self.clip_generator = ClipGenerator(
            output_dir=str(clip_output_dir),
            config=ClipConfig(use_gpu=use_gpu)
        )
    
    def search(
        self,
        query: str,
        video_path: str,
        top_k: int = 5,
        generate_clips: bool = True,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for scenes matching the query."""
        # Process query
        query_info = self.query_processor.process_query(query)
        query_embedding = query_info['embedding']
        
        # Index video if not already indexed
        self._ensure_video_indexed(video_path)
        
        # Search for matching scenes
        scene_matches = self.scene_indexer.search_scenes(
            query_embedding,
            filters=filters,
            top_k=top_k
        )
        
        # Convert matches to search results
        results = []
        for scene_id, score in scene_matches:
            metadata = self.scene_indexer.get_scene_metadata(scene_id)
            if metadata:
                result = SearchResult(
                    scene_id=scene_id,
                    movie_id=metadata.movie_id,
                    start_time=metadata.start_time,
                    end_time=metadata.end_time,
                    score=score,
                    metadata={
                        'characters': metadata.characters,
                        'dialogue': metadata.dialogue,
                        'actions': metadata.actions,
                        'emotions': metadata.emotions
                    }
                )
                
                # Generate clip if requested
                if generate_clips:
                    clip_path = self._generate_result_clip(
                        video_path,
                        result.start_time,
                        result.end_time,
                        scene_id
                    )
                    result.clip_path = clip_path
                
                results.append(result)
        
        return results
    
    def _ensure_video_indexed(self, video_path: str):
        """Ensure video scenes are indexed."""
        # Get video metadata
        video_metadata = self.scene_indexer.get_video_metadata(video_path)
        movie_id = video_metadata['filename']
        
        # Detect scenes if not already indexed
        scenes = self.scene_detector.detect_scenes(
            video_path,
            min_scene_length=self.min_scene_length,
            threshold=self.scene_threshold
        )
        
        # Index each scene
        for i, scene in enumerate(scenes):
            scene_id = f"{movie_id}_scene_{i}"
            
            # Check if scene is already indexed
            if self.scene_indexer.get_scene_metadata(scene_id):
                continue
            
            # Index the scene
            self.scene_indexer.index_scene(
                movie_id=movie_id,
                scene_id=scene_id,
                video_path=video_path,
                start_time=scene.start_time,
                end_time=scene.end_time,
                metadata={
                    'characters': scene.features.get('characters', []),
                    'dialogue': scene.features.get('dialogue', []),
                    'actions': scene.features.get('actions', []),
                    'emotions': scene.features.get('emotions', [])
                }
            )
    
    def _generate_result_clip(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        scene_id: str
    ) -> str:
        """Generate a clip for a search result."""
        clip_path = self.clip_output_dir / f"{scene_id}.mp4"
        
        if not clip_path.exists():
            clip = self.clip_generator.generate_clip(
                video_path=video_path,
                start_time=start_time,
                end_time=end_time,
                output_path=str(clip_path)
            )
        
        return str(clip_path)
