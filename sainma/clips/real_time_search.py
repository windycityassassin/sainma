"""Real-time video search and clip generation."""

from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np
from pathlib import Path
from .video_indexer import VideoIndexer, SceneIndex
from .clip_generator import ClipGenerator, ClipConfig

class RealTimeSearch:
    """Real-time video search and clip generation."""
    
    def __init__(self, index_dir: str):
        """Initialize real-time search.
        
        Args:
            index_dir: Directory containing video index
        """
        self.indexer = VideoIndexer(index_dir)
        self.clip_generator = ClipGenerator(output_dir=str(Path(index_dir) / "clips"))
        
        # Define scene type criteria mappings
        self.scene_criteria = {
            "car chase": {
                "motion_score": (0.7, 1.0),     # High motion
                "audio_level": (0.6, 1.0),      # High audio
                "avg_brightness": (0.3, 0.8),   # Varied lighting
                "scene_change_rate": (0.4, 1.0)  # Frequent scene changes
            },
            "explosion": {
                "motion_score": (0.8, 1.0),
                "audio_level": (0.8, 1.0),
                "avg_brightness": (0.7, 1.0)
            },
            # Add more scene types here
        }
    
    def add_video(self, video_path: str) -> str:
        """Add a video to the searchable index.
        
        Args:
            video_path: Path to video file
            
        Returns:
            video_id: Unique identifier for searching
        """
        return self.indexer.index_video(video_path)
    
    def search(
        self,
        video_id: str,
        search_type: str,
        threshold: float = 0.5,
        limit: int = 10
    ) -> List[SceneIndex]:
        """Search for scenes of a specific type.
        
        Args:
            video_id: ID of the indexed video
            search_type: Type of scene to search for (e.g., "drumming", "action")
            threshold: Minimum confidence threshold
            limit: Maximum number of results
            
        Returns:
            List of matching scenes
        """
        # Define search criteria based on scene type
        criteria = {}
        
        if search_type == "drumming":
            criteria = {
                'motion_score': (0.4, 1.0),    # High motion
                'audio_level': (0.6, 1.0)      # High audio level
            }
        elif search_type == "action":
            criteria = {
                'motion_score': (0.7, 1.0),    # Very high motion
                'brightness': (0.2, 0.8)       # Varied brightness
            }
        elif search_type == "quiet":
            criteria = {
                'motion_score': (0.0, 0.3),    # Low motion
                'audio_level': (0.0, 0.3)      # Low audio
            }
        # Add more scene types as needed
        
        return self.indexer.search_scenes(video_id, criteria, limit)
    
    def search_all_videos(
        self,
        query: str,
        threshold: float = 0.5,
        limit_per_video: int = 5,
        max_total_clips: int = 20
    ) -> List[Tuple[str, List[SceneIndex]]]:
        """Search for scenes across all indexed videos.
        
        Args:
            query: Natural language query (e.g., "car chase")
            threshold: Minimum confidence threshold
            limit_per_video: Maximum scenes per video
            max_total_clips: Maximum total scenes across all videos
            
        Returns:
            List of (video_id, scenes) tuples
        """
        # Get all indexed video IDs
        video_ids = self.indexer.get_all_video_ids()
        
        # Find best matching scene type from criteria
        scene_type = self._match_scene_type(query)
        if not scene_type:
            # If no exact match, use default criteria based on motion and audio
            criteria = {
                "motion_score": (0.5, 1.0),
                "audio_level": (0.5, 1.0)
            }
        else:
            criteria = self.scene_criteria[scene_type]
        
        all_results = []
        for video_id in video_ids:
            scenes = self.indexer.search_scenes(
                video_id=video_id,
                criteria=criteria,
                limit=limit_per_video
            )
            if scenes:
                all_results.append((video_id, scenes))
                
            if sum(len(scenes) for _, scenes in all_results) >= max_total_clips:
                break
                
        return all_results
    
    def generate_clip(
        self,
        video_id: str,
        scenes: List[SceneIndex],
        output_path: str
    ) -> str:
        """Generate a clip from the matched scenes.
        
        Args:
            video_id: ID of the video
            scenes: List of scenes to include
            output_path: Where to save the generated clip
            
        Returns:
            Path to the generated clip
        """
        # This would use FFmpeg to efficiently extract and combine the scenes
        # For now, it's just a placeholder
        pass
    
    def generate_playlist(
        self,
        search_results: List[Tuple[str, List[SceneIndex]]],
        output_path: str
    ) -> str:
        """Generate a playlist video from search results.
        
        Args:
            search_results: List of (video_id, scenes) tuples
            output_path: Where to save the playlist video
            
        Returns:
            Path to the generated playlist video
        """
        # Extract clips for each scene
        clip_paths = []
        for video_id, scenes in search_results:
            for scene in scenes:
                clip_path = self.indexer.get_clip(
                    video_id=video_id,
                    start_time=scene.start_time,
                    end_time=scene.end_time
                )
                if clip_path:
                    clip_paths.append(clip_path)
        
        # Combine clips into a playlist
        if clip_paths:
            return self.clip_generator.combine_clips(
                clip_paths=clip_paths,
                output_path=output_path,
                add_transitions=True
            )
        return ""
    
    def _match_scene_type(self, query: str) -> Optional[str]:
        """Match a natural language query to a predefined scene type."""
        # Simple exact matching for now
        query = query.lower().strip()
        return query if query in self.scene_criteria else None
