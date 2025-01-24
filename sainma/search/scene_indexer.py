"""Scene indexer for Sainma."""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from pathlib import Path
import json
import time
from dataclasses import dataclass
import cv2
from sentence_transformers import SentenceTransformer
from sainma.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class SceneMetadata:
    """Metadata for a movie scene."""
    movie_id: str
    scene_id: str
    start_time: float
    end_time: float
    characters: List[str]
    dialogue: List[str]
    actions: List[str]
    emotions: List[str]
    embedding: Optional[np.ndarray] = None

class SceneIndexer:
    """Indexes movie scenes for efficient search."""
    
    def __init__(self, index_dir: str = ".index/sainma"):
        """Initialize the scene indexer."""
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize sentence transformer for semantic indexing
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load existing index if available
        self.scene_index: Dict[str, SceneMetadata] = {}
        self._load_index()
    
    def index_scene(
        self,
        movie_id: str,
        scene_id: str,
        video_path: str,
        start_time: float,
        end_time: float,
        metadata: Dict[str, Any]
    ):
        """Index a new scene with its metadata."""
        # Extract scene features
        scene_features = self._extract_scene_features(
            video_path, start_time, end_time
        )
        
        # Create scene metadata
        scene_metadata = SceneMetadata(
            movie_id=movie_id,
            scene_id=scene_id,
            start_time=start_time,
            end_time=end_time,
            characters=metadata.get('characters', []),
            dialogue=metadata.get('dialogue', []),
            actions=metadata.get('actions', []),
            emotions=metadata.get('emotions', [])
        )
        
        # Generate scene embedding
        scene_text = self._create_scene_text(scene_metadata)
        scene_metadata.embedding = self.encoder.encode(scene_text)
        
        # Add to index
        self.scene_index[scene_id] = scene_metadata
        
        # Save index
        self._save_index()
        
        logger.info(f"Indexed scene {scene_id} from movie {movie_id}")
    
    def extract_features(self, video_path: str) -> Dict[str, Any]:
        """Extract features from a video file."""
        features = {}
        
        # Get basic video metadata
        cap = cv2.VideoCapture(video_path)
        features['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        features['fps'] = cap.get(cv2.CAP_PROP_FPS)
        features['duration'] = features['frame_count'] / features['fps']
        features['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        features['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Extract visual features from sample frames
        sample_frames = []
        frame_indices = np.linspace(0, features['frame_count']-1, num=5, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert to grayscale for feature extraction
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Extract SIFT features
                sift = cv2.SIFT_create()
                keypoints, descriptors = sift.detectAndCompute(gray, None)
                
                if descriptors is not None:
                    sample_frames.append(descriptors)
        
        cap.release()
        
        if sample_frames:
            # Aggregate features from all frames
            features['visual_features'] = np.mean(np.vstack(sample_frames), axis=0)
        else:
            features['visual_features'] = np.zeros(128)  # Default SIFT descriptor size
        
        return features
    
    def _extract_scene_features(
        self,
        video_path: str,
        start_time: float,
        end_time: float
    ) -> Dict[str, Any]:
        """Extract visual features from a scene."""
        features = {}
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            # Set position to start time
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
            
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or cap.get(cv2.CAP_PROP_POS_MSEC) > end_time * 1000:
                    break
                
                # Convert to grayscale for feature extraction
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Extract SIFT features
                sift = cv2.SIFT_create()
                keypoints, descriptors = sift.detectAndCompute(gray, None)
                
                if descriptors is not None:
                    frames.append(descriptors)
            
            cap.release()
            
            if frames:
                # Aggregate features from all frames
                features['visual_features'] = np.mean(np.vstack(frames), axis=0)
            else:
                features['visual_features'] = np.zeros(128)  # Default SIFT descriptor size
                
        except Exception as e:
            logger.error(f"Error extracting scene features: {e}")
            features['visual_features'] = np.zeros(128)  # Default SIFT descriptor size
        
        return features
    
    def get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Get metadata for a video file."""
        metadata = {
            'path': video_path,
            'filename': Path(video_path).name,
            'size_bytes': Path(video_path).stat().st_size,
            'created_at': Path(video_path).stat().st_ctime,
            'modified_at': Path(video_path).stat().st_mtime
        }
        
        # Add video features
        metadata.update(self.extract_features(video_path))
        
        return metadata
    
    def _create_scene_text(self, metadata: SceneMetadata) -> str:
        """Create a textual representation of the scene for embedding."""
        text_parts = []
        
        # Add characters
        if metadata.characters:
            text_parts.append(f"Characters: {', '.join(metadata.characters)}")
        
        # Add dialogue
        if metadata.dialogue:
            text_parts.append(f"Dialogue: {' '.join(metadata.dialogue)}")
        
        # Add actions
        if metadata.actions:
            text_parts.append(f"Actions: {', '.join(metadata.actions)}")
        
        # Add emotions
        if metadata.emotions:
            text_parts.append(f"Emotions: {', '.join(metadata.emotions)}")
        
        return " ".join(text_parts)
    
    def search_scenes(
        self,
        query_embedding: np.ndarray,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Search for scenes using a query embedding."""
        results = []
        
        for scene_id, metadata in self.scene_index.items():
            if metadata.embedding is None:
                continue
                
            # Apply filters if any
            if filters and not self._apply_filters(metadata, filters):
                continue
            
            # Calculate similarity
            similarity = np.dot(query_embedding, metadata.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(metadata.embedding)
            )
            
            results.append((scene_id, similarity))
        
        # Sort by similarity and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _apply_filters(
        self,
        metadata: SceneMetadata,
        filters: Dict[str, Any]
    ) -> bool:
        """Apply filters to scene metadata."""
        for key, value in filters.items():
            if key == 'min_duration':
                if metadata.end_time - metadata.start_time < value:
                    return False
            elif key == 'max_duration':
                if metadata.end_time - metadata.start_time > value:
                    return False
            elif key == 'characters' and value:
                if not any(char in metadata.characters for char in value):
                    return False
            elif key == 'emotions' and value:
                if not any(emotion in metadata.emotions for emotion in value):
                    return False
            elif key == 'actions' and value:
                if not any(action in metadata.actions for action in value):
                    return False
        return True
    
    def _save_index(self):
        """Save the scene index to disk."""
        index_path = self.index_dir / 'scene_index.json'
        
        # Convert embeddings to lists for JSON serialization
        serializable_index = {}
        for scene_id, metadata in self.scene_index.items():
            metadata_dict = metadata.__dict__.copy()
            if metadata.embedding is not None:
                metadata_dict['embedding'] = metadata.embedding.tolist()
            serializable_index[scene_id] = metadata_dict
        
        with open(index_path, 'w') as f:
            json.dump(serializable_index, f)
    
    def _load_index(self):
        """Load the scene index from disk."""
        index_path = self.index_dir / 'scene_index.json'
        if not index_path.exists():
            return
        
        with open(index_path) as f:
            serialized_index = json.load(f)
        
        # Convert back to SceneMetadata objects
        for scene_id, metadata_dict in serialized_index.items():
            if metadata_dict['embedding']:
                metadata_dict['embedding'] = np.array(metadata_dict['embedding'])
            self.scene_index[scene_id] = SceneMetadata(**metadata_dict)
    
    def get_scene_metadata(self, scene_id: str) -> Optional[SceneMetadata]:
        """Get metadata for a specific scene."""
        return self.scene_index.get(scene_id)
