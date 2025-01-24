from typing import List, Tuple, Optional, Iterator
import cv2
import numpy as np
from pathlib import Path
import torch
from dataclasses import dataclass
from sainma.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class Frame:
    """Represents a single video frame with metadata."""
    timestamp: float  # Time in seconds
    frame_number: int
    image: np.ndarray
    features: Optional[dict] = None

@dataclass
class VideoMetadata:
    """Metadata about a video file."""
    fps: float
    frame_count: int
    width: int
    height: int
    duration: float

class FrameExtractor:
    """Extracts and processes frames from video files."""
    
    def __init__(self, use_gpu: bool = True):
        """Initialize the frame extractor."""
        self.use_gpu = use_gpu and torch.cuda.is_available()
        if self.use_gpu:
            logger.info("Using GPU for frame extraction")
        else:
            logger.info("Using CPU for frame extraction")
    
    def get_video_metadata(self, video_path: str) -> VideoMetadata:
        """Get metadata about a video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            return VideoMetadata(
                fps=fps,
                frame_count=frame_count,
                width=width,
                height=height,
                duration=duration
            )
        finally:
            cap.release()
    
    def extract_frames(
        self,
        video_path: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        step_size: int = 1  # Extract every nth frame
    ) -> Iterator[Frame]:
        """Extract frames from a video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame ranges
            start_frame = 0
            if start_time is not None:
                start_frame = int(start_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            end_frame = total_frames
            if end_time is not None:
                end_frame = min(int(end_time * fps), total_frames)
            
            frame_number = start_frame
            while frame_number < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_number % step_size == 0:
                    timestamp = frame_number / fps
                    processed_frame = self._process_frame(frame)
                    
                    yield Frame(
                        timestamp=timestamp,
                        frame_number=frame_number,
                        image=processed_frame,
                        features=self._extract_frame_features(processed_frame)
                    )
                
                frame_number += 1
        
        finally:
            cap.release()
    
    def extract_keyframes(
        self,
        video_path: str,
        threshold: float = 0.5,
        min_scene_length: int = 15  # frames
    ) -> List[Frame]:
        """Extract key frames that represent significant visual changes."""
        keyframes = []
        prev_frame = None
        frame_buffer = []
        
        for frame in self.extract_frames(video_path):
            if prev_frame is None:
                keyframes.append(frame)
                prev_frame = frame
                continue
            
            # Calculate frame difference
            diff_score = self._calculate_frame_difference(
                prev_frame.image, frame.image
            )
            
            frame_buffer.append((frame, diff_score))
            
            # Process buffer when it reaches minimum scene length
            if len(frame_buffer) >= min_scene_length:
                # Find local maxima in difference scores
                scores = [score for _, score in frame_buffer]
                if max(scores) > threshold:
                    max_idx = scores.index(max(scores))
                    keyframes.append(frame_buffer[max_idx][0])
                
                # Keep last frame in buffer for next comparison
                frame_buffer = [frame_buffer[-1]]
            
            prev_frame = frame
        
        return keyframes
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a frame for better feature extraction."""
        # Convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize
        frame = frame.astype(np.float32) / 255.0
        
        return frame
    
    def _extract_frame_features(self, frame: np.ndarray) -> dict:
        """Extract visual features from a frame."""
        features = {}
        
        # Calculate average brightness
        features['brightness'] = np.mean(frame)
        
        # Calculate color histogram
        hist_r = cv2.calcHist([frame], [0], None, [256], [0, 1])
        hist_g = cv2.calcHist([frame], [1], None, [256], [0, 1])
        hist_b = cv2.calcHist([frame], [2], None, [256], [0, 1])
        features['color_hist'] = {
            'r': hist_r.flatten().tolist(),
            'g': hist_g.flatten().tolist(),
            'b': hist_b.flatten().tolist()
        }
        
        # Calculate edge intensity using Sobel
        gray = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        features['edge_intensity'] = np.mean(np.sqrt(sobelx**2 + sobely**2))
        
        return features
    
    def _calculate_frame_difference(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> float:
        """Calculate the difference between two frames."""
        # Convert to grayscale
        gray1 = cv2.cvtColor((frame1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor((frame2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Calculate mean difference
        return np.mean(diff) / 255.0  # Normalize to [0, 1]
