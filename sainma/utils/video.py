"""
Video processing utilities
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from pathlib import Path

def extract_frame(video_path: str, timestamp: float) -> Optional[np.ndarray]:
    """
    Extract a frame from a video at a specific timestamp
    
    Args:
        video_path: Path to the video file
        timestamp: Timestamp in seconds
        
    Returns:
        Frame as numpy array, or None if extraction fails
    """
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        cap.release()
        
        if ret:
            return frame
        return None
    except Exception as e:
        print(f"Error extracting frame: {e}")
        return None

def get_video_info(video_path: str) -> Tuple[int, int, float]:
    """
    Get basic information about a video file
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Tuple of (width, height, duration_in_seconds)
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    cap.release()
    return width, height, duration

def extract_audio(video_path: str, output_path: Optional[str] = None) -> str:
    """
    Extract audio from a video file
    
    Args:
        video_path: Path to the video file
        output_path: Optional path for the output audio file
        
    Returns:
        Path to the extracted audio file
    """
    if output_path is None:
        video_path = Path(video_path)
        output_path = str(video_path.with_suffix('.wav'))
    
    # TODO: Implement audio extraction using moviepy or ffmpeg
    return output_path

def detect_scene_changes(video_path: str, threshold: float = 30.0) -> List[float]:
    """
    Detect major scene changes in a video
    
    Args:
        video_path: Path to the video file
        threshold: Threshold for scene change detection
        
    Returns:
        List of timestamps (in seconds) where scene changes occur
    """
    scene_changes = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    ret, prev_frame = cap.read()
    if not ret:
        return scene_changes
    
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_count = 0
    
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
            
        curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(curr_frame, prev_frame)
        mean_diff = np.mean(diff)
        
        if mean_diff > threshold:
            timestamp = frame_count / fps
            scene_changes.append(timestamp)
            
        prev_frame = curr_frame
        frame_count += 1
    
    cap.release()
    return scene_changes
