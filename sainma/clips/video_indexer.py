"""Video indexing and real-time search module."""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import os
from dataclasses import dataclass, asdict
import threading
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from datetime import datetime

@dataclass
class SceneIndex:
    """Metadata for a scene in the index."""
    start_time: float
    end_time: float
    avg_brightness: float
    motion_score: float
    audio_level: float
    keyframe_path: str  # Path to stored keyframe
    features: Dict[str, float]

class VideoIndexer:
    """Indexes videos for fast searching and clip generation."""
    
    def __init__(self, index_dir: str):
        """Initialize the indexer.
        
        Args:
            index_dir: Directory to store index files and keyframes
        """
        self.index_dir = index_dir
        self.db_path = os.path.join(index_dir, "video_index.db")
        os.makedirs(index_dir, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def _init_database(self):
        """Initialize SQLite database for video index."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS videos (
                    video_id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    duration REAL,
                    indexed_at TIMESTAMP,
                    total_scenes INTEGER
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scenes (
                    scene_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT,
                    start_time REAL,
                    end_time REAL,
                    avg_brightness REAL,
                    motion_score REAL,
                    audio_level REAL,
                    keyframe_path TEXT,
                    features TEXT,
                    FOREIGN KEY(video_id) REFERENCES videos(video_id)
                )
            """)
            
            # Index for fast time-based queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_scenes_time ON scenes(video_id, start_time, end_time)")
    
    def index_video(self, video_path: str, batch_size: int = 30) -> str:
        """Index a video file for fast searching.
        
        Args:
            video_path: Path to video file
            batch_size: Number of frames to process in parallel
            
        Returns:
            video_id: Unique identifier for the indexed video
        """
        # Generate unique video ID
        video_id = os.path.basename(video_path).replace(".", "_") + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create video capture
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        # Store video info
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO videos (video_id, path, duration, indexed_at, total_scenes) VALUES (?, ?, ?, ?, 0)",
                (video_id, video_path, duration, datetime.now())
            )
        
        try:
            # Process video in batches
            scenes = []
            current_scene_start = 0
            prev_frame = None
            frame_buffer = []
            
            for frame_num in range(0, total_frames, batch_size):
                # Read batch of frames
                batch_frames = []
                for _ in range(batch_size):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    batch_frames.append(frame)
                
                if not batch_frames:
                    break
                
                # Process batch in parallel
                futures = []
                for frame in batch_frames:
                    future = self.executor.submit(self._process_frame, frame)
                    futures.append(future)
                
                # Get results
                frame_features = [f.result() for f in futures]
                frame_buffer.extend(frame_features)
                
                # Detect scene changes using simple threshold
                if len(frame_buffer) > 10:  # Need some history for reliable detection
                    avg_brightness = np.mean([f['brightness'] for f in frame_buffer[-10:]])
                    motion = np.mean([f['motion'] for f in frame_buffer[-10:]])
                    
                    # Scene change if significant change in brightness or motion
                    if len(scenes) == 0 or (
                        abs(avg_brightness - np.mean([f['brightness'] for f in frame_buffer[-20:-10]])) > 0.3 or
                        motion > 0.5
                    ):
                        if frame_num / fps - current_scene_start > 0.5:  # Minimum scene length
                            scene = SceneIndex(
                                start_time=current_scene_start,
                                end_time=frame_num / fps,
                                avg_brightness=avg_brightness,
                                motion_score=motion,
                                audio_level=np.mean([f['audio_level'] for f in frame_buffer[-10:]]),
                                keyframe_path=self._save_keyframe(video_id, len(scenes), batch_frames[0]),
                                features={}
                            )
                            scenes.append(scene)
                            current_scene_start = frame_num / fps
                
                # Keep buffer from growing too large
                if len(frame_buffer) > 30:
                    frame_buffer = frame_buffer[-30:]
            
            # Add final scene
            if current_scene_start < duration:
                scene = SceneIndex(
                    start_time=current_scene_start,
                    end_time=duration,
                    avg_brightness=np.mean([f['brightness'] for f in frame_buffer[-10:]]),
                    motion_score=np.mean([f['motion'] for f in frame_buffer[-10:]]),
                    audio_level=np.mean([f['audio_level'] for f in frame_buffer[-10:]]),
                    keyframe_path=self._save_keyframe(video_id, len(scenes), batch_frames[-1]),
                    features={}
                )
                scenes.append(scene)
            
            # Store scenes in database
            with sqlite3.connect(self.db_path) as conn:
                for scene in scenes:
                    conn.execute("""
                        INSERT INTO scenes 
                        (video_id, start_time, end_time, avg_brightness, motion_score, 
                         audio_level, keyframe_path, features)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        video_id, scene.start_time, scene.end_time, scene.avg_brightness,
                        scene.motion_score, scene.audio_level, scene.keyframe_path,
                        json.dumps(scene.features)
                    ))
                
                conn.execute(
                    "UPDATE videos SET total_scenes = ? WHERE video_id = ?",
                    (len(scenes), video_id)
                )
            
            return video_id
            
        finally:
            cap.release()
    
    def _process_frame(self, frame: np.ndarray) -> Dict[str, float]:
        """Extract features from a single frame."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate brightness
        brightness = np.mean(gray) / 255.0
        
        # Calculate motion (simplified)
        motion = np.std(gray) / 255.0
        
        # Simulate audio level (would need actual audio processing)
        audio_level = 0.0
        
        return {
            'brightness': brightness,
            'motion': motion,
            'audio_level': audio_level
        }
    
    def _save_keyframe(self, video_id: str, scene_num: int, frame: np.ndarray) -> str:
        """Save a keyframe image to disk."""
        keyframe_dir = os.path.join(self.index_dir, video_id, 'keyframes')
        os.makedirs(keyframe_dir, exist_ok=True)
        
        path = os.path.join(keyframe_dir, f"scene_{scene_num}.jpg")
        cv2.imwrite(path, frame)
        return path
    
    def search_scenes(
        self,
        video_id: str,
        criteria: Dict[str, Tuple[float, float]],
        limit: int = 10
    ) -> List[SceneIndex]:
        """Search for scenes matching given criteria.
        
        Args:
            video_id: ID of the video to search
            criteria: Dict of feature names and their (min, max) ranges
            limit: Maximum number of results to return
            
        Returns:
            List of matching scenes
        """
        query = "SELECT * FROM scenes WHERE video_id = ?"
        params = [video_id]
        
        # Add criteria to query
        for feature, (min_val, max_val) in criteria.items():
            if feature in ['avg_brightness', 'motion_score', 'audio_level']:
                query += f" AND {feature} BETWEEN ? AND ?"
                params.extend([min_val, max_val])
        
        query += f" LIMIT {limit}"
        
        # Execute search
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert rows to SceneIndex objects
            scenes = []
            for row in rows:
                scenes.append(SceneIndex(
                    start_time=row[2],
                    end_time=row[3],
                    avg_brightness=row[4],
                    motion_score=row[5],
                    audio_level=row[6],
                    keyframe_path=row[7],
                    features=json.loads(row[8])
                ))
            
            return scenes
    
    def get_all_video_ids(self) -> List[str]:
        """Get IDs of all indexed videos.
        
        Returns:
            List of video IDs
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT video_id FROM scenes")
            return [row[0] for row in cursor.fetchall()]
            
    def get_clip(self, video_id: str, start_time: float, end_time: float) -> Optional[str]:
        """Generate a clip from the video.
        
        Args:
            video_id: ID of the video
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Path to the generated clip, or None if failed
        """
        # Get video path from database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT path FROM videos WHERE video_id = ?",
                (video_id,)
            )
            result = cursor.fetchone()
            if not result:
                return None
            video_path = result[0]
        
        # Generate clip using VideoOps
        clip_name = f"{video_id}_{start_time:.2f}_{end_time:.2f}.mp4"
        clip_path = os.path.join(self.index_dir, "clips", clip_name)
        
        video_ops = VideoOps()
        success = video_ops.extract_clip(
            video_path=video_path,
            start_time=start_time,
            end_time=end_time,
            output_path=clip_path
        )
        
        return clip_path if success else None
