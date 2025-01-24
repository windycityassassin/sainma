"""Generate test videos for unit tests."""

import cv2
import numpy as np
from pathlib import Path
import subprocess
from typing import List, Tuple, Optional

class TestVideoGenerator:
    """Generate test videos with specific characteristics for testing."""
    
    def __init__(self, output_dir: str = "test_data"):
        """Initialize the video generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_scene_video(
        self,
        duration: float = 5.0,
        fps: float = 30.0,
        resolution: Tuple[int, int] = (640, 480),
        scene_changes: Optional[List[float]] = None,
        output_name: str = "scene_test.mp4"
    ) -> str:
        """Create a test video with scene changes at specified times."""
        frame_count = int(duration * fps)
        width, height = resolution
        output_path = str(self.output_dir / output_name)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            resolution
        )
        
        try:
            # Generate frames
            for i in range(frame_count):
                time = i / fps
                
                # Create base frame
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Add scene change if needed
                if scene_changes and any(abs(sc - time) < 1/fps for sc in scene_changes):
                    # Sudden change in content
                    color = (255, 255, 255)
                else:
                    # Gradual change
                    intensity = int(255 * (i / frame_count))
                    color = (intensity, intensity, intensity)
                
                # Draw content
                cv2.rectangle(
                    frame,
                    (width//4, height//4),
                    (3*width//4, 3*height//4),
                    color,
                    -1
                )
                
                # Add frame number and timestamp
                cv2.putText(
                    frame,
                    f"Frame {i} - {time:.2f}s",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )
                
                out.write(frame)
        
        finally:
            out.release()
        
        # Convert to H.264 for better compatibility
        temp_path = output_path.replace('.mp4', '_temp.mp4')
        subprocess.run([
            'ffmpeg', '-y',
            '-i', output_path,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            temp_path
        ])
        
        Path(output_path).unlink()
        Path(temp_path).rename(output_path)
        
        return output_path
    
    def create_character_video(
        self,
        duration: float = 5.0,
        fps: float = 30.0,
        resolution: Tuple[int, int] = (640, 480),
        character_positions: Optional[List[Tuple[float, Tuple[int, int]]]] = None,
        output_name: str = "character_test.mp4"
    ) -> str:
        """Create a test video with moving characters."""
        frame_count = int(duration * fps)
        width, height = resolution
        output_path = str(self.output_dir / output_name)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            resolution
        )
        
        try:
            # Generate frames
            for i in range(frame_count):
                time = i / fps
                
                # Create base frame
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Add characters if specified
                if character_positions:
                    for char_time, (x, y) in character_positions:
                        if abs(char_time - time) < 1/fps:
                            # Draw character
                            cv2.circle(
                                frame,
                                (x, y),
                                30,
                                (0, 255, 0),
                                -1
                            )
                            # Draw face-like features
                            cv2.circle(
                                frame,
                                (x-10, y-10),
                                5,
                                (255, 255, 255),
                                -1
                            )
                            cv2.circle(
                                frame,
                                (x+10, y-10),
                                5,
                                (255, 255, 255),
                                -1
                            )
                            cv2.ellipse(
                                frame,
                                (x, y+5),
                                (15, 10),
                                0,
                                0,
                                180,
                                (255, 255, 255),
                                2
                            )
                
                # Add frame number and timestamp
                cv2.putText(
                    frame,
                    f"Frame {i} - {time:.2f}s",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )
                
                out.write(frame)
        
        finally:
            out.release()
        
        # Convert to H.264
        temp_path = output_path.replace('.mp4', '_temp.mp4')
        subprocess.run([
            'ffmpeg', '-y',
            '-i', output_path,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            temp_path
        ])
        
        Path(output_path).unlink()
        Path(temp_path).rename(output_path)
        
        return output_path
