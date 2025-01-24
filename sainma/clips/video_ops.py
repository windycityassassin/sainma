"""Video operations using FFmpeg."""

import subprocess
import os
import json
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import tempfile
from pathlib import Path

@dataclass
class ClipRequest:
    """Represents a request to generate a video clip."""
    input_path: str
    start_time: float
    end_time: float
    output_path: Optional[str] = None

class VideoOps:
    """Handles video operations using FFmpeg."""
    
    def __init__(self, temp_dir: Optional[str] = None):
        """Initialize video operations.
        
        Args:
            temp_dir: Directory for temporary files. If None, uses system temp dir.
        """
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self._verify_ffmpeg()
    
    def _verify_ffmpeg(self):
        """Verify FFmpeg is installed and get capabilities."""
        try:
            # Check FFmpeg version and capabilities
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise RuntimeError("FFmpeg not found. Please install FFmpeg.")
            
            # Check hardware acceleration support
            result = subprocess.run(
                ['ffmpeg', '-hwaccels'],
                capture_output=True,
                text=True
            )
            self.hw_accels = [
                line.strip() for line in result.stdout.split('\n')
                if line.strip() and not line.startswith('Hardware acceleration methods:')
            ]
            
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found. Please install FFmpeg.")
    
    def generate_clip(
        self,
        clips: List[ClipRequest],
        output_path: Optional[str] = None,
        fast_mode: bool = True
    ) -> str:
        """Generate a video clip from multiple segments.
        
        Args:
            clips: List of clip requests to combine
            output_path: Path for output file. If None, generates temporary file.
            fast_mode: If True, uses faster encoding settings
            
        Returns:
            Path to generated clip
        """
        if not clips:
            raise ValueError("No clips provided")
        
        # Generate output path if not provided
        if output_path is None:
            output_path = os.path.join(
                self.temp_dir,
                f"clip_{os.urandom(4).hex()}.mp4"
            )
        
        # Create filter complex file
        filter_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.txt',
            dir=self.temp_dir,
            delete=False
        )
        
        try:
            # Generate filter complex for precise cutting
            filter_parts = []
            input_args = []
            
            # Add input for each unique video file
            input_files = {clip.input_path for clip in clips}
            file_to_idx = {path: idx for idx, path in enumerate(input_files)}
            
            for path in input_files:
                input_args.extend(['-i', path])
            
            # Generate filter complex
            for i, clip in enumerate(clips):
                input_idx = file_to_idx[clip.input_path]
                # Video part
                filter_parts.append(
                    f"[{input_idx}:v]trim=start={clip.start_time}:end={clip.end_time},"
                    f"setpts=PTS-STARTPTS[v{i}];"
                )
                # Audio part
                filter_parts.append(
                    f"[{input_idx}:a]atrim=start={clip.start_time}:end={clip.end_time},"
                    f"asetpts=PTS-STARTPTS[a{i}];"
                )
            
            # Concatenate all parts
            v_parts = ''.join(f'[v{i}]' for i in range(len(clips)))
            a_parts = ''.join(f'[a{i}]' for i in range(len(clips)))
            filter_parts.append(f"{v_parts}concat=n={len(clips)}:v=1:a=0[vout];")
            filter_parts.append(f"{a_parts}concat=n={len(clips)}:v=0:a=1[aout]")
            
            # Write filter complex to file
            filter_complex = ''.join(filter_parts)
            filter_file.write(filter_complex)
            filter_file.close()
            
            # Build FFmpeg command
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output if exists
            ]
            
            # Add all inputs
            cmd.extend(input_args)
            
            # Add filter complex
            cmd.extend([
                '-filter_complex_script', filter_file.name,
                '-map', '[vout]',
                '-map', '[aout]'
            ])
            
            # Add encoding settings
            if fast_mode:
                # Fast encoding settings
                cmd.extend([
                    '-c:v', 'libx264',
                    '-preset', 'ultrafast',
                    '-crf', '23',
                    '-c:a', 'aac',
                    '-movflags', '+faststart'  # Enable streaming
                ])
            else:
                # High quality settings
                cmd.extend([
                    '-c:v', 'libx264',
                    '-preset', 'medium',
                    '-crf', '18',
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-movflags', '+faststart'
                ])
            
            # Add output path
            cmd.append(output_path)
            
            # Run FFmpeg
            subprocess.run(cmd, check=True)
            
            return output_path
            
        finally:
            # Clean up filter file
            try:
                os.unlink(filter_file.name)
            except OSError:
                pass
    
    def extract_frame(
        self,
        video_path: str,
        timestamp: float,
        output_path: Optional[str] = None
    ) -> str:
        """Extract a single frame from video.
        
        Args:
            video_path: Path to video file
            timestamp: Time of frame to extract (seconds)
            output_path: Path for output file. If None, generates temporary file.
            
        Returns:
            Path to extracted frame image
        """
        if output_path is None:
            output_path = os.path.join(
                self.temp_dir,
                f"frame_{os.urandom(4).hex()}.jpg"
            )
        
        cmd = [
            'ffmpeg',
            '-y',
            '-ss', str(timestamp),
            '-i', video_path,
            '-vframes', '1',
            '-q:v', '2',  # High quality JPEG
            output_path
        ]
        
        subprocess.run(cmd, check=True)
        return output_path
    
    def get_video_info(self, video_path: str) -> Dict:
        """Get video metadata using FFprobe.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary of video metadata
        """
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return json.loads(result.stdout)
