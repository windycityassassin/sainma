from typing import List, Optional, Dict, Any
import cv2
import numpy as np
from pathlib import Path
import json
from dataclasses import dataclass
from sainma.clips.frame_extractor import FrameExtractor
from sainma.clips.scene_detector import SceneDetector, Scene
from sainma.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ClipConfig:
    """Configuration for clip generation."""
    max_duration: float = 30.0  # seconds
    min_duration: float = 3.0   # seconds
    target_fps: float = 30.0
    target_resolution: tuple = (1920, 1080)
    transition_duration: float = 0.5  # seconds
    quality_preset: str = 'high'  # low, medium, high
    enable_transitions: bool = True
    enable_stabilization: bool = True
    use_gpu: bool = True

@dataclass
class Clip:
    """Represents a generated video clip."""
    path: str
    start_time: float
    end_time: float
    duration: float
    scenes: List[Scene]
    metadata: Dict[str, Any]

class ClipGenerator:
    """Generates high-quality video clips."""
    
    def __init__(
        self,
        output_dir: str = "clips",
        config: Optional[ClipConfig] = None,
        use_gpu: bool = True
    ):
        """Initialize the clip generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or ClipConfig(use_gpu=use_gpu)
        self.scene_detector = SceneDetector(use_gpu=use_gpu)
        self.frame_extractor = FrameExtractor(use_gpu=use_gpu)
    
    def generate_clip(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        output_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Clip:
        """Generate a clip from the specified video segment."""
        logger.info(f"Generating clip from {video_path}")
        
        # Ensure duration limits
        duration = end_time - start_time
        if duration > self.config.max_duration:
            end_time = start_time + self.config.max_duration
            logger.warning(
                f"Clip duration exceeds maximum, truncating to {self.config.max_duration}s"
            )
        elif duration < self.config.min_duration:
            end_time = start_time + self.config.min_duration
            logger.warning(
                f"Clip duration below minimum, extending to {self.config.min_duration}s"
            )
        
        # Detect scenes
        scenes = self.scene_detector.detect_scenes(
            video_path,
            start_time=start_time,
            end_time=end_time
        )
        
        # Generate output path
        output_path = self.output_dir / f"{output_name}.mp4"
        
        # Create clip
        self._create_clip(
            video_path=video_path,
            scenes=scenes,
            output_path=str(output_path)
        )
        
        # Save metadata
        clip_metadata = {
            'source_video': video_path,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            'scenes': [
                {
                    'start_time': scene.start_time,
                    'end_time': scene.end_time,
                    'confidence': scene.confidence,
                    'features': scene.features
                }
                for scene in scenes
            ],
            'config': {
                'fps': self.config.target_fps,
                'resolution': self.config.target_resolution,
                'quality': self.config.quality_preset
            }
        }
        
        if metadata:
            clip_metadata.update(metadata)
        
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(clip_metadata, f, indent=2)
        
        return Clip(
            path=str(output_path),
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            scenes=scenes,
            metadata=clip_metadata
        )
    
    def generate_answer_clip(
        self,
        video_path: str,
        scenes: List[Scene],
        query: str,
        output_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Clip:
        """Generate a clip that answers a specific query."""
        try:
            # Sort scenes by relevance score if available
            if all('relevance_score' in scene.features for scene in scenes):
                scenes.sort(key=lambda x: x.features['relevance_score'], reverse=True)
            
            # Calculate total duration of selected scenes
            total_duration = sum(
                scene.end_time - scene.start_time
                for scene in scenes
            )
            
            # If total duration exceeds max, select highest scoring scenes
            if total_duration > self.config.max_duration:
                logger.info("Total scene duration exceeds maximum, selecting best scenes")
                selected_scenes = []
                current_duration = 0
                
                for scene in scenes:
                    scene_duration = scene.end_time - scene.start_time
                    if current_duration + scene_duration <= self.config.max_duration:
                        selected_scenes.append(scene)
                        current_duration += scene_duration
                    else:
                        break
                
                scenes = selected_scenes
            
            # Generate output name if not provided
            if output_name is None:
                import hashlib
                hash_input = f"{video_path}_{query}_{len(scenes)}"
                output_name = hashlib.md5(hash_input.encode()).hexdigest()[:10]
            
            # Add query info to metadata
            clip_metadata = metadata or {}
            clip_metadata.update({
                'query': query,
                'scene_count': len(scenes),
                'selected_scenes': [
                    {
                        'start_time': scene.start_time,
                        'end_time': scene.end_time,
                        'score': scene.features.get('relevance_score', 0.0)
                    }
                    for scene in scenes
                ]
            })
            
            # Generate the clip
            return self.generate_clip(
                video_path=video_path,
                start_time=scenes[0].start_time,
                end_time=scenes[-1].end_time,
                output_name=output_name,
                metadata=clip_metadata
            )
            
        except Exception as e:
            logger.error(f"Error generating answer clip: {str(e)}")
            raise
    
    def _create_clip(
        self,
        video_path: str,
        scenes: List[Scene],
        output_path: str
    ):
        """Create the actual video clip."""
        # Get video properties
        cap = cv2.VideoCapture(video_path)
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Calculate resize factors
        target_width, target_height = self.config.target_resolution
        scale_x = target_width / source_width
        scale_y = target_height / source_height
        scale = min(scale_x, scale_y)
        
        output_width = int(source_width * scale)
        output_height = int(source_height * scale)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            self.config.target_fps,
            (output_width, output_height)
        )
        
        try:
            prev_scene = None
            for scene in scenes:
                # Add transition from previous scene if enabled
                if (self.config.enable_transitions and prev_scene and
                    scene.start_time - prev_scene.end_time < 1.0):
                    self._add_transition(
                        video_path,
                        prev_scene,
                        scene,
                        out,
                        (output_width, output_height)
                    )
                
                # Add scene frames
                self._add_scene(
                    video_path,
                    scene,
                    out,
                    (output_width, output_height)
                )
                
                prev_scene = scene
        
        finally:
            out.release()
        
        # Optimize output if needed
        if self.config.quality_preset == 'high':
            self._optimize_clip(output_path)
    
    def _add_scene(
        self,
        video_path: str,
        scene: Scene,
        out: cv2.VideoWriter,
        output_size: tuple
    ):
        """Add a scene to the clip."""
        frames = self.frame_extractor.extract_frames(
            video_path,
            start_time=scene.start_time,
            end_time=scene.end_time
        )
        
        prev_frame = None
        for frame in frames:
            # Resize frame
            resized = cv2.resize(
                (frame.image * 255).astype(np.uint8),
                output_size
            )
            
            # Apply stabilization if enabled
            if self.config.enable_stabilization and prev_frame is not None:
                resized = self._stabilize_frame(prev_frame, resized)
            
            # Convert back to BGR for OpenCV
            bgr_frame = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
            
            # Write frame
            out.write(bgr_frame)
            prev_frame = resized
    
    def _add_transition(
        self,
        video_path: str,
        scene1: Scene,
        scene2: Scene,
        out: cv2.VideoWriter,
        output_size: tuple
    ):
        """Add a transition between two scenes."""
        transition_frames = int(
            self.config.transition_duration * self.config.target_fps
        )
        
        # Get last frame of first scene
        last_frame = next(self.frame_extractor.extract_frames(
            video_path,
            start_time=scene1.end_time - 0.1,
            end_time=scene1.end_time
        ))
        last_frame = cv2.resize(
            (last_frame.image * 255).astype(np.uint8),
            output_size
        )
        
        # Get first frame of second scene
        first_frame = next(self.frame_extractor.extract_frames(
            video_path,
            start_time=scene2.start_time,
            end_time=scene2.start_time + 0.1
        ))
        first_frame = cv2.resize(
            (first_frame.image * 255).astype(np.uint8),
            output_size
        )
        
        # Create transition frames
        for i in range(transition_frames):
            alpha = i / transition_frames
            blended = cv2.addWeighted(
                last_frame,
                1 - alpha,
                first_frame,
                alpha,
                0
            )
            out.write(cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    
    def _stabilize_frame(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray
    ) -> np.ndarray:
        """Stabilize current frame relative to previous frame."""
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Calculate transformation matrix
        h, w = curr_frame.shape[:2]
        flow_x = cv2.blur(flow[..., 0], (30, 30))
        flow_y = cv2.blur(flow[..., 1], (30, 30))
        
        dx = float(flow_x.mean())
        dy = float(flow_y.mean())
        
        transform = np.array([
            [1, 0, -dx],
            [0, 1, -dy]
        ])
        
        # Apply transformation
        stabilized = cv2.warpAffine(
            curr_frame,
            transform,
            (w, h),
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return stabilized
    
    def _optimize_clip(self, clip_path: str):
        """Optimize the generated clip for quality."""
        try:
            # Read the video
            cap = cv2.VideoCapture(clip_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create temporary output file
            temp_path = clip_path.replace('.mp4', '_optimized.mp4')
            
            # Configure optimization based on quality preset
            if self.config.quality_preset == 'high':
                # High quality settings
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(
                    temp_path,
                    fourcc,
                    fps,
                    (width, height),
                    isColor=True
                )
                
                # Process frames
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Apply denoising
                    denoised = cv2.fastNlMeansDenoisingColored(
                        frame,
                        None,
                        10,
                        10,
                        7,
                        21
                    )
                    
                    # Enhance contrast
                    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                    cl = clahe.apply(l)
                    enhanced = cv2.merge((cl,a,b))
                    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
                    
                    # Write frame
                    out.write(enhanced)
                
                out.release()
                
                # Replace original with optimized
                import os
                os.replace(temp_path, clip_path)
            
        except Exception as e:
            logger.error(f"Error optimizing clip: {str(e)}")
            raise
        finally:
            if 'cap' in locals():
                cap.release()
