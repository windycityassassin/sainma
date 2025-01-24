"""Scene detection and analysis module."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
import librosa
import torch
from torchvision import models, transforms
from PIL import Image
import io
import moviepy.editor as mp

@dataclass
class Scene:
    """Represents a detected scene in a video."""
    start_time: float
    end_time: float
    scene_type: Optional[str] = None
    confidence: float = 0.0
    features: Dict[str, Any] = field(default_factory=dict)
    visual_features: Dict[str, Any] = field(default_factory=dict)

class SceneDetector:
    """Detects and analyzes scenes in videos."""
    
    def __init__(self, use_gpu: bool = False):
        """Initialize scene detector."""
        self.use_gpu = use_gpu
        # Load pre-trained ResNet model for visual classification
        self.model = models.resnet18(pretrained=True)
        if self.use_gpu and torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        
        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def detect_scenes(
        self,
        video_path: str,
        min_scene_length: float = 0.5,  # Shorter minimum for real videos
        threshold: float = 0.15,  # Lower threshold for real transitions
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Scene]:
        """Detect scenes in a video."""
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Convert times to frame numbers
            start_frame = 0 if start_time is None else int(start_time * fps)
            end_frame = total_frames if end_time is None else int(end_time * fps)
            
            scenes = []
            current_scene_start = start_frame
            prev_frame = None
            diff_buffer = []
            buffer_size = 5  # Buffer for smoothing
            
            # Process frames
            for frame_num in range(start_frame, end_frame):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # Calculate frame difference
                    diff = cv2.absdiff(gray, prev_frame)
                    mean_diff = np.mean(diff) / 255.0
                    
                    # Calculate histogram difference
                    hist1 = cv2.calcHist([gray], [0], None, [256], [0, 256])
                    hist2 = cv2.calcHist([prev_frame], [0], None, [256], [0, 256])
                    hist_diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
                    hist_diff = min(1.0, hist_diff / 1000.0)
                    
                    # Calculate edge difference
                    edges1 = cv2.Canny(prev_frame, 100, 200)
                    edges2 = cv2.Canny(gray, 100, 200)
                    edge_diff = np.mean(cv2.absdiff(edges1, edges2)) / 255.0
                    
                    # Combine differences
                    combined_diff = (mean_diff + hist_diff + edge_diff) / 3.0
                    diff_buffer.append(combined_diff)
                    
                    # Keep buffer at fixed size
                    if len(diff_buffer) > buffer_size:
                        diff_buffer.pop(0)
                    
                    # Use smoothed difference for detection
                    if len(diff_buffer) == buffer_size:
                        smoothed_diff = sum(diff_buffer) / len(diff_buffer)
                        
                        # Calculate local maximum
                        is_local_max = smoothed_diff > threshold and (
                            len(diff_buffer) == 1 or
                            smoothed_diff > max(diff_buffer[:-1])
                        )
                        
                        # Detect scene change
                        if is_local_max and frame_num - current_scene_start >= fps * min_scene_length:
                            # Create scene
                            scene = Scene(
                                start_time=current_scene_start / fps,
                                end_time=frame_num / fps,
                                confidence=min(1.0, smoothed_diff / threshold)
                            )
                            
                            # Extract scene features
                            self._extract_scene_features(scene, frame)
                            
                            # Add confidence to features
                            scene.features['confidence'] = scene.confidence
                            scene.features['duration'] = scene.end_time - scene.start_time
                            scene.features['scene_type'] = scene.scene_type
                            scene.features.update(scene.visual_features)
                            
                            scenes.append(scene)
                            current_scene_start = frame_num
                            diff_buffer.clear()  # Reset buffer after scene change
                
                prev_frame = gray
            
            # Add final scene if needed
            if current_scene_start < end_frame - fps * min_scene_length:
                scene = Scene(
                    start_time=current_scene_start / fps,
                    end_time=end_frame / fps,
                    confidence=1.0
                )
                
                # Get last frame for feature extraction
                cap.set(cv2.CAP_PROP_POS_FRAMES, end_frame - 1)
                ret, frame = cap.read()
                if ret:
                    self._extract_scene_features(scene, frame)
                    
                    # Add confidence to features
                    scene.features['confidence'] = scene.confidence
                    scene.features['duration'] = scene.end_time - scene.start_time
                    scene.features['scene_type'] = scene.scene_type
                    scene.features.update(scene.visual_features)
                
                scenes.append(scene)
            
            return scenes
            
        finally:
            if 'cap' in locals():
                cap.release()
    
    def _extract_scene_features(self, scene: Scene, frame: np.ndarray):
        """Extract visual features from a scene."""
        # Convert to grayscale for some calculations
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate edge density
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges) / 255.0
        
        # Calculate motion density (approximation from single frame)
        motion_density = cv2.Laplacian(gray, cv2.CV_64F).var()
        motion_density = min(1.0, motion_density / 1000.0)  # Normalize
        
        # Face detection
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        face_count = len(faces)
        
        # Store features
        scene.visual_features.update({
            'edge_density': edge_density,
            'motion_density': motion_density,
            'face_count': face_count
        })
        
        # Determine scene type based on features
        if motion_density > 0.6:
            scene.scene_type = 'action'
        elif face_count > 2:
            scene.scene_type = 'group'
        elif face_count == 2:
            scene.scene_type = 'dialogue'
        elif edge_density < 0.1:
            scene.scene_type = 'transition'
        else:
            scene.scene_type = 'other'

    def _extract_audio_features(self, frame_audio: np.ndarray) -> Dict[str, float]:
        """Extract audio features including rhythm detection."""
        if frame_audio.size == 0:
            return {'rhythm_strength': 0.0, 'tempo': 0.0}
            
        # Convert to mono if stereo
        if len(frame_audio.shape) > 1:
            frame_audio = np.mean(frame_audio, axis=1)
            
        # Resample to 22050 Hz (librosa's default)
        frame_audio = librosa.resample(frame_audio, orig_sr=44100, target_sr=22050)
            
        # Extract onset strength (useful for drum detection)
        onset_env = librosa.onset.onset_strength(y=frame_audio, sr=22050)
        tempo, _ = librosa.beat.beat_track(y=frame_audio, sr=22050)
        
        # Get rhythm features
        mel_spec = librosa.feature.melspectrogram(y=frame_audio, sr=22050)
        rhythm_strength = np.mean(librosa.feature.rms(S=mel_spec)[0])
        
        return {
            'rhythm_strength': float(rhythm_strength),
            'tempo': float(tempo)
        }
    
    def _classify_frame(self, frame: np.ndarray) -> Dict[str, float]:
        """Classify the frame content using the pre-trained model."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        # Preprocess image
        input_tensor = self.preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        
        if self.use_gpu and torch.cuda.is_available():
            input_batch = input_batch.cuda()
            
        with torch.no_grad():
            output = self.model(input_batch)
            
        # Get probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        return {
            'action_score': float(probabilities[400:500].max()),  # Action-related classes
            'music_score': float(probabilities[500:600].max()),   # Music-related classes
        }
    
    def search_scenes(
        self,
        video_path: str,
        search_type: str,
        threshold: float = 0.15,
        min_scene_length: float = 0.5
    ) -> List[Scene]:
        """Search for specific types of scenes in the video."""
        
        # Load video
        video = mp.VideoFileClip(video_path)
        fps = video.fps
        
        scenes = []
        current_scene_start = 0
        current_scene_features = []
        
        # Process frames
        for t in range(0, int(video.duration * fps), 2):  # Process every 2nd frame for speed
            time = t / fps
            
            # Get frame and audio
            frame = video.get_frame(time)
            audio_segment = video.audio.subclip(time, time + 1/fps).to_soundarray()
            
            # Extract features
            visual_features = self._classify_frame(frame)
            audio_features = self._extract_audio_features(audio_segment)
            
            # Combine features
            features = {**visual_features, **audio_features}
            current_scene_features.append(features)
            
            # Detect scene type
            is_target_scene = False
            if search_type == "drumming":
                # Check for drumming characteristics
                is_target_scene = (
                    features['rhythm_strength'] > 0.6 and  # Strong rhythm
                    features['tempo'] > 100 and           # Fast tempo
                    features['music_score'] > 0.3         # Music-related visuals
                )
            
            # If scene type changes, mark scene boundary
            if len(current_scene_features) > 1:
                prev_features = current_scene_features[-2]
                if is_target_scene != (
                    prev_features['rhythm_strength'] > 0.6 and
                    prev_features['tempo'] > 100 and
                    prev_features['music_score'] > 0.3
                ):
                    if time - current_scene_start >= min_scene_length:
                        # Create scene if it matches search criteria
                        if is_target_scene:
                            scene = Scene(
                                start_time=current_scene_start,
                                end_time=time,
                                confidence=np.mean([f['rhythm_strength'] for f in current_scene_features])
                            )
                            scene.features = {
                                'avg_rhythm_strength': np.mean([f['rhythm_strength'] for f in current_scene_features]),
                                'avg_tempo': np.mean([f['tempo'] for f in current_scene_features]),
                                'avg_music_score': np.mean([f['music_score'] for f in current_scene_features])
                            }
                            scenes.append(scene)
                        
                        current_scene_start = time
                        current_scene_features = []
        
        # Add final scene if needed
        if current_scene_features and time - current_scene_start >= min_scene_length:
            scene = Scene(
                start_time=current_scene_start,
                end_time=video.duration,
                confidence=np.mean([f['rhythm_strength'] for f in current_scene_features])
            )
            scene.features = {
                'avg_rhythm_strength': np.mean([f['rhythm_strength'] for f in current_scene_features]),
                'avg_tempo': np.mean([f['tempo'] for f in current_scene_features]),
                'avg_music_score': np.mean([f['music_score'] for f in current_scene_features])
            }
            scenes.append(scene)
        
        video.close()
        return scenes
