"""Scene embedding generation and processing."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoFeatureExtractor
import librosa
import cv2
from PIL import Image
from sainma.utils.logger import get_logger

logger = get_logger(__name__)

class SceneEmbedder:
    """Generates rich scene embeddings combining visual, audio, and text features."""
    
    def __init__(
        self,
        use_gpu: bool = True,
        frame_sample_rate: int = 5,  # Sample every N frames
        audio_duration: float = 0.5,  # Duration of audio segments in seconds
        embedding_dim: int = 768,
        batch_size: int = 8
    ):
        """Initialize scene embedder components."""
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.frame_sample_rate = frame_sample_rate
        self.audio_duration = audio_duration
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        
        # Initialize visual embedding model (CLIP)
        self.visual_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.visual_model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        self.visual_model.to(self.device)
        
        # Initialize audio embedding model (wav2vec)
        self.audio_processor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        self.audio_model = AutoModel.from_pretrained("facebook/wav2vec2-base")
        self.audio_model.to(self.device)
        
        # Initialize text embedding model (all-mpnet-base-v2)
        self.text_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        self.text_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        self.text_model.to(self.device)
    
    async def embed_scene(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        scene_info: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Generate a combined embedding for a scene."""
        try:
            # Extract frames and audio
            frames = await self._extract_frames(video_path, start_time, end_time)
            audio = await self._extract_audio(video_path, start_time, end_time)
            
            # Generate individual embeddings
            visual_emb = await self._generate_visual_embedding(frames)
            audio_emb = await self._generate_audio_embedding(audio)
            text_emb = await self._generate_text_embedding(scene_info)
            
            # Combine embeddings (weighted average)
            weights = np.array([0.4, 0.3, 0.3])  # Visual, audio, text weights
            combined_emb = np.average(
                [visual_emb, audio_emb, text_emb],
                weights=weights,
                axis=0
            )
            
            # Normalize the combined embedding
            combined_emb = combined_emb / np.linalg.norm(combined_emb)
            
            return combined_emb.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error generating scene embedding: {str(e)}")
            raise
    
    async def _extract_frames(
        self,
        video_path: str,
        start_time: float,
        end_time: float
    ) -> List[np.ndarray]:
        """Extract frames from the video segment."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            # Extract frames
            frames = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            for frame_idx in range(start_frame, end_frame, self.frame_sample_rate):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            raise
        finally:
            if 'cap' in locals():
                cap.release()
    
    async def _extract_audio(
        self,
        video_path: str,
        start_time: float,
        end_time: float
    ) -> np.ndarray:
        """Extract audio from the video segment."""
        try:
            # Load audio using librosa
            y, sr = librosa.load(
                video_path,
                offset=start_time,
                duration=end_time - start_time,
                sr=16000  # wav2vec2 expects 16kHz
            )
            
            # Split into segments
            segment_length = int(self.audio_duration * sr)
            segments = []
            
            for i in range(0, len(y), segment_length):
                segment = y[i:i + segment_length]
                if len(segment) == segment_length:
                    segments.append(segment)
            
            return np.array(segments)
            
        except Exception as e:
            logger.error(f"Error extracting audio: {str(e)}")
            raise
    
    async def _generate_visual_embedding(
        self,
        frames: List[np.ndarray]
    ) -> np.ndarray:
        """Generate visual embedding from frames."""
        try:
            frame_embeddings = []
            
            # Process frames in batches
            for i in range(0, len(frames), self.batch_size):
                batch_frames = frames[i:i + self.batch_size]
                
                # Convert frames to PIL images
                pil_images = [Image.fromarray(frame) for frame in batch_frames]
                
                # Process images
                inputs = self.visual_processor(
                    images=pil_images,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                # Generate embeddings
                with torch.no_grad():
                    outputs = self.visual_model(**inputs)
                    embeddings = outputs.pooler_output.cpu().numpy()
                
                frame_embeddings.extend(embeddings)
            
            # Average frame embeddings
            avg_embedding = np.mean(frame_embeddings, axis=0)
            return avg_embedding
            
        except Exception as e:
            logger.error(f"Error generating visual embedding: {str(e)}")
            raise
    
    async def _generate_audio_embedding(
        self,
        audio_segments: np.ndarray
    ) -> np.ndarray:
        """Generate audio embedding from audio segments."""
        try:
            segment_embeddings = []
            
            # Process audio segments in batches
            for i in range(0, len(audio_segments), self.batch_size):
                batch_segments = audio_segments[i:i + self.batch_size]
                
                # Process audio
                inputs = self.audio_processor(
                    batch_segments,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                # Generate embeddings
                with torch.no_grad():
                    outputs = self.audio_model(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                
                segment_embeddings.extend(embeddings)
            
            # Average segment embeddings
            avg_embedding = np.mean(segment_embeddings, axis=0)
            return avg_embedding
            
        except Exception as e:
            logger.error(f"Error generating audio embedding: {str(e)}")
            raise
    
    async def _generate_text_embedding(
        self,
        scene_info: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """Generate text embedding from scene information."""
        try:
            if not scene_info:
                # Return zero embedding if no text info
                return np.zeros(self.embedding_dim)
            
            # Combine relevant text information
            text_elements = []
            
            if 'scene_type' in scene_info:
                text_elements.append(f"This is a {scene_info['scene_type']} scene.")
            
            if 'description' in scene_info:
                text_elements.append(scene_info['description'])
            
            if 'dialogue' in scene_info:
                text_elements.append(scene_info['dialogue'])
            
            if 'characters' in scene_info:
                chars = scene_info['characters']
                if chars:
                    text_elements.append(
                        f"Characters present: {', '.join(chars)}."
                    )
            
            if 'actions' in scene_info:
                actions = scene_info['actions']
                if actions:
                    text_elements.append(
                        f"Actions: {', '.join(actions)}."
                    )
            
            # Combine all text elements
            combined_text = " ".join(text_elements)
            
            # Tokenize and generate embedding
            inputs = self.text_tokenizer(
                combined_text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.text_model(**inputs)
                # Use CLS token embedding
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embedding[0]
            
        except Exception as e:
            logger.error(f"Error generating text embedding: {str(e)}")
            raise
