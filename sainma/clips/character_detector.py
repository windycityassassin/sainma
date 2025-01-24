"""Character detection and tracking in video scenes."""

from typing import Dict, List, Optional, Tuple, Set
import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from deepface import DeepFace
import face_recognition
from concurrent.futures import ThreadPoolExecutor
from thefuzz import fuzz
from collections import defaultdict
from sainma.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class CharacterAlias:
    """Character name alias with confidence score."""
    name: str
    confidence: float
    source: str  # Where this alias came from (exact, fuzzy, face_match, etc.)

@dataclass
class Character:
    """Information about a detected character."""
    id: str  # Unique identifier
    name: Optional[str]  # Primary character name if known
    aliases: List[CharacterAlias]  # Alternative names with confidence scores
    face_encodings: List[np.ndarray]  # Multiple face encodings for better matching
    appearances: List[Tuple[float, float]]  # List of (start_time, end_time)
    confidence: float
    attributes: Dict[str, any]  # Additional attributes (age, gender, emotion, etc.)

class CharacterMatcher:
    """Matches character names using various similarity metrics."""
    
    def __init__(
        self,
        min_fuzzy_ratio: int = 80,
        min_partial_ratio: int = 85,
        min_token_sort_ratio: int = 85
    ):
        """Initialize character matcher."""
        self.min_fuzzy_ratio = min_fuzzy_ratio
        self.min_partial_ratio = min_partial_ratio
        self.min_token_sort_ratio = min_token_sort_ratio
        
        # Common name variations
        self.name_patterns = {
            r'dr\.?\s*': 'doctor ',
            r'mr\.?\s*': 'mister ',
            r'mrs\.?\s*': 'missus ',
            r'ms\.?\s*': 'miss ',
            r'prof\.?\s*': 'professor ',
            r'rev\.?\s*': 'reverend ',
            r'sr\.?\s*': 'senior ',
            r'jr\.?\s*': 'junior '
        }
        
        # Cache for normalized names
        self._name_cache: Dict[str, str] = {}
    
    def normalize_name(self, name: str) -> str:
        """Normalize a character name for comparison."""
        if name in self._name_cache:
            return self._name_cache[name]
        
        # Convert to lowercase
        normalized = name.lower().strip()
        
        # Replace common titles and suffixes
        import re
        for pattern, replacement in self.name_patterns.items():
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        # Remove special characters
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        self._name_cache[name] = normalized
        return normalized
    
    def match_names(
        self,
        query_name: str,
        target_names: List[str],
        threshold: float = 0.7
    ) -> List[Tuple[str, float, str]]:
        """Match a query name against a list of target names."""
        matches = []
        normalized_query = self.normalize_name(query_name)
        
        for target in target_names:
            normalized_target = self.normalize_name(target)
            
            # Calculate various similarity scores
            ratios = {
                'exact': 1.0 if normalized_query == normalized_target else 0.0,
                'fuzzy': fuzz.ratio(normalized_query, normalized_target) / 100,
                'partial': fuzz.partial_ratio(normalized_query, normalized_target) / 100,
                'token_sort': fuzz.token_sort_ratio(normalized_query, normalized_target) / 100
            }
            
            # Get best matching method and score
            best_method = max(ratios.items(), key=lambda x: x[1])
            
            if best_method[1] >= threshold:
                matches.append((target, best_method[1], best_method[0]))
        
        return sorted(matches, key=lambda x: x[1], reverse=True)

class CharacterDetector:
    """Detects and tracks characters in video scenes."""
    
    def __init__(
        self,
        use_gpu: bool = True,
        min_face_size: int = 20,
        recognition_threshold: float = 0.6,
        max_workers: int = 4
    ):
        """Initialize character detector."""
        self.use_gpu = use_gpu
        self.min_face_size = min_face_size
        self.recognition_threshold = recognition_threshold
        self.max_workers = max_workers
        
        # Initialize MediaPipe for pose estimation
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize character matcher
        self.matcher = CharacterMatcher()
        
        # Character database
        self.characters: Dict[str, Character] = {}
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def detect_characters(
        self,
        video_path: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        known_characters: Optional[Dict[str, List[np.ndarray]]] = None
    ) -> Dict[str, Character]:
        """Detect and track characters in a video segment."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Convert times to frames
            start_frame = int(start_time * fps) if start_time else 0
            end_frame = int(end_time * fps) if end_time else total_frames
            
            # Initialize character tracking
            current_characters: Dict[str, Dict] = {}
            frame_buffer = []
            buffer_size = 5  # Process faces every N frames
            
            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            current_frame = start_frame
            while cap.isOpened() and current_frame < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Add frame to buffer
                frame_buffer.append((current_frame, frame))
                
                # Process buffer when full
                if len(frame_buffer) >= buffer_size:
                    await self._process_frame_buffer(
                        frame_buffer,
                        current_characters,
                        fps,
                        known_characters
                    )
                    frame_buffer = []
                
                current_frame += 1
            
            # Process remaining frames
            if frame_buffer:
                await self._process_frame_buffer(
                    frame_buffer,
                    current_characters,
                    fps,
                    known_characters
                )
            
            # Convert tracking data to Character objects
            characters = {}
            for char_id, data in current_characters.items():
                characters[char_id] = Character(
                    id=char_id,
                    name=data.get('name'),
                    aliases=data.get('aliases', []),
                    face_encodings=data['face_encodings'],
                    appearances=data['appearances'],
                    confidence=data['confidence'],
                    attributes=data.get('attributes', {})
                )
            
            return characters
            
        except Exception as e:
            logger.error(f"Error detecting characters: {str(e)}")
            raise
        finally:
            if 'cap' in locals():
                cap.release()
    
    async def match_character_query(
        self,
        query: str,
        characters: Dict[str, Character],
        threshold: float = 0.6
    ) -> List[Tuple[Character, float]]:
        """Match a character query against detected characters."""
        try:
            matches = []
            
            for char in characters.values():
                # Check primary name
                if char.name:
                    name_matches = self.matcher.match_names(
                        query,
                        [char.name],
                        threshold
                    )
                    if name_matches:
                        matches.append((char, name_matches[0][1]))
                        continue
                
                # Check aliases
                if char.aliases:
                    alias_matches = self.matcher.match_names(
                        query,
                        [alias.name for alias in char.aliases],
                        threshold
                    )
                    if alias_matches:
                        # Weight alias matches by alias confidence
                        best_match = alias_matches[0]
                        alias_conf = next(
                            (a.confidence for a in char.aliases if a.name == best_match[0]),
                            0.0
                        )
                        matches.append((char, best_match[1] * alias_conf))
            
            return sorted(matches, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            logger.error(f"Error matching character query: {str(e)}")
            return []
    
    async def _process_frame_buffer(
        self,
        frame_buffer: List[Tuple[int, np.ndarray]],
        current_characters: Dict[str, Dict],
        fps: float,
        known_characters: Optional[Dict[str, List[np.ndarray]]] = None
    ):
        """Process a buffer of frames for character detection."""
        try:
            # Detect faces in parallel
            face_futures = []
            for frame_num, frame in frame_buffer:
                future = self.executor.submit(
                    self._detect_faces_and_poses,
                    frame
                )
                face_futures.append((frame_num, future))
            
            # Process detection results
            for frame_num, future in face_futures:
                faces, poses = future.result()
                timestamp = frame_num / fps
                
                # Match faces with existing characters
                for face_encoding, face_location, pose in zip(faces, poses):
                    matched_id = await self._match_face(
                        face_encoding,
                        current_characters,
                        known_characters
                    )
                    
                    if matched_id:
                        # Update existing character
                        char_data = current_characters[matched_id]
                        char_data['face_encodings'].append(face_encoding)
                        char_data['appearances'].append((timestamp, timestamp))
                        char_data['confidence'] = max(
                            char_data['confidence'],
                            0.8  # Increase confidence with more appearances
                        )
                    else:
                        # Create new character
                        new_id = f"char_{len(current_characters)}"
                        current_characters[new_id] = {
                            'face_encodings': [face_encoding],
                            'appearances': [(timestamp, timestamp)],
                            'confidence': 0.6,  # Initial confidence
                            'attributes': {}
                        }
                        
                        # Try to analyze attributes
                        try:
                            attributes = DeepFace.analyze(
                                face_location,
                                actions=['age', 'gender', 'emotion']
                            )
                            current_characters[new_id]['attributes'] = attributes
                        except Exception as e:
                            logger.warning(f"Error analyzing face attributes: {str(e)}")
            
            # Merge close appearances
            for char_data in current_characters.values():
                char_data['appearances'] = self._merge_appearances(
                    char_data['appearances']
                )
            
        except Exception as e:
            logger.error(f"Error processing frame buffer: {str(e)}")
            raise
    
    def _detect_faces_and_poses(
        self,
        frame: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Detect faces and poses in a frame."""
        try:
            # Convert to RGB for face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = face_recognition.face_locations(
                rgb_frame,
                model="cnn" if self.use_gpu else "hog"
            )
            face_encodings = face_recognition.face_encodings(
                rgb_frame,
                face_locations
            )
            
            # Detect poses
            pose_results = self.pose.process(rgb_frame)
            poses = []
            
            if pose_results.pose_landmarks:
                for _ in face_locations:
                    poses.append(pose_results.pose_landmarks)
            
            return face_encodings, poses
            
        except Exception as e:
            logger.error(f"Error detecting faces and poses: {str(e)}")
            return [], []
    
    async def _match_face(
        self,
        face_encoding: np.ndarray,
        current_characters: Dict[str, Dict],
        known_characters: Optional[Dict[str, List[np.ndarray]]] = None
    ) -> Optional[str]:
        """Match a face encoding with existing characters."""
        try:
            best_match = None
            best_score = float('inf')
            
            # Check known characters first
            if known_characters:
                for name, encodings in known_characters.items():
                    for encoding in encodings:
                        score = np.linalg.norm(face_encoding - encoding)
                        if score < self.recognition_threshold and score < best_score:
                            best_score = score
                            best_match = name
            
            # Check current characters
            for char_id, char_data in current_characters.items():
                for encoding in char_data['face_encodings']:
                    score = np.linalg.norm(face_encoding - encoding)
                    if score < self.recognition_threshold and score < best_score:
                        best_score = score
                        best_match = char_id
            
            return best_match
            
        except Exception as e:
            logger.error(f"Error matching face: {str(e)}")
            return None
    
    def _merge_appearances(
        self,
        appearances: List[Tuple[float, float]],
        max_gap: float = 1.0  # Maximum gap in seconds to merge
    ) -> List[Tuple[float, float]]:
        """Merge close character appearances."""
        if not appearances:
            return []
        
        # Sort by start time
        sorted_appearances = sorted(appearances, key=lambda x: x[0])
        merged = [sorted_appearances[0]]
        
        for current in sorted_appearances[1:]:
            previous = merged[-1]
            
            # If current start is close to previous end, merge them
            if current[0] - previous[1] <= max_gap:
                merged[-1] = (previous[0], max(previous[1], current[1]))
            else:
                merged.append(current)
        
        return merged
    
    def match_character_name(
        self,
        face_encoding: np.ndarray,
        character_names: Set[str],
        threshold: float = 0.6
    ) -> Optional[str]:
        """Match a face encoding with known character names."""
        try:
            # Use fuzzy matching to find closest name
            matches = self.matcher.match_names(
                face_encoding,
                list(character_names),
                threshold
            )
            
            if matches:
                return matches[0][0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error matching character name: {str(e)}")
            return None
