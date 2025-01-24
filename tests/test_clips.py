import pytest
import numpy as np
import cv2
from pathlib import Path
from sainma.clips.frame_extractor import FrameExtractor
from sainma.clips.scene_detector import SceneDetector
from sainma.clips.clip_generator import ClipGenerator, ClipConfig

def test_frame_extractor_metadata(sample_video_path):
    """Test frame extractor metadata retrieval."""
    extractor = FrameExtractor(use_gpu=False)
    metadata = extractor.get_video_metadata(sample_video_path)
    
    # Check basic metadata
    assert metadata.fps == 30.0
    assert metadata.frame_count == 300  # 10 seconds * 30 fps
    assert metadata.width == 640
    assert metadata.height == 480
    assert metadata.duration == 10.0

def test_frame_extractor_frames(sample_video_path):
    """Test frame extraction functionality."""
    extractor = FrameExtractor(use_gpu=False)
    
    # Test full video extraction
    frames = list(extractor.extract_frames(
        sample_video_path,
        start_time=0,
        end_time=10.0
    ))
    assert len(frames) == 300  # 10 seconds * 30 fps
    
    # Test partial extraction
    frames = list(extractor.extract_frames(
        sample_video_path,
        start_time=2.0,
        end_time=4.0
    ))
    assert len(frames) == 60  # 2 seconds * 30 fps
    
    # Test frame properties
    for frame in frames[:10]:  # Check first 10 frames
        assert isinstance(frame.image, np.ndarray)
        assert frame.image.shape == (480, 640, 3)
        assert frame.timestamp >= 2.0
        assert frame.timestamp <= 4.0

def test_scene_detector_basic(sample_video_path):
    """Test basic scene detection."""
    detector = SceneDetector(use_gpu=False)
    scenes = detector.detect_scenes(
        sample_video_path,
        start_time=0,
        end_time=10.0
    )
    
    # We created scene changes at 2.0, 5.0, and 8.0
    assert len(scenes) == 4  # Should detect all 3 changes + initial scene
    
    # Verify scene boundaries
    scene_boundaries = [s.start_time for s in scenes]
    expected_boundaries = [0.0, 2.0, 5.0, 8.0]
    
    for actual, expected in zip(scene_boundaries, expected_boundaries):
        assert abs(actual - expected) < 0.1  # Allow small timing differences

def test_scene_detector_confidence(sample_video_path):
    """Test scene detection confidence scores."""
    detector = SceneDetector(use_gpu=False)
    scenes = detector.detect_scenes(
        sample_video_path,
        start_time=0,
        end_time=10.0
    )
    
    for scene in scenes:
        assert 'confidence' in scene.features
        assert 0 <= scene.features['confidence'] <= 1.0

def test_clip_generator_basic(test_data_dir, sample_video_path, test_config):
    """Test basic clip generation."""
    config = ClipConfig(**test_config)
    generator = ClipGenerator(
        output_dir=str(test_data_dir),
        config=config,
        use_gpu=False
    )
    
    # Generate a basic clip
    clip = generator.generate_clip(
        video_path=sample_video_path,
        start_time=0,
        end_time=5.0,
        output_name="test_basic"
    )
    
    # Verify clip properties
    assert clip.duration == 5.0
    assert Path(clip.path).exists()
    assert clip.path.endswith("test_basic.mp4")
    
    # Verify clip metadata
    assert clip.metadata['source_video'] == sample_video_path
    assert clip.metadata['start_time'] == 0
    assert clip.metadata['end_time'] == 5.0

def test_clip_generator_quality(test_data_dir, sample_video_path, test_config):
    """Test clip generation with different quality settings."""
    # Test high quality
    config = ClipConfig(**{**test_config, 'quality_preset': 'high'})
    generator = ClipGenerator(
        output_dir=str(test_data_dir),
        config=config,
        use_gpu=False
    )
    
    clip_high = generator.generate_clip(
        video_path=sample_video_path,
        start_time=0,
        end_time=2.0,
        output_name="test_high_quality"
    )
    
    # Verify high quality settings were applied
    cap = cv2.VideoCapture(clip_high.path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    assert width == 640
    assert height == 480
    assert fps == 30.0

def test_clip_generator_duration_limits(test_data_dir, sample_video_path, test_config):
    """Test clip duration limits."""
    config = ClipConfig(**test_config)
    generator = ClipGenerator(
        output_dir=str(test_data_dir),
        config=config,
        use_gpu=False
    )
    
    # Test minimum duration enforcement
    clip_short = generator.generate_clip(
        video_path=sample_video_path,
        start_time=0,
        end_time=0.5,  # Less than min_duration
        output_name="test_short"
    )
    assert clip_short.duration == config.min_duration
    
    # Test maximum duration enforcement
    clip_long = generator.generate_clip(
        video_path=sample_video_path,
        start_time=0,
        end_time=40.0,  # More than max_duration
        output_name="test_long"
    )
    assert clip_long.duration == config.max_duration
