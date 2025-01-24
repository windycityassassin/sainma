import pytest
import os
import tempfile
from pathlib import Path
from .fixtures.video_generator import TestVideoGenerator

@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture(scope="session")
def video_generator(test_data_dir):
    """Create a video generator instance."""
    return TestVideoGenerator(output_dir=str(test_data_dir))

@pytest.fixture(scope="session")
def sample_video_path(video_generator):
    """Create a sample video file with scene changes."""
    return video_generator.create_scene_video(
        duration=10.0,
        fps=30.0,
        scene_changes=[2.0, 5.0, 8.0],
        output_name="sample.mp4"
    )

@pytest.fixture(scope="session")
def character_video_path(video_generator):
    """Create a sample video file with character movements."""
    character_positions = [
        (1.0, (320, 240)),  # Center
        (3.0, (160, 240)),  # Left
        (5.0, (480, 240)),  # Right
        (7.0, (320, 120)),  # Top
        (9.0, (320, 360))   # Bottom
    ]
    return video_generator.create_character_video(
        duration=10.0,
        fps=30.0,
        character_positions=character_positions,
        output_name="character.mp4"
    )

@pytest.fixture(scope="session")
def sample_subtitle_path(test_data_dir):
    """Create a sample subtitle file for testing."""
    subtitle_path = test_data_dir / "sample.srt"
    with open(subtitle_path, "w") as f:
        f.write("""1
00:00:01,000 --> 00:00:04,000
Hello, this is a test subtitle.

2
00:00:04,100 --> 00:00:08,000
It contains multiple lines for testing.

3
00:00:08,100 --> 00:00:12,000
There's a character moving around.

4
00:00:12,100 --> 00:00:16,000
The scene changes at specific times.""")
    yield str(subtitle_path)

@pytest.fixture(scope="session")
def test_config():
    """Create test configuration."""
    return {
        'max_duration': 30.0,
        'min_duration': 1.0,
        'target_fps': 30.0,
        'target_resolution': (640, 480),
        'quality_preset': 'medium',
        'use_gpu': False
    }
