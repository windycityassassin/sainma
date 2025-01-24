"""
Configuration management for Sainma.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()

class Config:
    # Project Paths
    BASE_DIR = Path(__file__).parent
    MOVIE_DATA_DIR = Path(os.getenv('MOVIE_DATA_DIR', BASE_DIR / 'data' / 'movies'))
    CACHE_DIR = Path(os.getenv('CACHE_DIR', BASE_DIR / 'data' / 'cache'))
    MODEL_DIR = Path(os.getenv('MODEL_DIR', BASE_DIR / 'data' / 'models'))
    TEMP_DIR = Path(os.getenv('TEMP_DIR', BASE_DIR / 'data' / 'temp'))
    LOG_FILE = Path(os.getenv('LOG_FILE', BASE_DIR / 'logs' / 'sainma.log'))

    # API Keys
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    # Processing Settings
    MAX_CLIP_LENGTH = int(os.getenv('MAX_CLIP_LENGTH', 120))
    MIN_CLIP_LENGTH = int(os.getenv('MIN_CLIP_LENGTH', 5))
    MAX_PARALLEL_TASKS = int(os.getenv('MAX_PARALLEL_TASKS', 4))
    CACHE_EXPIRY = int(os.getenv('CACHE_EXPIRY', 3600))

    # Search Settings
    SEARCH_THRESHOLD = float(os.getenv('SEARCH_THRESHOLD', 0.75))
    MAX_RESULTS = int(os.getenv('MAX_RESULTS', 10))
    CONTEXT_WINDOW = int(os.getenv('CONTEXT_WINDOW', 30))

    # API Settings
    API_HOST = os.getenv('API_HOST', 'localhost')
    API_PORT = int(os.getenv('API_PORT', 8000))
    API_DEBUG = os.getenv('API_DEBUG', 'true').lower() == 'true'

    @classmethod
    def setup(cls):
        """Create necessary directories and setup logging."""
        # Create directories if they don't exist
        for directory in [cls.MOVIE_DATA_DIR, cls.CACHE_DIR, 
                         cls.MODEL_DIR, cls.TEMP_DIR, cls.LOG_FILE.parent]:
            directory.mkdir(parents=True, exist_ok=True)

        # Setup logging
        logger.add(
            cls.LOG_FILE,
            rotation="1 day",
            retention="7 days",
            level=os.getenv('LOG_LEVEL', 'INFO')
        )

        logger.info("Sainma configuration initialized")
        
        # Validate critical settings
        cls._validate_settings()

    @classmethod
    def _validate_settings(cls):
        """Validate critical configuration settings."""
        if not cls.DEEPSEEK_API_KEY:
            logger.warning("DEEPSEEK_API_KEY not set")
        
        if not cls.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not set")

        # Validate numerical settings
        if cls.MAX_CLIP_LENGTH < cls.MIN_CLIP_LENGTH:
            raise ValueError("MAX_CLIP_LENGTH must be greater than MIN_CLIP_LENGTH")
        
        if cls.SEARCH_THRESHOLD < 0 or cls.SEARCH_THRESHOLD > 1:
            raise ValueError("SEARCH_THRESHOLD must be between 0 and 1")

        logger.info("Configuration validation complete")
