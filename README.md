# Sainma

A powerful movie answer engine leveraging AI agents for real-time search and clip generation.

## Features

### Intelligent Search
- Natural language query understanding
- Scene-level search with semantic understanding
- Character and dialogue-based search
- Emotion and action recognition
- Context-aware results

### Smart Clip Generation
- Automatic scene detection
- High-quality clip generation
- Smooth scene transitions
- Video stabilization
- Quality optimization

### AI Agent Coordination
- Crew of specialized AI agents
- Efficient resource management
- Smart caching system
- Error recovery mechanisms

## Project Structure

```
sainma/
├── sainma/                # Main package
│   ├── agents/           # AI Agents
│   │   ├── chief_agent.py       # Coordination and planning
│   │   ├── movie_expert.py      # Movie knowledge
│   │   ├── visual_analyst.py    # Visual analysis
│   │   ├── dialogue_expert.py   # Dialogue processing
│   │   └── clip_director.py     # Clip generation
│   │
│   ├── search/          # Search System
│   │   ├── query_processor.py   # Query understanding
│   │   ├── scene_indexer.py     # Scene indexing
│   │   └── search_engine.py     # Search coordination
│   │
│   ├── clips/           # Clip Generation
│   │   ├── frame_extractor.py   # Frame processing
│   │   ├── scene_detector.py    # Scene detection
│   │   └── clip_generator.py    # Clip creation
│   │
│   ├── coordination/    # System Coordination
│   │   ├── coordinator.py       # Main coordinator
│   │   ├── resource_manager.py  # Resource management
│   │   ├── cache_manager.py     # Caching system
│   │   ├── error_handler.py     # Error handling
│   │   └── context_manager.py   # Context sharing
│   │
│   └── utils/          # Utilities
│       └── logger.py          # Logging system
│
├── tests/              # Test Suite
│   ├── test_search.py        # Search tests
│   ├── test_clips.py         # Clip tests
│   └── test_coordination.py  # Coordination tests
│
├── data/               # Data Directory
│   ├── movies/        # Movie files
│   ├── cache/         # Cache storage
│   ├── index/         # Search indices
│   └── clips/         # Generated clips
│
└── logs/              # Log files
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sainma.git
cd sainma
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy and configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

1. Start the system:
```python
from sainma.coordination.coordinator import SainmaCoordinator

coordinator = SainmaCoordinator()

# Search for a scene
results = coordinator.search("Show me the fight scene between Iron Man and Thanos")

# Generate a clip
clip = coordinator.generate_clip(results[0])
```

## Development

### Running Tests
```bash
pytest tests/
```

### Code Style
We follow PEP 8 guidelines. Run linting:
```bash
flake8 sainma/
```

## Configuration

Key configuration options in `.env`:
- `MOVIE_DATA_DIR`: Directory containing movie files
- `CACHE_DIR`: Directory for caching results
- `MAX_CLIP_LENGTH`: Maximum clip duration in seconds
- `SEARCH_THRESHOLD`: Minimum similarity score for search results
- `GPU_MEMORY_LIMIT`: Maximum GPU memory usage (0.0-1.0)

See `.env.example` for all configuration options.

## Dependencies

Core dependencies:
- CrewAI: Agent coordination
- PyTorch: ML operations
- OpenCV: Video processing
- FAISS: Similarity search
- Sentence Transformers: Text embeddings

See `requirements.txt` for complete list.

## License

MIT License. See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Roadmap

See `DEVELOPMENT_ROADMAP.md` for detailed development plans.
