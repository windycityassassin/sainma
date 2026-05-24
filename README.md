# Sainma

A movie answer engine. You ask "the scene where the lightsaber duel happens on the lava planet" and it returns the actual clip, not a Google result and a timestamp guess.

[Live demo](https://huggingface.co/spaces/windycityassassin/sainma-demo) (slim public version: SigLIP 2 visual retrieval over auto-cut shots from four Blender open-movie shorts, no agent crew, no API key required)

## The problem

Finding a specific scene in a movie is brittle. You either remember the exact dialogue (rare), scrub the timeline (slow), or search YouTube and hope someone uploaded that exact moment (often not). Existing video search tools index transcripts, which fails the moment the scene is visual, emotional, or wordless.

Sainma treats a movie as a searchable corpus of scenes, where each scene has a visual fingerprint, a dialogue track, a character roster, and an emotional shape. Natural language queries hit all four signals at once.

## Approach

A crew of specialized agents, orchestrated with CrewAI. Each agent owns one slice of the problem and the chief agent routes the query.

- **Chief Agent.** Parses the query, decomposes it into subtasks, dispatches to specialists, compiles results.
- **Movie Expert.** Owns plot, characters, scene context. Resolves "the duel" to candidate scenes by knowledge.
- **Visual Analyst.** Runs CLIP for scene understanding, YOLOv8 for objects, scene-boundary detection on raw frames.
- **Dialogue Expert.** Runs Whisper for speech-to-text, RoBERTa for sentiment, matches on quoted lines.
- **Clip Director.** Takes the winning scenes, stitches boundaries with ffmpeg, exports a playlist.

The specialists run in parallel. The chief agent compiles their outputs into a ranked list with weighted scoring: semantic similarity (30%), character relevance (20%), scene type (20%), emotion match (15%), temporal context (15%).

## What makes it work

Multimodal scene embeddings are the load-bearing piece. Each scene gets a fused vector from three encoders:

- Visual via CLIP (40% weight) over sampled frames.
- Audio via Wav2Vec2 (30%) over 16kHz segments.
- Text via MPNet (30%) over scene type, character presence, actions, dialogue.

Indexed with FAISS for sub-second retrieval over hours of footage. Caching at four layers: character detection, embeddings, temporal context, name matching. A 2-hour film indexes once, queries forever.

## Run locally

You need ffmpeg installed and a Deepseek or OpenAI API key for the chief agent's reasoning.

```bash
git clone https://github.com/windycityassassin/sainma.git
cd sainma
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: set DEEPSEEK_API_KEY or OPENAI_API_KEY, MOVIE_DATA_DIR
```

Drop a video into your `MOVIE_DATA_DIR` and run the scene search demo:

```bash
# Edit demo_scene_search.py to point video_paths at real files, then:
python demo_scene_search.py
```

Or test clip generation against a short video without the full search stack:

```bash
python demo_clip_generation.py
```

First indexing pass on a 2-hour film takes 10 to 20 minutes on CPU, a few minutes with CUDA. Subsequent queries are sub-second.

## What I learned

- **Agent orchestration is not free.** CrewAI gives you a clean mental model, but the latency tax of routing a query through five LLM calls is real. The parallel-fanout pattern (chief decomposes, specialists run concurrently, chief recombines) was the only way to keep the end-to-end under 30 seconds.
- **Semantic search quality lives or dies on the embedding mix.** Visual-only retrieval ranks fight scenes by camera motion and misses the dialogue payoff. Text-only retrieval misses the wordless beats. The 40/30/30 visual/audio/text weighting was tuned empirically and is still the most fragile number in the system.
- **Video processing has no fast path.** Frame extraction, scene-boundary detection, and audio resampling each cost real time. Aggressive caching at every layer was the difference between a 20-minute first-run and a 6-hour first-run.
- **Character name matching is its own NLP problem.** "Tony", "Stark", "Iron Man", "Mr. Stark" all point to one entity. A naive exact-match recall was around 40%. Fuzzy matching (token-sort ratio + alias table + title normalization) pushed it past 90%.
- **Specialist agents beat one big agent.** Early versions had a single CrewAI agent with every tool attached. It hallucinated tool calls and lost context. Splitting by domain (visual, dialogue, plot) made each agent's prompt tractable and the failure modes legible.

## Architecture detail

See `AGENT_ARCHITECTURE.md` for the full agent crew layout, tool assignments, and the parallel-processing flow. See `DEVELOPMENT_ROADMAP.md` for the per-feature implementation status.

## License

MIT.
