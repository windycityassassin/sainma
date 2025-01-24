#!/usr/bin/env python3
"""Demo script for searching scenes across multiple videos and generating playlists."""

import os
from pathlib import Path
from sainma.clips.real_time_search import RealTimeSearch

def main():
    # Initialize search engine
    index_dir = "video_index"
    search = RealTimeSearch(index_dir)
    
    # Add some sample videos (you would replace these with your actual video paths)
    video_paths = [
        "path/to/movie1.mp4",
        "path/to/movie2.mp4",
        "path/to/movie3.mp4"
    ]
    
    # Index videos
    for video_path in video_paths:
        if os.path.exists(video_path):
            video_id = search.add_video(video_path)
            print(f"Indexed video: {video_path} -> {video_id}")
    
    # Search for car chase scenes across all videos
    query = "car chase"
    results = search.search_all_videos(
        query=query,
        threshold=0.6,
        limit_per_video=3,
        max_total_clips=10
    )
    
    # Generate a playlist from the results
    if results:
        output_path = os.path.join(index_dir, "playlists", f"{query.replace(' ', '_')}_playlist.mp4")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        playlist_path = search.generate_playlist(
            search_results=results,
            output_path=output_path
        )
        
        if playlist_path:
            print(f"\nGenerated playlist: {playlist_path}")
            print(f"Contains {sum(len(scenes) for _, scenes in results)} clips from {len(results)} videos")
        else:
            print("Failed to generate playlist")
    else:
        print(f"No matching scenes found for query: {query}")

if __name__ == "__main__":
    main()
