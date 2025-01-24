"""Demo of real-time clip generation."""

from sainma.clips.video_ops import VideoOps, ClipRequest
import time
import os

def main():
    # Initialize video operations
    video_ops = VideoOps()
    
    # Path to test video
    video_path = "The Wait  - 1 Minute Short Film ｜ Award Winning.mp4"
    
    # Get video info
    print("Getting video info...")
    info = video_ops.get_video_info(video_path)
    duration = float(info['format']['duration'])
    print(f"Video duration: {duration:.2f} seconds")
    
    # Generate some test clips
    print("\nGenerating test clips...")
    
    # Test 1: Extract first 10 seconds
    print("\nTest 1: First 10 seconds")
    start_time = time.time()
    clip_path = video_ops.generate_clip(
        [ClipRequest(video_path, 0, 10)],
        output_path="first_10s.mp4"
    )
    print(f"Generated clip: {clip_path}")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    
    # Test 2: Extract multiple segments
    print("\nTest 2: Multiple segments")
    clips = [
        ClipRequest(video_path, 0, 5),    # First 5 seconds
        ClipRequest(video_path, 20, 25),  # 20-25 seconds
        ClipRequest(video_path, 40, 45)   # 40-45 seconds
    ]
    start_time = time.time()
    clip_path = video_ops.generate_clip(
        clips,
        output_path="multiple_segments.mp4"
    )
    print(f"Generated clip: {clip_path}")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    
    # Test 3: Extract frames
    print("\nTest 3: Extracting frames")
    timestamps = [0, 10, 20, 30]
    for ts in timestamps:
        start_time = time.time()
        frame_path = video_ops.extract_frame(
            video_path,
            ts,
            output_path=f"frame_{ts}s.jpg"
        )
        print(f"Extracted frame at {ts}s: {frame_path}")
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
