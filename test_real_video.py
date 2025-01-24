from sainma.clips.scene_detector import SceneDetector
import cv2

def main():
    # Initialize scene detector
    detector = SceneDetector(use_gpu=False)
    
    # Path to your video
    video_path = "The Wait  - 1 Minute Short Film ｜ Award Winning.mp4"
    
    # Get video duration
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()
    
    print(f"Video Info:")
    print(f"- Duration: {duration:.2f} seconds")
    print(f"- FPS: {fps}")
    print(f"- Total Frames: {frame_count}")
    print("\nDetecting scenes...")
    
    # Try different thresholds
    thresholds = [0.05, 0.1, 0.15]
    
    for threshold in thresholds:
        print(f"\nTesting with threshold = {threshold}")
        # Detect scenes
        scenes = detector.detect_scenes(
            video_path,
            min_scene_length=0.5,  # Allow shorter scenes
            threshold=threshold  # Try different thresholds
        )
        
        # Print scene information
        print(f"Found {len(scenes)} scenes:")
        for i, scene in enumerate(scenes):
            print(f"\nScene {i+1}:")
            print(f"- Time: {scene.start_time:.2f}s to {scene.end_time:.2f}s")
            print(f"- Duration: {scene.end_time - scene.start_time:.2f}s")
            print(f"- Confidence: {scene.confidence:.2f}")

if __name__ == "__main__":
    main()
