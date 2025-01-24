from sainma.clips.scene_detector import SceneDetector

def main():
    # Initialize scene detector
    detector = SceneDetector(use_gpu=False)
    
    # Path to your video
    video_path = "The Wait  - 1 Minute Short Film ｜ Award Winning.mp4"
    
    print("Searching for drumming scenes...")
    scenes = detector.search_scenes(
        video_path,
        search_type="drumming",
        min_scene_length=0.5  # Allow shorter scenes
    )
    
    print(f"\nFound {len(scenes)} drumming scenes:")
    for i, scene in enumerate(scenes):
        print(f"\nDrumming Scene {i+1}:")
        print(f"- Time: {scene.start_time:.2f}s to {scene.end_time:.2f}s")
        print(f"- Duration: {scene.end_time - scene.start_time:.2f}s")
        print(f"- Confidence: {scene.confidence:.2f}")
        print("- Features:")
        for key, value in scene.features.items():
            print(f"  - {key}: {value:.2f}")

if __name__ == "__main__":
    main()
