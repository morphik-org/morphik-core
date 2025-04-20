import json
import logging
import os
from datetime import datetime

from core.parser.video.parse_video import VideoParser
from core.tests import setup_test_logging

# Configure test logging
setup_test_logging()
logger = logging.getLogger(__name__)


async def main():
    # Logging is already configured in the setup_test_logging() function

    # Get the current directory where test_parser.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(current_dir, "assets/trial.mp4")
    logger.info(f"Video path: {video_path}")

    # Initialize parser with video path
    logger.info("Initializing VideoParser")
    parser = VideoParser(video_path)

    # Process video to get transcript and frame descriptions
    logger.info("Processing video...")
    results = await parser.process_video()
    logger.info("Video processing complete")

    # Create output directory if it doesn't exist
    output_dir = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Using output directory: {output_dir}")

    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save metadata
    metadata_path = os.path.join(output_dir, f"metadata_{timestamp}.json")
    logger.info(f"Saving metadata to {metadata_path}")
    with open(metadata_path, "w") as f:
        json.dump(results["metadata"], f, indent=4)

    # Save transcript data
    transcript_data = {
        "timestamps": results["transcript"].timestamps,
        "contents": results["transcript"].contents,
    }
    transcript_path = os.path.join(output_dir, f"transcript_{timestamp}.json")
    logger.info(f"Saving transcript to {transcript_path}")
    with open(transcript_path, "w") as f:
        json.dump(transcript_data, f, indent=4)

    # Save frame descriptions
    frame_data = {
        "timestamps": results["frame_descriptions"].timestamps,
        "contents": results["frame_descriptions"].contents,
    }
    frames_path = os.path.join(output_dir, f"frame_descriptions_{timestamp}.json")
    logger.info(f"Saving frame descriptions to {frames_path}")
    with open(frames_path, "w") as f:
        json.dump(frame_data, f, indent=4)

    # Print metadata
    logger.info("Video Metadata:")
    logger.info(f"Duration: {results['metadata']['duration']:.2f} seconds")
    logger.info(f"FPS: {results['metadata']['fps']}")
    logger.info(f"Total Frames: {results['metadata']['total_frames']}")
    logger.info(f"Frame Sample Rate: {results['metadata']['frame_sample_rate']}")

    # Print sample of transcript
    logger.info("Transcript Sample (first 3 segments):")
    transcript_data = results["transcript"].at_time(0, padding=10)
    for time, text in transcript_data[:3]:
        logger.info(f"{time:.2f}s: {text}")

    # Print sample of frame descriptions
    logger.info("Frame Descriptions Sample (first 3 frames):")
    frame_data = results["frame_descriptions"].at_time(0, padding=10)
    for time, desc in frame_data[:3]:
        logger.info(f"{time:.2f}s: {desc}")

    logger.info(f"Output files saved in: {output_dir}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
