import os
from ultralytics import YOLO
import cv2
import pytubefix

# Process every Nth frame
FRAME_SAMPLE_RATE = 5  # Process every 30th frame

model = YOLO("yolo11m.pt")
yt = pytubefix.YouTube(
    "https://www.youtube.com/watch?v=kDMOZSCGSSo&t=10s&pp=ygUVbGl2ZXJwb29sIHZzIHdlc3QgaGFt"
)
stream = yt.streams.get_highest_resolution()
if not stream:
    raise Exception("No stream found")
stream.download(output_path=os.path.dirname(__file__), filename="downloaded_video.mp4")
source_video = os.path.join(os.path.dirname(__file__), "downloaded_video.mp4")
# source_video = os.path.join(os.path.dirname(__file__), "trial_inperson.mp4")

# Open video and create output video with every Nth frame
cap = cv2.VideoCapture(source_video)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create output path in same directory as source
output_path = os.path.join(os.path.dirname(source_video), "sampled_video_new.mp4")
fourcc = cv2.VideoWriter.fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps / FRAME_SAMPLE_RATE, (width, height), isColor=True)

# Set video writer parameters to avoid compression
out.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)  # Set quality to maximum

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % FRAME_SAMPLE_RATE == 0:
        # Write frame without any compression
        out.write(frame)
    frame_count += 1

cap.release()
out.release()

# Update source video path to sampled video
source_video = output_path

results = model.track(
    source=source_video,
    # stream=True,
    persist=True,
    save=True,
)

import json

# Convert results to JSON-serializable format
json_results = []
for i, r in enumerate(results):
    frame_data = {
        "frame_idx": i,
        "boxes": r.boxes.data.tolist() if r.boxes is not None else [],
        "classes": r.boxes.cls.tolist() if r.boxes is not None else [],
        "ids": r.boxes.id.tolist() if r.boxes is not None and r.boxes.id is not None else [],
        "names": [r.names[int(c)] for c in r.boxes.cls.tolist()] if r.boxes is not None else [],
    }
    json_results.append(frame_data)

# Save to JSON file in same directory as video
output_json = os.path.join(os.path.dirname(source_video), "tracking_results.json")
with open(output_json, "w") as f:
    json.dump(json_results, f, indent=2)
