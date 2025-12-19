import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import sys
import torch
from time import time
from collections import deque

# Configuration
if len(sys.argv) > 2:
    MODEL_PATH = sys.argv[1]
    INPUT_VIDEO = sys.argv[2]
else:
    exit("Usage: python detection.py <model_path> <input_video>")

OUTPUT_CSV = f"./annotations/{INPUT_VIDEO.rsplit('/', 1)[-1].rsplit('.', 1)[0]}.csv"
OUTPUT_VIDEO = f"./results/{INPUT_VIDEO.rsplit('/', 1)[-1].rsplit('.', 1)[0]}.mp4"
CONF_THRESHOLD = 0.25
MAX_TRAJECTORY_LENGTH = 200  # max number of points to keep in trajectory
MAX_INTERPOLATION_GAP = 50  # max frames to interpolate across
MAX_DETECTION_JUMP = 200  # max pixel distance between consecutive detections (outlier rejection)
start = time()

# Load model
model = YOLO(MODEL_PATH)

# Video IO
cap = cv2.VideoCapture(INPUT_VIDEO)
assert cap.isOpened(), "Failed to open input video"

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

rows = []
detections = deque(maxlen=MAX_TRAJECTORY_LENGTH) 
frame_idx = 0

def is_valid_detection(new_x, new_y, detections, max_jump):
    """Check if new detection is valid (not an outlier)"""
    if len(detections) == 0:
        return True
    
    # Check distance from last detection
    last_frame, last_x, last_y = detections[-1]
    distance = np.sqrt((new_x - last_x)**2 + (new_y - last_y)**2)
    
    return distance <= max_jump

def interpolate_gap(detections, current_frame):
    """Interpolate missing detections for small gaps"""
    if len(detections) < 1:
        return None
    
    last_frame, last_x, last_y = detections[-1]
    gap = current_frame - last_frame
    
    if gap <= 1 or gap > MAX_INTERPOLATION_GAP:
        return None
    
    # Linear interpolation
    if len(detections) >= 2:
        prev_frame, prev_x, prev_y = detections[-2]
        # Estimate velocity
        vx = (last_x - prev_x) / (last_frame - prev_frame)
        vy = (last_y - prev_y) / (last_frame - prev_frame)
    else:
        vx, vy = 0, 0
    
    # Generate interpolated points
    interpolated = []
    for f in range(last_frame + 1, current_frame):
        interp_x = last_x + vx * (f - last_frame)
        interp_y = last_y + vy * (f - last_frame)
        interpolated.append((f, interp_x, interp_y))
    
    return interpolated

# Processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
    visible = 0
    cx, cy = None, None

    if results.boxes is not None and len(results.boxes) > 0:
        # Select highest confidence detection
        boxes = results.boxes
        best_idx = int(torch.argmax(boxes.conf))
        cx, cy, _, _ = boxes.xywh[best_idx].cpu().numpy()

        # Validate detection (reject outliers)
        if is_valid_detection(cx, cy, detections, MAX_DETECTION_JUMP):
            visible = 1
            print(f"Accepted detection at frame {frame_idx}: ({cx}, {cy})")
            # Interpolate if there was a small gap
            interpolated = interpolate_gap(detections, frame_idx)
            if interpolated:
                for interp_frame, interp_x, interp_y in interpolated:
                    detections.append((interp_frame, interp_x, interp_y))
            
            # Add current detection
            detections.append((frame_idx, cx, cy))
        else:
            # Rejected as outlier
            visible = 0
            cx, cy = None, None

    # Draw current detection
    if visible:
        cv2.circle(frame, (int(round(cx)), int(round(cy))), 10, (0, 0, 255), -1)

    # Draw direct trajectory
    if len(detections) > 1:
        for i in range(1, len(detections)):
            cv2.line(
                frame,
                (int(round(detections[i - 1][1])), int(round(detections[i - 1][2]))),
                (int(round(detections[i][1])), int(round(detections[i][2]))),
                (0, 255, 0),  # Green trajectory
                thickness=2
            )

    # Write CSV row
    rows.append({
        "frame_index": frame_idx,
        "x_centroid": cx if visible else -1,
        "y_centroid": cy if visible else -1,
        "visibility_flag": visible
    })

    writer.write(frame)
    frame_idx += 1

# Cleanup
cap.release()
writer.release()

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved CSV to {OUTPUT_CSV}")
print(f"Saved annotated video to {OUTPUT_VIDEO}")
end_time = time()
print(f"Processing time: {end_time - start:.2f} seconds")