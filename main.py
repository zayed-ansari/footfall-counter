# Name: Zayed Ansari
# Internship Assessment: Computer Vision - Footfall Counter
# Footfall Counter Project
# Python 3.11
# Dependencies: ultralytics, opencv-python, numpy, torch

from ultralytics import YOLO
import cv2
import numpy as np
import time

# ===== CONFIGURABLE PATHS =====
video_path = "people.mp4"            # Input video
output_path = "footfall_output.mp4"  # Output video

# ===== LOAD YOLO MODEL =====
model = YOLO("yolov8n.pt")  # YOLOv8 Nano (fast + lightweight)

# ===== VIDEO SETUP =====
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0,
                      (int(cap.get(3)), int(cap.get(4))))

# ===== COUNTING VARIABLES =====
line_y = 300                 # ROI line (horizontal)
entered, exited = set(), set()
track_memory = {}            # stores previous y position per ID
trajectory_memory = {}       # stores list of previous points per ID

# ===== FPS VARIABLES =====
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ===== DETECTION & TRACKING =====
    results = model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml")
    
    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for box, id_ in zip(boxes, ids):
            x1, y1, x2, y2 = box
            cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)

            # ===== COUNTING LOGIC =====
            if id_ in track_memory:
                prev_y = track_memory[id_]
                if prev_y < line_y <= cy:
                    entered.add(id_)
                elif prev_y > line_y >= cy:
                    exited.add(id_)
            track_memory[id_] = cy

            # ===== TRAJECTORY VISUALIZATION =====
            if id_ not in trajectory_memory:
                trajectory_memory[id_] = []
            trajectory_memory[id_].append((cx, cy))

            # Draw trajectory line
            for j in range(1, len(trajectory_memory[id_])):
                cv2.line(frame, trajectory_memory[id_][j-1], trajectory_memory[id_][j], (255, 255, 0), 2)

            # Draw bounding box and ID
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
            cv2.putText(frame, f'ID {id_}', (cx, cy-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ===== DRAW ROI LINE & COUNTS =====
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)
    cv2.putText(frame, f'IN: {len(entered)}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'OUT: {len(exited)}', (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # ===== FPS CALCULATION =====
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {int(fps)}', (frame.shape[1] - 150, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # ===== SHOW & SAVE FRAME =====
    cv2.imshow("Footfall Counter", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

# ===== CLEANUP =====
cap.release()
out.release()
cv2.destroyAllWindows()

# ===== FINAL COUNTS =====
print(f"Total Entered: {len(entered)}")
print(f"Total Exited: {len(exited)}")
