# Footfall Counter Project

## Overview
This project is a computer vision system that counts the number of people entering and exiting a specific area in a video. The goal is to demonstrate object detection, tracking, counting logic, and trajectory visualization using Python.

## Video Source
The input video used for testing is from YouTube: [People Entering And Exiting Mall Stock Footage](https://www.youtube.com/watch?v=gAuJlwnUqMs).  
For testing, I downloaded a short clip and saved it as `people.mp4`.

## Approach
1. **Detection**:  
   - YOLOv8 (nano version) from Ultralytics is used to detect people.  
   - Detection is limited to humans only (class 0).

2. **Tracking**:  
   - ByteTrack algorithm assigns unique IDs to each detected person.  
   - This ensures that individuals are tracked across frames and prevents double-counting.

3. **Counting Logic**:  
   - A horizontal ROI line is defined in the frame.  
   - Moving downward across the line → counted as **Entered**.  
   - Moving upward across the line → counted as **Exited**.  
   - Previous positions of each ID are stored in a dictionary for comparison.

4. **Visualization**:  
   - Bounding boxes, IDs, ROI line, and live counts (IN / OUT) are overlaid on each frame.  
   - FPS (frames per second) is displayed to monitor performance.  
   - **Trajectory visualization:** the movement path of each person is drawn with light blue lines connecting previous centroids.

5. **Output**:  
   - Processed video is saved as `footfall_output.mp4`.  
   - Total counts are printed at the end of processing.  
   - The demo video shows bounding boxes, IDs, ROI line, counts, and trajectories in action.

6. **Optional / Extensions**:  
   - The script can be adapted for **real-time webcam input** by changing the video source to `0`.  
   - Can be extended to handle more advanced occlusions, heatmaps, or deployed as an API for automated video input.

## Files
- `main.py` → Main Python script with comments and modular structure.  
- `footfall_output.mp4` → Processed video demonstrating bounding boxes, counts, and trajectories.  
- `requirements.txt` → Python dependencies.  

## Setup Instructions
1. Clone or download the project folder.  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
