from collections import deque
import time

import cv2
import numpy as np
from mss import mss

# Configuration
FPS = 60
DURATION_SEC = 10
TOTAL_FRAMES = FPS * DURATION_SEC
FRAME_SIZE = (1920, 1080)
OUTPUT_FILE = "capture.mp4"
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")  # Use 'XVID' for AVI if needed

bounding_box = {"top": 0, "left": 0, "width": FRAME_SIZE[0], "height": FRAME_SIZE[1]}
sct = mss()

# Initialize VideoWriter for grayscale
out = cv2.VideoWriter(
    filename=OUTPUT_FILE,
    fourcc=FOURCC,
    fps=FPS,
    frameSize=FRAME_SIZE,
    isColor=False,  # Critical for black & white output
)

t0 = time.perf_counter()
frame_count = 0
frames = deque()

try:
    # Capture frames with precise timing
    for _ in range(TOTAL_FRAMES):
        # Capture frame
        sct_img = sct.grab(bounding_box)

        # Convert to grayscale (direct conversion to 2D array)
        gray_frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2GRAY)
        frames.append(gray_frame)

        # Write frame
        frame_count += 1

        # Maintain precise FPS
        next_frame_time = (frame_count) / FPS
        while (time.perf_counter() - t0) < next_frame_time:
            pass

    for frame in frames:
        out.write(frame)


finally:
    # Cleanup resources
    out.release()
    print(f"Saved {frame_count} grayscale frames to {OUTPUT_FILE}")
