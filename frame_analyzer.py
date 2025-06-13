import time

import cv2
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture("./capture.mp4")  # Replace with your video path

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
fps = cap.get(cv2.CAP_PROP_FPS)

if not cap.isOpened():
    print("Error opening video file")
    exit()

# Store center positions

# Parameters for stabilization
MIN_DOT_AREA = 48  # Minimum expected dot size in pixels
CIRCULARITY_THRESHOLD = 0.7  # Minimum circularity (1 = perfect circle)
GAUSSIAN_BLUR_SIZE = (3, 3)  # Noise reduction kernel size
MORPH_KERNEL_SIZE = 5  # Size for morphological operations

# Create morphological kernel
kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
ret, frame = cap.read()
height, width = frame.shape[:2]
buffer = np.empty((height, width), dtype=np.uint8)

centers = np.empty(1801)
weights = np.arange(width, dtype=np.uint8)

i = 0

t0 = time.perf_counter()
while True:
    if not ret:
        break  # End of video

    buffer[:] = frame[:, :, 0]
    cv2.threshold(buffer, 200, 255, cv2.THRESH_BINARY, dst=buffer)
    M = cv2.moments(buffer, binaryImage=True)

    centers[i] = M["m10"] / M["m00"]
    i += 1

    ret, frame = cap.read()
print(time.perf_counter() - t0)
cap.release()

np.savetxt("capture.npy", centers)
