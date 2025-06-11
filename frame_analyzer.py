import cv2
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture('./capture.mp4')  # Replace with your video path

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
fps = cap.get(cv2.CAP_PROP_FPS)
print(width)
print(fps)

if not cap.isOpened():
    print("Error opening video file")
    exit()

# Store center positions
centers = []

# Parameters for stabilization
MIN_DOT_AREA = 48          # Minimum expected dot size in pixels
CIRCULARITY_THRESHOLD = 0.7 # Minimum circularity (1 = perfect circle)
GAUSSIAN_BLUR_SIZE = (3, 3) # Noise reduction kernel size
MORPH_KERNEL_SIZE = 5       # Size for morphological operations

# Create morphological kernel
kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Preprocessing pipeline
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_SIZE, 0)

    # Thresholding - adjust value based on your video
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

    # Morphological operations to clean up the image
    # cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_contour = None
    best_circularity = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        # Skip small contours
        if area < MIN_DOT_AREA:
            continue

        # Calculate circularity
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # Find the most circular contour
            if circularity > CIRCULARITY_THRESHOLD and circularity > best_circularity:
                best_contour = contour
                best_circularity = circularity

    if best_contour is not None:
        # Calculate centroid
        M = cv2.moments(best_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            x = cX / width * 2 - 1
            centers.append(x)
            # print(x)


            # Draw contour outline
            # cv2.drawContours(frame, [best_contour], -1, (0, 255, 0), 1)

            # Draw centroid point (for reference)
            # cv2.circle(frame, (cX, cY), 1, (0, 0, 255), -1)

    # Display processing
    # cv2.imshow('Stabilized Dot Tracking', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
        # break

cap.release()
# cv2.destroyAllWindows()

np.savetxt("capture.npy", np.array(centers))
