#!/usr/bin/env python3

import time
from collections import deque
from math import tau

import cv2
import mss
import mss.tools
import numpy as np

# 1. Verify server is reachable
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.os_manager import ChromeType

# 2. Configure Chrome with essential flags
chrome_options = Options()
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option("useAutomationExtension", False)
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--ignore-certificate-errors")
chrome_options.add_argument("--allow-insecure-localhost")
chrome_options.add_argument("--disable-web-security")
chrome_options.add_argument("start-maximized")
chrome_options.add_argument("disable-infobars")
chrome_options.add_argument("--remote-debugging-port=9222")
chrome_options.add_argument("--host-resolver-rules=MAP * 127.0.0.1")
chrome_options.add_argument("--disable-blink-features=AutomationControlled")


# 3. Initialize driver with explicit service
service = Service(ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install())
driver = webdriver.Chrome(service=service, options=chrome_options)

driver.maximize_window()
driver.get("http://localhost:5000")
size = driver.get_window_size()
width = size["width"]
height = size["height"]

mx = 0.5 * (width - 1)
my = 0.5 * (height - 1)
x0 = 0.5 * width
y0 = 0.5 * height

canvas = driver.find_element("tag name", "canvas")

driver.execute_script("""
window.__moveCursorAbsolute = function(x, y) {
    // Ensure coordinates are within viewport
    x = Math.max(0, Math.min(x, window.innerWidth - 1));
    y = Math.max(0, Math.min(y, window.innerHeight - 1));

    // Get element at position
    var element = document.elementFromPoint(x, y);
    if (!element) element = document.body;

    // Create and dispatch event
    var event = new MouseEvent('mousemove', {
        view: window,
        bubbles: true,
        cancelable: true,
        clientX: x,
        clientY: y,
        screenX: x + window.screenX,
        screenY: y + window.screenY
    });
    element.dispatchEvent(event);
};
""")


def move_to_position(x, y):
    px = int(0.5 * (width - 1) * (x + 1))
    py = int(0.5 * (height - 1) * (y + 1))
    driver.execute_script(f"window.__moveCursorAbsolute({px}, {py});")


f0 = 0.1
N = int(1 / f0) + 1
w = np.arange(N) * f0 * tau
dt = 0.001
phi = np.linspace(0, 0.25 * tau, N)

# Configuration
FPS = 60
DURATION_SEC = 3 / f0
TOTAL_FRAMES = int(FPS * DURATION_SEC) + 1
OUTPUT_FILE = "capture.mp4"
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")  # Use 'XVID' for AVI if needed

t0 = time.perf_counter()
frame_count = 0
buffer = np.array(1)
frames = deque()


def main():
    move_to_position(0, 0)

    with mss.mss() as sct:
        # Capture frame
        bb = sct.monitors[1]

        SCREENSHOT_REGION = {
            "top": int(bb["top"] + bb["height"] * 0.55),
            "left": int(bb["left"] + bb["width"] * 0.4),
            "height": int(bb["height"] * 0.05),
            "width": int(bb["width"] * 0.2),
        }

        # Capture frame
        sct_img = sct.grab(SCREENSHOT_REGION)

        # Initialize VideoWriter for grayscale
        out = cv2.VideoWriter(
            filename=OUTPUT_FILE,
            fourcc=FOURCC,
            fps=FPS,
            frameSize=(SCREENSHOT_REGION["width"], SCREENSHOT_REGION["height"]),
            isColor=False,  # Critical for black & white output
        )

        scale = 0.1 / 0.6904998132600674
        frame_count = 1
        PRELOADED_FRAMES = 20
        t0 = time.perf_counter()
        try:
            # Capture frames with precise timing
            for _ in range(PRELOADED_FRAMES):
                next_frame_time = (frame_count) / FPS

                t = (frame_count - 1) / FPS
                x = (scale * np.sin(w * t + phi)).sum() / N
                move_to_position(x, 0)

                frame_count += 1

                while (time.perf_counter() - t0) < next_frame_time:
                    pass

            for _ in range(TOTAL_FRAMES + PRELOADED_FRAMES):
                t = (frame_count - 1) / FPS
                x = (scale * np.sin(w * t + phi)).sum() / N
                move_to_position(x, 0)

                # Capture frame
                sct_img = sct.grab(SCREENSHOT_REGION)

                # Convert to grayscale (direct conversion to 2D array)
                frames.append(np.asarray(sct_img))

                # Write frame
                frame_count += 1

                # Maintain precise FPS
                next_frame_time = (frame_count) / FPS
                while (time.perf_counter() - t0) < next_frame_time:
                    pass

            for frame in frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY))
        finally:
            # Cleanup resources
            out.release()
            print(f"Saved {TOTAL_FRAMES} grayscale frames to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
