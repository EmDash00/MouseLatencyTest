#!/usr/bin/env python3

"""
System latency measurement tool that analyzes the delay between input signals
and on-screen response using frequency domain analysis.

Methodology:
1. Generates a broad band signal at, sampled at 0.1 Hz intervals
2. Captures screen response
3. Computes group delay via FFT analysis
"""

import threading
import time
from collections import deque
from enum import IntEnum
from math import tau

import cv2
import mss
import mss.tools
import numpy as np
import pyautogui
from cv2.typing import MatLike
from mss.models import Monitor
from scipy.fft import rfft, rfftfreq

from util import find_closest, precision_sleep, spin


class Keys(IntEnum):
    Q = ord("q")
    W = ord("w")
    A = ord("a")
    S = ord("s")
    D = ord("d")
    ENTER = 13
    UP = 82
    DOWN = 84
    LEFT = 81
    RIGHT = 83


# Configuration
BASE_FREQUENCY = 0.1
NUM_SAMPLES = int(1 / BASE_FREQUENCY) + 1
FREQUENCIES = np.arange(NUM_SAMPLES) * BASE_FREQUENCY * tau
PHASES = np.linspace(0, 0.25 * tau, NUM_SAMPLES)
OUTPUT_SCALE = 0.03
NORMALIZATION_FACTOR = 0.6904998132600674
AMPLITUDE = OUTPUT_SCALE / NORMALIZATION_FACTOR
frame_count = 1
NUM_PRELOADED_FRAMES = 20

FPS = 60
NUM_PERIODS = 3  # Record 3 full periods of the lowest frequency
DURATION_SEC = NUM_PERIODS / BASE_FREQUENCY
TOTAL_FRAMES = int(FPS * DURATION_SEC) + 1

# Crop the region we record to reduce latency and processing work.
BUFFER_AMOUNT = 0.01
WIDTH_CROP_RATIO = OUTPUT_SCALE + BUFFER_AMOUNT
HEIGHT_CROP_RATIO = 0.01 + BUFFER_AMOUNT

# Globals

capture_semaphore = threading.Semaphore(value=0)
frame_deltas = deque()
capture_request_times = deque()

t = np.arange(NUM_PRELOADED_FRAMES, TOTAL_FRAMES + NUM_PRELOADED_FRAMES) / FPS
x = FREQUENCIES[:, np.newaxis] * t + PHASES[:, np.newaxis]
input_signal = AMPLITUDE * np.sin(x).sum(axis=0) / NUM_SAMPLES
fft_freqs = rfftfreq(TOTAL_FRAMES, 1 / FPS)
input_frequency_indicies = find_closest(
    fft_freqs, BASE_FREQUENCY * np.arange(1, NUM_SAMPLES)
)
input_frequencies = fft_freqs[input_frequency_indicies]
input_signal_fft = rfft(input_signal)[input_frequency_indicies]


def process_frames(frames: deque[MatLike]):
    """
    Takes a set of frames and calculates the group delay from them.
    """

    centers = np.empty(len(frames))
    height, width = np.asarray(frames[0]).shape[:2]
    buffer = np.empty((height, width), dtype=np.uint8)

    for i, frame in enumerate(frames):
        buffer[:] = np.asarray(frame)[:, :, 0]
        cv2.threshold(buffer, 200, 255, cv2.THRESH_BINARY, dst=buffer)
        M = cv2.moments(buffer, binaryImage=True)
        center_px = M["m10"] / M["m00"]
        centers[i] = OUTPUT_SCALE * (2 * center_px / width - 1)

    output_signal_fft = rfft(centers)[input_frequency_indicies]

    # Compute transfer function of input to output.
    # Input is typically denoted u and output y, so H_yu is the transfer
    # function

    H_yu = output_signal_fft / input_signal_fft  # type: ignore

    # Compute the phase delay
    H_yu_phase = np.angle(H_yu)

    # The group delay is the derivative of the phase delay and represents
    # the time delay for that frequency component. Constant time delay
    # leads to constant group delay for all frequencies (linear phase)
    # Fit a line to this. Any y intercept corresponds to start timing
    # inconsistency.
    group_delay, b = np.polyfit(input_frequencies * tau, H_yu_phase, deg=1)

    # Compute the r^2 value for the fit
    r_squared = 1 - (
        (H_yu_phase - (group_delay * input_frequencies * tau + b)) ** 2
    ).sum() / (H_yu_phase.var() * (H_yu_phase.size - 1))

    print(f"Calculated group delay: {-group_delay:.6f} seconds (r^2 = {r_squared:.6f})")
    np.savetxt("capture.npy", centers)


def select_monitor() -> tuple[int, Monitor]:
    """
    Displays labeled screenshots of several monitors. The user is
    prompted to select one.
    """

    with mss.mss() as sct:
        selected_monitor_index = 0

        def get_monitor_selection():
            nonlocal selected_monitor_index

            while True:
                print(
                    "Pick the monitor you will play the game on. Previews are "
                    f"shown above [1-{len(sct.monitors) - 1}]: ",
                    end="",
                )
                selection = input()

                if len(selection) > 1:
                    print("Please input a monitor number.")
                    continue

                if not selection.isdigit():
                    print("Please input a monitor number.")
                    continue

                index = int(selection)
                if index <= 0 or index >= len(sct.monitors):
                    print("Invalid monitor number.")
                    continue

                selected_monitor_index = index

                break

        if len(sct.monitors) == 2:
            return 1, sct.monitors[1]

        # Skip the first monitor as that's the whole screen (all monitors)
        for i, monitor in enumerate(sct.monitors[1:]):
            screenshot = np.array(sct.grab(monitor))

            # Resize the screenshots to be smallish while maintaining aspect ratio
            aspect_ratio = screenshot.shape[1] / screenshot.shape[0]
            resize_height = 300
            resize_width = int(resize_height * aspect_ratio)
            cv2.imshow(
                f"Monitor {i + 1}",
                cv2.resize(
                    screenshot,
                    (resize_width, resize_height),
                    interpolation=cv2.INTER_AREA,
                ),
            )

        thread = threading.Thread(target=get_monitor_selection)
        thread.start()

        # Keep the windows open until the monitor is selected
        while selected_monitor_index == 0:
            # sleep for 20 ms at a time by waiting for a key with timeout of 20 ms
            cv2.waitKey(20)

        cv2.destroyAllWindows()

        return selected_monitor_index, sct.monitors[selected_monitor_index]


def normalized_coordinates_to_pixels(
    x: float, y: float, monitor: Monitor
) -> tuple[int, int]:
    px = int(0.5 * (monitor["width"] - 1) * (x + 1) + monitor["left"])
    py = int(0.5 * (monitor["height"] - 1) * (y + 1) + monitor["top"])

    return (px, py)


def get_screenshot_region(monitor: Monitor, monitor_idx: int):
    SCREENSHOT_REGION = {
        "top": int(monitor["top"] + monitor["height"] * 0.5 * (1 - HEIGHT_CROP_RATIO)),
        "left": int(monitor["left"] + monitor["width"] * 0.5 * (1 - WIDTH_CROP_RATIO)),
        "height": int(monitor["height"] * HEIGHT_CROP_RATIO),
        "width": int(monitor["width"] * WIDTH_CROP_RATIO),
    }

    with mss.mss() as sct:
        frame = np.asarray(sct.grab(sct.monitors[monitor_idx]))
        display_scale = int(frame.shape[1] / monitor["width"])
        print(display_scale)
        while True:
            match cv2.waitKey(40) & 0xFF:
                case Keys.ENTER:
                    break
                case Keys.UP | Keys.W:
                    SCREENSHOT_REGION["top"] -= 5
                case Keys.DOWN | Keys.S:
                    SCREENSHOT_REGION["top"] += 5
                case Keys.LEFT | Keys.A:
                    SCREENSHOT_REGION["left"] -= 5
                case Keys.RIGHT | Keys.D:
                    SCREENSHOT_REGION["left"] += 5

            frame = np.asarray(sct.grab(sct.monitors[monitor_idx]))

            x0 = SCREENSHOT_REGION["left"] - monitor["left"]
            y0 = SCREENSHOT_REGION["top"] - monitor["top"]
            frame = cv2.rectangle(
                frame,
                (display_scale * x0, display_scale * y0),
                (
                    display_scale * (x0 + SCREENSHOT_REGION["width"]),
                    display_scale * (y0 + SCREENSHOT_REGION["height"]),
                ),
                color=(0, 0, 255),  # red
                thickness=2,  # 2 px thick line
            )

            aspect_ratio = monitor["width"] / monitor["height"]

            # Show a 600 px preview
            frame = cv2.resize(
                frame,
                (600, int(600 / aspect_ratio)),
                interpolation=cv2.INTER_AREA,
            )
            cv2.imshow("Preview", frame)

        cv2.destroyWindow("Preview")

        return SCREENSHOT_REGION


def preload_frames(t0: float, num_preload_frames: int, monitor: Monitor):
    frame_count = 0
    for _ in range(num_preload_frames):
        next_frame_time = (frame_count) / FPS + t0

        t = (frame_count - 1) / FPS
        x = (AMPLITUDE * np.sin(FREQUENCIES * t + PHASES)).sum() / NUM_SAMPLES

        pyautogui.moveTo(
            *normalized_coordinates_to_pixels(x, 0, monitor),
            duration=0,
            _pause=False,
        )

        frame_count += 1
        spin(next_frame_time - time.perf_counter())


def run_latency_test(
    t0: float, num_preload_frames: int, total_frames: int, monitor: Monitor
):
    time_frame_begin = 0
    frame_count = num_preload_frames + 1
    print(total_frames)

    for _ in range(total_frames):
        time_frame_begin = time.perf_counter()
        t = (frame_count - 1) / FPS
        x = (AMPLITUDE * np.sin(FREQUENCIES * t + PHASES)).sum() / NUM_SAMPLES

        pyautogui.moveTo(
            *normalized_coordinates_to_pixels(x, 0, monitor),
            duration=0,
            _pause=False,
        )

        # Capture frame
        capture_request_times.append(time.perf_counter())
        capture_semaphore.release()

        frame_count += 1

        # Maintain precise FPS
        next_frame_time = frame_count / FPS + t0

        precision_sleep(next_frame_time - time.perf_counter())
        frame_delta = time.perf_counter() - time_frame_begin
        frame_deltas.append(frame_delta)
        # print(frame_delta)
        # print(frame_count)


def main():
    capture_latency = deque()
    frames = deque()

    is_running = True

    monitor_idx, monitor = select_monitor()
    print("Calculated resolution:", monitor)

    SCREENSHOT_REGION = get_screenshot_region(monitor, monitor_idx)

    # Create a thread to capture frames.
    def capture():
        with mss.mss() as sct:
            while is_running:
                # Sleep until a capture is requested.
                capture_semaphore.acquire()

                # If we are still running, capture a screenshot
                if is_running:
                    capture_latency.append(
                        time.perf_counter() - capture_request_times.popleft()
                    )
                    frames.append(sct.grab(SCREENSHOT_REGION))

    thread = threading.Thread(target=capture)
    thread.start()

    # Click on window the game is on to focus it.
    pyautogui.click(
        *normalized_coordinates_to_pixels(0, 0, monitor),
    )

    t0 = time.perf_counter()
    # Give a 20 frame head start to avoid "jumps" in the data.
    preload_frames(t0, NUM_PRELOADED_FRAMES, monitor)

    run_latency_test(t0, NUM_PRELOADED_FRAMES, TOTAL_FRAMES, monitor)

    # Cleanup
    is_running = False
    capture_semaphore.release()
    thread.join()

    # Compute and log results.
    process_frames(frames)
    print("Mean frame delta:", np.mean(frame_deltas))
    print("Mean capture latency:", np.mean(capture_latency))
    np.savetxt("dt.npy", frame_deltas)
    np.savetxt("dt2.npy", capture_latency)


if __name__ == "__main__":
    main()
