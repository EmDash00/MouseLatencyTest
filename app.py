#!/usr/bin/env python3

import asyncio
import textwrap
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
from mss.base import MSSBase
from mss.models import Monitor
from scipy.fft import rfft, rfftfreq

from util import find_closest

f0 = 0.1
N = int(1 / f0) + 1
w = np.arange(N) * f0 * tau
phi = np.linspace(0, 0.25 * tau, N)
scale = 0.03 / 0.6904998132600674
frame_count = 1
PRELOADED_FRAMES = 20


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
FPS = 60
DURATION_SEC = 3 / f0
TOTAL_FRAMES = int(FPS * DURATION_SEC) + 1

t = np.arange(PRELOADED_FRAMES, TOTAL_FRAMES + PRELOADED_FRAMES) / FPS
x = w[:, np.newaxis] * t + phi[:, np.newaxis]
u = scale * np.sin(x).sum(axis=0) / N
fft_freqs = rfftfreq(TOTAL_FRAMES, 1 / FPS)
fi = find_closest(fft_freqs, f0 * np.arange(1, N))
f = fft_freqs[fi]
U = rfft(u)[fi]

t0 = time.perf_counter()
frame_count = 0
frames = deque()

# Crop the region we record to reduce latency and processing work.
WIDTH_CROP_RATIO = 0.031
HEIGHT_CROP_RATIO = 0.011


def process_frames(frames: deque[MatLike]):
    centers = np.empty(len(frames))
    height, width = np.asarray(frames[0]).shape[:2]
    buffer = np.empty((height, width), dtype=np.uint8)

    for i, frame in enumerate(frames):
        buffer[:] = np.asarray(frame)[:, :, 0]
        cv2.threshold(buffer, 200, 255, cv2.THRESH_BINARY, dst=buffer)
        M = cv2.moments(buffer, binaryImage=True)
        center_px = M["m10"] / M["m00"]
        centers[i] = WIDTH_CROP_RATIO * (2 * center_px / width - 1)

    Y = rfft(centers)[fi]
    H_yu = Y / U  # type: ignore
    H_yu_phase = np.angle(H_yu)
    group_delay, b = np.polyfit(f * tau, H_yu_phase, deg=1)

    r_squared = 1 - ((H_yu_phase - (group_delay * f * tau + b)) ** 2).sum() / (
        H_yu_phase.var() * (H_yu_phase.size - 1)
    )

    print(f"Calculated group delay: {-group_delay:.6f} seconds (r^2 = {r_squared:.6f})")
    np.savetxt("capture.npy", centers)


def select_monitor(sct: MSSBase) -> tuple[int, Monitor]:
    """
    Displays labeled screenshots of several monitors. The user is
    prompted to select one.
    """
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
                screenshot, (resize_width, resize_height), interpolation=cv2.INTER_AREA
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


def main():
    dt = deque()
    with mss.mss() as sct:
        monitor_idx, monitor = select_monitor(sct)
        print("Calculated resolution:", monitor)

        def click_mouse(x, y):
            px = 0.5 * (monitor["width"] - 1) * (x + 1) + monitor["left"]
            py = 0.5 * (monitor["height"] - 1) * (y + 1) + monitor["top"]
            pyautogui.click(px, py)

        def move_mouse(x, y):
            px = 0.5 * (monitor["width"] - 1) * (x + 1) + monitor["left"]
            py = 0.5 * (monitor["height"] - 1) * (y + 1) + monitor["top"]
            pyautogui.moveTo(px, py, _pause=False)

        SCREENSHOT_REGION = {
            "top": int(
                monitor["top"] + monitor["height"] * 0.5 * (1 - HEIGHT_CROP_RATIO)
            ),
            "left": int(
                monitor["left"] + monitor["width"] * 0.5 * (1 - WIDTH_CROP_RATIO)
            ),
            "height": int(monitor["height"] * HEIGHT_CROP_RATIO),
            "width": int(monitor["width"] * WIDTH_CROP_RATIO),
        }

        frame = np.asarray(sct.grab(sct.monitors[monitor_idx]))
        display_scale = frame.shape[0] / monitor["width"]
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
            print(frame.shape)

            frame = cv2.rectangle(
                frame,
                (
                    int(SCREENSHOT_REGION["left"] * display_scale),
                    int(SCREENSHOT_REGION["top"] * display_scale),
                ),
                (
                    int(
                        display_scale
                        * (SCREENSHOT_REGION["left"] + SCREENSHOT_REGION["width"])
                    ),
                    int(
                        display_scale
                        * (SCREENSHOT_REGION["top"] + SCREENSHOT_REGION["height"])
                    ),
                ),
                (0, 0, 255),
                2,
            )

            ratio = monitor["height"] / monitor["width"]
            frame = cv2.resize(
                frame,
                (600, int(600 * ratio)),
                interpolation=cv2.INTER_AREA,
            )
            cv2.imshow("Preview", frame)

        cv2.destroyWindow("Preview")
        click_mouse(0, 0)

        frame_count = 1
        t0 = time.perf_counter()
        # Capture frames with precise timing

        for _ in range(PRELOADED_FRAMES):
            next_frame_time = (frame_count) / FPS

            t = (frame_count - 1) / FPS
            x = (scale * np.sin(w * t + phi)).sum() / N
            move_mouse(x, 0)

            frame_count += 1

            while (time.perf_counter() - t0) < next_frame_time:
                pass

        t1 = 0
        for _ in range(TOTAL_FRAMES):
            t1 = time.perf_counter()
            t = (frame_count - 1) / FPS
            x = (scale * np.sin(w * t + phi)).sum() / N

            move_mouse(x, 0)

            # Capture frame
            frames.append(sct.grab(SCREENSHOT_REGION))

            frame_count += 1

            dt.append(time.perf_counter() - t1)
            # Maintain precise FPS
            next_frame_time = frame_count / FPS
            while (time.perf_counter() - t0) < next_frame_time:
                pass

    process_frames(frames)
    print(np.mean(dt))
    np.savetxt("dt.npy", dt)


if __name__ == "__main__":
    main()
