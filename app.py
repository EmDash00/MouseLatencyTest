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
from playwright.async_api import Page, async_playwright
from scipy.fft import rfft, rfftfreq

from util import find_closest

f0 = 0.1
N = int(1 / f0) + 1
w = np.arange(N) * f0 * tau
phi = np.linspace(0, 0.25 * tau, N)
scale = 0.1 / 0.6904998132600674
frame_count = 1
PRELOADED_FRAMES = 20


class Keys(IntEnum):
    Q = ord("q")
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
WIDTH_CROP_RATIO = 0.2
HEIGHT_CROP_RATIO = 0.0125


async def activate_fullscreen(page: Page):
    # Create a transparent button for user gesture
    await page.evaluate("""() => {
        const btn = document.createElement('button');
        btn.id = '__fullscreen_trigger';
        btn.style.position = 'fixed';
        btn.style.top = '0';
        btn.style.left = '0';
        btn.style.opacity = '0';
        btn.style.zIndex = '9999';
        document.body.appendChild(btn);
    }""")

    # Click the button to establish user context
    await page.click("#__fullscreen_trigger")

    # Activate fullscreen on document element
    await page.evaluate("""() => {
        const elem = document.documentElement;
        const requestFullscreen =
            elem.requestFullscreen ||
            elem.mozRequestFullScreen ||
            elem.webkitRequestFullscreen ||
            elem.msRequestFullscreen;

        if (requestFullscreen) {
            requestFullscreen.call(elem).catch(e => console.log(e));
            return true;
        }
        return false;
    }""")

    # Clean up the trigger button
    await page.evaluate("""() => {
        const btn = document.getElementById('__fullscreen_trigger');
        btn?.parentNode?.removeChild(btn);
    }""")


def process_frames(frames: deque[MatLike]):
    centers = np.empty(len(frames))
    height, width = frames[0].shape[:2]
    buffer = np.empty((height, width), dtype=np.uint8)

    for i, frame in enumerate(frames):
        buffer[:] = frame[:, :, 0]
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

    print(f"Calculated group delay: {-group_delay:.3f} seconds (r^2 = {r_squared:.3f})")
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


DEFAULT_BROWSER_ARGS = {
    "chromium": [
        # Disable automation detection
        "--disable-blink-features=AutomationControlled",
        "--disable-infobars",
        # Performance and stability
        "--disable-dev-shm-usage",
        "--disable-gpu",
        # Privacy/security
        "--ignore-certificate-errors",  # Bypass HTTPS errors
        "--unsafely-treat-insecure-origin-as-secure=http://localhost:5000",  # Treat HTTP as secure
        "--disable-features=IsolateOrigins,site-per-process",
        "--disable-web-security",  # Disable CORS & other security policies
        "--allow-running-insecure-content",  # Allow mixed content
        "--disable-extensions",
        "--disable-notifications",
        "--disable-popup-blocking",
        "--disable-web-security",
        # UI/display
        "--hide-scrollbars",
        "--mute-audio",
        # Misc
        "--disable-background-networking",
        "--disable-background-timer-throttling",
        "--disable-backgrounding-occluded-windows",
        "--disable-breakpad",
        "--disable-client-side-phishing-detection",
        "--disable-component-extensions-with-background-pages",
        "--disable-default-apps",
        "--disable-hang-monitor",
        "--disable-ipc-flooding-protection",
        "--disable-renderer-backgrounding",
        "--disable-sync",
        "--metrics-recording-only",
        "--no-first-run",
        "--password-store=basic",
        "--use-mock-keychain",
    ],
    "firefox": [
        # Privacy/security
        "--no-remote",
        "--disable-extensions",
        "--disable-default-apps",
        "--disable-popup-blocking",
        # Performance
        "--disable-gpu",
        # Automation stealth
        "--disable-browser-side-navigation",
        "--disable-web-security",
        # UI/display
        "--hide-scrollbars",
        "--mute-audio",
    ],
    "webkit": [],
}


def select_browser() -> str:
    selection_list = [
        "chromium",
        "webkit",
        "firefox",
    ]
    while True:
        print(
            textwrap.dedent("""
            Which browser type will you be playing on?
            [1] Chromium (Chrome, Edge, Brave, Opera)
            [2] Apple Webkit (Safari)
            [3] Firefox
            Your choice [1-3]: """),
            end="",
        )
        selection = input()

        if len(selection) != 1:
            print("Selection should be a single digit.")
            continue

        if not selection[0].isdigit():
            print("Selection should be a single digit.")
            continue

        index = int(selection)

        if not (index >= 1 and index <= 3):
            print("Selection should be a number [1-3].")
            continue

        return selection_list[index - 1]


async def main():
    dt = deque()
    with mss.mss() as sct:
        async with async_playwright() as p:
            monitor_idx, monitor = select_monitor(sct)
            browser_type = select_browser()

            browser = await {
                "firefox": p.firefox,
                "webkit": p.webkit,
                "chromium": p.chromium,
            }.get(browser_type, p.chromium).launch(
                headless=False,
                args=DEFAULT_BROWSER_ARGS[browser_type],
            )
            context = await browser.new_context(no_viewport=True)

            page = await context.new_page()
            viewport = await page.evaluate(""" () => ({
                width: window.screen.width,
                height: window.screen.height
            });""")
            await page.close()

            context = await browser.new_context(viewport=viewport)
            page = await context.new_page()

            print("Monitor information: ", monitor)
            print("Calculated real screen resolution: ", viewport)

            # Create initial page
            await page.goto("http://localhost:5000")
            await page.wait_for_timeout(1)
            await activate_fullscreen(page)

            async def move_mouse(x, y):
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

            while True:
                match cv2.waitKey(40) & 0xFF:
                    case Keys.ENTER:
                        cv2.destroyAllWindows()
                        break
                    case Keys.UP:
                        SCREENSHOT_REGION["top"] -= 5
                        print(SCREENSHOT_REGION["top"] / monitor["height"])
                    case Keys.DOWN:
                        SCREENSHOT_REGION["top"] += 5
                        print(SCREENSHOT_REGION["top"] / monitor["height"])
                    case Keys.LEFT:
                        SCREENSHOT_REGION["left"] -= 5
                        print(SCREENSHOT_REGION["left"] / monitor["width"])
                    case Keys.RIGHT:
                        SCREENSHOT_REGION["top"] += 5
                        print(SCREENSHOT_REGION["left"] / monitor["width"])

                frame = np.asarray(sct.grab(sct.monitors[monitor_idx]))

                frame = cv2.rectangle(
                    frame,
                    (SCREENSHOT_REGION["left"], SCREENSHOT_REGION["top"]),
                    (
                        SCREENSHOT_REGION["left"] + SCREENSHOT_REGION["width"],
                        SCREENSHOT_REGION["top"] + SCREENSHOT_REGION["height"],
                    ),
                    (0, 0, 255),
                    2,
                )

                frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
                cv2.imshow("PREVIEW", frame)

            frame_count = 1
            t0 = time.perf_counter()
            try:
                # Capture frames with precise timing
                for _ in range(PRELOADED_FRAMES):
                    next_frame_time = (frame_count) / FPS

                    t = (frame_count - 1) / FPS
                    x = (scale * np.sin(w * t + phi)).sum() / N
                    await move_mouse(x, 0)

                    frame_count += 1

                    while (time.perf_counter() - t0) < next_frame_time:
                        await asyncio.sleep(0)

                t1 = 0
                for _ in range(TOTAL_FRAMES):
                    if t1 != 0:
                        dt.append(time.perf_counter() - t1)
                    t1 = time.perf_counter()

                    t = (frame_count - 1) / FPS
                    x = (scale * np.sin(w * t + phi)).sum() / N
                    await move_mouse(x, 0)

                    # Capture frame
                    frames.append(np.asarray(sct.grab(SCREENSHOT_REGION)))

                    frame_count += 1

                    # Maintain precise FPS
                    next_frame_time = (frame_count) / FPS
                    while (time.perf_counter() - t0) < next_frame_time:
                        # This is a trick. Sleeps/waits will always yield
                        # control, even if the value is 0.
                        # This yields control back to the browser for one
                        # update cycle to allow it to continue working.
                        await asyncio.sleep(0)

            finally:
                await browser.close()

    process_frames(frames)
    np.savetxt("dt.npy", dt)
    print(np.mean(dt))


if __name__ == "__main__":
    asyncio.run(main())
