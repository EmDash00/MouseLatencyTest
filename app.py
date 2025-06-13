#!/usr/bin/env python3

import asyncio
import textwrap
import threading
import time
from collections import deque
from math import tau

import cv2
import mss
import mss.tools
import numpy as np
from cv2.typing import MatLike
from mss.base import MSSBase
from playwright.async_api import async_playwright
from scipy.fft import rfft, rfftfreq

from util import find_closest

f0 = 0.1
N = int(1 / f0) + 1
w = np.arange(N) * f0 * tau
phi = np.linspace(0, 0.25 * tau, N)
scale = 0.1 / 0.6904998132600674
frame_count = 1
PRELOADED_FRAMES = 20


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
HEIGHT_CROP_RATIO = 0.05


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


def select_monitor(sct: MSSBase):
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

    return sct.monitors[selected_monitor_index]


DEFAULT_BROWSER_ARGS = {
    "chromium": [
        # Disable automation detection
        "--disable-blink-features=AutomationControlled",
        "--disable-infobars",
        # Performance and stability
        "--disable-dev-shm-usage",
        "--disable-gpu",
        # Privacy/security
        "--disable-extensions",
        "--disable-notifications",
        "--disable-popup-blocking",
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
            monitor = select_monitor(sct)
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

            context = await browser.new_context(
                viewport=viewport, device_scale_factor=1
            )

            # Create initial page
            page = await context.new_page()
            await page.goto("http://localhost:5000")
            cdp_session = None

            try:
                cdp_session = await context.new_cdp_session(page)
            except Exception:
                pass

            async def move_mouse(x, y):
                px = 0.5 * (viewport["width"] - 1) * (x + 1)
                py = 0.5 * (viewport["height"] - 1) * (y + 1)

                if cdp_session is not None:
                    await cdp_session.send('Input.dispatchMouseEvent', {
                        'type': 'mouseMoved',
                        'x': px,
                        'y': py
                    })
                    return


                await page.evaluate(
                    """([px, py]) => {
                     document.dispatchEvent(
                         new MouseEvent('mousemove', {
                             clientX: px,
                             clientY: py,
                             bubbles: true,
                             cancelable: true,
                         })
                     );
                 }""",
                    [px, py],
                )

            SCREENSHOT_REGION = {
                "top": int(monitor["top"] + monitor["height"] * 0.575),
                "left": int(monitor["left"] + monitor["width"] * 0.405),
                "height": int(monitor["height"] * HEIGHT_CROP_RATIO),
                "width": int(monitor["width"] * WIDTH_CROP_RATIO),
            }

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
