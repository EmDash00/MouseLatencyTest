import os
import subprocess
import sys
from pathlib import Path

from mss import mss
from mss.tools import to_png
from playwright.sync_api import sync_playwright


class MonitorHandler:
    def __init__(self):
        self.temp_files = []
        self.sct = mss()

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        for file in self.temp_files:
            try:
                if file and Path(file).exists():
                    Path(file).unlink()
            except:
                pass
        self.sct.close()

    def get_monitors(self):
        """Get monitor info using MSS only"""
        return self.sct.monitors[1:]

    def capture_preview(self, monitor):
        """Capture monitor screenshot using MSS"""
        try:
            screenshot = self.sct.grab(monitor)
            temp_dir = Path(
                os.environ.get("TEMP", "/tmp" if not sys.platform == "win32" else ".")
            )
            filepath = temp_dir / f"monitor_{monitor['id']}.png"
            to_png(screenshot.rgb, screenshot.size, output=str(filepath))
            self.temp_files.append(str(filepath))
            return str(filepath)
        except Exception as e:
            print(f"Capture failed for monitor {monitor['id']}: {str(e)}")
            return None

    def show_preview(self, filepath):
        """Open preview with native viewer"""
        if not filepath or not Path(filepath).exists():
            return

        try:
            if sys.platform == "win32":
                os.startfile(filepath)
            elif sys.platform == "darwin":
                subprocess.run(["open", filepath])
            else:
                subprocess.run(["xdg-open", filepath])
        except:
            pass

    def select_monitor_interactive(self):
        """Interactive monitor selection"""
        monitors = self.get_monitors()

        if len(monitors) == 1:
            return monitors[0]

        print("\nAvailable Monitors:")
        screenshots = []

        for monitor in monitors:
            screenshot = self.capture_preview(monitor)
            screenshots.append(screenshot)
            print(
                f"[{monitor['id']}] {monitor['width']}x{monitor['height']} at ({monitor['x']},{monitor['y']})"
            )
            self.show_preview(screenshot)

        while True:
            try:
                choice = input("\nSelect monitor number: ")
                selected = next((m for m in monitors if str(m["id"]) == choice), None)
                if selected:
                    return selected
                print("Invalid choice. Try again.")
            except KeyboardInterrupt:
                sys.exit(0)


def launch_browser(browser_type="chromium", url="https://example.com"):
    handler = MonitorHandler()
    monitor = handler.select_monitor_interactive()

    with sync_playwright() as p:
        browser = {
            "firefox": p.firefox,
            "webkit": p.webkit,
            "chromium": p.chromium,
        }.get(browser_type.lower(), p.chromium).launch(
            headless=False,
            args=[
                f"--window-position={monitor['x']},{monitor['y']}",
                f"--window-size={monitor['width'] // 2},{monitor['height'] // 2}",
                *(["--disable-win32k-lockdown"] if sys.platform == "win32" else []),
            ],
        )

        page = browser.new_page(no_viewport=True)
        page.goto(url)
        return browser


if __name__ == "__main__":
    launch_browser()
