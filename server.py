import os
from typing import Final

from flask import Flask, send_from_directory

app = Flask(__name__, static_folder="dist", template_folder="templates")
DIST_DIR = "dist"
PORT = 5000
CONFIG_BASE_DIRECTORY = "src/assets/configs"
VALIDATE: Final[bool] = True
DEBUG_VALIDATE: Final[bool] = True


@app.route("/<path:path>")
def serve_static(path):
    # Handle all static files (JS, CSS, images, etc.)
    if not os.path.exists(os.path.join(DIST_DIR, path)):
        # Try with .gz for compressed assets
        if os.path.exists(os.path.join(DIST_DIR, f"{path}.gz")):
            return send_from_directory(DIST_DIR, f"{path}.gz")
    return send_from_directory(DIST_DIR, path)


@app.route("/")
def serve_index():
    return send_from_directory(DIST_DIR, "index.html")


def main():
    if not os.path.exists(DIST_DIR):
        print(f"Error: {DIST_DIR} directory not found. Run webpack build first.")
        print("Try: npm run build or yarn build")
    else:
        print(f"Serving Webpack build from {DIST_DIR} on port {PORT}")
        print(f"Access at: http://localhost:{PORT}")
        app.run(port=PORT)


if __name__ == "__main__":
    main()
