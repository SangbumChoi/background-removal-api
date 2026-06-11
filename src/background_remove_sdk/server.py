"""Optional HTTP API for the SDK (successor of the original Flask app).

Requires the ``server`` extra: ``pip install "background-remove-sdk[server]"``.
Run with ``bg-remove-server`` and POST a multipart image::

    curl -X POST http://127.0.0.1:5000/api/remove -F "image=@photo.jpg" -o photo_no_bg.png
    curl -X POST http://127.0.0.1:5000/api/mask -F "image=@photo.jpg" -o photo_mask.png
    curl -X POST http://127.0.0.1:5000/api/extract -F "image=@photo.jpg" -F "x=120" -F "y=80" -o object.png
"""

from __future__ import annotations

import io

from background_remove_sdk import __version__
from background_remove_sdk.core import BackgroundRemover


def create_app(model: str = "inspyrenet", device: str | None = None):
    try:
        from flask import Flask, jsonify, request, send_file
    except ImportError as exc:
        raise SystemExit(
            "The HTTP server requires flask. "
            'Install it with: pip install "background-remove-sdk[server]"'
        ) from exc

    app = Flask("background_remove_sdk")
    remover = BackgroundRemover(model=model, device=device)

    def _read_image():
        if "image" not in request.files:
            return None, (jsonify({"error": "No 'image' file in request"}), 400)
        return request.files["image"].read(), None

    def _png_response(image):
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        buffer.seek(0)
        return send_file(buffer, mimetype="image/png")

    @app.get("/")
    def index():
        return jsonify(
            {
                "name": "background-remove-sdk",
                "version": __version__,
                "model": model,
                "endpoints": ["/api/remove", "/api/mask", "/api/extract"],
            }
        )

    @app.post("/api/remove")
    def remove():
        data, error = _read_image()
        if error:
            return error
        return _png_response(remover.remove(data))

    @app.post("/api/mask")
    def mask():
        data, error = _read_image()
        if error:
            return error
        return _png_response(remover.mask(data))

    @app.post("/api/extract")
    def extract():
        data, error = _read_image()
        if error:
            return error
        if "x" not in request.form or "y" not in request.form:
            return jsonify({"error": "Missing 'x' or 'y' coordinate in request"}), 400
        try:
            x, y = int(request.form["x"]), int(request.form["y"])
        except ValueError:
            return jsonify({"error": "'x' and 'y' must be integers"}), 400
        try:
            return _png_response(remover.extract_object(data, x, y))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 422

    return app


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="bg-remove-server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--model", default="inspyrenet", help='model spec "backend[:variant]"')
    parser.add_argument("--device", default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    create_app(model=args.model, device=args.device).run(
        host=args.host, port=args.port, debug=args.debug
    )


if __name__ == "__main__":
    main()
