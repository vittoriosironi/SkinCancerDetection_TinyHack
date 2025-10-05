"""Flask server that receives alerts from Nicla Vision and stores incoming images.

Run with:
    python app.py --host 110.100.15.27 --port 8000
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

LOGGER = logging.getLogger("nicla.server")
app = Flask(__name__)

DEFAULT_UPLOAD_DIR = Path(__file__).resolve().parent / "incoming"
DEFAULT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _print_image_to_terminal(image_path: Path, width: int = 80) -> None:
    """Stampa una preview ASCII dell'immagine nel terminale."""
    if not HAS_PIL:
        print("📷 [Immagine ricevuta - installa Pillow per vedere la preview]")
        return
    
    try:
        img = Image.open(image_path)
        
        # Ridimensiona mantenendo aspect ratio
        aspect_ratio = img.height / img.width
        height = int(width * aspect_ratio * 0.5)  # 0.5 perché i caratteri sono più alti che larghi
        img_resized = img.resize((width, height), Image.Resampling.LANCZOS)
        
        # Converti in grayscale
        img_gray = img_resized.convert('L')
        
        # Caratteri ASCII dal più scuro al più chiaro
        ascii_chars = ' .:-=+*#%@'
        
        print("\n📷 PREVIEW IMMAGINE:")
        print("┌" + "─" * width + "┐")
        
        for y in range(height):
            row = "│"
            for x in range(width):
                pixel_value = img_gray.getpixel((x, y))
                # Mappa il valore del pixel (0-255) a un carattere ASCII
                char_index = int((pixel_value / 255) * (len(ascii_chars) - 1))
                row += ascii_chars[char_index]
            row += "│"
            print(row)
        
        print("└" + "─" * width + "┘")
        print(f"Dimensioni originali: {img.width}x{img.height} px\n")
        
    except Exception as e:
        print(f"⚠️  Impossibile mostrare preview: {e}")


def _parse_metadata(metadata_raw: str | None) -> Dict[str, Any]:
    if not metadata_raw:
        return {}
    try:
        metadata = json.loads(metadata_raw)
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object")
        return metadata
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"invalid metadata JSON: {exc}") from exc


def _save_image(image_stream, original_filename: str, upload_dir: Path) -> Tuple[Path, str]:
    sanitized_name = secure_filename(original_filename) or "capture.jpg"
    suffix = Path(sanitized_name).suffix or ".jpg"
    capture_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    destination = upload_dir / f"{capture_id}{suffix}"
    image_stream.save(destination)
    return destination, capture_id


@app.post("/ingest")
def ingest():
    upload_dir = Path(app.config.get("UPLOAD_DIR", DEFAULT_UPLOAD_DIR))
    upload_dir.mkdir(parents=True, exist_ok=True)

    received_at = datetime.utcnow().isoformat() + "Z"

    if "image" not in request.files:
        return jsonify({"status": "error", "message": "missing 'image' field"}), 400

    image_file = request.files["image"]
    metadata_raw = request.form.get("metadata")

    try:
        metadata = _parse_metadata(metadata_raw)
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400

    destination, capture_id = _save_image(image_file, image_file.filename, upload_dir)

    metadata.update(
        {
            "capture_id": capture_id,
            "received_at_utc": received_at,
            "original_filename": image_file.filename,
            "content_type": image_file.content_type,
            "content_length": request.content_length,
        }
    )

    metadata_path = destination.with_suffix(destination.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Log dettagliato per verificare la comunicazione
    print("\n" + "=" * 80)
    print("📸 NUOVA IMMAGINE RICEVUTA!")
    print("=" * 80)
    print(f"🆔 Capture ID:       {capture_id}")
    print(f"📁 File salvato:     {destination.name}")
    print(f"📊 Dimensione:       {request.content_length:,} bytes ({request.content_length / 1024:.2f} KB)")
    print(f"🎨 Tipo contenuto:   {image_file.content_type}")
    print(f"📝 Nome originale:   {image_file.filename}")
    print(f"⏰ Ricevuto alle:    {received_at}")
    
    if metadata.get("device_id"):
        print(f"🔧 Device ID:        {metadata['device_id']}")
    if metadata.get("score") is not None:
        print(f"⚠️  Score sospetto:   {metadata['score']:.3f}")
    if metadata.get("timestamp"):
        print(f"⏱️  Timestamp device: {metadata['timestamp']} ms")
    
    print(f"💾 Metadata salvati: {metadata_path.name}")
    print("=" * 80 + "\n")
    
    # Stampa preview dell'immagine a terminale
    _print_image_to_terminal(destination, width=60)

    LOGGER.info("Stored capture %s -> %s", capture_id, destination.name)

    return (
        jsonify(
            {
                "status": "ok",
                "capture_id": capture_id,
                "stored_filename": destination.name,
                "metadata_path": metadata_path.name,
            }
        ),
        201,
    )


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s in %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Nicla Vision ingestion server")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument(
        "--upload-dir",
        type=Path,
        default=DEFAULT_UPLOAD_DIR,
        help="Directory where incoming captures will be stored",
    )
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args(argv)

    _setup_logging(args.verbose)
    app.config["UPLOAD_DIR"] = args.upload_dir

    LOGGER.info("Starting Nicla ingestion server on %s:%s", args.host, args.port)
    LOGGER.info("Saving captures to %s", args.upload_dir)

    # Ensure upload directory exists ahead of time
    args.upload_dir.mkdir(parents=True, exist_ok=True)

    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        LOGGER.info("Server interrupted; shutting down")
        try:
            # Give Flask a chance to shut down cleanly
            os.kill(os.getpid(), signal.SIGINT)
        except OSError:
            pass
        sys.exit(0)
