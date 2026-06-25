# This is a proprietary development of ROBOTIX Hands-on Learning, protected and not to be sold or transferred to third parties.
"""Download of the dlib face model (shape_predictor_68_face_landmarks.dat).

The model is ~95 MB and is NOT stored in git. It is downloaded the first time
(automatically from Eyes(), or by hand with `python download_model.py`).
"""

import bz2
import sys
import urllib.request
from pathlib import Path

MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
MODEL_PATH = (
    Path(__file__).parent / "gaze_tracking" / "trained_models" / "shape_predictor_68_face_landmarks.dat"
)
_MIN_BYTES = 90_000_000  # the real model is ~99 MB; smaller than this means an incomplete download

_last_pct = -1


def _progress(block_num, block_size, total_size):
    global _last_pct
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        if pct != _last_pct:  # print only when the % changes (keeps the console clean)
            _last_pct = pct
            sys.stdout.write(f"\r  downloading... {pct}%  ({downloaded // (1 << 20)} MB)")
            sys.stdout.flush()


def ensure_model(verbose=True):
    """Return the path to the model, downloading it if missing or incomplete."""
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > _MIN_BYTES:
        return MODEL_PATH

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Face model not found. Downloading from {MODEL_URL}")

    tmp_bz2 = MODEL_PATH.with_name(MODEL_PATH.name + ".bz2.part")
    urllib.request.urlretrieve(MODEL_URL, tmp_bz2, _progress if verbose else None)
    if verbose:
        print("\n  decompressing...")

    with bz2.open(tmp_bz2, "rb") as src, open(MODEL_PATH, "wb") as dst:
        while True:
            chunk = src.read(1 << 20)
            if not chunk:
                break
            dst.write(chunk)
    tmp_bz2.unlink(missing_ok=True)

    if verbose:
        print(f"  model ready: {MODEL_PATH}  ({MODEL_PATH.stat().st_size // (1 << 20)} MB)")
    return MODEL_PATH
