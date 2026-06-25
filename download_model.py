# This is a proprietary development of ROBOTIX Hands-on Learning, protected and not to be sold or transferred to third parties.
"""Download the face model (dlib) needed to detect the gaze.

You normally do NOT need to run this: Eyes() downloads it the first time.
But if you want it ready beforehand (or the room won't have internet later), run:

    python download_model.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from gaze.model import ensure_model

if __name__ == "__main__":
    ensure_model()
    print("Done.")
