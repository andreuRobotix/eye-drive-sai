"""
TRAIN YOUR EYES: teach a tiny AI model to read your gaze.

It will ask you to look LEFT, CENTER and RIGHT for a few seconds each, learn a
model from your data, tell you how accurate it is, and save it to
eye_model.json. After that, play.py uses YOUR model automatically.

Run from the project folder:
    python train_eyes.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from gaze.trainer import run_calibration

if __name__ == "__main__":
    print("We'll train a tiny model to read YOUR eyes.")
    print("Look where the camera window tells you (LEFT / CENTER / RIGHT).\n")
    model, acc, counts = run_calibration()
    print("\nTrained! Your personal eye model:")
    print(f"  samples collected : {counts}")
    print(f"  learned gaze       : right={model.right:.2f}  center={model.center:.2f}  left={model.left:.2f}")
    print(f"  decision rule      : right if ratio<= {model.t_right:.2f}   left if ratio>= {model.t_left:.2f}")
    print(f"  accuracy on your data: {acc * 100:.0f}%")
    print("\n  saved to eye_model.json  ->  play.py will use it automatically.")
