"""
STAGE 1 - test ONLY the vision (no robot needed).

Opens the webcam and shows the detected direction and BRAKE.
Use it to CALIBRATE: look at the `ratio=` value on screen and tune
LOOK_RIGHT_BELOW / LOOK_LEFT_ABOVE in gaze/eyes.py if needed.

Run from the project folder:
    python solutions/tests/vision_demo.py
Quit: 'q' or ESC.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from gaze import Eyes

with Eyes() as eyes:
    while True:
        key = eyes.update()
        state = "BRAKE" if eyes.is_braking() else eyes.direction()
        print(state)
        if key in (ord('q'), 27):
            break
