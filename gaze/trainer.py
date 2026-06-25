"""Eye Trainer: a tiny machine-learning model that learns to read YOUR eyes.

Idea (same family as the Day-1 perceptron): you give it labeled examples by
looking LEFT / CENTER / RIGHT for a few seconds each. It "trains" by learning
the typical gaze value of each class, then classifies new gazes into
left / center / right. It also reports its accuracy on your own data.

Run `python train_eyes.py` to train; play.py then uses the model automatically.
"""

import json
import math
import sys
import time
from pathlib import Path

import cv2

from .gaze_tracking import GazeTracking
from .model import ensure_model

# the model is saved next to the project (works no matter where you run from)
DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parents[1] / "eye_model.json")

DEADZONE = 0.5        # how far (0..1) toward a side you must look to trigger it; bigger = less sensitive
COLLECT_SECONDS = 3.0       # seconds of samples per direction (the actual recording)
COUNTDOWN_SECONDS = 10.0    # "LOOK <direction>" countdown shown before each recording
INTRO_SECONDS = 10.0        # intro before the first direction (long enough to read)

# Capture/window size -- big enough that the on-screen instructions are never cut off.
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720

# full-screen tints so you can tell the phase at a glance (out of the corner of your eye)
GREEN = (0, 200, 0)    # GREEN  = recording this direction
GRAY = (130, 130, 130)  # GRAY  = get ready / waiting
_TINT_ALPHA = 0.45

# DirectShow opens the webcam fast on Windows; Mac/Linux use the default backend.
_CAMERA_BACKEND = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY


class EyeModel:
    """A 1-feature classifier: learns the mean gaze for left/center/right and
    classifies a new gaze by comparing against learned decision thresholds."""

    def __init__(self, left, center, right, deadzone=DEADZONE, t_left=None, t_right=None):
        self.left = left
        self.center = center
        self.right = right
        self.deadzone = deadzone
        # Decision thresholds. Hand-tuned thresholds (dragged on the calibration
        # bar) win if given; otherwise they're derived from the learned means
        # (personalized dead-zone). With manual-only calibration there are no
        # means, so explicit thresholds are required.
        if t_left is not None and t_right is not None:
            self.t_left = t_left
            self.t_right = t_right
        else:
            self.t_left = center + deadzone * (left - center)
            self.t_right = center - deadzone * (center - right)

    def set_thresholds(self, t_left, t_right):
        """Override the decision thresholds (used by the on-screen calibration bar)."""
        self.t_left = t_left
        self.t_right = t_right

    def predict(self, ratio):
        if ratio is None:
            return "none"
        if ratio >= self.t_left:
            return "left"
        if ratio <= self.t_right:
            return "right"
        return "center"

    def to_dict(self):
        return {
            "left": self.left, "center": self.center, "right": self.right,
            "deadzone": self.deadzone,
            "t_left": self.t_left, "t_right": self.t_right,
        }

    def save(self, path=DEFAULT_MODEL_PATH):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path=DEFAULT_MODEL_PATH):
        d = json.loads(Path(path).read_text())
        return cls(d.get("left"), d.get("center"), d.get("right"),
                   d.get("deadzone", DEADZONE), d.get("t_left"), d.get("t_right"))

    @staticmethod
    def exists(path=DEFAULT_MODEL_PATH):
        return Path(path).exists()


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def train(samples_by_label, deadzone=DEADZONE):
    """Learn the model from labeled gaze samples. Raises ValueError if the data
    is unusable (eyes not seen, or the three looks aren't separable)."""
    left, center, right = _mean(samples_by_label.get("left", [])), \
        _mean(samples_by_label.get("center", [])), _mean(samples_by_label.get("right", []))
    if None in (left, center, right):
        raise ValueError("Not enough samples — make sure your eyes were visible the whole time.")
    # gaze ratio is LOW when looking right and HIGH when looking left, so we expect right < center < left
    if not (right < center < left):
        raise ValueError(
            f"Calibration looks off (right={right:.2f}, center={center:.2f}, left={left:.2f}). "
            "Re-run and look clearly to each side."
        )
    return EyeModel(left, center, right, deadzone)


def accuracy(model, samples_by_label):
    """Fraction of the training samples the model classifies correctly."""
    total = correct = 0
    for label, xs in samples_by_label.items():
        for r in xs:
            total += 1
            if model.predict(r) == label:
                correct += 1
    return correct / total if total else 0.0


def _fit_centered_text(frame, text, y, color, target_scale, thickness, max_width_frac=0.92):
    """Draw `text` horizontally centered at height `y`, shrinking the font if it
    would be too wide so the message is ALWAYS fully visible (never cut off).
    Returns the drawn text height in pixels."""
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = target_scale
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    max_w = int(frame.shape[1] * max_width_frac)
    if tw > max_w:                       # too wide for the frame -> scale it down to fit
        scale *= max_w / tw
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, (frame.shape[1] - tw) // 2)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return th


def _phase(video, gaze, window, big, small, seconds, tint, collect=False):
    """Show a message for `seconds` with a full-frame color tint; optionally
    record gaze samples. tint = GREEN (recording) or GRAY (waiting)."""
    samples = []
    end = time.time() + seconds
    while time.time() < end:
        ok, frame = video.read()
        if not ok:
            continue
        gaze.refresh(frame)
        ratio = gaze.horizontal_ratio() if gaze.pupils_located else None
        if collect and ratio is not None:
            samples.append(ratio)
        # tint the WHOLE frame so the phase is obvious even from the corner of your eye
        overlay = frame.copy()
        overlay[:] = tint
        frame = cv2.addWeighted(overlay, _TINT_ALPHA, frame, 1 - _TINT_ALPHA, 0)
        rem = max(0.0, end - time.time())
        h = frame.shape[0]
        # big instruction (e.g. "LOOK LEFT"), centered and auto-fitted so it never gets cut
        big_h = _fit_centered_text(frame, big, int(h * 0.46), (255, 255, 255), 2.6, 5)
        # small status / countdown line, centered just below the instruction
        small_text = small.format(rem=rem, n=len(samples), secs=math.ceil(rem))
        _fit_centered_text(frame, small_text, int(h * 0.46) + big_h + 55,
                           (255, 255, 255), 1.3, 2)
        cv2.imshow(window, frame)
        if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
            break
    return samples


def _screen_size(default=(1280, 720)):
    """Best-effort screen resolution so we can center the window. Falls back to a
    sane default if it can't be determined (e.g. tkinter unavailable)."""
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        size = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
        return size
    except Exception:
        return default


def run_calibration(camera_index=0, seconds=COLLECT_SECONDS, model_path=DEFAULT_MODEL_PATH):
    """Guided calibration in order LEFT -> CENTER -> RIGHT, then train and save.
    The whole webcam goes GREEN while recording and GRAY while getting ready."""
    ensure_model()
    gaze = GazeTracking()
    video = cv2.VideoCapture(camera_index, _CAMERA_BACKEND)
    if not video.isOpened():
        raise RuntimeError(f"Could not open the camera (index={camera_index}). Try 1 or 2.")
    # capture big so the on-screen instructions are never cut off
    video.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    window = "Train your eyes (q = cancel)"
    # open a BIG window (~95% of the screen) and CENTER it, keeping 16:9 so the
    # video isn't distorted and nothing is clipped off-edge
    sw, sh = _screen_size()
    scale = min(sw * 0.95 / CAPTURE_WIDTH, sh * 0.95 / CAPTURE_HEIGHT)
    win_w = int(CAPTURE_WIDTH * scale)
    win_h = int(CAPTURE_HEIGHT * scale)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, win_w, win_h)
    cv2.moveWindow(window, max(0, (sw - win_w) // 2), max(0, (sh - win_h) // 2))
    labels = ("left", "center", "right")  # calibration order
    # WHERE to look for each one: the top corners of your screen and the webcam
    # (top-center). Looking at the screen corners gives clean, well-separated gazes.
    targets = {
        "left":   ("LOOK TOP-LEFT",      "the TOP-LEFT corner of your screen"),
        "center": ("LOOK AT THE WEBCAM", "the WEBCAM (top-center of your screen)"),
        "right":  ("LOOK TOP-RIGHT",     "the TOP-RIGHT corner of your screen"),
    }
    data = {}
    try:
        _phase(video, gaze, window, "EYE CALIBRATION",
               "you'll look at 3 spots on your screen -- starting in {secs}s", INTRO_SECONDS, GRAY)
        for label in labels:
            big, where = targets[label]
            # GRAY countdown: tell them WHERE to look; they start when it turns GREEN
            _phase(video, gaze, window, big,
                   "look at " + where + " when the screen turns GREEN  ({secs}s)",
                   COUNTDOWN_SECONDS, GRAY)
            # GREEN recording: name the exact spot and HOLD until the green screen ends, then next
            data[label] = _phase(video, gaze, window, big,
                                 "look at " + where + " -- HOLD until this GREEN ends  ({secs}s)",
                                 seconds, GREEN, collect=True)
    finally:
        video.release()
        cv2.destroyAllWindows()
    model = train(data)
    acc = accuracy(model, data)
    model.save(model_path)
    return model, acc, {k: len(v) for k, v in data.items()}
