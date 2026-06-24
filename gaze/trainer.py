"""Eye Trainer: a tiny machine-learning model that learns to read YOUR eyes.

Idea (same family as the Day-1 perceptron): you give it labeled examples by
looking LEFT / CENTER / RIGHT for a few seconds each. It "trains" by learning
the typical gaze value of each class, then classifies new gazes into
left / center / right. It also reports its accuracy on your own data.

Run `python train_eyes.py` to train; play.py then uses the model automatically.
"""

import json
import sys
import time
from pathlib import Path

import cv2

from .gaze_tracking import GazeTracking
from .model import ensure_model

# the model is saved next to the project (works no matter where you run from)
DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parents[1] / "eye_model.json")

DEADZONE = 0.5        # how far (0..1) toward a side you must look to trigger it; bigger = less sensitive
COLLECT_SECONDS = 3.0  # seconds of samples per direction

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
        cv2.putText(frame, big, (30, 70), cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 255, 255), 3)
        cv2.putText(frame, small.format(rem=rem, n=len(samples)), (30, 120),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
        cv2.imshow(window, frame)
        if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
            break
    return samples


def run_calibration(camera_index=0, seconds=COLLECT_SECONDS, model_path=DEFAULT_MODEL_PATH):
    """Guided calibration in order LEFT -> CENTER -> RIGHT, then train and save.
    The whole webcam goes GREEN while recording and GRAY while getting ready."""
    ensure_model()
    gaze = GazeTracking()
    video = cv2.VideoCapture(camera_index, _CAMERA_BACKEND)
    if not video.isOpened():
        raise RuntimeError(f"Could not open the camera (index={camera_index}). Try 1 or 2.")
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    window = "Train your eyes (q = cancel)"
    labels = ("left", "center", "right")  # calibration order
    data = {}
    try:
        _phase(video, gaze, window, "Get ready...", "starting in {rem:0.0f}s", 2.0, GRAY)
        for idx, label in enumerate(labels, 1):
            _phase(video, gaze, window, f"[{idx}/3] Get ready: look {label.upper()}",
                   "{rem:0.1f}s", 1.5, GRAY)
            data[label] = _phase(video, gaze, window, f"[{idx}/3] LOOK {label.upper()}",
                                 "GREEN = recording   {rem:0.1f}s   samples={n}", seconds, GREEN, collect=True)
    finally:
        video.release()
        cv2.destroyAllWindows()
    model = train(data)
    acc = accuracy(model, data)
    model.save(model_path)
    return model, acc, {k: len(v) for k, v in data.items()}
