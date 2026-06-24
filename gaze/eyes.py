"""Eyes: a simple wrapper around GazeTracking to drive a robot with your gaze.

Usage:
    from gaze import Eyes
    eyes = Eyes()
    eyes.update()         # grab a webcam frame, analyze it and draw the preview
    eyes.direction()      # "center" | "left" | "right" | "none"
    eyes.is_braking()     # True if you blink repeatedly (handbrake)
    eyes.release()        # release the camera

To CALIBRATE you only touch the CONSTANTS below.
"""

import sys
import time

import cv2

from .gaze_tracking import GazeTracking
from .model import ensure_model
from .trainer import DEFAULT_MODEL_PATH, EyeModel

# ===================== CALIBRATION =====================
# GazeTracking horizontal_ratio: 0.0 = looking RIGHT, 0.5 = center, 1.0 = looking LEFT
LOOK_RIGHT_BELOW = 0.35   # ratio <= this -> "right"  (lower = you must look further right)
LOOK_LEFT_ABOVE = 0.65    # ratio >= this -> "left"   (higher = you must look further left)
#  (between 0.35 and 0.65 -> "center": a wide dead-zone so a tiny glance doesn't turn)

# Smoothing: a new direction must hold this many frames before it counts (less twitchy)
DIRECTION_STABLE_FRAMES = 3

# Handbrake by SUSTAINED blinking (not a single natural blink)
BLINK_WINDOW_SEC = 2.0        # time window we look at (a bit generous so it's easy to trigger)
BLINK_COUNT_THRESHOLD = 3     # number of blinks within the window to brake
BRAKE_HOLD_SEC = 1.0          # once braking, hold it at least this long

# Camera and window
# DirectShow opens the webcam fast on Windows; Mac/Linux use the default backend.
CAMERA_BACKEND = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
CAMERA_INDEX = 0
FRAME_WIDTH = 640
SHOW_PREVIEW = True
PREVIEW_WINDOW = "Eyes - look to drive (q = quit)"
# =======================================================


class Eyes:
    def __init__(self, camera_index=CAMERA_INDEX, show_preview=SHOW_PREVIEW, model_path=DEFAULT_MODEL_PATH):
        ensure_model()  # download the face model the first time if missing
        self.gaze = GazeTracking()
        # If you trained a personal model (python train_eyes.py), use it.
        self._model_path = model_path
        self.model = None
        if model_path and EyeModel.exists(model_path):
            try:
                self.model = EyeModel.load(model_path)
                print("[eyes] using your trained eye model (eye_model.json)")
            except Exception:
                self.model = None

        # Live decision thresholds shown on the calibration bar and editable by
        # dragging. Seed them from the trained model if present, else from the
        # fixed constants above. (right < center < left in ratio terms, so
        # t_right < t_left.) Reference means are only used to draw little ticks.
        if self.model is not None and self.model.center is not None:
            self._means = (self.model.right, self.model.center, self.model.left)
        else:
            self._means = None
        if self.model is not None:
            self._t_left = self.model.t_left
            self._t_right = self.model.t_right
        else:
            self._t_left = LOOK_LEFT_ABOVE
            self._t_right = LOOK_RIGHT_BELOW
        # remember the seeds so 'r' can reset the bar to where calibration started
        self._t_left_default = self._t_left
        self._t_right_default = self._t_right
        # calibration-bar geometry + drag state (set while rendering)
        self._bar = None        # (x0, y0, x1, y1) pixel rect of the bar
        self._drag = None       # which handle is being dragged: "left" | "right" | None
        self.video = cv2.VideoCapture(camera_index, CAMERA_BACKEND)
        if not self.video.isOpened():
            raise RuntimeError(
                f"Could not open the camera (CAMERA_INDEX={camera_index}). "
                "Try changing CAMERA_INDEX to 1 or 2 in gaze/eyes.py."
            )
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.show_preview = show_preview
        if self.show_preview:
            cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(PREVIEW_WINDOW, self._on_mouse)

        self._prev_closed = False
        self._blink_times = []
        self._brake_until = 0.0
        self._braking = False
        self._last_ratio = None
        # smoothed direction state
        self._dir = "none"
        self._cand = "none"
        self._cand_count = 0
        self.last_key = -1

    def update(self):
        """Grab and analyze a frame; draw the preview. Returns the pressed key."""
        ok, frame = self.video.read()
        if not ok or frame is None or getattr(frame, "size", 0) == 0:
            # transient camera hiccup: keep the previous state
            self.last_key = (cv2.waitKey(1) & 0xFF) if self.show_preview else -1
            return self.last_key
        try:
            self.gaze.refresh(frame)
        except Exception:
            # GazeTracking can choke on a bad frame (e.g. an empty eye crop) and
            # raise deep inside OpenCV. Don't let that kill the game: drop this
            # frame and treat it as "no face" until a good one comes in.
            self.gaze.eye_left = None
            self.gaze.eye_right = None
        self._update_brake()
        self._update_direction()
        self._last_ratio = self.gaze.horizontal_ratio() if self.gaze.pupils_located else None
        if self.show_preview:
            self._render(frame)
            if self.last_key == ord('r'):
                self._reset_calibration()
        return self.last_key

    def _update_brake(self):
        now = time.time()
        try:
            closed = bool(self.gaze.is_blinking())
        except Exception:
            closed = self._prev_closed
        # open->closed edge = one blink (so a long blink counts as a SINGLE one)
        if closed and not self._prev_closed:
            self._blink_times.append(now)
        self._prev_closed = closed
        # keep only the blinks inside the window
        cutoff = now - BLINK_WINDOW_SEC
        self._blink_times = [t for t in self._blink_times if t >= cutoff]
        if len(self._blink_times) >= BLINK_COUNT_THRESHOLD:
            self._brake_until = now + BRAKE_HOLD_SEC
        self._braking = now < self._brake_until

    def _raw_direction(self):
        """Instant direction from the current frame, before smoothing."""
        if not self.gaze.pupils_located:
            return "none"
        r = self.gaze.horizontal_ratio()
        if r is None:
            return "none"
        # Single source of truth = the live thresholds shown on the calibration
        # bar. They start from your trained model (or the fixed constants) and
        # update the instant you drag a handle.
        if r >= self._t_left:
            return "left"
        if r <= self._t_right:
            return "right"
        return "center"

    def _update_direction(self):
        """Only commit a new direction after it holds for DIRECTION_STABLE_FRAMES."""
        raw = self._raw_direction()
        if raw == self._dir:
            self._cand = raw
            self._cand_count = 0
            return
        if raw == self._cand:
            self._cand_count += 1
        else:
            self._cand = raw
            self._cand_count = 1
        if self._cand_count >= DIRECTION_STABLE_FRAMES:
            self._dir = raw
            self._cand_count = 0

    def direction(self):
        """'center', 'left', 'right' or 'none' (smoothed, less twitchy)."""
        return self._dir

    def is_braking(self):
        return self._braking

    def _render(self, frame):
        annotated = self.gaze.annotated_frame()
        self._draw_state_badge(annotated)
        self._draw_bar(annotated)
        cv2.imshow(PREVIEW_WINDOW, annotated)
        self.last_key = cv2.waitKey(1) & 0xFF

    # ===================== calibration bar (the HUD) =====================
    # The model is 1-D: the gaze ratio (0=right .. 0.5=center .. 1=left) is split
    # into three zones by two thresholds. We draw that line, show your live gaze
    # on it, and let you drag the two boundaries to recalibrate (saved to disk).
    _C_LEFT = (246, 130, 59)    # BGR -> blue   (looking LEFT)
    _C_CENTER = (94, 197, 34)   # BGR -> green  (center / go straight)
    _C_RIGHT = (11, 158, 245)   # BGR -> amber  (looking RIGHT)
    _C_BRAKE = (68, 68, 240)    # BGR -> red
    _C_PANEL = (40, 32, 28)     # BGR -> dark slate (translucent HUD panel)
    _FONT = cv2.FONT_HERSHEY_SIMPLEX

    @staticmethod
    def _round_rect(img, x0, y0, x1, y1, color, r, left=True, right=True):
        """Filled rectangle with rounded ends (round either/both short sides)."""
        if x1 <= x0 or y1 <= y0:
            return
        r = max(1, min(r, (y1 - y0) // 2, (x1 - x0) // 2))
        xa = x0 + (r if left else 0)
        xb = x1 - (r if right else 0)
        cv2.rectangle(img, (xa, y0), (xb, y1), color, -1, cv2.LINE_AA)
        if left:
            cv2.rectangle(img, (x0, y0 + r), (x0 + r, y1 - r), color, -1)
            cv2.circle(img, (x0 + r, y0 + r), r, color, -1, cv2.LINE_AA)
            cv2.circle(img, (x0 + r, y1 - r), r, color, -1, cv2.LINE_AA)
        if right:
            cv2.rectangle(img, (x1 - r, y0 + r), (x1, y1 - r), color, -1)
            cv2.circle(img, (x1 - r, y0 + r), r, color, -1, cv2.LINE_AA)
            cv2.circle(img, (x1 - r, y1 - r), r, color, -1, cv2.LINE_AA)

    @staticmethod
    def _dim(color, f=0.40):
        return tuple(int(c * f) for c in color)

    def _ratio_to_x(self, r):
        """ratio 1.0 (LEFT) -> left edge of the bar; 0.0 (RIGHT) -> right edge."""
        x0, _, x1, _ = self._bar
        return int(round(x1 - r * (x1 - x0)))

    def _x_to_ratio(self, x):
        x0, _, x1, _ = self._bar
        return (x1 - x) / float(x1 - x0)

    def _draw_state_badge(self, img):
        """A clean rounded pill, top-left, coloured by what you're doing."""
        if self._braking:
            label, color = "BRAKE", self._C_BRAKE
        else:
            label = self._dir.upper() if self._dir != "none" else "NO FACE"
            color = {"LEFT": self._C_LEFT, "CENTER": self._C_CENTER,
                     "RIGHT": self._C_RIGHT}.get(label, (90, 90, 90))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
        x, y, pad = 22, 22, 12
        self._round_rect(img, x, y, x + tw + 2 * pad, y + th + 2 * pad, color,
                         (th + 2 * pad) // 2)
        cv2.putText(img, label, (x + pad, y + pad + th), cv2.FONT_HERSHEY_DUPLEX,
                    0.9, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_bar(self, img):
        h, w = img.shape[:2]
        mx = 28
        panel_y0, panel_y1 = max(0, h - 116), h - 10

        # translucent rounded HUD panel behind everything
        px0, px1 = mx - 12, w - mx + 12
        roi = img[panel_y0:panel_y1, px0:px1]
        if roi.size:
            shade = roi.copy()
            shade[:] = self._C_PANEL
            cv2.addWeighted(shade, 0.55, roi, 0.45, 0, dst=roi)

        x0, x1 = mx + 6, w - mx - 6
        by1 = h - 60
        by0 = by1 - 24
        self._bar = (x0, by0, x1, by1)
        r = (by1 - by0) // 2

        xL = self._ratio_to_x(self._t_left)    # LEFT | CENTER boundary
        xR = self._ratio_to_x(self._t_right)   # CENTER | RIGHT boundary
        # while braking the robot is stopped, so dim every zone (nothing glows)
        active = None if self._braking else self._dir

        # three zones: the one you're in glows, the others are muted
        cL = self._C_LEFT if active == "left" else self._dim(self._C_LEFT)
        cC = self._C_CENTER if active == "center" else self._dim(self._C_CENTER)
        cR = self._C_RIGHT if active == "right" else self._dim(self._C_RIGHT)
        self._round_rect(img, x0, by0, xL, by1, cL, r, left=True, right=False)
        cv2.rectangle(img, (xL, by0), (xR, by1), cC, -1)
        self._round_rect(img, xR, by0, x1, by1, cR, r, left=False, right=True)

        # zone names inside the bar
        cy = (by0 + by1) // 2
        for text, xa, xb in (("LEFT", x0, xL), ("CENTER", xL, xR), ("RIGHT", xR, x1)):
            (tw, th), _ = cv2.getTextSize(text, self._FONT, 0.48, 1)
            if xb - xa > tw + 12:
                cv2.putText(img, text, ((xa + xb) // 2 - tw // 2, cy + th // 2),
                            self._FONT, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # faint ticks for your trained means (only if you ran train_eyes.py)
        if self._means is not None:
            for m in self._means:
                xm = self._ratio_to_x(m)
                cv2.line(img, (xm, by1 + 4), (xm, by1 + 9), (170, 170, 170), 1, cv2.LINE_AA)

        # two draggable handles (line + white knob) with their values underneath
        for xh, val in ((xL, self._t_left), (xR, self._t_right)):
            cv2.line(img, (xh, by0 - 5), (xh, by1 + 5), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(img, (xh, by1 + 6), 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (xh, by1 + 6), 6, (70, 70, 70), 1, cv2.LINE_AA)
            s = f"{val:.2f}"
            (tw, _t), _ = cv2.getTextSize(s, self._FONT, 0.42, 1)
            cv2.putText(img, s, (xh - tw // 2, by1 + 28), self._FONT, 0.42,
                        (215, 215, 215), 1, cv2.LINE_AA)

        # live gaze marker: white needle + a coloured value pill floating above
        if self._last_ratio is not None:
            xm = max(x0, min(x1, self._ratio_to_x(self._last_ratio)))
            accent = {"left": self._C_LEFT, "center": self._C_CENTER,
                      "right": self._C_RIGHT}.get(active, (235, 235, 235))
            cv2.line(img, (xm, by0 - 4), (xm, by1), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(img, (xm, by0 - 4), 4, accent, -1, cv2.LINE_AA)
            cv2.circle(img, (xm, by0 - 4), 4, (255, 255, 255), 1, cv2.LINE_AA)
            s = f"{self._last_ratio:.2f}"
            (tw, th), _ = cv2.getTextSize(s, self._FONT, 0.5, 1)
            pw, ph = tw + 16, th + 12
            ppx = int(max(x0, min(x1 - pw, xm - pw // 2)))
            ppy = by0 - 10 - ph
            self._round_rect(img, ppx, ppy, ppx + pw, ppy + ph, accent, ph // 2)
            cv2.putText(img, s, (ppx + 8, ppy + ph - 6), self._FONT, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

        # hint, centered along the panel's bottom edge
        hint = "drag handles to calibrate    r = reset"
        (tw, _t), _ = cv2.getTextSize(hint, self._FONT, 0.4, 1)
        cv2.putText(img, hint, ((x0 + x1) // 2 - tw // 2, panel_y1 - 6),
                    self._FONT, 0.4, (165, 165, 165), 1, cv2.LINE_AA)

    def _on_mouse(self, event, x, y, flags, param):
        """Grab a threshold handle and drag it to move the zone boundary."""
        if self._bar is None:
            return
        x0, y0, x1, y1 = self._bar
        if event == cv2.EVENT_LBUTTONDOWN:
            if y0 - 14 <= y <= y1 + 14:
                xL, xR = self._ratio_to_x(self._t_left), self._ratio_to_x(self._t_right)
                if abs(x - xL) <= 16 and abs(x - xL) <= abs(x - xR):
                    self._drag = "left"
                elif abs(x - xR) <= 16:
                    self._drag = "right"
        elif event == cv2.EVENT_MOUSEMOVE and self._drag is not None:
            if not (flags & cv2.EVENT_FLAG_LBUTTON):   # button released elsewhere
                self._drag = None
                return
            self._set_threshold(self._drag, self._x_to_ratio(x))
        elif event == cv2.EVENT_LBUTTONUP and self._drag is not None:
            self._drag = None
            self._save_calibration()

    def _set_threshold(self, which, ratio):
        """Move one boundary, keeping right < left with a small minimum gap."""
        gap = 0.04
        ratio = max(0.02, min(0.98, ratio))
        if which == "left":
            self._t_left = max(ratio, self._t_right + gap)
        else:
            self._t_right = min(ratio, self._t_left - gap)

    def _save_calibration(self):
        """Persist the current thresholds to eye_model.json so they stick."""
        if not self._model_path:
            return
        try:
            if self.model is not None:
                self.model.set_thresholds(self._t_left, self._t_right)
            else:
                self.model = EyeModel(None, None, None,
                                      t_left=self._t_left, t_right=self._t_right)
            self.model.save(self._model_path)
            print(f"[eyes] calibration saved (right<= {self._t_right:.2f}, "
                  f"left>= {self._t_left:.2f})")
        except Exception as e:  # noqa: BLE001 - never crash the game over a save
            print("[eyes] could not save calibration:", e)

    def _reset_calibration(self):
        """Restore the thresholds to where this session started, and save."""
        self._t_left = self._t_left_default
        self._t_right = self._t_right_default
        self._save_calibration()
        print("[eyes] calibration reset")

    def release(self):
        self.video.release()
        cv2.destroyAllWindows()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False
