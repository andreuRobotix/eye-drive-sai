# Eye Drive — the two-person obstacle circuit

No hands, no remote: you drive a robot from the **Computer Science and AI kit** **with your eyes**. Look left, center or right at the webcam and the robot turns, drives forward, or stops.

It's a **team activity for two**:

- **The Driver (the eyes).** Sits at the laptop and looks **left / center / right** at the webcam to steer. They're focused on the screen, so they can't really watch the course.
- **The Navigator (the guide).** Stands by the circuit, watches the robot and the obstacles, and calls out the moves — *"look left… center… brake!"*.

Your mission: get the robot through the obstacle circuit without knocking anything over. Swap roles each run. But first, **you write the code that makes it work** — three small pieces.

## What you need

- A laptop with a webcam (Windows or Mac).
- Python 3.13 installed.
- The robot from the Computer Science and AI kit and its connection card (you'll use the number printed on the card).
- Bluetooth turned on.
- The obstacle circuit we've set up on the floor.

## Get it running (once)

Create and activate a virtual environment.

On Windows:

    py -3.13 -m venv .venv
    .venv\Scripts\activate

On Mac:

    python3.13 -m venv .venv
    source .venv/bin/activate

Install everything:

    pip install -r requirements.txt

The first time you run it, it also downloads a face-detection model, so give it a minute.

---

# The activity: write the brain of the game

You write **three pieces** of code; everything else (camera, eye-tracking, game loop) is done for you:

| Step | What you write | File | Function |
|------|----------------|------|----------|
| **1. Calibration** | the **average** of each direction's samples | `gaze/trainer.py` | `_mean` (used by `train()`) |
| **2. Prediction** | the **`if`** that says left / center / right | `gaze/trainer.py` | `predict()` |
| **3. Driving** | move the robot for each direction | `play.py` | `decide()` |

The webcam gives one **gaze number** between `0.0` and `1.0`:

    0.0 ─────────── 0.5 ─────────── 1.0
   looking RIGHT  looking CENTER  looking LEFT

---

## Step 1 — Calibration: average your samples

**File: `gaze/trainer.py`.**

When you calibrate, the program records you looking each way and groups the gaze numbers into three lists (one number per camera frame):

    samples_by_label["left"]     # numbers recorded while you looked LEFT
    samples_by_label["center"]   # ... CENTER
    samples_by_label["right"]    # ... RIGHT

Your job: turn each list into **one** number, its **arithmetic mean** — add the numbers up and divide by how many there are (`sum(list) / len(list)`). Fill in the `_mean` helper:

```python
def _mean(values):
    if not values:                 # nothing recorded -> leave it empty
        return None
    return sum(values) / len(values)   # <-- the average: add them up, divide by how many
```

That's the whole "training". `train()` already calls `_mean` on your three lists to build the model:

```python
def train(samples_by_label, deadzone=DEADZONE):
    left   = _mean(samples_by_label["left"])
    center = _mean(samples_by_label["center"])
    right  = _mean(samples_by_label["right"])
    ...
    return EyeModel(left, center, right, deadzone)
```

Look right ≈ `0.30`, center ≈ `0.50`, left ≈ `0.70` → your model is `(0.70, 0.50, 0.30)`.

---

## Step 2 — Prediction: left, center or right?

**File: `gaze/trainer.py`.**

A new gaze number arrives (`ratio`) and you say which direction it is. The two thresholds are already given to you as `self.t_left` (≈ `0.60`) and `self.t_right` (≈ `0.40`). You only write the two `if` conditions:

```python
def predict(self, ratio):
    if ratio is None:        # the camera can't see your eyes
        return "none"
    if ______________:       # YOU: when is it "left"?
        return "left"
    if ______________:       # YOU: when is it "right"?
        return "right"
    return "center"
```

Bigger number = more to the left, smaller = more to the right. So it's **left** at or above the left threshold, **right** at or below the right threshold:

```python
    if ratio >= self.t_left:
        return "left"
    if ratio <= self.t_right:
        return "right"
```

With that model, `0.66` is left, `0.35` is right, `0.50` is center.

**Now record your eyes** (this runs Steps 1 and 2 on your real data):

    python train_eyes.py

Look where the screen tells you — **left, center, right** (green = recording, ~15 s). It uses your `train()` and `predict()`, prints the accuracy, and saves your model to `eye_model.json`; the game loads it automatically. Re-run it if the room's lighting changes.

> **Tip:** while the game runs, drag the handles on the **calibration bar** at the bottom of the camera window to make turns trigger more or less easily (saved instantly; press **r** to reset).

---

## Step 3 — Driving: move the robot

**File: `play.py`.**

Move the robot with the **tank** command — each value is a wheel's speed, from `-100` to `100`:

    motor.movement_move_tank(LEFT, RIGHT)
    motor.stop()

`decide()` receives `state` — one of `"center"`, `"left"`, `"right"` or `"none"` (no face). (Braking is automatic: blinking a lot stops the robot for you, so it never reaches here.)

Set your **`CARD_SERIAL`** at the top of `play.py`, then write `decide()`:

```python
def decide(motor, state):
    if state == "center":
        motor.movement_move_tank(SPEED, SPEED)        # forward
    elif state == "left":
        motor.movement_move_tank(TURN_INNER, SPEED)   # turn left
    elif state == "right":
        motor.movement_move_tank(SPEED, TURN_INNER)   # turn right
    else:      # "none" (no face)
        motor.stop()
```

`SPEED` and `TURN_INNER` are at the top of `play.py` — lower them for tighter control.

**Try it without the robot first:** keep `MOCK = True` and run `python play.py`. It prints what the robot *would* do as you move your eyes. When it looks right, set `MOCK = False` and drive for real.

---

## Controls & circuit

- Look **left / center / right** to move; **blink several times quickly** to brake.
- Drag the calibration-bar handles to adjust your turns (**r** = reset). Press **q** to quit.
- Drive the circuit as a team — weave the cones, take the corners, reach the finish without knocking obstacles over. Time your runs and swap roles.

## If something goes wrong

- **Camera doesn't open:** in `gaze/eyes.py`, change `CAMERA_INDEX` from `0` to `1` (or `2`).
- **Robot won't connect:** check Bluetooth is on, the robot is on and nearby, and not already connected elsewhere (e.g. a browser tab). To find your card number, run `python solutions/tests/find_serial.py`.
- **Directions feel swapped:** flip the values in your `decide()`, or ask a mentor.
