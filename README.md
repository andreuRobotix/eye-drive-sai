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

Clone the repository and enter the folder:

    git clone https://github.com/andreuRobotix/eye-drive-sai.git
    cd eye-drive-sai

Then create and activate a virtual environment.

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

The webcam gives one **gaze number** between `0.0` and `1.0` (same orientation as the calibration bar in the app — LEFT on the left):

    1.0 ─────────── 0.5 ─────────── 0.0
   looking LEFT   looking CENTER  looking RIGHT

---

## Step 1 — Calibration: average your samples

**File: `gaze/trainer.py`.**

**Goal:** teach the program what *your* eyes look like when you look left, center and right.

**What you're given (already done).** When you calibrate, the camera films you looking each way for a few seconds. Every frame gives one **gaze number** (`0.0` = far right … `1.0` = far left). All those numbers are collected into three lists, e.g.:

    samples_by_label["left"]   = [0.71, 0.69, 0.70, 0.72, 0.70, ...]   # while you looked LEFT
    samples_by_label["center"] = [0.50, 0.49, 0.51, 0.50, ...]         # ... CENTER
    samples_by_label["right"]  = [0.31, 0.30, 0.29, 0.30, ...]         # ... RIGHT

**What you write.** Each list has many numbers, but you want **one** number that represents each direction — its **average** (arithmetic mean): add the numbers up and divide by how many there are.

    average of [0.71, 0.69, 0.70, 0.72, 0.70]  =  3.52 / 5  =  0.704

Write the `_mean` helper yourself so it takes a list and returns that average:

```python
def _mean(values):
    # TODO 1: if the list is empty (no samples), return None so we don't divide by zero
    # TODO 2: otherwise, return the average of the numbers in `values`
    ...
```

Two built-in functions are all you need: `sum(values)` adds up the list, and `len(values)` tells you how many numbers it has. The average is one of them divided by the other.

**That trio is your model.** `train()` is already written: it calls your `_mean` once per direction and keeps the three averages. Those three numbers together *are* your personal eye model:

```python
def train(samples_by_label, deadzone=DEADZONE):
    left   = _mean(samples_by_label["left"])     # e.g. 0.70
    center = _mean(samples_by_label["center"])   # e.g. 0.50
    right  = _mean(samples_by_label["right"])    # e.g. 0.30
    ...
    return EyeModel(left, center, right, deadzone)
```

With the lists above, your model is `(left=0.70, center=0.50, right=0.30)`. Step 2 uses those three numbers to decide where a new gaze is pointing.

---

## Step 2 — Prediction: left, center or right?

**File: `gaze/trainer.py`.**

A new gaze number arrives (`ratio`) and you have to say which direction it is. You write the two `if` conditions inside `predict()`. To compare against, you're **given two thresholds**: `self.t_left` (≈ `0.60`) and `self.t_right` (≈ `0.40`) — the lines a gaze has to cross to count as a turn.

Fill in the two blank conditions yourself:

```python
def predict(self, ratio):
    if ratio is None:        # the camera can't see your eyes
        return "none"
    if ______________:       # TODO: when is it "left"?
        return "left"
    if ______________:       # TODO: when is it "right"?
        return "right"
    return "center"          # not far enough either way
```

Hints:

- Remember the scale: **bigger `ratio` = more to the left, smaller = more to the right.**
- It's **left** when `ratio` has reached the *left* threshold, and **right** when it has reached the *right* threshold — so each condition compares `ratio` against one of the two `self.t_...` values (think `>=` and `<=`).

Quick check: with those thresholds, `0.66` should come out **left**, `0.35` **right**, and `0.50` **center**.

**Now record your eyes** (this runs Steps 1 and 2 on your real data):

    python train_eyes.py

Look where the screen tells you — **left, center, right** (green = recording, ~15 s). It uses your `train()` and `predict()`, prints the accuracy, and saves your model to `eye_model.json`; the game loads it automatically. Re-run it if the room's lighting changes.

> **Tip:** while the game runs, drag the handles on the **calibration bar** at the bottom of the camera window to make turns trigger more or less easily (saved instantly; press **r** to reset).

---

## Step 3 — Driving: move the robot

**File: `play.py`.**

Move the robot with the **tank** command — each value is a wheel's speed, from `-100` to `100`. And there's a stop:

    motor.movement_move_tank(LEFT_WHEEL, RIGHT_WHEEL)
    motor.stop()

The trick: to go **straight**, both wheels spin at the **same** speed; to **turn**, make **one wheel slower** than the other; to **stop**, call `motor.stop()`.

`decide()` receives `state` — one of `"center"`, `"left"`, `"right"` or `"none"` (no face). We've solved the **left** case for you as an example; fill in the other three (`SPEED` and `TURN_INNER` are set at the top of `play.py`):

```python
def decide(motor, state):
    if state == "left":
        motor.movement_move_tank(TURN_INNER, SPEED)   # turn left: slow the LEFT (inner) wheel
    elif state == "center":
        # TODO: go forward -> both wheels at the SAME speed
        ...
    elif state == "right":
        # TODO: turn right -> the mirror image of "left"
        ...
    else:      # "none" (no face)
        # TODO: stop the robot, for safety
        ...
```

(Braking is automatic: blinking a lot stops the robot for you, so that case never reaches `decide()`.)

When `decide()` is done, connect the robot and drive for real:

    python play.py

> ⚠️ **VERY IMPORTANT — set your own card number first.** At the top of `play.py`, change `CARD_SERIAL = "...."` to **your robot's number** (the one printed on its connection card). Every robot has a different number, so if it isn't yours the program won't connect to *your* robot (it may connect to someone else's, or to none). Not sure of your number? Run `python solutions/tests/find_serial.py`.

---

## Controls & circuit

- Look **left / center / right** to move; **blink several times quickly** to brake.
- Drag the calibration-bar handles to adjust your turns (**r** = reset). Press **q** to quit.
- Drive the circuit as a team — weave the cones, take the corners, reach the finish without knocking obstacles over. Time your runs and swap roles.

## If something goes wrong

- **Camera doesn't open:** in `gaze/eyes.py`, change `CAMERA_INDEX` from `0` to `1` (or `2`).
- **Robot won't connect:** check Bluetooth is on, the robot is on and nearby, and not already connected elsewhere (e.g. a browser tab). To find your card number, run `python solutions/tests/find_serial.py`.
- **Directions feel swapped:** flip the values in your `decide()`, or ask a mentor.
