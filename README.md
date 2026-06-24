# Eye Drive — the two-person obstacle circuit

No hands, no remote: you drive a LEGO SAI robot **with your eyes**. Look left, center or right at the webcam and the robot turns, drives forward, or stops.

This is a **team activity for two people**. One of you steers with their eyes; the other guides them. Together your mission is to get the robot through the obstacle circuit we've laid out for you and reach the finish line without knocking obstacles over.

- **The Driver (the eyes).** Sits at the laptop and looks **left / center / right** at the webcam to steer. The Driver is busy concentrating on the screen and on controlling their own eyes, so they can't really watch the course.
- **The Navigator (the guide).** Stands next to the circuit with a full view of the robot and the obstacles, and calls out instructions to the Driver — *"look left… now center… brake!"*. The Navigator is the eyes on the track; the Driver is the hands on the wheel.

Swap roles after each run so everyone does both jobs.

And before you can play, **you write the code that makes it all work**: the part that learns your eyes, the part that decides where you're looking, and the part that moves the robot.

## What you need

- A laptop with a webcam. Windows, Mac and Linux all work.
- Python 3.12 installed.
- A SAI robot and its connection card (you'll use the number printed on the card).
- Bluetooth turned on.
- The obstacle circuit we've set up (cones, cups, blocks…) on the floor.

## Get it running (you only do this once)

Create a virtual environment and turn it on.

On Windows:

    py -3.12 -m venv .venv
    .venv\Scripts\activate

On Mac or Linux:

    python3.12 -m venv .venv
    source .venv/bin/activate

Install everything:

    pip install -r requirements.txt

The first time you run the program it also downloads a face-detection model, so give it a minute.

---

# The activity: write the brain of the game

There are **three pieces of code you have to write yourself**. We give you everything around them — the camera, the eye-tracking, the game loop — but the actual logic is yours:

| Step | What you write | File | Function |
|------|----------------|------|----------|
| **1. Calibration** | the **average** of each direction's samples | `gaze/trainer.py` | `train()` (via `_mean`) |
| **2. Prediction** | the **`if`** that says left / center / right | `gaze/trainer.py` | `predict()` |
| **3. Driving** | move the robot for each direction | `play.py` | `decide()` |

Do them **in order**: Step 1 and Step 2 first (they're the "brain"), then run the calibration, then Step 3.

Here's the idea in one line: the webcam gives a single **gaze number** between `0.0` and `1.0`.

    0.0  ──────────────  0.5  ──────────────  1.0
   looking RIGHT        looking CENTER       looking LEFT

Step 1 learns what *your* "right", "center" and "left" numbers look like. Step 2 turns a new gaze number into a direction. Step 3 turns that direction into wheel movement.

---

## Step 1 — Calibration: turn your recorded samples into a model

**File: `gaze/trainer.py`.**

To learn your eyes, the program first **records you**: while you look in each direction for a few seconds, it captures *many* gaze numbers and groups them into three lists:

    samples_by_label["left"]     # all the numbers recorded while you looked LEFT
    samples_by_label["center"]   # ... while you looked CENTER
    samples_by_label["right"]    # ... while you looked RIGHT

Each list has **N numbers** in it (one per camera frame). Your job is to squeeze each list down to **one** number — its **arithmetic mean** (its average). That average is your "typical" gaze for that direction, and the three averages together *are* the model.

The arithmetic mean is just: **add all the numbers up, then divide by how many there are.** Two built-in Python functions do exactly that:

- `sum(a_list)` adds up all the numbers. `sum([0.70, 0.72, 0.68])` is `2.10`.
- `len(a_list)` is how many numbers there are. `len([0.70, 0.72, 0.68])` is `3`.

So the mean is `sum(a_list) / len(a_list)`. In the example: `2.10 / 3 = 0.70`.

**What you write.** In `gaze/trainer.py` there is a small helper that the `train()` function uses to average each list. Fill in its body:

```python
def _mean(values):
    if not values:                  # no samples recorded -> can't average, leave it empty
        return None
    return sum(values) / len(values)   # <-- the arithmetic mean: add them up, divide by how many
```

That single line is the whole "training". The `train()` function right below it already calls `_mean` on each of your three lists and bundles the results into the model:

```python
def train(samples_by_label, deadzone=DEADZONE):
    left   = _mean(samples_by_label["left"])     # average of all the LEFT numbers
    center = _mean(samples_by_label["center"])   # average of all the CENTER numbers
    right  = _mean(samples_by_label["right"])    # average of all the RIGHT numbers
    ...
    return EyeModel(left, center, right, deadzone)
```

If you looked right at about `0.30`, center at about `0.50` and left at about `0.70`, your model is `(left=0.70, center=0.50, right=0.30)`. That's it — three averages.

---

## Step 2 — Prediction: left, center or right?

**File: `gaze/trainer.py`.**

Now a brand-new gaze number arrives (called `ratio`) and you have to say which direction it is. We've already worked out **two thresholds** for you from your model and saved them as `self.t_left` and `self.t_right`:

- `self.t_left` — the line you must cross to count as **left** (about `0.60` for the example model).
- `self.t_right` — the line you must cross to count as **right** (about `0.40` for the example model).

(They sit halfway out toward each side, so a tiny glance doesn't accidentally count as a turn. You don't compute them — they're handed to you.)

Your job is only the **`if`**: decide whether the new `ratio` is **above** the left threshold or **below** the right threshold. Fill in the two blank conditions inside `predict()`:

```python
def predict(self, ratio):
    if ratio is None:          # the camera can't see your eyes -> no direction
        return "none"
    if _____________:          # YOU: when is it "left"?
        return "left"
    if _____________:          # YOU: when is it "right"?
        return "right"
    return "center"            # not far enough to either side -> center
```

Think about which way the gaze number goes: **bigger = more to the left, smaller = more to the right.** So:

- it's **left** when the number is **at or above** the left threshold → `ratio >= self.t_left`
- it's **right** when the number is **at or below** the right threshold → `ratio <= self.t_right`

The completed function:

```python
def predict(self, ratio):
    if ratio is None:
        return "none"
    if ratio >= self.t_left:      # at or past the LEFT line
        return "left"
    if ratio <= self.t_right:     # at or past the RIGHT line
        return "right"
    return "center"
```

With the example model the thresholds are `0.60` and `0.40`, so `0.66` is **left**, `0.35` is **right**, and `0.50` is **center**. Make sure you cover all four answers: `none`, `left`, `right`, `center` — the function above already does.

---

## Now calibrate: record your eyes and save the model

With Step 1 and Step 2 written, run the trainer so it can record you and build your personal model:

    python train_eyes.py

The whole screen turns **green** while it is recording you and **gray** while it is getting ready, so you can tell what's happening even out of the corner of your eye. Look where it tells you, in this order: **left, then center, then right** (about 15 seconds total).

Behind the scenes it calls **your** `train()` to average the samples (Step 1), checks the model with **your** `predict()` (Step 2), prints how accurate it is, and saves everything to a file called `eye_model.json` in the project folder. From then on the game loads that file automatically. Re-run it whenever the light in the room changes.

> **Tip — fine-tune live.** While the game is running, the camera window shows a **calibration bar** along the bottom: a line split into **LEFT / CENTER / RIGHT** zones, a yellow marker that follows your gaze in real time, and two white handles at the zone boundaries. **Drag a handle** to make a turn trigger more or less easily (it saves instantly), or press **r** to reset. You can even calibrate with this alone — just watch the marker and drag the handles to sit just inside each side.

---

## Step 3 — Driving: move the robot

**File: `play.py`.**

Last, write `decide()` so each direction actually moves the robot. You move it with the **tank** command — each value is the speed of one wheel, from `-100` to `100`:

    motor.movement_move_tank(LEFT, RIGHT)   # left wheel speed, right wheel speed
    motor.stop()

`decide()` receives the direction in `state`, which is one of `"center"`, `"left"`, `"right"` or `"none"` (no face). You don't have to worry about braking — blinking a lot is the handbrake and the game loop stops the robot for you, so that case never reaches `decide()`.

Open `play.py`, set your **card number** (`CARD_SERIAL`) at the top, and fill in `decide()` so the robot does this:

- look **center** → drive **forward** (both wheels the same speed)
- look **left** → **turn left** (slow the inner wheel)
- look **right** → **turn right** (slow the other inner wheel)
- **no face** → **stop** (for safety)

The completed function (the two speeds `SPEED` and `TURN_INNER` are set at the top of `play.py`):

```python
def decide(motor, state):
    if state == "center":
        motor.movement_move_tank(SPEED, SPEED)        # forward
    elif state == "left":
        motor.movement_move_tank(TURN_INNER, SPEED)   # turn left (inner = left wheel)
    elif state == "right":
        motor.movement_move_tank(SPEED, TURN_INNER)   # turn right (inner = right wheel)
    else:      # "none" (no face)
        motor.stop()
```

Cover all four cases — `center`, `left`, `right` and `none` — so the robot always knows what to do. There are extra hints and the full answer in `activities/01_drive.md`.

### Try it without the robot first

No robot connected yet? Keep `MOCK = True` at the top of `play.py` and run:

    python play.py

It prints what the robot *would* do (`[FAKE] tank(...)`) as you move your eyes, so the Driver and Navigator can rehearse the controls and check the logic before connecting anything. When it looks right, set `MOCK = False` and drive for real.

---

## How the Driver controls the robot

- Look **left**, **center** or **right** to move.
- **Blink several times quickly** to brake. A normal blink does nothing, so don't worry about it.
- **Drag the handles** on the bottom calibration bar to adjust how far you must look to turn (and **r** to reset).
- Press **q** in the camera window to quit.

The Navigator's job is to watch the robot and the obstacles and call these out clearly and early — *"a bit left… hold center… brake now!"* — because the Driver is looking at the screen, not the floor.

## Drive the circuit

We've laid out an obstacle course on the floor — weave between the cones, take the tight corners without clipping them, and drive straight through the gates. As a team, get the robot from the start line to the finish without knocking obstacles over.

Tune `SPEED` and `TURN_INNER` at the top of `play.py` to the course: a **slower speed** and a **smaller `TURN_INNER`** make the robot much easier to control on tight sections. Time each run and try to beat it, or count how many obstacles you knock over — fewest wins. Then **swap roles** and go again.

## If something goes wrong

- **The camera doesn't open:** open `gaze/eyes.py` and change `CAMERA_INDEX` from `0` to `1` (or `2`).
- **The robot won't connect:** make sure Bluetooth is on, the robot is on and nearby, and it isn't already connected to something else (like a browser tab). To find your card number, run `python solutions/tests/find_serial.py`.
- **Forward and backward, or left and right, feel swapped:** just flip the signs/values in your `decide()`, or ask a mentor.

That's everything. Write the three pieces, set up your roles, and drive it home as a team. Good luck.
