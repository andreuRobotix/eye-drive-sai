# Train your eyes: a tiny AI that learns YOUR gaze

## Goal
Instead of fixed numbers deciding when you're "looking left", you **train a small machine-learning model on your own eyes**. You give it examples (look left, center, right); it learns your personal gaze and then classifies where you look. It's the same idea as the Day-1 perceptron: learn from labeled data, then predict.

## Run it
```
python train_eyes.py
```
Look where the window says (LEFT / CENTER / RIGHT) for a few seconds each. At the end it prints the gaze it learned for each direction, its **accuracy** on your data, and saves `eye_model.json`. From then on, `play.py` automatically uses YOUR model (you'll see `[eyes] using your trained eye model`).

## How it works (the ML part)
<details>
<summary>Read how the model works</summary>

- **Data**: while you look in each direction, the program records `horizontal_ratio` (a number from GazeTracking: ~0.0 looking right, ~0.5 center, ~1.0 looking left). Each recorded number is a *labeled example*.
- **Training**: it computes the average gaze for each class → three learned values (right, center, left). Those are the model's *parameters*, learned from your data instead of hard-coded.
- **Prediction**: a new gaze is classified by which side it falls on, using thresholds placed between the learned values and scaled by `DEADZONE` (a personalized dead-zone). With one input feature this is a linear classifier — the same family as a single neuron.
- **Evaluation**: it predicts the very samples it collected and reports the % it gets right (training accuracy).

</details>

## Make it less / more sensitive

Two ways:

**Live, on the calibration bar (no re-training).** While `play.py` runs, the bottom of the camera window shows your model as a line: **LEFT / CENTER / RIGHT** zones, a yellow marker for your live gaze, and two white handles at the zone boundaries. Drag a handle to move a boundary — a narrower CENTER turns more easily, a wider CENTER is less twitchy. Changes save to `eye_model.json` instantly; press **r** to reset. This is literally editing the model's decision thresholds by hand while watching the data.

**In code (the default dead-zone).** Open `gaze/trainer.py` and change `DEADZONE`:
- bigger (e.g. `0.6`) → you must look further to turn (less sensitive),
- smaller (e.g. `0.35`) → turns more easily.

Re-run `python train_eyes.py` to apply (re-training resets any hand-dragged boundaries).
