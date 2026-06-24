# Drive the circuit with your gaze

## Goal
Program the robot to drive with your eyes through the obstacle course:

- Look **center** → drive forward
- Look **left** → turn left
- Look **right** → turn right
- **Blink several times in a row** → brake
- No face → stop

Open `play.py`, set your `CARD_SERIAL` and fill in the `decide()` block (the `TODO` lines).

## Tips
<details>
<summary>Read tips</summary>

- The robot moves with `motor.movement_move_tank(left, right)`: each value goes from **-100 to 100** and is the speed of the left and right wheel.
- To go **straight**, both wheels at the same speed: `motor.movement_move_tank(SPEED, SPEED)`.
- To **turn**, one wheel faster than the other. Which one should be faster to turn left?
- To **brake / stop**: `motor.stop()`.
- The `"none"` state means the camera can't see your eyes. For safety, it's best to stop.
- Driving an obstacle course is all about control: lower `SPEED` and a smaller `TURN_INNER` (at the top of `play.py`) make tight turns much easier.

</details>

<details>
<summary>Example solution</summary>

```python
def decide(motor, state):
    if state == "center":
        motor.movement_move_tank(SPEED, SPEED)        # forward
    elif state == "left":
        motor.movement_move_tank(TURN_INNER, SPEED)   # turn left
    elif state == "right":
        motor.movement_move_tank(SPEED, TURN_INNER)   # turn right
    else:  # "none" (no face); the blink-brake is handled automatically by the game loop
        motor.stop()
```

</details>

## Try it without a robot first
Before connecting the SAI, keep `MOCK = True` and run:

```
python play.py
```

In the console you'll see what the robot would do (`[FAKE] tank(...)`) as you move your eyes. When the behavior is correct, set `MOCK = False` and connect the real robot.

## The challenge
Lay out a circuit on the floor with obstacles (cones, cups, blocks). Set a start and a finish line, then drive the robot around it with your eyes without knocking obstacles over. Time your runs and try to beat your record, or compete for the fewest obstacles hit.
