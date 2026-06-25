"""
PLAY: drive the SAI robot with your EYES through the obstacle course.

How to play:
  - Look CENTER to go forward.
  - Look LEFT / RIGHT to turn.
  - Blink SEVERAL times in a row to BRAKE (this is automatic — you don't program it).
  - Drag the handles on the bottom calibration bar to adjust your turns (r = reset).
  - Your mission: steer around the obstacles and finish the circuit.

Steps:
  1) Set your CARD_SERIAL (the number printed on your SAI card).
  2) Keep MOCK = True to test WITHOUT a robot (it prints what it would do).
  3) Fill in the decide() block following the hints (the TODO lines).
  4) When it works in MOCK, set MOCK = False and drive for real.

You only need to edit THIS file.
"""

from gaze import Eyes
from robot import get_motor
from robot.game import run_game

# ===================== CONFIGURE THIS =====================
CARD_SERIAL = "3664"   # the serial of YOUR SAI card (the number on it) -- CHANGE IT
MOCK = False           # True = no robot (prints);  False = real robot
# =========================================================

# Speeds -- tune these to your circuit (smaller = slower & easier to control)
SPEED = 25             # forward speed
TURN_INNER = 7         # inner wheel while turning (smaller = sharper turn)


def decide(motor, state):
    """Receives the state of your eyes and moves the robot.

    state is one of: "center", "left", "right" or "none" (no face).
    (Blinking a lot = BRAKE; the game loop stops the robot for you, so it never
    reaches here -- you don't need to handle it.)

    Move the robot with:
        motor.movement_move_tank(LEFT, RIGHT)   # speeds from -100 to 100
        motor.stop()
    """
    # ----- center=forward, left/right=turn, no face=stop -----
    if state == "center":
        motor.movement_move_tank(SPEED, SPEED)        # forward
    elif state == "left":
        motor.movement_move_tank(TURN_INNER, SPEED)   # turn left
    elif state == "right":
        motor.movement_move_tank(SPEED, TURN_INNER)   # turn right
    else:      # "none" (no face)
        motor.stop()


eyes = Eyes()
motor = get_motor(mock=MOCK, card_serial=CARD_SERIAL)
try:
    run_game(eyes, motor, decide)
finally:
    eyes.release()
