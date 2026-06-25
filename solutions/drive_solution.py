# This is a proprietary development of ROBOTIX Hands-on Learning, protected and not to be sold or transferred to third parties.
"""
FULL SOLUTION - drive the obstacle circuit.

Drive the robot with your eyes:
    center -> forward
    left   -> turn left
    right  -> turn right
    repeated blinking / no face -> brake / stop
"""

import pathlib
import sys

# allows running this file directly from the project folder
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from gaze import Eyes
from robot import get_motor
from robot.game import run_game

CARD_SERIAL = "XXXX"   # serial of YOUR robot (the number on the SAI card) -- CHANGE IT
MOCK = False           # False = real robot;  True = no robot (prints to console)

SPEED = 25             # forward speed
TURN_INNER = 7         # inner wheel while turning (smaller = sharper turn)


def decide(motor, state):
    if state == "center":
        motor.movement_move_tank(SPEED, SPEED)        # forward
    elif state == "left":
        motor.movement_move_tank(TURN_INNER, SPEED)   # turn left
    elif state == "right":
        motor.movement_move_tank(SPEED, TURN_INNER)   # turn right
    else:  # "none" (no face); the blink-brake is handled automatically by the game loop
        motor.stop()


eyes = Eyes()
motor = get_motor(mock=MOCK, card_serial=CARD_SERIAL)
try:
    run_game(eyes, motor, decide)
finally:
    eyes.release()
