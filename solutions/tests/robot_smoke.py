# This is a proprietary development of ROBOTIX Hands-on Learning, protected and not to be sold or transferred to third parties.
"""
STAGE 3 - test ONLY the robot (no camera needed).

>>> LIFT THE WHEELS OFF THE GROUND before running it. <<<

Use it to fix the tank SIGNS: see which one is FORWARD and which way each
combination curves. Write the good values into play.py
(SPEED / TURN_INNER and the turn signs).

Run from the project folder:
    python solutions/tests/robot_smoke.py
"""

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from robot import get_motor

CARD_SERIAL = "3664"   # change to YOUR card serial


def demo(name, left, right, t=1.5):
    print(name)
    motor.movement_move_tank(left, right)
    time.sleep(t)
    motor.stop()
    time.sleep(0.5)


motor = get_motor(mock=False, card_serial=CARD_SERIAL)
motor.set_speed(40)

print("run() -> forward")
motor.run()
time.sleep(1.5)
motor.stop()
time.sleep(0.5)

demo("tank(40, 40)  -> should go straight", 40, 40)
demo("tank(10, 45)  -> curve one way", 10, 45)
demo("tank(45, 10)  -> curve the other", 45, 10)
demo("tank(-40, 40) -> spin in place", -40, 40)
demo("tank(40, -40) -> spin the other way", 40, -40)

print("done")
