"""run_game: the main game loop.

You do NOT need to edit this. Your logic goes in the decide() function in play.py.
The loop:
  1. reads the eyes,
  2. computes the state ("center"/"left"/"right"/"brake"/"none"),
  3. if you are blinking a lot (state == "brake") it STOPS the robot itself
     (the handbrake always works, no matter what decide() does);
     otherwise it calls your decide(motor, state) ONLY when the state changes
     (so we don't flood Bluetooth with repeated commands),
  4. on exit, stops the robot no matter what.
"""

import time


def run_game(eyes, motor, decide, sleep_s=0.02):
    last_state = None
    try:
        while True:
            key = eyes.update()
            if key in (ord('q'), 27):  # 'q' or ESC to quit
                break
            state = "brake" if eyes.is_braking() else eyes.direction()
            if state != last_state:
                if state == "brake":
                    motor.stop()       # blinking = handbrake: ALWAYS stop
                else:
                    decide(motor, state)
                last_state = state
            time.sleep(sleep_s)
    finally:
        motor.stop()


__all__ = ["run_game"]
