"""FakeDoubleMotor: a pretend robot that PRINTS what it would do, with no Bluetooth.

Use it to test the whole game with MOCK = True without the SAI in front of you.
It implements the same methods as the real doubleMotor, so your code works the
same on the real robot by switching MOCK to False.
"""


class FakeDoubleMotor:
    def __init__(self, card_serial="1151"):
        self.card_serial = card_serial
        self.speed = 50
        print(f"[FAKE] doubleMotor(card_serial={card_serial}) created (no robot).")

    def connect(self, card_serial="1151", card_color=None):
        self.card_serial = card_serial
        print(f"[FAKE] connect(card_serial={card_serial}) OK (simulated).")

    def set_speed(self, speed):
        self.speed = speed
        print(f"[FAKE] set_speed({speed})")

    def run(self):
        print(f"[FAKE] run() -> FORWARD (speed={self.speed})")

    def stop(self):
        print("[FAKE] stop() -> BRAKE")

    def turn_left(self, degrees=90):
        print(f"[FAKE] turn_left({degrees})")

    def turn_right(self, degrees=90):
        print(f"[FAKE] turn_right({degrees})")

    def movement_move_tank(self, leftPercent, rightPercent):
        print(f"[FAKE] tank(L={leftPercent}, R={rightPercent})")

    def movement_set_speed(self, speed):
        self.speed = speed
        print(f"[FAKE] movement_set_speed({speed})")
