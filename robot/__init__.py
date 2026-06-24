"""Access to the SAI robot.

get_motor() gives you a doubleMotor ready to use:
  - mock=True  -> FakeDoubleMotor (prints to the console, no Bluetooth)
  - mock=False -> the real lelib doubleMotor, already connected by card_serial
"""


def get_motor(mock=False, card_serial="1151"):
    if mock:
        from .fake import FakeDoubleMotor
        return FakeDoubleMotor(card_serial=card_serial)
    # lazy import: in mock mode you don't need legoeducation/bleak installed
    from .lelib import doubleMotor
    motor = doubleMotor()
    motor.connect(card_serial=card_serial)
    return motor


__all__ = ["get_motor"]
