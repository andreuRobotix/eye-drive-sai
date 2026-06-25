# This is a proprietary development of ROBOTIX Hands-on Learning, protected and not to be sold or transferred to third parties.
"""
Find the CARD SERIAL of nearby SAI / LEGO Education devices.

Use this when connecting fails ("Could not find device matching Card serial ...").
It scans Bluetooth and prints the serial each device is advertising, so you can
put the right number in CARD_SERIAL.

Important before running:
  - Turn the SAI device ON.
  - Make sure it is NOT connected to anything else (close the Blockly browser
    tab / any other app; if it's paired in Windows Bluetooth settings, remove it).
  - Keep it close to the computer.

Run from the project folder:
    python solutions/tests/find_serial.py
"""

import asyncio

from bleak import BleakScanner

LEGO_COMPANY_ID = 0x0397  # LEGO manufacturer id in the BLE advertisement
SCAN_SECONDS = 12

# product_id -> device type (so you can tell the motor from the color sensor)
PRODUCT_NAMES = {
    512: "Single Motor",
    513: "Double Motor",
    514: "Color Sensor",
    515: "Controller",
}


def decode_lego(manufacturer_data):
    """Return (product_id, card_color, card_serial, raw_hex) or None."""
    data = manufacturer_data.get(LEGO_COMPANY_ID)
    if not data or len(data) < 5:
        return None
    product_id = (data[0] << 8) | data[1]
    card_color = data[2]
    card_serial = data[3] | (data[4] << 8)
    return product_id, card_color, card_serial, bytes(data).hex()


async def main():
    print(f"Scanning Bluetooth for {SCAN_SECONDS}s... (turn the SAI on and keep it close)\n")
    lego = {}
    others = {}

    def callback(device, adv):
        info = decode_lego(adv.manufacturer_data or {})
        name = adv.local_name or device.name
        if info:
            lego[device.address] = (name, device.address, adv.rssi, *info)
        else:
            others[device.address] = (name, device.address, adv.rssi)

    scanner = BleakScanner(detection_callback=callback)
    await scanner.start()
    await asyncio.sleep(SCAN_SECONDS)
    await scanner.stop()

    if lego:
        print(f"Found {len(lego)} LEGO/SAI device(s):")
        for name, addr, rssi, pid, color, serial, raw in lego.values():
            ptype = PRODUCT_NAMES.get(pid, f"product_id={pid}")
            # card names can contain emoji (e.g. a colour dot) that the Windows
            # console can't print -> strip to ASCII so we never crash on output
            safe_name = str(name).encode("ascii", "replace").decode("ascii")
            print(f"  >> CARD_SERIAL = \"{serial:04d}\"   [{ptype}]   (card_color={color}  rssi={rssi}  name={safe_name})")
        print('\nPut the [Double Motor] serial in CARD_SERIAL and run again.')
    else:
        print("No LEGO/SAI device is advertising right now.")
        print("  - Is the SAI on?")
        print("  - Is it still connected to the Blockly app or another program? Disconnect it.")
        print("  - If it's paired in Windows Bluetooth settings, remove the pairing and retry.")
        print(f"\n(For reference: {len(others)} other Bluetooth devices were seen, so scanning works.)")


if __name__ == "__main__":
    asyncio.run(main())
