#!/usr/bin/env python3

from pymavlink import mavutil
import time

# Pixhawk connection through Raspberry Pi serial telemetry port
# Common options:
#   /dev/serial0
#   /dev/ttyAMA0
#   /dev/ttyS0
CONNECTION = "/dev/serial0"
BAUD = 57600

MOTOR_NUMBER = 1          # Motor 1 = output 1
THROTTLE_PERCENT = 10     # Keep low for testing
TEST_DURATION_SECONDS = 3


def main():
    print("Connecting to Pixhawk...")

    master = mavutil.mavlink_connection(CONNECTION, baud=BAUD)

    print("Waiting for heartbeat...")
    master.wait_heartbeat()

    print(
        f"Heartbeat from system {master.target_system}, "
        f"component {master.target_component}"
    )

    print(
        f"Testing motor {MOTOR_NUMBER} at "
        f"{THROTTLE_PERCENT}% for {TEST_DURATION_SECONDS} seconds"
    )

    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_MOTOR_TEST,
        0,
        MOTOR_NUMBER,          # param1: motor number
        0,                     # param2: throttle type, 0 = percent
        THROTTLE_PERCENT,      # param3: throttle value
        TEST_DURATION_SECONDS, # param4: duration
        0, 0, 0                # unused
    )

    time.sleep(TEST_DURATION_SECONDS + 1)

    print("Motor test command sent.")


if __name__ == "__main__":
    main()