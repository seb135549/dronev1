#!/usr/bin/env python3

from pymavlink import mavutil
import time

CONNECTION = "/dev/serial0"
BAUD = 57600

MOTOR_NUMBER = 1
THROTTLE_PWM = 2000
DURATION_SECONDS = 3


def wait_for_ack(master, command, timeout=5):
    start = time.time()

    while time.time() - start < timeout:
        msg = master.recv_match(type=["COMMAND_ACK", "STATUSTEXT"], blocking=True, timeout=1)

        if msg is None:
            continue

        msg_type = msg.get_type()

        if msg_type == "STATUSTEXT":
            print(f"STATUSTEXT: {msg.text}")

        elif msg_type == "COMMAND_ACK":
            if msg.command == command:
                print(f"COMMAND_ACK result: {msg.result}")
                return msg.result

    print("No COMMAND_ACK received.")
    return None


def main():
    print("Connecting to Pixhawk...")
    master = mavutil.mavlink_connection(CONNECTION, baud=BAUD)

    print("Waiting for heartbeat...")
    master.wait_heartbeat()

    print(f"Connected to system {master.target_system}, component {master.target_component}")

    print("Requesting data stream...")
    master.mav.request_data_stream_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL,
        2,
        1
    )

    print("Make sure:")
    print("- propellers are removed")
    print("- main battery is connected")
    print("- safety switch is pressed/unlocked")
    print("- vehicle is disarmed")
    time.sleep(2)

    print(f"Sending motor test: motor {MOTOR_NUMBER}, {THROTTLE_PWM}us, {DURATION_SECONDS}s")

    command = mavutil.mavlink.MAV_CMD_DO_MOTOR_TEST

    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        command,
        0,
        MOTOR_NUMBER,          # motor number: 1, 2, 3, 4
        1,                     # throttle type: 1 = PWM in microseconds
        THROTTLE_PWM,          # PWM value to output
        DURATION_SECONDS,      # duration
        0,                     # motor count, 0 = one motor
        0,
        0
    )

    result = wait_for_ack(master, command)

    print("Done.")
    print(f"ACK result code: {result}")


if __name__ == "__main__":
    main()
