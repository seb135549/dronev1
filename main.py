from pymavlink import mavutil
import time
from tracking import get_pid_outputs

CONNECTION = "/dev/serial0"
BAUD = 57600


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

def wait_until_armed(master):
    print("Waiting for vehicle to arm...")

    while True:
        msg = master.recv_match(type="HEARTBEAT", blocking=True)

        if msg is None:
            continue

        if msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
            print("Vehicle is armed.")
            return

        time.sleep(0.1)

def wait_until_mode(master, mode):
    print(f"Waiting for vehicle to enter {mode} mode...")

    while True:
        msg = master.recv_match(type="HEARTBEAT", blocking=True)

        if msg is None:
            continue

        current_mode = mavutil.mode_string_v10(msg)

        if current_mode == mode:
            print(f"Vehicle is in {mode} mode.")
            return

        time.sleep(0.1)

def wait_for_takeoff(master, target_altitude, tolerance=0.5):
    while True:

        msg = master.recv_match(
            type="GLOBAL_POSITION_INT",
            blocking=True
        )

        altitude = msg.relative_alt / 1000.0

        print(f"{altitude:.2f} m")

        if altitude >= target_altitude * 0.95: #if withing 5% of target altitude, consider takeoff complete
            print("Takeoff complete")
            break

        time.sleep(0.1)


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

    master.set_mode_apm("GUIDED") #set Pixhawk to GUIDED mode and wait until; mode is switched

    wait_until_mode(master, "GUIDED") #wait until Pixhawk is in GUIDED mode before arming

    master.arducopter_arm() #arm motors and wait until armed

    wait_until_armed(master) #wait until Pixhawk is armed before sending takeoff command

    TARGET_ALT = 2  # meters

    master.mav.command_long_send( #send takeoff command to Pixhawk for TARGET_ALT meters
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0, 0, 0, 0,
        0, 0,
        TARGET_ALT
    )

    result = wait_for_ack(master, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF)

    if result != mavutil.mavlink.MAV_RESULT_ACCEPTED:
        print("Takeoff command rejected!")
        #TODO: handle rejection (e.g., retry, abort, etc.)
        return

    wait_for_takeoff(master, TARGET_ALT) #wait until drone reaches target altitude

if __name__ == "__main__":
    main()