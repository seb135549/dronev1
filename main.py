from pymavlink import mavutil
import time
from tracking import update_pid_outputs
from vision import start_video_recording

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
    
    start_video_recording() #start recording video feed with bounding boxes and labels
    
    while True:
        turn_output, vertical_output, forward_output = update_pid_outputs() #get PID outputs for turning and vertical movement

        print(f"Turn Output: {turn_output:.2f}, Vertical Output: {vertical_output:.2f}, Forward Output: {forward_output:.2f}")

        #send the turn_output as yaw_rate and vertical_output as z velocity and forward_output as x velocity to Pixhawk using SET_POSITION_TARGET_LOCAL_NED message and MAV_CMD_CONDITION_YAW
        master.mav.set_position_target_local_ned_send(
            0,  # time_boot_ms (not used)
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,  # coordinate frame
            0b0000111111000111,  # type_mask (only velocities enabled)
            0, 0, 0,  # x, y, z positions (not used)
            forward_output,  # x velocity (forward/backward)
            0,  # y velocity (left/right)
            vertical_output,  # z velocity (up/down)
            0, 0, 0,  # x, y, z accelerations (not supported)
            0, 0)  # yaw, yaw_rate (not supported)
        
        direction = 1 if turn_output >= 0 else -1  # Determine direction based on the sign of turn_output
        #send the turn_output as yaw_rate to Pixhawk using MAV_CMD_CONDITION_YAW
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            0,  # confirmation
            0,  # target angle (degrees)
            abs(turn_output),  # speed (degrees/second)
            direction,  # direction (-1: counter-clockwise, 1: clockwise)
            0,  # relative offset (0: absolute angle, 1: relative angle)
            0, 0, 0)  # unused parameters

        time.sleep(0.1)  # Adjust the sleep time as needed

if __name__ == "__main__":
    main()