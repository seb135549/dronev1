from vision import get_latest_boxes, get_feed_dimensions
import time

forward_kP = 0.1  # Proportional gain for the controller
forward_kI = 0.01  # Integral gain for the controller
forward_kD = 0.05  # Derivative gain for the controller

prev_forward_error = 0
forward_integral = 0

vertical_kP = 0.1  # Proportional gain for the controller
vertical_kI = 0.01  # Integral gain for the controller
vertical_kD = 0.05  # Derivative gain for the controller

prev_vertical_error = 0
vertical_integral = 0

turn_kP = 0.1  # Proportional gain for the controller
turn_kI = 0.01  # Integral gain for the controller
turn_kD = 0.05  # Derivative gain for the controller

dt = 0.01  # Time step for the controller

prev_turn_error = 0
turn_integral = 0

turn_output = 0
vertical_output = 0
forward_output = 0

DESIRED_HEIGHT = 300  # Desired distance from the person 


def update_pid_outputs():
    latest_boxes = get_latest_boxes()
    width, height = get_feed_dimensions()
    
    if latest_boxes:
        # Calculate the center of the frame
        center_x = width / 2
        center_y = height / 2
        
        closest_person = min(latest_boxes, key=lambda box: ((box[4] - center_x) ** 2 + (box[5] - center_y) ** 2) ** 0.5) #find the closest person to the center of the frame 
        
        turn_error = closest_person[4] - center_x  # Calculate the horizontal error
        vertical_error = closest_person[5] - center_y  # Calculate the vertical error
        #calculate the forward error based on the size of the bounding box (assuming larger boxes mean closer objects)
        forward_error = DESIRED_HEIGHT - (closest_person[3] - closest_person[1])  # Height of the bounding box

        turn_integral += turn_error * dt
        turn_derivative = (turn_error - prev_turn_error) / dt
        prev_turn_error = turn_error

        vertical_integral += vertical_error * dt
        vertical_derivative = (vertical_error - prev_vertical_error) / dt
        prev_vertical_error = vertical_error
        
        forward_integral += forward_error * dt
        forward_derivative = (forward_error - prev_forward_error) / dt
        prev_forward_error = forward_error
        
        #anti-windup for integral term
        if abs(turn_integral) > 100:
            turn_integral = 100 * (turn_integral / abs(turn_integral))  # Limit the integral term to prevent windup
        
        if abs(vertical_integral) > 100:
            vertical_integral = 100 * (vertical_integral / abs(vertical_integral))  # Limit the integral term to prevent windup
            
        if abs(forward_integral) > 100:
            forward_integral = 100 * (forward_integral / abs(forward_integral))  # Limit the integral term to prevent windup

        turn_output = turn_kP * turn_error + turn_kI * turn_integral + turn_kD * turn_derivative
        vertical_output = vertical_kP * vertical_error + vertical_kI * vertical_integral + vertical_kD * vertical_derivative
        forward_output = forward_kP * forward_error + forward_kI * forward_integral + forward_kD * forward_derivative
        
        #clamp the outputs to a reasonable range
        turn_output = max(min(turn_output, 30), -30)  # Clamp turn output to [-30, 30] degrees per second
        vertical_output = max(min(vertical_output, 2), -2)  # Clamp vertical output to [-2, 2] m/s
        forward_output = max(min(forward_output, 2), -2)  # Clamp forward output to [-2, 2] m/s
        
        return turn_output, vertical_output, forward_output
    
    time.sleep(dt)  # Wait for the next iteration

    
            
        