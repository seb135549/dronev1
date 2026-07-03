from vision import get_latest_boxes, get_feed_dimensions
import time

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

def get_pid_outputs():
    return turn_output, vertical_output

while True:
    latest_boxes = get_latest_boxes()
    width, height = get_feed_dimensions()
    
    if latest_boxes:
        # Calculate the center of the frame
        center_x = width / 2
        center_y = height / 2
        
        closest_person = min(latest_boxes, key=lambda box: ((box[0] - center_x) ** 2 + (box[1] - center_y) ** 2) ** 0.5) #find the closest person to the center of the frame 
        
        turn_error = closest_person[0] - center_x  # Calculate the horizontal error
        vertical_error = closest_person[1] - center_y  # Calculate the vertical error
        
        turn_integral += turn_error * dt
        turn_derivative = (turn_error - prev_turn_error) / dt
        prev_turn_error = turn_error

        vertical_integral += vertical_error * dt
        vertical_derivative = (vertical_error - prev_vertical_error) / dt
        prev_vertical_error = vertical_error
        
        #anti-windup for integral term
        if abs(turn_integral) > 100:
            turn_integral = 100 * (turn_integral / abs(turn_integral))  # Limit the integral term to prevent windup
        
        if abs(vertical_integral) > 100:
            vertical_integral = 100 * (vertical_integral / abs(vertical_integral))  # Limit the integral term to prevent windup

        turn_output = turn_kP * turn_error + turn_kI * turn_integral + turn_kD * turn_derivative
        vertical_output = vertical_kP * vertical_error + vertical_kI * vertical_integral + vertical_kD * vertical_derivative
    
    time.sleep(dt)  # Wait for the next iteration

    
            
        