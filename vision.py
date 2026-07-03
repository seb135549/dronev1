import cv2
from ultralytics import YOLO
import time

from pathlib import Path

model = YOLO("yolov8n.pt")  # Load a pre-trained YOLOv8 model
save_dir = Path("/home/pi/flights") #creates dirctory for saving videos if it doesn't already exist
save_dir.mkdir(exist_ok=True)
filename = save_dir / f"{time.strftime('%Y%m%d_%H%M%S')}.mp4" #filename for saved video, named timestamp of when the video was recorded

people_xy = []  # Initialize the people positions list
width, height = 0, 0  # Initialize the feed dimensions

def get_latest_boxes():
    return people_xy

def get_feed_dimensions():
    return width, height

def start_video_recording():
    cap = cv2.VideoCapture(0)  # Open the default camera

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #get width / height of frame
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))



    fourcc = cv2.VideoWriter_fourcc(*'mp4v') #fourcc code for MP4 video format

    fps = cap.get(cv2.CAP_PROP_FPS) #get the frame rate of the camera

    writer = cv2.VideoWriter(
        filename,           #filename for saved video
        fourcc,             #fourcc code for video format
        fps,                 #frame rate in fps
        (width, height)     #height / width of video in px 
    )
    
    while True:
        ret, frame = cap.read()  #Read a frame from the camera
        
        if not ret:
            break

        results = model(frame)  #Perform object detection on the frame

        annotated_frame = results[0].plot()  #Get the annotated frame with bounding boxes and labels

        people_xy = []
        
        for box in results[0].boxes:

            cls = int(box.cls[0])

            name = model.names[cls]

            if name != "person":
                continue

            x1, y1, x2, y2 = box.xyxy[0]

            cx = int((x1 + x2) / 2) #get center of bounding box
            cy = int((y1 + y2) / 2)

            people_xy.append((x1, y1, x2, y2, cx, cy))  # Append the bounding box coordinates and center to the list
            
            cv2.circle(annotated_frame, (cx, cy), 5, (0,255,0), -1)

        writer.write(annotated_frame)  #Write the annotated frame to the video file
        
        if stopCondition: #TODO: create stop condition for video recording ideally something that happens at end of drone flight such as disarming or landing
            break
        
    cap.release()
    writer.release()
