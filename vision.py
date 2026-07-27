import cv2
import numpy as np
import time

from pathlib import Path
from main import armed

# Load MobileNet SSD
net = cv2.dnn.readNetFromCaffe(
    "MobileNetSSD_deploy.prototxt",
    "mobilenet_iter_73000.caffemodel"
)

# Class labels used by MobileNet SSD
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor"
]

PERSON_CLASS = 15

save_dir = Path("/home/pi/flights")
save_dir.mkdir(exist_ok=True)

filename = save_dir / f"{time.strftime('%Y%m%d_%H%M%S')}.mp4"

people_xy = []
width = 0
height = 0


def get_latest_boxes():
    return people_xy


def get_feed_dimensions():
    return width, height


def start_video_recording():

    global people_xy
    global width
    global height

    cap = cv2.VideoCapture(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    writer = cv2.VideoWriter(
        str(filename),
        fourcc,
        fps,
        (width, height)
    )

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        people_xy = []

        # Create input blob
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            scalefactor=0.007843,
            size=(300, 300),
            mean=127.5
        )

        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence < 0.45:
                continue

            class_id = int(detections[0, 0, i, 1])

            if class_id != PERSON_CLASS:
                continue

            box = detections[0, 0, i, 3:7] * np.array(
                [width, height, width, height]
            )

            x1, y1, x2, y2 = box.astype(int)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            people_xy.append((x1, y1, x2, y2, cx, cy))

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Draw centre point
            cv2.circle(
                frame,
                (cx, cy),
                5,
                (0, 0, 255),
                -1
            )

            # Draw confidence
            cv2.putText(
                frame,
                f"Person {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        writer.write(frame)

        if not armed:
            break

    cap.release()
    writer.release()