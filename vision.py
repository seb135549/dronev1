```python
import cv2
import numpy as np
import time

from pathlib import Path
from picamera2 import Picamera2


# ============================================================
# CONSTANTS
# ============================================================

WIDTH = 640
HEIGHT = 480

# YOLOv8n standard ONNX input size
INPUT_SIZE = 640

CONFIDENCE_THRESHOLD = 0.40
NMS_THRESHOLD = 0.45

# COCO class 0 = person
PERSON_CLASS = 0


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = Path(__file__).parent / "yolov8n.onnx"

print("Loading YOLOv8n ONNX model...")
print(f"Model: {MODEL_PATH}")

net = cv2.dnn.readNetFromONNX(str(MODEL_PATH))

# CPU
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

print("YOLOv8n ONNX model loaded successfully.")


# ============================================================
# SHARED DETECTION DATA
# ============================================================

people_xy = []

width = WIDTH
height = HEIGHT


def get_latest_boxes():
    return people_xy


def get_feed_dimensions():
    return width, height


# ============================================================
# VIDEO STORAGE
# ============================================================

save_dir = Path("/home/admin/Videos")
save_dir.mkdir(parents=True, exist_ok=True)

filename = save_dir / f"{time.strftime('%Y%m%d_%H%M%S')}.mp4"


# ============================================================
# PICAMERA2 SETUP
# ============================================================

def create_camera():

    print("Starting Pi Camera 2...")

    picam2 = Picamera2()

    config = picam2.create_video_configuration(
        main={
            "size": (WIDTH, HEIGHT),
            "format": "RGB888"
        },
        controls={
            "FrameRate": 30
        }
    )

    picam2.configure(config)

    picam2.start()

    # Give the camera a moment to start
    time.sleep(2)

    print(f"Pi Camera 2 started: {WIDTH}x{HEIGHT}")

    return picam2


# ============================================================
# YOLO INFERENCE
# ============================================================

def run_yolo(frame):

    """
    Runs YOLOv8n inference on one OpenCV frame.

    Returns:
        boxes
        confidences
    """

    # --------------------------------------------------------
    # CREATE YOLO INPUT
    # --------------------------------------------------------

    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1.0 / 255.0,
        size=(INPUT_SIZE, INPUT_SIZE),
        swapRB=True,
        crop=False
    )

    net.setInput(blob)

    # --------------------------------------------------------
    # RUN INFERENCE
    # --------------------------------------------------------

    outputs = net.forward()

    # --------------------------------------------------------
    # HANDLE YOLOv8 OUTPUT
    #
    # Normal YOLOv8 output:
    #
    # (1, 84, 8400)
    #
    # 84 =
    #   4 box values
    #   80 COCO class scores
    # --------------------------------------------------------

    if isinstance(outputs, tuple):
        output = outputs[0]
    else:
        output = outputs

    if output.ndim == 3:
        output = output[0]

    # Convert:
    #
    # (84, 8400)
    #
    # into:
    #
    # (8400, 84)
    #
    if output.shape[0] < output.shape[1]:
        output = output.transpose()

    boxes = []
    confidences = []

    # --------------------------------------------------------
    # PROCESS DETECTIONS
    # --------------------------------------------------------

    for detection in output:

        # First four values:
        #
        # x center
        # y center
        # width
        # height

        x_center = float(detection[0])
        y_center = float(detection[1])

        box_width = float(detection[2])
        box_height = float(detection[3])

        class_scores = detection[4:]

        # Find class with highest confidence
        class_id = int(np.argmax(class_scores))
        class_confidence = float(class_scores[class_id])

        # Only detect people
        if class_id != PERSON_CLASS:
            continue

        if class_confidence < CONFIDENCE_THRESHOLD:
            continue

        # ----------------------------------------------------
        # SCALE FROM YOLO INPUT TO CAMERA
        # ----------------------------------------------------

        x_center *= width / INPUT_SIZE
        y_center *= height / INPUT_SIZE

        box_width *= width / INPUT_SIZE
        box_height *= height / INPUT_SIZE

        # ----------------------------------------------------
        # CONVERT CENTER FORMAT TO CORNERS
        # ----------------------------------------------------

        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)

        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        # ----------------------------------------------------
        # CLAMP TO IMAGE
        # ----------------------------------------------------

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))

        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        box_w = x2 - x1
        box_h = y2 - y1

        if box_w <= 0 or box_h <= 0:
            continue

        boxes.append([x1, y1, box_w, box_h])
        confidences.append(class_confidence)

    return boxes, confidences


# ============================================================
# VIDEO RECORDING + DETECTION
# ============================================================

def start_video_recording(is_armed_callback):

    global people_xy
    global width
    global height

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    picam2 = create_camera()

    width = WIDTH
    height = HEIGHT

    # --------------------------------------------------------
    # VIDEO WRITER
    # --------------------------------------------------------

    fps = 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(filename),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        print("ERROR: Could not open video writer.")
        picam2.stop()
        return

    print(f"Recording video to: {filename}")


    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:

        # ----------------------------------------------------
        # GET FRAME FROM PICAMERA2
        # ----------------------------------------------------

        frame = picam2.capture_array()

        if frame is None:
            print("ERROR: Could not read frame.")
            break

        # Picamera2 is configured for RGB888.
        #
        # OpenCV normally works with BGR.
        #
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR
        )

        # ----------------------------------------------------
        # CLEAR PREVIOUS DETECTIONS
        # ----------------------------------------------------

        people_xy = []

        # ----------------------------------------------------
        # RUN YOLO
        # ----------------------------------------------------

        try:

            boxes, confidences = run_yolo(frame)

        except cv2.error as e:

            print("YOLO OpenCV inference error:")
            print(e)

            break

        # ----------------------------------------------------
        # NON-MAXIMUM SUPPRESSION
        # ----------------------------------------------------

        indices = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            CONFIDENCE_THRESHOLD,
            NMS_THRESHOLD
        )

        # ----------------------------------------------------
        # PROCESS FINAL DETECTIONS
        # ----------------------------------------------------

        if len(indices) > 0:

            for index in indices:

                # OpenCV versions can return:
                #
                # [0]
                # [[0]]
                # np.array([0])
                #

                if isinstance(
                    index,
                    (list, tuple, np.ndarray)
                ):
                    index = index[0]

                index = int(index)

                x, y, w, h = boxes[index]

                x2 = x + w
                y2 = y + h

                # ------------------------------------------------
                # CENTER
                # ------------------------------------------------

                cx = int((x + x2) / 2)
                cy = int((y + y2) / 2)

                # ------------------------------------------------
                # SAVE DETECTION
                #
                # Same format as your old code:
                #
                # x1, y1, x2, y2, cx, cy
                # ------------------------------------------------

                people_xy.append(
                    (
                        x,
                        y,
                        x2,
                        y2,
                        cx,
                        cy
                    )
                )

                # ------------------------------------------------
                # DRAW BOUNDING BOX
                # ------------------------------------------------

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # ------------------------------------------------
                # DRAW CENTER
                # ------------------------------------------------

                cv2.circle(
                    frame,
                    (cx, cy),
                    5,
                    (0, 0, 255),
                    -1
                )

                # ------------------------------------------------
                # DRAW LABEL
                # ------------------------------------------------

                label = (
                    f"Person "
                    f"{confidences[index]:.2f}"
                )

                cv2.putText(
                    frame,
                    label,
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        # ----------------------------------------------------
        # WRITE VIDEO
        # ----------------------------------------------------

        writer.write(frame)

        # ----------------------------------------------------
        # CHECK ARMED STATUS
        # ----------------------------------------------------

        if not is_armed_callback():

            print(
                "Drone disarmed. "
                "Stopping video recording."
            )

            break


    # ========================================================
    # CLEANUP
    # ========================================================

    print("Stopping Pi Camera 2...")

    picam2.stop()

    writer.release()

    print("Video recording stopped.")