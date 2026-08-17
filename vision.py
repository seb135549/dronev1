import cv2
import numpy as np
import time

from pathlib import Path


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "/home/pi/dronev1/ssd_mobilenet_v1_12.onnx"

print("Loading SSD-MobileNet model...")

net = cv2.dnn.readNetFromONNX(MODEL_PATH)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

print("SSD-MobileNet model loaded.")


# ============================================================
# COCO CLASS NAMES
# ============================================================

CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush"
]

PERSON_CLASS = 1


# ============================================================
# SETTINGS
# ============================================================

CONFIDENCE_THRESHOLD = 0.45

INPUT_WIDTH = 300
INPUT_HEIGHT = 300


# ============================================================
# VIDEO STORAGE
# ============================================================

save_dir = Path("/home/pi/flights")
save_dir.mkdir(exist_ok=True)

filename = save_dir / f"{time.strftime('%Y%m%d_%H%M%S')}.mp4"


# ============================================================
# SHARED DETECTION DATA
# ============================================================

people_xy = []

width = 0
height = 0


def get_latest_boxes():
    return people_xy


def get_feed_dimensions():
    return width, height


# ============================================================
# VIDEO RECORDING
# ============================================================

def start_video_recording(is_armed_callback):

    global people_xy
    global width
    global height

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    # 640x480 is a good starting point for the Pi 4
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Camera resolution: {width}x{height}")


    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    print(f"Camera FPS: {fps}")


    # --------------------------------------------------------
    # VIDEO WRITER
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(filename),
        fourcc,
        fps,
        (width, height)
    )

    print(f"Recording video to: {filename}")


    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret:
            print("ERROR: Failed to read camera frame.")
            break


        # Clear previous detections
        people_xy = []


        # ----------------------------------------------------
        # CREATE INPUT BLOB
        # ----------------------------------------------------

        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 127.5,
            size=(INPUT_WIDTH, INPUT_HEIGHT),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
            crop=False
        )


        # ----------------------------------------------------
        # RUN MODEL
        # ----------------------------------------------------

        net.setInput(blob)

        detections = net.forward()


        # ----------------------------------------------------
        # PROCESS DETECTIONS
        # ----------------------------------------------------

        # SSD-MobileNet output:
        #
        # [image_id, class_id, confidence,
        #  x_min, y_min, x_max, y_max]
        #

        for detection in detections[0, 0]:

            class_id = int(detection[1])

            confidence = float(detection[2])


            # Ignore anything that isn't a person
            if class_id != PERSON_CLASS:
                continue


            # Ignore low-confidence detections
            if confidence < CONFIDENCE_THRESHOLD:
                continue


            # ------------------------------------------------
            # CONVERT NORMALIZED COORDINATES TO PIXELS
            # ------------------------------------------------

            x1 = int(detection[3] * width)
            y1 = int(detection[4] * height)

            x2 = int(detection[5] * width)
            y2 = int(detection[6] * height)


            # ------------------------------------------------
            # KEEP BOX INSIDE IMAGE
            # ------------------------------------------------

            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))

            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))


            # ------------------------------------------------
            # CENTER OF PERSON
            # ------------------------------------------------

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)


            # ------------------------------------------------
            # STORE DETECTION
            #
            # Same format as your original YOLO program:
            #
            # x1, y1, x2, y2, cx, cy
            # ------------------------------------------------

            people_xy.append(
                (
                    x1,
                    y1,
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
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # ------------------------------------------------
            # DRAW CENTER POINT
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

            label = f"Person {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )


        # ----------------------------------------------------
        # WRITE FRAME
        # ----------------------------------------------------

        writer.write(frame)


        # ----------------------------------------------------
        # CHECK ARMED STATUS
        # ----------------------------------------------------

        if not is_armed_callback():

            print("Drone disarmed. Stopping recording.")

            break


    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    writer.release()

    print("Video recording stopped.")