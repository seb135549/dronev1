import cv2
import numpy as np
import time

from pathlib import Path


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "/home/pi/dronev1/yolov5n.onnx"

net = cv2.dnn.readNetFromONNX(MODEL_PATH)

# Use CPU
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# ============================================================
# COCO CLASSES
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

PERSON_CLASS = 0


# ============================================================
# SETTINGS
# ============================================================

INPUT_SIZE = 320

CONFIDENCE_THRESHOLD = 0.40
NMS_THRESHOLD = 0.45


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

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Camera resolution: {width}x{height}")


    # --------------------------------------------------------
    # VIDEO FPS
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
            print("Failed to read camera frame.")
            break


        # Clear previous detections
        people_xy = []


        # ----------------------------------------------------
        # PREPARE IMAGE FOR YOLOv5
        # ----------------------------------------------------

        blob = cv2.dnn.blobFromImage(
            frame,
            1 / 255.0,
            (INPUT_SIZE, INPUT_SIZE),
            swapRB=True,
            crop=False
        )

        net.setInput(blob)


        # ----------------------------------------------------
        # RUN NEURAL NETWORK
        # ----------------------------------------------------

        outputs = net.forward()


        # ----------------------------------------------------
        # YOLOv5 OUTPUT
        # ----------------------------------------------------

        detections = outputs[0]


        boxes = []
        confidences = []


        # ----------------------------------------------------
        # PROCESS DETECTIONS
        # ----------------------------------------------------

        for detection in detections:

            # Objectness score
            objectness = float(detection[4])

            if objectness < CONFIDENCE_THRESHOLD:
                continue


            # Class probabilities
            class_scores = detection[5:]

            class_id = np.argmax(class_scores)

            class_confidence = float(class_scores[class_id])

            confidence = objectness * class_confidence


            # Only interested in people
            if class_id != PERSON_CLASS:
                continue


            if confidence < CONFIDENCE_THRESHOLD:
                continue


            # ------------------------------------------------
            # CONVERT BOX
            # ------------------------------------------------

            center_x = float(detection[0]) * width
            center_y = float(detection[1]) * height

            box_width = float(detection[2]) * width
            box_height = float(detection[3]) * height


            x1 = int(center_x - box_width / 2)
            y1 = int(center_y - box_height / 2)

            x2 = int(center_x + box_width / 2)
            y2 = int(center_y + box_height / 2)


            # Keep inside image
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))

            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))


            boxes.append(
                [x1, y1, x2 - x1, y2 - y1]
            )

            confidences.append(confidence)


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
        # STORE PEOPLE
        # ----------------------------------------------------

        if len(indices) > 0:

            for index in indices:

                # OpenCV versions can return different formats
                if isinstance(index, (list, tuple, np.ndarray)):
                    index = index[0]

                index = int(index)

                x, y, w, h = boxes[index]

                x2 = x + w
                y2 = y + h


                # Center of person
                cx = int((x + x2) / 2)
                cy = int((y + y2) / 2)


                # SAME FORMAT AS YOUR OLD YOLO PROGRAM
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
                # DRAW BOX
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

                label = f"Person {confidences[index]:.2f}"

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

            print("Drone disarmed. Stopping recording.")

            break


    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    writer.release()

    print("Video recording stopped.")