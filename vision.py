import cv2
import numpy as np
import time

from pathlib import Path


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = Path(__file__).parent / "yolov8n.onnx"

print("Loading YOLOv8n ONNX model...")

net = cv2.dnn.readNetFromONNX(str(MODEL_PATH))

# Use CPU
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

print("YOLOv8n ONNX model loaded successfully.")


# ============================================================
# YOLO SETTINGS
# ============================================================

INPUT_SIZE = 320

CONFIDENCE_THRESHOLD = 0.40
NMS_THRESHOLD = 0.45

# COCO class 0 = person
PERSON_CLASS = 0


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

    # 640x480 keeps the camera workload reasonable
    # for the Raspberry Pi 4B.
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
            print("ERROR: Could not read frame.")
            break


        # Clear detections from previous frame
        people_xy = []


        # ----------------------------------------------------
        # CREATE YOLO INPUT
        # ----------------------------------------------------

        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1 / 255.0,
            size=(INPUT_SIZE, INPUT_SIZE),
            swapRB=True,
            crop=False
        )

        net.setInput(blob)


        # ----------------------------------------------------
        # RUN YOLO
        # ----------------------------------------------------

        outputs = net.forward()


        # ----------------------------------------------------
        # GET OUTPUT
        # ----------------------------------------------------

        output = outputs[0]

        # YOLOv8 OpenCV output is normally:
        #
        # (1, 84, 8400)
        #
        # We want:
        #
        # (8400, 84)
        #
        if len(output.shape) == 3:
            output = output[0]

        if output.shape[0] < output.shape[1]:
            output = output.transpose()


        # ----------------------------------------------------
        # DETECTIONS
        # ----------------------------------------------------

        boxes = []
        confidences = []


        for detection in output:

            # YOLOv8 format:
            #
            # x
            # y
            # width
            # height
            # class scores...
            #

            x_center = detection[0]
            y_center = detection[1]

            box_width = detection[2]
            box_height = detection[3]

            class_scores = detection[4:]


            # ------------------------------------------------
            # FIND BEST CLASS
            # ------------------------------------------------

            class_id = int(np.argmax(class_scores))

            class_confidence = float(
                class_scores[class_id]
            )


            # ------------------------------------------------
            # ONLY LOOK FOR PEOPLE
            # ------------------------------------------------

            if class_id != PERSON_CLASS:
                continue


            if class_confidence < CONFIDENCE_THRESHOLD:
                continue


            # ------------------------------------------------
            # SCALE BOX TO CAMERA RESOLUTION
            # ------------------------------------------------

            x_center *= width / INPUT_SIZE
            y_center *= height / INPUT_SIZE

            box_width *= width / INPUT_SIZE
            box_height *= height / INPUT_SIZE


            # Convert center/width/height
            # into corner coordinates

            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)

            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)


            # ------------------------------------------------
            # CLAMP BOX
            # ------------------------------------------------

            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))

            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))


            # ------------------------------------------------
            # STORE BOX
            # ------------------------------------------------

            boxes.append(
                [x1, y1, x2 - x1, y2 - y1]
            )

            confidences.append(
                class_confidence
            )


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

                # Handle different OpenCV versions
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
                # Same format as your original YOLO code:
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

    cap.release()
    writer.release()

    print("Video recording stopped.")