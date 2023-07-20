import cv2
import numpy as np
from app import redis_client, video
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import time
import io
import base64
from scipy import signal
import math
import pytesseract
import face_recognition
from pathlib import Path
import orjson as json


lines_lw = None
lines_dp = {}
circles_dp = None
DEFAULT_FPS = 30
LINES_DISTANCE_SCALE = 300.0
FACE_DETECT_THRESHOLD = 0.6
FACE_DETECT_THRESHOLD_FOR_REPLACE_RAW = 0.2
FACE_DETECT_THRESHOLD_FOR_SAVE_RECENT = 0.4


def close_video():
    global video
    if video is not None:
        time.sleep(0.5) # wait until last frame done
        video.release()
        cv2.destroyAllWindows()
        video = None


def open_video(video_fileName=0):
    global video
    time.sleep(0.5) # wait for switch pages
    if video is None:
        video = cv2.VideoCapture(video_fileName)

    if not video.isOpened():
        print("Could not open video")


def read_frame(video, sleep=0):
    time.sleep(sleep)
    ok, frame = video.read()
    if not ok:
        cv2.putText(
            frame,
            "Cannot read video file",
            (100, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
        )

    return frame


def tracker_create(frame, choice, key, table="rect"):
    tracker_types = [
        "BOOSTING",
        "MIL",
        "KCF",
        "TLD",
        "MEDIANFLOW",
        "GOTURN",
        "MOSSE",
        "CSRT",
    ]
    tracker_type = tracker_types[choice]

    if tracker_type == "BOOSTING":
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == "MIL":
        tracker = cv2.TrackerMIL_create()
    if tracker_type == "KCF":
        tracker = cv2.TrackerKCF_create()
    if tracker_type == "TLD":
        tracker = cv2.TrackerTLD_create()
    if tracker_type == "MEDIANFLOW":
        tracker = cv2.legacy.TrackerMedianFlow_create()
    if tracker_type == "GOTURN":
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == "MOSSE":
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()

    data = redis_client.hget(table, key)
    if data:
        bbox = tuple(json.loads(data).values())

    ok = tracker.init(frame, bbox)
    if not ok:
        print("tracker initialize failed")

    return tracker_type, tracker, bbox


def tracker_update(tracker_type, tracker, frame):
    ok, bbox = tracker.update(frame)
    # Draw bounding box
    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else:
        cv2.putText(
            frame,
            "Tracking failure detected",
            (100, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
        )
    # Display on frame
    cv2.putText(
        frame,
        tracker_type + " Tracker",
        (100, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (50, 170, 50),
        2,
    )
    # cv2.putText(frame, "Frames/sec : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

    # Display result
    cv2.imshow("Tracking", frame)

    return bbox


def track_box():
    open_video()
    global centroids_arr, centroids_arr_fixed, fps, bbox
    centroids_arr = []
    centroids_arr_fixed = []
    fps = DEFAULT_FPS
    bbox = ()

    # Read first frame.
    frame = read_frame(video)

    tracker_type, tracker, bbox = tracker_create(frame, choice=4, key=1)
    _, tracker_fixed, bbox_fixed = tracker_create(frame, choice=4, key=2)

    # Start clock cycles
    timer = cv2.getTickCount()

    # refresh button
    refresh = 0

    while True:
        data = redis_client.hget("refresh", "1")
        if data:
            refresh = json.loads(data)
        if refresh and timer > 1:
            refresh = 0

            tracker_type, tracker, bbox = tracker_create(frame, choice=4, key=1)
            _, tracker_fixed, bbox_fixed = tracker_create(frame, choice=4, key=2)

            # Start clock cycles
            timer = cv2.getTickCount()
            centroids_arr = []
            centroids_arr_fixed = []

        # Define centroid of bbox
        centroid = int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)
        centroids_arr.append(centroid)

        centroid_fixed = int(bbox_fixed[0] + bbox_fixed[2] / 2), int(
            bbox_fixed[1] + bbox_fixed[3] / 2
        )
        centroids_arr_fixed.append(centroid_fixed)

        # Read a new frame
        frame = read_frame(video)

        # Get frames per second
        fps = video.get(cv2.CAP_PROP_FPS)

        # Calculate Time
        t = (cv2.getTickCount() - timer) / cv2.getTickFrequency()

        bbox = tracker_update(tracker_type, tracker, frame)
        bbox_fixed = tracker_update(tracker_type, tracker_fixed, frame)

        cv2.putText(
            frame,
            "Timer: " + str(int(t)),
            (100, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (50, 170, 50),
            2,
        )

        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


def butter_lowpass(cutoff, nyq_freq, order=4):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype="lowpass")
    return b, a


def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def cal_vibration():
    global centroids_arr, centroids_arr_fixed, fps, bbox

    x_1 = np.array(centroids_arr)[:, 0]
    y_1 = np.array(centroids_arr)[:, 1]
    x_fixed = np.array(centroids_arr_fixed)[:, 0]
    y_fixed = np.array(centroids_arr_fixed)[:, 1]
    x_1 = np.abs(np.subtract(x_1, x_fixed))
    y_1 = np.abs(np.subtract(y_1, y_fixed))
    x_len = bbox[2]
    y_len = bbox[3]
    # object real size
    data = redis_client.hget("rect", "real")
    if data:
        temp = json.loads(data)
    try:
        real_x = float(temp["length"])
        real_y = float(temp["width"])
    except:
        print("lack of real dimension")
    sx = real_x / x_len
    sy = real_y / y_len
    x_axis = np.arange(len(x_1)) / fps
    y_axis = x_axis
    # plt.figure(figsize=(25,6))
    # plt.plot(x_axis, x_1*sx, label="X")
    # plt.plot(y_axis, y_1*sy, label="Y")
    cutoff_frequency = 3
    a_xf = butter_lowpass_filter(x_1 * sx, cutoff_frequency, fps / 2)
    a_yf = butter_lowpass_filter(y_1 * sy, cutoff_frequency, fps / 2)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_axis, a_xf, label="X")
    ax.plot(y_axis, a_yf, label="Y")
    ax.legend()
    ax.set_title(f"Filtered low-pass with cutoff frequency of {cutoff_frequency} Hz")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mm)")
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    plt.close(fig)

    encoded = base64.b64encode(output.getvalue()).decode("utf-8")

    return f"""<img src="data:image/png;base64,{encoded}" alt="plot">"""


# read pre-trained model and config file
net = cv2.dnn.readNet(
    "./app/static/yolov3/yolov3.weights", "./app/static/yolov3/yolov3.cfg"
)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open("./app/static/yolov3/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]


def detecting():
    open_video()

    while 1:
        frame = read_frame(video, 0.1)

        height, width, _ = frame.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=0.00392,
            size=(416, 416),
            mean=(0, 0, 0),  # subtract mean
            swapRB=True,  # swap red blue channel
            crop=False,
        )

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]  # probability of all classes
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # detection[4]: object exist confidence

                    # Rectangle coordinates
                    x = int(center_x - w / 1.8)
                    y = int(center_y - h / 1.8)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            score_threshold=0.4,  # filter lower confidences boxes
            nms_threshold=0.3,  # higher: keep more overlap boxes
        )

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                if class_ids[i] == 0:  # human -> 0
                    label = str(classes[class_ids[i]])
                    confidence = confidences[i]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(
                        frame,
                        label + " " + str(round(confidence, 2)),
                        (x, y + 30),
                        cv2.FONT_HERSHEY_PLAIN,
                        2,
                        (255, 0, 0),
                        2,
                    )

        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


def messuring_r():
    open_video()
    global lines_lw

    while 1:
        # frame = read_frame(video)
        frame = cv2.imread("./app/static/images/liquid_level.jpg")

        frame = cv2.bilateralFilter(
            frame,
            d=9,  # 直径，较大的值会导致更强的去噪效果，但也会损失一些细节
            sigmaColor=75,  # 较大的值表示更强的颜色相似性考虑
            sigmaSpace=75,  # 较大的值表示距离较远的像素对于滤波器的计算也具有较大的贡献
        )

        minVal = 0
        maxVal = 255
        data = redis_client.hget("rect", "filter")
        if data:
            temp = json.loads(data)
        try:
            minVal = float(temp["minVal"])
            maxVal = float(temp["maxVal"])
        except:
            print("lack of filter parameters")

        frame = cv2.Canny(frame, minVal, maxVal)

        threshold = 150
        data = redis_client.hget("rect", "threshold")
        if data:
            temp = json.loads(data)
        try:
            threshold = int(temp["threshold"])
        except:
            print("lack of houghLines' parameters")

        lines_lw = cv2.HoughLines(
            frame,
            rho=1,  # distance accuracy
            theta=np.pi / 180,  # degree accuracy
            threshold=threshold,  # minimum number of points to detect a line
            # srn=0, effect distance computation speed，[0, rho]
            # stn=0, effect degree computation speed，[0, pi]
            min_theta=88 / 180 * np.pi,  # 0
            max_theta=92 / 180 * np.pi,  # np.pi
        )

        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


def lines_distance(lines, benchmark):
    scale = LINES_DISTANCE_SCALE
    data = redis_client.hget("rect", "scale")
    if data:
        temp = json.loads(data)
    try:
        scale = float(temp["scale"])
    except:
        print("lack of scale")

    # if len(lines) >= 2:
    #     max_dis = max(lines, key=lambda x: x[0][0])[0][0]
    #     min_dis = min(lines, key=lambda x: x[0][0])[0][0]
    #     return (max_dis - min_dis) / scale
    if len(lines) == 1:
        return (benchmark - lines[0][0][0]) / scale


def plot_scales(frame):
    height, width = frame.shape[:2]
    thickness = 2
    color = (0, 0, 255)

    for i in range(100, width + 1, 100):
        cv2.line(
            frame, (i, height), (i, height - 10), color, thickness
        )  # draw horizontal tick marks
        # cv2.putText(frame, str(i), (i-10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness) # draw scale number

    for i in range(100, height + 1, 100):
        cv2.line(
            frame, (width, i), (width - 10, i), color, thickness
        )  # draw vertical tick marks
        # cv2.putText(frame, str(i), (width-30, i+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness) # draw scale number


def messuring_l():
    global lines_lw
    open_video()

    fps = 10
    threshold = 5
    i = 0
    length = None

    while 1:
        i += 1
        i %= 3 * fps

        # frame = read_frame(video, 1 / fps)
        frame = cv2.imread("./app/static/images/liquid_level.jpg")

        height, width = frame.shape[:2]
        benchmark = height - 10
        cv2.line(frame, (width, benchmark), (1, benchmark), (0, 0, 255), 2)

        if lines_lw is not None:
            horizontal = []
            vertical = []
            for line in lines_lw:
                # rho 表示直线距离左上角的距离，theta 表示直线的法线与 x 轴之间的夹角（弧度）
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                # x0, y0为垂足坐标
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                # (x1, y1), (x2, y2)为从垂足起沿直线向下，上移动1000的点
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                if i == 0:
                    angle = math.degrees(theta)
                    if abs(angle - 90) < threshold:
                        horizontal.append(line)
                    elif abs(angle) < threshold:
                        vertical.append(line)
                    # print("Number of horizontal lines: ", len(horizontal))
                    length = lines_distance(horizontal, benchmark)

                if length:
                    cv2.putText(
                        frame,
                        f"Length of object : {length}m",
                        (100, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 0, 255),
                        2,
                    )

        else:
            cv2.putText(
                frame,
                "no line is detected",
                (100, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
            )

        plot_scales(frame)

        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


def theta_map(theta):
    if 315 > theta > 225:
        return 0
    if theta >= 315:
        theta -= 360

    # [-45°，225] -> 0~100%
    x1, y1 = 225, 0
    x2, y2 = -45, 100

    if theta > x1:
        return 0
    else:
        return (y2 - y1) / (x2 - x1) * (theta - x1) + y1


def dial_plate_r():
    open_video()
    global circles_dp, lines_dp

    while 1:
        frame = read_frame(video, 0.5)
        frame = cv2.imread("./app/static/images/instrument_panel.jpeg")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        minRadius = 80
        data = redis_client.hget("rect", "minRadius_dial")
        if data:
            temp = json.loads(data)
        try:
            minRadius = int(temp["minRadius"])
        except:
            print("lack of real minRadius")

        circles_dp = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,  # minDist between circle centres
            # param1=100 high threshold for Canny detection, low threshold is the half
            param2=100,  # threshold of accumulator to consider as a circle
            minRadius=minRadius
            # maxRadius
        )

        if circles_dp is not None:
            circles_dp = np.uint16(np.around(circles_dp)) # convert to int for index

            for i, cir in enumerate(circles_dp[0, :]):
                roi = frame[
                    cir[1] - cir[2] : cir[1] + cir[2], cir[0] - cir[2] : cir[0] + cir[2]
                ]

                # print(lines_dp)
                try:
                    for line in lines_dp[i]:
                        x1, y1, x2, y2 = line[0]
                        cv2.line(roi, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        frame[
                            cir[1] - cir[2] : cir[1] + cir[2],
                            cir[0] - cir[2] : cir[0] + cir[2],
                        ] = roi
                except:
                    pass

        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


def dial_plate_l():
    open_video()
    global circles_dp, lines_dp
    lines_dp = {}

    while 1:
        frame = read_frame(video, 0.5)
        frame = cv2.imread("./app/static/images/instrument_panel.jpeg")

        if circles_dp is not None:
            threshold = 80
            minLineLength = 120
            data = redis_client.hget("rect", "lines_dial")
            if data:
                temp = json.loads(data)
            try:
                threshold = int(temp["threshold"])
                minLineLength = int(temp["minLineLength"])
            except:
                print("lack of real line params")

            for i, cir in enumerate(circles_dp[0, :]):
                roi = frame[
                    cir[1] - cir[2] : cir[1] + cir[2], cir[0] - cir[2] : cir[0] + cir[2]
                ]
                # print(roi.shape)
                # cv2.imwrite("x.jpg", roi)
                cv2.circle(
                    frame, (cir[0], cir[1]), cir[2], (0, 255, 0), 2, cv2.LINE_AA
                )  # draw circle
                cv2.circle(
                    frame, (cir[0], cir[1]), 2, (0, 255, 0), 2, cv2.LINE_AA
                )  # draw circle center
                roi_temp = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi_temp = cv2.Canny(roi_temp, 50, 150)

                lines_dp[i] = cv2.HoughLinesP(
                    roi_temp,
                    rho=1,
                    theta=np.pi / 180,
                    threshold=threshold,  # higher means filter more strict
                    minLineLength=minLineLength,
                    maxLineGap=10,  # two lines consider as one if their distance < maxLineGap
                )
                if lines_dp[i] is not None:
                    # print(len(lines_dp))
                    long = max(
                        lines_dp[i],
                        key=lambda x: (x[0][0] - x[0][2]) ** 2
                        + (x[0][1] - x[0][3]) ** 2,
                    )
                    # for j, line in enumerate(lines_dp):
                    x1, y1, x2, y2 = long[0]
                    theta = -math.degrees(math.atan2(y1 - y2, x1 - x2))
                    # [0, 180 and -180, 0] -> [0, 360]
                    if theta < 0:
                        theta += 360
                    # print(f"θ{j}: {theta}")
                    percent = theta_map(theta)
                    cv2.line(roi, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        roi,
                        f"{theta=:.2f},{percent=:.2f}",
                        (x1, y1),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )
                    frame[
                        cir[1] - cir[2] : cir[1] + cir[2],
                        cir[0] - cir[2] : cir[0] + cir[2],
                    ] = roi

        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


def dial_plate_square():
    open_video()

    while 1:
        frame = read_frame(video, 0.5)
        frame = cv2.imread("./app/static/images/square_gauge.jpeg")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 使用OTSU减少了不必要的轮廓检测，加速了findContours
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

        contours, _ = cv2.findContours(
            edges,
            mode=cv2.RETR_EXTERNAL,  # only detect external contours
            method=cv2.CHAIN_APPROX_SIMPLE,  # line segments only store endpoints
        )

        i = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > 1000:
                i += 1
                x, y, w, h = cv2.boundingRect(cnt)
                offset = 35

                roi = frame[y + offset : y + h - offset, x + offset : x + w - offset]
                roi_temp = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi_temp = cv2.Canny(roi_temp, 50, 150)
                lines_dp[i] = cv2.HoughLinesP(
                    roi_temp,
                    rho=1,
                    theta=np.pi / 180,
                    threshold=50,
                    minLineLength=30,
                    maxLineGap=10,
                )

                if lines_dp[i] is not None:
                    long = max(
                        lines_dp[i],
                        key=lambda x: (x[0][0] - x[0][2]) ** 2
                        + (x[0][1] - x[0][3]) ** 2,
                    )
                    x1, y1, x2, y2 = long[0]
                    theta = -math.degrees(math.atan2(y1 - y2, x1 - x2))
                    if theta < 0:
                        theta += 360
                    cv2.line(roi, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    cv2.putText(
                        roi,
                        f"{theta=:.2f}",
                        (x1, y1 + 10),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.25,
                        (0, 0, 255),
                        1,
                    )
                    frame[
                        y + offset : y + h - offset, x + offset : x + w - offset
                    ] = roi

                _ = cv2.rectangle(
                    frame,
                    (x + offset, y + offset),
                    (x + w - offset, y + h - offset),
                    (0, 255, 0),
                    2,
                )

        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


def displayBbox(im, bbox):
    if bbox is not None:
        bbox = [bbox[0].astype(int)]
        n = len(bbox[0])
        for i in range(n):
            cv2.line(im, tuple(bbox[0][i]), tuple(bbox[0][(i + 1) % n]), (0, 255, 0), 3)


detector = cv2.wechat_qrcode_WeChatQRCode(
    "./app/static/qr/detect.prototxt",
    "./app/static/qr/detect.caffemodel",
    "./app/static/qr/sr.prototxt",
    "./app/static/qr/sr.caffemodel",
)


def qrcode():
    open_video()

    while 1:
        frame = read_frame(video, 0.1)
        # is detected, 4 conner points
        res, points = detector.detectAndDecode(frame)

        # Detected outputs.
        if len(res) > 0:
            cv2.putText(
                frame,
                f"output: {res[0]}",
                (100, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
            )
            displayBbox(frame, points)
        else:
            cv2.putText(
                frame,
                "QRCode not detected",
                (100, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
            )

        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


def characters_detection():
    open_video()

    pytesseract.pytesseract.tesseract_cmd = r"E:/Tesseract-OCR/tesseract"

    while 1:
        frame = read_frame(video, 0.1)

        # '--oem '
        #  0    Legacy engine only.
        #  1    Neural nets LSTM engine only.
        #  2    Legacy + LSTM engines.
        #  3    Default, based on what is available.
        #
        #  '--psm 3' sets the Page Segmentation Mode (psm) to auto.
        # 只返回数字 -c "tessedit_char_whitelist=0123456789"
        config = "--oem 1 --psm 3"
        text = pytesseract.image_to_string(frame, lang="eng+chi_sim", config=config)
        if text:
            print(text)

        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


def process_distance(data, image_to_test_encoding, frame, path):
    if data is None:
        return False

    known_face_encodings = list(data.values())
    known_face_names = list(data.keys())
    distances = face_recognition.face_distance(
        known_face_encodings, image_to_test_encoding
    )
    min_index = np.argmin(distances)
    min_distances = distances[min_index]

    name = known_face_names[min_index].split("_")[0]

    detected = False
    if min_distances < FACE_DETECT_THRESHOLD:
        cv2.putText(
            frame,
            f"Hello, {name}",
            (100, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
        )
        time.sleep(0.1)
        detected = True

    if min_distances < FACE_DETECT_THRESHOLD_FOR_REPLACE_RAW:
        with open(f"{path}/faces.json", "wb") as f:
            data[name] = image_to_test_encoding.tolist()
            f.write(json.dumps(data))
    elif min_distances < FACE_DETECT_THRESHOLD_FOR_SAVE_RECENT:
        with open(f"{path}/faces_recent.json", "wb") as f:
            data[name] = image_to_test_encoding.tolist()
            f.write(json.dumps(data))

    return detected


path = "./app/static/faces"
images_names = set(pic.stem for pic in list(Path(f"{path}/images").glob("*.*")))

try:
    with open(f"{path}/faces.json", "rb") as f:
        data = json.loads(f.read())
except:
    data = {}

features_names = set(data.keys())
new_images = images_names - features_names

if new_images:
    for name in new_images:
        image = face_recognition.load_image_file(f"{path}/images/{name}.jpg")
        encoding = face_recognition.face_encodings(image)[0]
        data[name] = encoding.tolist()
        # print(data)

    with open(f"{path}/faces.json", "w+b") as f:
        f.write(json.dumps(data))


def face_rec():
    open_video()

    try:
        with open(f"{path}/faces_recent.json", "rb") as f:
            data_new = json.loads(f.read())
    except:
        data_new = None

    while 1:
        frame = read_frame(video, 0.1)

        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])  # C连续存储
        # 使用 face_recognition 库检测出当前帧中所有的人脸，未识别到返回[]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            if not process_distance(data_new, face_encoding, frame, path):
                process_distance(data, face_encoding, frame, path)

        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )
