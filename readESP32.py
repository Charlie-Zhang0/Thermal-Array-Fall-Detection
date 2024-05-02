import serial
from datetime import datetime
import ast
import numpy as np
import cv2
import paho.mqtt.client as mqtt
from collections import deque


port = 'COM3'
baud_rate = 921600

# All in terms of frames, FPS ~= 6
input_fifo_size = 3
detect_fifo_size = 3 # How "Fast" the fall is
background_delay = 10 # Sample length of background
demo_time = background_delay
Threshold = 24 # "Hot or "Cold"

fall_portion = 0.30 # How "Big" the fall is
smallBox = (6, 2) # Minimum size to be considered as a human
minIoU = 0.25 # Minimum portion of overlaping between frames



class ReadLine:
    def __init__(self, s):
        self.buf = bytearray()
        self.s = s

    def readline(self):
        i = self.buf.find(b"\n")
        if i >= 0:
            r = self.buf[:i+1]
            self.buf = self.buf[i+1:]
            return r
        while True:
            i = max(1, min(2048, self.s.in_waiting))
            data = self.s.read(i)
            i = data.find(b"\n")
            if i >= 0:
                r = self.buf + data[:i+1]
                self.buf[0:] = data[i+1:]
                return r
            else:
                self.buf.extend(data)


def calculate_iou(box1, box2):
    x1, y1, w1, h1, _ = box1
    x2, y2, w2, h2 = box2

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)

    inter_area = max(inter_x2 - inter_x1, 0) * max(inter_y2 - inter_y1, 0)
    box1_area = w1 * h1
    box2_area = w2 * h2

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area


def draw_boxes(image, filtered_boxes, color=(0, 255, 0)):
    for (x, y, w, h, isTarget) in filtered_boxes:
        x = 32 * x
        y = 32 * y
        w = 32 * w
        h = 32 * h
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        center = (x + w // 2, y + h // 2)

        cv2.rectangle(image, top_left, bottom_right, color, 5)
        cv2.circle(image, center, radius=5, color=color, thickness=5)

        head = ""
        if isTarget:
            head = "Target"

        # Draw the label
        label_position = (x, y-10)
        cv2.putText(image, head, label_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


def filter_boxes_by_size(bounding_boxes, min_size):
    filtered_boxes = []
    for x, y, w, h in bounding_boxes:
        if w > min_size[1] and h > min_size[0]:
            filtered_boxes.append((x, y, w, h))
    return filtered_boxes


def average_fifo(fifo):
    if len(fifo) == 0:
        return None
    sum_array = np.zeros(fifo[0].shape)
    for temp_array in fifo:
        sum_array += temp_array
    return sum_array / len(fifo)


input_fifo = deque(maxlen=input_fifo_size)
background = np.zeros((24, 32, background_delay))
i,j = 0,0
ser = serial.Serial(port, baud_rate, timeout=1)
box_buffer = ()  # (x, y, w, h)
detect_fifo = deque(maxlen=detect_fifo_size)
mqttBroker = "172.20.10.2"
client = mqtt.Client(client_id="board")
client.connect(mqttBroker, 1883)

client.publish("flask/fall","hello")



while True:

    # start_time = time.time()
    data = ReadLine(ser).readline()

    # end_time = time.time()
    # print(1/(end_time-start_time))

    if len(data) != 0:
        try:
            msg_str = str(data.decode('utf-8'))
            dict_data = ast.literal_eval(msg_str)
            Onboard_timestamp = int(dict_data["loc_ts"])
            Ambient_temperature = float(dict_data["AT"])
            Detected_temperature = np.array(
                dict_data["data"]).reshape((24, 32))
        except:
            continue

    input_fifo.append(Detected_temperature)
    if i < background_delay:
        background[:, :, i] = Detected_temperature
        i += 1
        ira_expand = Detected_temperature
    else:
        # Time-Domain Windowing
        Detected_temperature = average_fifo(input_fifo)

        # Background Removal
        Detected_temperature -= np.average(background, axis=2)
        if j > demo_time:
        # Naive CFAR
        # This if is for demo
            Detected_temperature = np.where(Detected_temperature < Threshold, 0, 1)
            # Graphics coloring
            ira_expand = Detected_temperature*160 + 32
        else:
            ira_expand = Detected_temperature + 36
            j+=1

    # Rectangle Bounding
    Detected_temperature = (Detected_temperature).astype(np.uint8)
    contours, _ = cv2.findContours(
        Detected_temperature, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = [cv2.boundingRect(c) for c in contours]

    # Discard small boxes
    # Bool IsTarget = False
    filtered_boxes = [(x, y, w, h, False) for (x, y, w, h) in filter_boxes_by_size(
        bounding_boxes, min_size=smallBox)]

    # Target Tracking
    if filtered_boxes != []:
        # First encountering Objs
        if box_buffer == ():
            # The target is the biggest component that appears first
            x, y, w, h, _ = max(
                filtered_boxes, key=lambda box: box[2] * box[3])
            box_buffer = (x, y, w, h)

        # Need to determine which of the detected boxes is the previous target
        else:
            max_iou = 0
            TargetBox = None
            for current_box in filtered_boxes:
                iou = calculate_iou(current_box, box_buffer)
                if iou > max_iou and iou > minIoU:
                    max_iou = iou
                    TargetBox = current_box
            # Target is gone
            if max_iou == 0:
                box_buffer = ()
                detect_fifo.clear()
                
            else:
                x, y, w, h, _ = TargetBox
                box_buffer = (x, y, w, h)
                filtered_boxes[0] = (x, y, w, h, True)
                filtered_boxes = [(a, b, c, d, True) if (a, b, c, d) == (
                    x, y, w, h) else (a, b, c, d, e) for (a, b, c, d, e) in filtered_boxes]
                detect_fifo.append((y,h))
    # Objs are all gone
    else:
        box_buffer = ()
        detect_fifo.clear()
        
    if len(detect_fifo) == detect_fifo_size:
        if ((detect_fifo[-1][0] - detect_fifo[0][0]) / detect_fifo[0][1]) > fall_portion:
            print((detect_fifo[-1][0] - detect_fifo[0][0])/ detect_fifo[0][1]) 
            c = datetime.now()
            current_time = c.strftime('%H:%M:%S')
            current_date = c.strftime("%d/%m/%Y")
            client.publish("flask/fall", (current_date+" "+current_time))

    # Graphics
    ira_expand = np.repeat(ira_expand, 32, 0)
    ira_expand = np.repeat(ira_expand, 32, 1)
    ira_img_colored = cv2.applyColorMap(
        (ira_expand).astype(np.uint8), cv2.COLORMAP_JET)
    draw_boxes(ira_img_colored, filtered_boxes)
    cv2.namedWindow('All', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('All', ira_img_colored)

    key = cv2.waitKey(1)
    if key == 27:
        break
