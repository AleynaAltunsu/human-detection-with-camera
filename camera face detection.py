import os
import urllib.request

# Create a directory for YOLO files if it doesn't exist
os.makedirs('yolo', exist_ok=True)

# URLs to the YOLO files
weights_url = 'https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights'
config_url = 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg'
coco_names_url = 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'

# Paths to save the files
weights_path = 'yolo/yolov4.weights'
config_path = 'yolo/yolov4.cfg'
coco_names_path = 'yolo/coco.names'

# Download the YOLOv4 weights
print("Downloading YOLOv4 weights...")
urllib.request.urlretrieve(weights_url, weights_path)
print("Downloaded YOLOv4 weights.")

# Download the YOLOv4 configuration file
print("Downloading YOLOv4 configuration file...")
urllib.request.urlretrieve(config_url, config_path)
print("Downloaded YOLOv4 configuration file.")

# Download the COCO names file
print("Downloading COCO names file...")
urllib.request.urlretrieve(coco_names_url, coco_names_path)
print("Downloaded COCO names file.")





import cv2
import numpy as np
import os

# Paths to YOLO files
weights_path = 'yolo/yolov4.weights'
config_path = 'yolo/yolov4.cfg'
coco_names_path = 'yolo/coco.names'

# Ensure the paths are correct
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"YOLO weights file not found: {weights_path}")
if not os.path.exists(config_path):
    raise FileNotFoundError(f"YOLO config file not found: {config_path}")
if not os.path.exists(coco_names_path):
    raise FileNotFoundError(f"COCO names file not found: {coco_names_path}")

# Load YOLO model
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO names
with open(coco_names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Prepare the image for the neural network
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Processing detections
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == 'person':
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Drawing bounding boxes
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), font, 1, color, 2)

    # Display the resulting frame
    cv2.imshow('Human Detection', frame)

    # Break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
