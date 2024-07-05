# human-detection-with-camera
uses yolo v4

# Human Detection using YOLOv4

## Project Overview

This project utilizes the YOLOv4 (You Only Look Once version 4) deep learning model for real-time human detection using live camera input. YOLOv4 is a highly efficient object detection model known for its speed and accuracy in detecting multiple objects simultaneously.

### Components

- **YOLOv4 Model**: A deep neural network trained on the COCO (Common Objects in Context) dataset, optimized for real-time object detection tasks.
  
- **OpenCV**: A computer vision library used to capture real-time video from the camera, process frames, and display the results of object detection.

### Directory Structure

```
human_detection_yolov4/
├── download_yolo_files.py
├── camera_face_detection.py
└── yolo/
    ├── coco.names
    ├── yolov4.cfg
    └── yolov4.weights
```

- **download_yolo_files.py**: Python script to download YOLOv4 weights, configuration file, and COCO names file.
- **camera_face_detection.py**: Main script that initializes the YOLOv4 model, captures webcam frames, performs human detection, and displays bounding boxes around detected humans.

## Step-by-Step Explanation

### 1. Installation and Setup

- **Clone the Repository**: Start by cloning the GitHub repository containing the project scripts.

  ```bash
  git clone https://github.com/AleynaAltunsu/human_detection_yolov4.git
  cd human_detection_yolov4
  ```

- **Download YOLOv4 Files**: Use the `download_yolo_files.py` script to download the necessary YOLOv4 model files (`yolov4.weights`, `yolov4.cfg`, and `coco.names`).

  ```bash
  python download_yolo_files.py
  ```

### 2. Loading the YOLOv4 Model

- **Model Initialization**: The YOLOv4 model is initialized using OpenCV's `cv2.dnn.readNet` function, which loads the pre-trained weights and configuration file.

  ```python
  net = cv2.dnn.readNet('yolo/yolov4.weights', 'yolo/yolov4.cfg')
  layer_names = net.getLayerNames()
  output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
  ```

### 3. Real-time Object Detection with Webcam

- **Capture Webcam Feed**: OpenCV captures frames from the webcam using `cv2.VideoCapture(0)`, where `0` indicates the default webcam.

  ```python
  cap = cv2.VideoCapture(0)
  ```

- **Process Each Frame**: For each frame captured:

  - **Preprocessing**: Resize and normalize the frame to match the input dimensions expected by YOLOv4.
  
  - **Forward Pass**: Pass the preprocessed frame through the YOLOv4 model to obtain predictions (`outs`).
  
  - **Postprocessing**: Interpret the model's output to extract bounding box coordinates, confidence scores, and class predictions.

    ```python
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    ```

  - **Draw Bounding Boxes**: Draw bounding boxes around detected humans on the original frame and label each box with the class name ("person") and confidence score.

    ```python
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), font, 1, color, 2)
    ```

### 4. Display Results

- **Real-time Display**: Display the processed frame with bounding boxes drawn around detected humans using `cv2.imshow`.

  ```python
  cv2.imshow('Human Detection', frame)
  ```

### 5. User Interaction

- **Control Loop**: The script runs continuously, processing each frame from the webcam feed until the user presses 'q' to quit the application.

  ```python
  if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  ```

## Conclusion

This project provides a practical demonstration of using YOLOv4 for real-time human detection using live camera input. It leverages the capabilities of OpenCV for capturing, processing, and displaying video frames, making it suitable for applications in surveillance, security, and human-computer interaction.
