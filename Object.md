# Real-time Object Detection Workshop with YOLOv5 and OpenCV

## Overview
This workshop guides you through building a real-time object detection system using YOLOv5 and OpenCV. By the end, you'll have a working system that can detect objects through your computer's camera in real-time.

## Prerequisites
- Basic Python programming knowledge
- Computer with a webcam
- Internet connection for downloading required packages

## Workshop Sections

### Part 1: Environment Setup (15-20 minutes)
1. Install Anaconda & Jupyter
   - Download from [Anaconda's website](https://www.anaconda.com/products/distribution)
   - Follow installation instructions for your operating system
   - Launch Jupyter Notebook through Anaconda Navigator

2. Install PyTorch
   - Visit [PyTorch's website](https://pytorch.org/get-started/locally/)
   - Select your configuration (OS, package manager)
   - Run the provided installation command
   - Verify installation:
     ```python
     import torch
     print(torch.__version__)
     ```

### Part 2: Understanding the Components (10-15 minutes)
1. YOLOv5
   - Pre-trained object detection model
   - Can detect multiple objects in a single frame
   - Provides bounding boxes and class predictions

2. OpenCV
   - Handles video capture and image processing
   - Manages real-time video feed
   - Draws detection boxes and labels

### Part 3: Implementation (30-40 minutes)

#### Step 1: Import Libraries
```python
import cv2
import torch
```

#### Step 2: Setup Model
```python
# Configure device (GPU if available, otherwise CPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

#### Step 3: Real-time Detection
```python
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Perform detection
    results = model(frame)
    
    # Draw results
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result
        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                     (0, 255, 0), 2)
        # Add label
        cv2.putText(frame, f"{model.names[int(cls)]} {conf:.2f}", 
                    (int(x1), int(y1) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display output
    cv2.imshow('Real-time Object Detection', frame)
    
    # Exit on 'q' press
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
```

### Part 4: Understanding the Code (15-20 minutes)
1. Video Capture
   - `cv2.VideoCapture(0)`: Opens default camera
   - `cap.read()`: Captures individual frames

2. Object Detection
   - Model processes each frame
   - Returns coordinates and class predictions

3. Visualization
   - Draw rectangles around detected objects
   - Display class names and confidence scores
   - Show processed frame in real-time

## Troubleshooting
- If camera doesn't open, check if another application is using it
- For CUDA errors, verify PyTorch installation matches your GPU
- If detection is slow, consider using CPU version on less powerful machines

## Next Steps
- Experiment with different YOLOv5 model sizes
- Modify detection confidence thresholds
- Add custom object classes
- Save detected frames to video file

## Resources
- [YOLOv5 Documentation](https://docs.ultralytics.com/)
- [OpenCV Python Tutorials](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
- [PyTorch Getting Started](https://pytorch.org/get-started/locally/)