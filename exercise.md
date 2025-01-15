# Object Detection with Automatic Image Saving

## Overview
This project implements real-time object detection using YOLOv5 and OpenCV, with the additional functionality of automatically saving detected objects as individual images. Each detected object is saved with its class name and a unique identifier.

## Features
- Real-time object detection
- Automatic cropping of detected objects
- Organized file saving system
- Unique identification for each detected object
- Confidence threshold filtering

## Prerequisites
- Python 3.7+
- PyTorch
- OpenCV
- YOLOv5
- NumPy

## Installation
```bash
pip install torch torchvision
pip install opencv-python
pip install numpy
```

## Implementation

### Directory Setup
```python
import os
import cv2
import torch
import numpy as np
from datetime import datetime

# Create base directory for saved images
def create_directories():
    base_dir = "detected_objects"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    return base_dir

# Create class-specific directories
def create_class_directory(base_dir, class_name):
    class_dir = os.path.join(base_dir, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    return class_dir
```

### Main Detection and Saving Code
```python
def detect_and_save_objects():
    # Initialize model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model = model.to(device)
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    # Create base directory
    base_dir = create_directories()
    
    # Initialize counter for unique IDs
    object_counters = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Perform detection
        results = model(frame)
        
        # Process each detection
        for result in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = result
            
            # Convert to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Get class name
            class_name = model.names[int(cls)]
            
            # Only process if confidence is above threshold
            if conf > 0.5:  # You can adjust this threshold
                # Create class directory if it doesn't exist
                class_dir = create_class_directory(base_dir, class_name)
                
                # Update counter for this class
                if class_name not in object_counters:
                    object_counters[class_name] = 0
                object_counters[class_name] += 1
                
                # Crop the detected object
                object_img = frame[y1:y2, x1:x2]
                
                # Generate unique filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{class_name}_{object_counters[class_name]}_{timestamp}.jpg"
                filepath = os.path.join(class_dir, filename)
                
                # Save the cropped image
                cv2.imwrite(filepath, object_img)
                
                # Draw rectangle and label on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, 
                          f"{class_name} {conf:.2f}", 
                          (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.5,
                          (0, 255, 0),
                          2)
        
        # Display the frame
        cv2.imshow('Object Detection', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

# Run the detection
if __name__ == "__main__":
    detect_and_save_objects()
```

## File Structure
```
detected_objects/
├── person/
│   ├── person_1_20240115_121501.jpg
│   ├── person_2_20240115_121502.jpg
├── car/
│   ├── car_1_20240115_121503.jpg
│   ├── car_2_20240115_121504.jpg
├── dog/
    ├── dog_1_20240115_121505.jpg
```

## Features Explanation

### 1. Object Detection
- Uses YOLOv5 for real-time object detection
- Processes each frame from the webcam
- Applies confidence threshold filtering (default: 0.5)

### 2. Image Saving System
- Creates organized directory structure
- Saves each detected object as a separate image
- Naming convention: `{class_name}_{id}_{timestamp}.jpg`
- Automatically creates directories for new object classes

### 3. Unique Identification
- Maintains separate counters for each object class
- Includes timestamps in filenames
- Prevents filename conflicts

### 4. Error Handling
- Checks for directory existence
- Validates frame capture
- Ensures proper cleanup of resources

## Customization Options
1. Adjust confidence threshold:
   ```python
   if conf > 0.5:  # Change this value
   ```

2. Modify save directory:
   ```python
   base_dir = "detected_objects"  # Change this name
   ```

3. Change filename format:
   ```python
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Modify format
   ```

## Common Issues and Solutions

1. **Memory Usage**
   - If running for long periods, consider periodic cleanup
   - Implement maximum image count per class

2. **Storage Management**
   ```python
   # Add to the main loop to check storage
   if os.path.getsize(class_dir) > 1_000_000_000:  # 1GB limit
       cleanup_old_images(class_dir)
   ```

3. **Performance Optimization**
   - Reduce frame processing frequency
   - Implement batch processing
   - Use GPU acceleration when available

## Next Steps
- Add image compression options
- Implement object tracking to avoid duplicate saves
- Add metadata logging
- Create image preview functionality