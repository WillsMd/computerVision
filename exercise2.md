# Object Detection Workshop Exercise

## Exercise Overview
In this exercise, you'll build a smart security system that can detect and track specific objects of interest (e.g., people, bags, laptops) and generate alerts based on certain conditions.

## Exercise Requirements

### Basic Requirements (Must Complete)
1. Detect people and bags in real-time video feed
2. Save images when both a person AND a bag are detected in the same frame
3. Add timestamp and detection confidence to saved images
4. Implement a simple counting system for people and bags

### Advanced Requirements (Optional)
1. Generate alerts when specific conditions are met
2. Track objects across multiple frames
3. Create a summary report of detections
4. Implement a basic user interface

## Starter Code
```python
import cv2
import torch
import os
from datetime import datetime

# TODO: Initialize your model and video capture
# TODO: Create necessary directories
# TODO: Implement detection logic
# TODO: Add counting system
# TODO: Save images when conditions are met
```

## Detailed Solution Guide

### Step 1: Setup and Initialization
```python
import cv2
import torch
import os
from datetime import datetime
import numpy as np
from pathlib import Path

class SecuritySystem:
    def __init__(self):
        # Initialize model
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.model = self.model.to(self.device)
        
        # Initialize counters and storage
        self.counts = {
            'person': 0,
            'bag': 0,
            'alerts': 0
        }
        
        # Create directories
        self.base_dir = Path('security_logs')
        self.alert_dir = self.base_dir / 'alerts'
        self.report_dir = self.base_dir / 'reports'
        
        self._create_directories()
        
    def _create_directories(self):
        """Create necessary directories for logging"""
        self.base_dir.mkdir(exist_ok=True)
        self.alert_dir.mkdir(exist_ok=True)
        self.report_dir.mkdir(exist_ok=True)
```

### Step 2: Detection and Analysis
```python
    def analyze_frame(self, frame):
        """Analyze a single frame for objects of interest"""
        results = self.model(frame)
        
        # Initialize detection flags
        detections = {
            'person': False,
            'bag': False,
            'person_coords': [],
            'bag_coords': []
        }
        
        # Process detections
        for result in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = result
            class_name = self.model.names[int(cls)]
            
            if conf > 0.5:  # Confidence threshold
                if class_name == 'person':
                    detections['person'] = True
                    detections['person_coords'].append(
                        (int(x1), int(y1), int(x2), int(y2), float(conf))
                    )
                elif class_name in ['backpack', 'handbag', 'suitcase']:
                    detections['bag'] = True
                    detections['bag_coords'].append(
                        (int(x1), int(y1), int(x2), int(y2), float(conf))
                    )
                
                # Draw bounding box
                self._draw_box(frame, x1, y1, x2, y2, class_name, conf)
        
        return frame, detections
    
    def _draw_box(self, frame, x1, y1, x2, y2, class_name, conf):
        """Draw detection box with label"""
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
```

### Step 3: Alert System
```python
    def check_alert_conditions(self, detections):
        """Check if alert conditions are met"""
        should_alert = False
        alert_message = ""
        
        # Alert if person and bag detected together
        if detections['person'] and detections['bag']:
            should_alert = True
            alert_message = "Person with bag detected"
            self.counts['alerts'] += 1
        
        return should_alert, alert_message
    
    def save_alert(self, frame, message):
        """Save alert frame with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"alert_{self.counts['alerts']}_{timestamp}.jpg"
        filepath = self.alert_dir / filename
        
        # Add alert message to frame
        cv2.putText(frame, message, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imwrite(str(filepath), frame)
```

### Step 4: Main Monitoring Loop
```python
    def start_monitoring(self):
        """Start the security monitoring system"""
        cap = cv2.VideoCapture(0)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Analyze frame
                processed_frame, detections = self.analyze_frame(frame)
                
                # Update counts
                if detections['person']:
                    self.counts['person'] += len(detections['person_coords'])
                if detections['bag']:
                    self.counts['bag'] += len(detections['bag_coords'])
                
                # Check for alerts
                should_alert, alert_message = self.check_alert_conditions(detections)
                if should_alert:
                    self.save_alert(processed_frame, alert_message)
                
                # Display counts on frame
                self._display_counts(processed_frame)
                
                # Show frame
                cv2.imshow('Security Monitor', processed_frame)
                
                if cv2.waitKey(1) == ord('q'):
                    break
                
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.generate_report()
    
    def _display_counts(self, frame):
        """Display detection counts on frame"""
        y_position = 30
        for item, count in self.counts.items():
            cv2.putText(frame, f"{item}: {count}", (10, y_position),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_position += 25
```

### Step 5: Reporting System
```python
    def generate_report(self):
        """Generate summary report of monitoring session"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.report_dir / f"report_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("Security Monitoring Report\n")
            f.write("=" * 25 + "\n\n")
            f.write(f"Session Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Detection Summary:\n")
            for item, count in self.counts.items():
                f.write(f"- Total {item}: {count}\n")
            
            f.write(f"\nAlert images saved: {self.counts['alerts']}\n")
            f.write(f"Alert directory: {self.alert_dir}\n")
```

## Running the Complete System
```python
if __name__ == "__main__":
    # Initialize and start the security system
    security_system = SecuritySystem()
    security_system.start_monitoring()
```

## Exercise Completion Checklist

### Basic Requirements
- [ ] Real-time detection of people and bags
- [ ] Image saving system
- [ ] Timestamp and confidence recording
- [ ] Object counting system

### Advanced Requirements
- [ ] Alert generation system
- [ ] Multi-frame tracking
- [ ] Summary reporting
- [ ] User interface elements

## Testing Your Implementation

1. Basic Functionality Test
   ```python
   security_system = SecuritySystem()
   # Should create directories
   assert os.path.exists('security_logs')
   assert os.path.exists('security_logs/alerts')
   ```

2. Detection Test
   ```python
   # Test with sample image
   import numpy as np
   test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
   processed_frame, detections = security_system.analyze_frame(test_frame)
   assert isinstance(detections, dict)
   assert 'person' in detections
   ```

## Common Issues and Solutions

1. Model Loading Issues
   ```python
   # Add error handling for model loading
   try:
       model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
   except Exception as e:
       print(f"Error loading model: {e}")
       # Fall back to local model or exit gracefully
   ```

2. Memory Management
   ```python
   # Add periodic cleanup
   if len(os.listdir(alert_dir)) > 1000:
       # Remove oldest files
       oldest_files = sorted(Path(alert_dir).glob('*.jpg'))[:100]
       for file in oldest_files:
           file.unlink()
   ```

## Extension Ideas

1. Add Multiple Camera Support
   ```python
   def __init__(self, camera_ids=[0]):
       self.cameras = {
           id: cv2.VideoCapture(id) for id in camera_ids
       }
   ```

2. Implement Motion Detection
   ```python
   def detect_motion(self, frame1, frame2):
       diff = cv2.absdiff(frame1, frame2)
       return np.mean(diff) > MOTION_THRESHOLD
   ```

3. Add Remote Notification System
   ```python
   def send_alert(self, message):
       # Implement email or SMS notification
       pass
   ```

## Evaluation Criteria

1. Code Quality (30%)
   - Clean, well-organized code
   - Proper error handling
   - Good documentation

2. Functionality (40%)
   - Accurate detection
   - Reliable alert system
   - Proper image saving

3. Performance (30%)
   - Processing speed
   - Resource usage
   - System stability