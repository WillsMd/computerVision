# Functional Object Detection Implementation
A modular, function-based implementation of the security monitoring system without using classes.

## Overview
This implementation uses purely functional programming approach with separate functions for each component of the system.

## Code Implementation

### 1. Setup and Configuration
```python
import cv2
import torch
import os
from datetime import datetime
from pathlib import Path

# Global configurations
CONFIG = {
    'confidence_threshold': 0.5,
    'base_dir': Path('security_logs'),
    'alert_dir': Path('security_logs/alerts'),
    'report_dir': Path('security_logs/reports'),
    'counts': {
        'person': 0,
        'bag': 0,
        'alerts': 0
    }
}

def setup_directories():
    """Create necessary directories for logging"""
    CONFIG['base_dir'].mkdir(exist_ok=True)
    CONFIG['alert_dir'].mkdir(exist_ok=True)
    CONFIG['report_dir'].mkdir(exist_ok=True)

def initialize_model():
    """Initialize the YOLOv5 model"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    return model.to(device)
```

### 2. Detection Functions
```python
def analyze_frame(frame, model):
    """Analyze a single frame for objects of interest"""
    results = model(frame)
    
    detections = {
        'person': False,
        'person_coords': [],
        'bag': False,
        'bag_coords': []
    }
    
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result
        class_name = model.names[int(cls)]
        
        if conf > CONFIG['confidence_threshold']:
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
            
            draw_detection_box(frame, x1, y1, x2, y2, class_name, conf)
    
    return frame, detections

def draw_detection_box(frame, x1, y1, x2, y2, class_name, conf):
    """Draw bounding box and label on frame"""
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{class_name} {conf:.2f}"
    cv2.putText(frame, label, (x1, y1 - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
```

### 3. Alert System Functions
```python
def check_alert_conditions(detections):
    """Check if alert conditions are met"""
    should_alert = False
    alert_message = ""
    
    if detections['person'] and detections['bag']:
        should_alert = True
        alert_message = "Person with bag detected"
        CONFIG['counts']['alerts'] += 1
    
    return should_alert, alert_message

def save_alert(frame, message):
    """Save alert frame with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"alert_{CONFIG['counts']['alerts']}_{timestamp}.jpg"
    filepath = CONFIG['alert_dir'] / filename
    
    # Add alert message to frame
    cv2.putText(frame, message, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imwrite(str(filepath), frame)
```

### 4. Display and Counting Functions
```python
def update_counts(detections):
    """Update object counts"""
    if detections['person']:
        CONFIG['counts']['person'] += len(detections['person_coords'])
    if detections['bag']:
        CONFIG['counts']['bag'] += len(detections['bag_coords'])

def display_counts(frame):
    """Display current counts on frame"""
    y_position = 30
    for item, count in CONFIG['counts'].items():
        cv2.putText(frame, f"{item}: {count}", (10, y_position),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_position += 25
```

### 5. Reporting Function
```python
def generate_report():
    """Generate summary report of monitoring session"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = CONFIG['report_dir'] / f"report_{timestamp}.txt"
    
    with open(report_path, 'w') as f:
        f.write("Security Monitoring Report\n")
        f.write("=" * 25 + "\n\n")
        f.write(f"Session Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Detection Summary:\n")
        for item, count in CONFIG['counts'].items():
            f.write(f"- Total {item}: {count}\n")
        
        f.write(f"\nAlert images saved: {CONFIG['counts']['alerts']}\n")
        f.write(f"Alert directory: {CONFIG['alert_dir']}\n")
```

### 6. Main Monitoring Function
```python
def start_monitoring():
    """Main function to run the security monitoring system"""
    # Initialize
    setup_directories()
    model = initialize_model()
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame, detections = analyze_frame(frame, model)
            
            # Update counts and check for alerts
            update_counts(detections)
            should_alert, alert_message = check_alert_conditions(detections)
            
            if should_alert:
                save_alert(processed_frame, alert_message)
            
            # Display information
            display_counts(processed_frame)
            cv2.imshow('Security Monitor', processed_frame)
            
            if cv2.waitKey(1) == ord('q'):
                break
                
    except Exception as e:
        print(f"Error during monitoring: {e}")
        
    finally:
        cap.release()
        cv2.destroyAllWindows()
        generate_report()

# Run the system
if __name__ == "__main__":
    start_monitoring()
```

### 7. Utility Functions
```python
def cleanup_old_files():
    """Remove oldest files if storage limit is reached"""
    max_files = 1000
    alert_files = list(CONFIG['alert_dir'].glob('*.jpg'))
    
    if len(alert_files) > max_files:
        # Sort by creation time and remove oldest
        alert_files.sort(key=lambda x: x.stat().st_ctime)
        for file in alert_files[:100]:  # Remove oldest 100 files
            file.unlink()

def format_timestamp():
    """Generate formatted timestamp"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
```

## Usage Example
```python
# Basic usage
start_monitoring()

# With custom configuration
CONFIG['confidence_threshold'] = 0.7
start_monitoring()

# With cleanup
cleanup_old_files()
start_monitoring()
```

## Key Differences from Class-based Implementation

1. Global Configuration
   - Uses a global CONFIG dictionary instead of instance variables
   - Easier to modify settings across functions

2. Function Independence
   - Each function is independent and can be tested separately
   - Functions can be easily modified or replaced

3. Data Flow
   - Data is passed explicitly between functions
   - State is managed through the CONFIG dictionary

4. Error Handling
   - Centralized error handling in the main monitoring function
   - Each function can handle its own specific errors

## Best Practices
1. Keep functions small and focused
2. Use meaningful function names
3. Document function parameters and return values
4. Handle errors appropriately
5. Use type hints for better code clarity (optional)

Would you like me to explain any particular function in more detail or add additional functionality to this implementation?