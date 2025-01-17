import cv2
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = model.to(device)

cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Perform detection
    results = model(frame)
    
    print(results)
    
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