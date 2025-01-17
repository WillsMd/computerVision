import cv2
import numpy as np


def analyze_face(image):
    face = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
    eyes = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')
    smile = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_smile.xml')
    """
    Analyze face features in an image and return detected features and annotated image
    Returns: tuple (features_dict, processed_image)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create copy for drawing
    annotated_image = image.copy()
    
    # Initialize features dictionary
    features = {
        'face_detected': False,
        'eyes_count': 0,
        'smile_detected': False
    }
    
    # Detect faces
    faces = face.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Process each detected face
    for (x, y, w, h) in faces:
        features['face_detected'] = True
        
        # Draw rectangle around face
        cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Define regions of interest (ROI)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = annotated_image[y:y+h, x:x+w]
        
        # Detect eyes
        detected_eyes = eyes.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        
        features['eyes_count'] = len(detected_eyes)
        
        # Draw rectangles around eyes
        for (ex, ey, ew, eh) in detected_eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        # Detect smile
        smile_roi_gray = roi_gray[h//2:, :]
        smile_roi_color = roi_color[h//2:, :]
        
        smiles = smile.detectMultiScale(
            smile_roi_gray,
            scaleFactor=1.7,
            minNeighbors=20,
            minSize=(25, 25)
        )
        
        features['smile_detected'] = len(smiles) > 0
        
        # Draw rectangles around detected smiles
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(smile_roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
    
    return features, annotated_image

if __name__ == '__main__':
    try:
        # Initialize video capture
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Could not open video capture device")
        
        cv2.namedWindow('Face Analysis')  # Create a named window
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Analyze frame and get features and annotated image
            features, annotated_frame = analyze_face(frame)
            
            # Display feature detection results
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(annotated_frame, f"Eyes: {features['eyes_count']}", (10, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Smile: {features['smile_detected']}", (10, 70), font, 1, (255, 255, 255), 2)
            cv2.putText(annotated_frame, "Press 'q' to quit", (10, 110), font, 1, (255, 255, 255), 2)
            
            # Show the frame
            cv2.imshow('Face Analysis', annotated_frame)
            
            # Check for 'q' key press - now using a shorter wait time
            key = cv2.waitKey(1)
            if key == ord('q') or key == ord('Q') or cv2.getWindowProperty('Face Analysis', cv2.WND_PROP_VISIBLE) < 1:
                print("Exiting...")
                break
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    
    finally:
        # Clean up
        print("Cleaning up...")
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Additional waitKey to ensure windows are closed