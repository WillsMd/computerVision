
---

## 🎥 Real-Time Face Detection with OpenCv

Now that we have successfully performed face detection on a static image with OpenCV, let’s see how to do the same on a live video stream. 

### Step 1: Pre-Requisites 🖥️  
First, let’s go ahead and import the OpenCV library and load the Haar Cascade model just like we did in the previous section. You can skip this block of code if you already ran it previously:
```python
import cv2

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
```


### Step 2: Access the WebCam
Now, we need to access our device’s camera to read a live stream of video data. This can be done with the following code:

```python
video_capture = cv2.VideoCapture(0)
```  
Notice that we have passed the parameter 0 to the VideoCapture() function. This tells OpenCV to use the default camera on our device. If you have multiple cameras attached to your device, you can change this parameter value accordingly.


### Step 3: Identifying Faces in the Video Stream
Now, let’s create a function to detect faces in the video stream and draw a bounding box around them:

```python
def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces
```
The detect_bounding_box function takes the video frame as input.

In this function, we are using the same codes as we did earlier to convert the frame into grayscale before performing face detection.

Then, we are also detecting the face in this image using the same parameter values for scaleFactor, minNeighbors, and minSize as we did previously.

Finally, we draw a green bounding box of thickness 4 around the frame.

### Step 4: Creating a Loop for Real-Time Face Detection
Now, we need to create an indefinite while loop that will capture the video frame from our webcam and apply the face detection function to it:

```python
while True:

    result, video_frame = video_capture.read()  # read frames from the video
    if result is False:
        break  # terminate the loop if the frame is not read successfully

    faces = detect_bounding_box(
        video_frame
    )  # apply the function we created to the video frame

    cv2.imshow(
        "My Face Detection Project", video_frame
    )  # display the processed frame in a window named "My Face Detection Project"

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
```