

# Face Detection Fun with OpenCV 😎  

Today we will start with a casual face detection, we will  cover both static and real time basics of the face detection with opencv

---

## 📸 Static Image Face Detection  

Wanna find faces in a picture? Here's how:  

### Step 1: Import the OpenCV Package  
Now, let’s import OpenCV and enter the input image path with the following lines of code:
   ```bash
import cv2
imagePath = 'input_image.jpg'
   ```  


### Step 2: Read the Image
Then, we need to read the image with OpenCV’s imread() function:
```python
img = cv2.imread(imagePath)
```  
This will load the image from the specified file path and return it in the form of a Numpy array. 

Let’s print the dimensions of this array:
```python
img.shape
```  
Notice that this is a 3-dimensional array. The array’s values represent the picture’s height, width, and channels respectively. Since this is a color image, there are three channels used to depict it - blue, green, and red (BGR). 

Note that while the conventional sequence used to represent images is RGB (Red, Blue, Green), the OpenCV library uses the opposite layout (Blue, Green, Red).

### Step 3: Convert the Image to Grayscale  
To improve computational efficiency, we first need to convert this image to grayscale before performing face detection on it:

```python
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```  
Let’s now examine the dimensions of this grayscale image:

```python
gray_image.shape
```  
Notice that this array only has two values since the image is grayscale and no longer has the third color channel.

### Step 4: Load the Classifier
Let’s load the pre-trained Haar Cascade classifier that is built into OpenCV:

```python
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
```  
Notice that we are using a file called haarcascade_frontalface_default.xml. This classifier is designed specifically for detecting frontal faces in visual input. 

OpenCV also provides other pre-trained models to detect different objects within an image - such as a person’s eyes, smile, upper body, and even a vehicle’s license plate. You can learn more about the different classifiers built into OpenCV by examining the library’s https://github.com/opencv/opencv/tree/master/data/haarcascades.


### Step 5: Perform the Face Detection
We can now perform face detection on the grayscale image using the classifier we just loaded:
```python
face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)
```
0. detectMultiScale():
The detectMultiScale() method is used to identify faces of different sizes in the input image.

1. grey image:
The first parameter in this method is called grey_image, which is the grayscale image we created previously.

2. scaleFactor:
This parameter is used to scale down the size of the input image to make it easier for the algorithm to detect larger faces. In this case, we have specified a scale factor of 1.1, indicating that we want to reduce the image size by 10%.

3. minNeighbors:
The cascade classifier applies a sliding window through the image to detect faces in it. You can think of these windows as rectangles. 
Initially, the classifier will capture a large number of false positives. These are eliminated using the minNeighbors parameter, which specifies the number of neighboring rectangles that need to be identified for an object to be considered a valid detection.
To summarize, passing a small value like 0 or 1 to this parameter would result in a high number of false positives, whereas a large number could lead to losing out on many true positives.
The trick here is to find a tradeoff that allows us to eliminate false positives while also accurately identifying true positives.

4. minSize:
Finally, the minSize parameter sets the minimum size of the object to be detected. The model will ignore faces that are smaller than the minimum size specified.

### Step 6: Drawing a Bounding Box
Now that the model has detected the faces within the image, let’s run the following lines of code to create a bounding box around these faces:

```python
for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
```
The face variable is an array with four values: the x and y axis in which the faces were detected, and their width and height. The above code iterates over the identified faces and creates a bounding box that spans across these measurements.

The parameter 0,255,0 represents the color of the bounding box, which is green, and 4 indicates its thickness.

### Step 7: Displaying the image
To display the image with the detected faces, we first need to convert the image from the BGR format to RGB:

```python
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

```   
Now, let’s use the Matplotlib library to display the image:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plt.imshow(img_rgb)
plt.axis('off')
```


