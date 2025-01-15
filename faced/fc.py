import cv2
import matplotlib.pyplot as plt 

imagePath = 'facoo.jpeg'

img = cv2.imread(imagePath)

print(img.shape)

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(gray_image)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.2, minNeighbors=10, minSize=(20, 20)
)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# cv2.imwrite("detected.jpeg", img_rgb)
plt.imshow(img_rgb)
plt.title("Detected Faces")
plt.axis("off")
plt.show()