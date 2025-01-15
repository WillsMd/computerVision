import cv2
import matplotlib.pyplot as plt

# Load the input image
imagePath = 'input_image.jpg'
img = cv2.imread(imagePath)

# Check if the image is loaded
if img is None:
    raise FileNotFoundError(f"Image at {imagePath} not found!")

print("Original image shape:", img.shape)

# Convert image to grayscale for face detection
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load the Haar cascade classifier
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


faces = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)


img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


plt.imshow(img_rgb)
plt.title("Detected Faces")
plt.axis("off")
plt.show()

print(f"Number of faces detected: {len(faces)}")
