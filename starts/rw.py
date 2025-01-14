import cv2
import os


try:
    image = cv2.imread(r'..\Wallpapers\AppBreweryWallpaper 1.png')

    if image is None:
        raise FileNotFoundError("Image file could not be found or image could not be read")
    
    output_dir = r'..\Wallpapers\cv2images'
    save_path = os.path.join(output_dir, 'cv2image.png')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    success = cv2.imwrite(save_path, image)

    if not success:
        raise IOError("Failed to write the image to the specified location")
    
    print("image is succefully saved to ", save_path)
    
except Exception as e:
    print("Error ", e)
