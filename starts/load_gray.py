import cv2
import os


try:
    image = cv2.imread(r'..\Wallpapers\AppBreweryWallpaper 1.png', cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError("Image file could not be found or image could not be read")
    
    output_dir = r'..\Wallpapers\cv2images'
    save_path = os.path.join(output_dir, 'graycv2image.png')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    success = cv2.imwrite(save_path, image)

    if not success:
        raise IOError("Failed to write the image to the specified location")
    
    print("image is succefully saved to ", save_path)

    # print(image.shape)
    # print(image.size)
    # print(image.itemsize)
    # print(image.dtype)
    #print(len(image[0]))
    
except Exception as e:
    print("Error ", e)
