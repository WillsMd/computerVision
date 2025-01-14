import numpy as np
import os
import cv2

randomByteArray = bytearray(os.urandom(120000))
flatNUmpyArray = np.array(randomByteArray)

grayImage = flatNUmpyArray.reshape(300, 400)

output_path = r'..\Wallpapers\cv2rawImages'
save_path = os.path.join(output_path, 'randomGray.png')

if not os.path.exists(output_path):
    os.makedirs(output_path)

cv2.imwrite(save_path, grayImage)