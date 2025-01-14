import numpy as np
import os
import cv2

randomByteArray = bytearray(os.urandom(120000))
flatNUmpyArray = np.array(randomByteArray)

bgrImage = flatNUmpyArray.reshape(100, 400, 3)

output_path = r'..\Wallpapers\cv2rawImages'

save_path = os.path.join(output_path, 'randomColredGray.png')

if not os.path.exists(output_path):
    os.makedirs(output_path)

cv2.imwrite(save_path, bgrImage)