import numpy as np
import cv2

'''
Each pixel is represented by a single 8-bit integer, which means that the values for 
each pixel are in the 0-255 range.
'''
img = np.array(
    [[0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],], dtype=np.uint8) 

'''
Let's now convert this image into Blue-green-red (BGR) using cv2.cvtColor:
'''
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

print(img)

'''
You can check the structure of an image by inspecting the shape property, which 
returns rows, columns, and the number of channels (if there is more than one)
'''

img2 = np.zeros((3,3), dtype=np.uint8)
print(img2.shape)

print(img.shape)
