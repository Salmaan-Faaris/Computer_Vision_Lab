import cv2
import numpy as np

obj = cv2.imread("D:\\Salman\\Semesters\\Sem8\\Computer Vision\\Class Work\\Image Registration\\image1.jfif")
scene = np.ones((600, 800, 3), dtype=np.uint8) * 255

# Resize object
obj_small = cv2.resize(obj, (300, 400))

# Place object inside scene
scene[100:500, 250:550] = obj_small

cv2.imwrite("scene.jpg", scene)