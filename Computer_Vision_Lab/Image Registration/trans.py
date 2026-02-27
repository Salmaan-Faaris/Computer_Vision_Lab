import cv2
import numpy as np

img = cv2.imread("D:\\Salman\\Semesters\\Sem8\\Computer Vision\\Class Work\\Image Registration\\image1.jfif")
h, w = img.shape[:2]

# Create affine transform
M = cv2.getRotationMatrix2D((w/2, h/2), 15, 1.1)
M[0,2] += 30
M[1,2] += 20

transformed = cv2.warpAffine(img, M, (w, h))
cv2.imwrite("image2.jpg", transformed)