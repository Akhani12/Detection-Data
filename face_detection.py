# Face Detection

# pip install cvlib

import cvlib as cv
import sys
import cv2
import os
from google.colab.patches import cv2_imshow

image = cv2.imread("/content/face_detection_input.jpg")
faces, confidences = cv.detect_face(image)
print(faces)
print(confidences)
for face,conf in zip(faces,confidences):
    (startX,startY) = face[0],face[1]
    (endX,endY) = face[2],face[3]
    cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 2)
cv2_imshow(image)
cv2.waitKey()
cv2.imwrite("face_detection.jpg", image)
cv2.destroyAllWindows()
