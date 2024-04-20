import json
import os
# THIS WAS ALL FOR TESTING PLEASE IGNORE THIS FILE THERE IS NOTHING IN HERE
import cv2
import base64
import numpy as np
import requests
import time
import pytesseract
from PIL import Image
from io import BytesIO

from roboflow import Roboflow

# Set your tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Set your API Key
rf = Roboflow(api_key="roboflow api key")
project = rf.workspace().project("license-plate-recognition-rxg4e")
model = project.version(4).model

# Get webcam interface via opencv-python
video = cv2.VideoCapture(0)

# Define ROBOFLOW_SIZE
ROBOFLOW_SIZE = 416

video = cv2.VideoCapture(0)

def infer():
    ret, img = video.read()

    height, width, channels = img.shape
    scale = ROBOFLOW_SIZE / max(height, width)
    img = cv2.resize(img, (round(scale * width), round(scale * height)))

    retval, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer)

    resp = requests.post(upload_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    })

    predictions = resp.json()['predictions']

    for pred in predictions:
        x = pred['x']
        y = pred['y']
        w = pred['width']
        h = pred['height']
        label = pred['class']
        confidence = pred['confidence']

        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{label}: {confidence:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img

while 1:
    if(cv2.waitKey(1) == ord('q')):
        break

    image = infer()
    cv2.imshow('image', image)

video.release()
cv2.destroyAllWindows()