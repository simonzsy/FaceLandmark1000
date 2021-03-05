import sys
import torch
from face_detector import *
from face_landmark import *
import os
from flask import Flask

app = Flask(__name__)
@app.route('/')


def image_run():
    return "ok"
    face_detector_handle = FaceDetector()
    face_landmark_handle = FaceLandmark()

    image = cv2.imread('data/1.jpg')
    detections, _ = face_detector_handle.run(image)

    face_detector_handle.show_result(image, detections)

    if len(detections) == 0:
        return

    for detection in detections:
        landmarks, states = face_landmark_handle.run(image, detection)
        # face_landmark_handle.show_result(image, landmarks)
        return "landmarks ok"



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
