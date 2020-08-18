from imutils.video import VideoStream
import argparse
import cv2
import imutils
import time
import os
import urllib.request
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="Path to Caffe deploy txt file")
ap.add_argument("-m", "--model", required=True, help="Path to Caffe-pretrained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-o", "--output", required=True, help="path to output directory")
args = vars(ap.parse_args())

url = "http://192.168.43.1:8080/shot.jpg"
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
f = os.mkdir(args["output"].split()[-1])
total = 0
print("[INFO] starting video stream...")
while True:
    frameO = urllib.request.urlopen(url)
    frameNp = np.array(bytearray(frameO.read()), dtype=np.uint8)
    frame = cv2.imdecode(frameNp, -1)
    frame = cv2.resize(frame, (500, 400))
    orig = frame.copy()
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 107.0, 123.0))
    
    net.setInput(blob)
    detections = net.forward()
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < args["confidence"]:
            continue
        else:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            text = "{:.2f}%".format(confidence * 100)
            (startX, startY, endX, endY) = box.astype('int')
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('k'):
        p = os.path.sep.join([args["output"], "{}.png".format(str(total).zfill(5))])
        cv2.imwrite(p, orig)
        total += 1
    elif key == ord('q'):
        break
print("[INFO] {} face images stored".format(total))
print("[INFO] cleaning up...")
cv2.destroyAllWindows()