from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
from imutils import paths
import os

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="Path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True, help="Path to the data image")
ap.add_argument("-o", "--output", required=True, help="Path to output data image")
args = vars(ap.parse_args())

total = 0
print("[INFO] loading facial landmark...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
fa = FaceAligner(predictor, desiredFaceWidth=256)

print("[INFO] loading dataset...")
imagePaths = list(paths.list_images(args["image"]))

labels = []

for (i, imagePath) in enumerate(imagePaths):
    print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]
    
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray, 2)
    
    for rect in rects:
        (x, y, w, h) = rect_to_bb(rect)
        faceAligened = fa.align(image, gray, rect)
        z = os.path.sep.join([args["output"], "{}.png".format(str(total).zfill(5))])
        cv2.imwrite(z, faceAligened)
        total += 1