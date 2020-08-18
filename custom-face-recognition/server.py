from flask import Flask
from flask import request, jsonify
from flask_cors import CORS, cross_origin
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import numpy as np
import cv2
import base64
import os
import sqlite3
import face_recognition
import imutils
import pickle

# Khai bao cong cua server
my_port = '8000'
protoPath = "{base_path}/detect_face_model/deploy.prototxt".format(base_path=os.path.abspath(os.path.dirname(__file__)))
modelPath = "{base_path}/detect_face_model/res10_300x300_ssd_iter_140000.caffemodel".format(base_path=os.path.abspath(os.path.dirname(__file__)))
embeddingPath = "{base_path}/output/embeddings.pickle".format(base_path=os.path.abspath(os.path.dirname(__file__)))
embedderPath = "{base_path}/openface_nn4.small2.v1.t7".format(base_path=os.path.abspath(os.path.dirname(__file__)))
recogPath = "{base_path}/output/recognizer.pickle".format(base_path=os.path.abspath(os.path.dirname(__file__)))
lePath = "{base_path}/output/le.pickle".format(base_path=os.path.abspath(os.path.dirname(__file__)))
total=1
step = 0
dataset = "dataset"
detection_method = "cnn"

def build_return(name, x, y, x_plus_w, y_plus_h, age):
    return str(name) + "," + str(x) + "," + str(y) + "," + str(x_plus_w) + "," + str(y_plus_h) + "," + str(age)

def checking():
    percent=(step/total)*100
    if percent>100:
        print("xxxxxxxxxxxxxxxxxxxxxxx")
    return str(int(percent))

# Hàm cập nhật tên và ID vào CSDL
def insertOrUpdate(id, name, Age):
    conn=sqlite3.connect("D:\PracticePY\Project\Study\client-server\custom-face-recognition\FaceBaseNew.db")
    cursor=conn.execute('SELECT * FROM People WHERE Id='+str(id))
    isRecordExist=0
    for row in cursor:
        isRecordExist = 1
        break

    if isRecordExist==1:
        cmd="UPDATE People SET Name=' "+str(name)+" ' WHERE ID="+str(id)
        cmd="UPDATE People SET Age=' "+str(Age)+" ' WHERE ID="+str(id)
    else:
        cmd="INSERT INTO People(ID,Name,Age) Values("+str(id)+",'"+str(name)+"','"+str(Age)+"')"

    conn.execute(cmd)
    conn.commit()
    conn.close()

def getInfor(id):
	conn=sqlite3.connect("D:\PracticePY\Project\Study\client-server\custom-face-recognition\FaceBaseNew.db")
	conn.row_factory = sqlite3.Row
	cursor = conn.cursor()
	cursor= conn.execute('SELECT * FROM People')
	rows= cursor.fetchall()
	for row in rows:
		if row["Id"] == int(id):
			return row
   
def encode():
    global step
    global total
    # load our serialized face detector from disk
    print("[INFO] loading face detector...")
    # define the path to the face detector
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch(embedderPath)

    # grab the paths to the input images in our dataset
    print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images(dataset))

    # initialize our lists of extracted facial embeddings and
    # corresponding people names
    knownEmbeddings = []
    knownNames = []

    # initialize the total number of faces processed
    total = len(imagePaths)

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]

        nameTemp = name.split("_")
        id = nameTemp[0]
        nameId = nameTemp[1]
        age = nameTemp[2]
        insertOrUpdate(id, nameId, age)

        step = step + 1

        # load the image, resize it to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()

        # ensure at least one face was found
        if len(detections) > 0:
            # we're making the assumption that each image has only ONE
            # face, so find the bounding box with the largest probability
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            # ensure that the detection with the largest probability also
            # means our minimum probability test (thus helping filter out
            # weak detections)
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI and grab the ROI dimensions
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                    (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # add the name of the person + corresponding face
                # embedding to their respective lists
                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                
    # dump the facial embeddings + names to disk
    print("[INFO] serializing {} encodings...".format(total))
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    f = open("output/embeddings.pickle", "wb")
    f.write(pickle.dumps(data))
    f.close()

# Doan ma khoi tao server
app = Flask(__name__)
CORS(app)

# Khai bao ham xu ly request index
@app.route('/')
@cross_origin()
def index():
    return "Welcome to flask API!"

# Khai bao ham xu ly request hello_word
@app.route('/hello_world', methods=['GET'])
@cross_origin()
def hello_world():
    # Lay staff id cua client gui len
    staff_id = request.args.get('staff_id')
    # Tra ve cau chao Hello
    return "Hello "  + str(staff_id)

@app.route('/train',methods=['GET'])
def train():
    global step,total
    step=0
    total=1
    # t = threading.Thread(encode())
    # t.start()
    # t.join()
    encode()
    
    # load the face embeddings
    print("[INFO] loading face embeddings...")
    data = pickle.loads(open(embeddingPath, "rb").read())
    
    # encode the labels
    print("[INFO] encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])
    
    print("[INFO] training model...")
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    # write the actual face recognition model to disk
    f = open(recogPath, "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    # write the label encoder to disk
    f = open(lePath, "wb")
    f.write(pickle.dumps(le))
    f.close()
    print("TRAINED !")
    return str("TRAINED !")

@app.route('/progress',methods=['GET'])
def check():
    k=checking()
    return str(k)

@app.route('/capture',methods=['POST'])
def detect():
    image_b64 = request.form.get('image')
    name = request.args.get("name")
    id = request.args.get("id")
    pic = request.args.get("pic")
    age = request.args.get("age").strip()
    print(id+" : "+name)

    #insertOrUpdate(id, name, age)

    image = np.fromstring(base64.b64decode(image_b64), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_ANYCOLOR)
    print(image.shape)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detect the (x, y)-coordinates of the bounding boxes corresponding
    # to each face in the input image, then compute the facial embeddings
    # for each face
    print("[INFO] detecting faces...")
    boxes = face_recognition.face_locations(rgb,
                                           model=detection_method)  # tra ve so face trong hinh
    print(len(boxes))
    retString = ""
    if len(boxes)!=0:
        data_dir = os.path.sep.join([dataset, id + "_" + name + "_" + age])
        for dir in os.listdir(dataset):
            if int(id) == int(dir.split("_")[0]) and name != dir.split("_")[1]:      #same id, diferent name, break
                return build_return(0, 0, 2, 3, 4, 5)   #id existed

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        img_name = name +'_' +pic +".png"
        imgPath = os.path.sep.join([data_dir,img_name])
        cv2.imwrite(imgPath,image)
        print("Save {} successfully !!!".format(img_name))

        for (top, right, bottom, left) in boxes:
            x = left
            y = top
            w = right - left
            h = bottom - top
            # Xay dung chuoi tra ve client
            retString = build_return(len(boxes), round(x), round(y), round(w), round(h), age)
        return retString
    else:
        retString = build_return(0, 1, 2, 3, 4, 5)
        return retString

@app.route('/detect', methods=['POST'])
@cross_origin()
def recog():
    image_b64 = request.form.get('image')
    image = np.fromstring(base64.b64decode(image_b64), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_ANYCOLOR)
    print(image.shape)
    # image = imutils.resize(image, width=800)
    # image = cv2.copyMakeBorder(image, 100, 100, 100, 100, borderType=cv2.BORDER_CONSTANT)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...")

    embedder = cv2.dnn.readNetFromTorch(embedderPath)
    
    # load the actual face recognition model along with the label encoder
    recognizerPath = "{base_path}/output/recognizer.pickle".format(base_path=os.path.abspath(os.path.dirname(__file__)))
    lePath = "{base_path}/output/le.pickle".format(base_path=os.path.abspath(os.path.dirname(__file__)))
    recognizer = pickle.loads(open(recognizerPath, "rb").read())
    le = pickle.loads(open(lePath, "rb").read())

    (h, w) = image.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

        # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the
            # face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            print(startX, startY, endX, endY)
            # extract the face ROI
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            print(preds)
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            
            # Get ID and INFORMATION in database
            id = name.split("_")[0]
            Age = getInfor(id)["Age"]

        retString = build_return(name, round(startX), round(startY), round(endX - startX), round(endY - startY), Age)
    return retString

# Thuc thi server
if __name__ == '__main__':
    app.run(debug=True,port=my_port)