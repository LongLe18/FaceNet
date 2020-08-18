import cv2
import argparse
import os
import numpy as np
import sqlite3

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

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True, help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
#ap.add_argument("-o", "--output", required=True, help="path to save output image")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt.txt"])
modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

id = input("Nhập mã nhân viên: ")
name = input("Nhập tên sinh viên: ")
age = input("Nhập tuổi sinh viên: ")
print("Bắt đầu chụp ảnh nhân viên, nhấn q để thoát!")

insertOrUpdate(id, name, age)

# Tạo thư mục mới với tên nhân viên tương ứng
os.chdir("dataset/")
if os.path.isdir(name) == True:
    os.rmdir(name)
    path = os.mkdir(name)
    os.chdir(name)
else:
    path = os.mkdir(name)
    os.chdir(name)

cam = cv2.VideoCapture(0)

sample = 0

while True:
    # Đọc ảnh từ camera
    ret, img=cam.read()

    (h ,w) = img.shape[:2]
    # blob image
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 107.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence < args["confidence"]:
            continue
        else:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
            text = "{:.2f} %".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 255), 2)
            if cv2.waitKey(1) & 0xFF == ord('a'):
                cv2.imwrite(id + '.' + str(sample) + ".jpg", img)
                sample = sample + 1
    
    cv2.imshow("SHow", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif sample > 22:
        print("[INFO] Đã đủ số lượng ảnh!!!")
        break
cam.release()
cv2.destroyAllWindows()