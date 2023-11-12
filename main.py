import cv2
import tensorflow as tf
import numpy as np

MODEL=tf.keras.models.load_model("saved_models/1")# Loading our Model
CLASSES=["Average","Beautiful"]

cap=cv2.VideoCapture(0)

haar_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    ret,frame=cap.read()
    if not ret:
        print("Cannot Open Camera !")

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for x,y,w,h in haar_cascade.detectMultiScale(gray_frame):
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2,cv2.LINE_AA)

        img=frame[y:y+h,x:x+w]
        img=cv2.resize(img,(256,256))
        # cv2.imshow("Img_",img)

        img_batch=np.expand_dims(img,axis=0)
        y_pred=MODEL.predict(img_batch)
        belong_class=CLASSES[round(y_pred[0][0])]

        frame=cv2.putText(frame,belong_class,(x,w),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2,cv2.LINE_AA)

    cv2.imshow("Frame",frame)

    if cv2.waitKey(1)==ord("q"):
        print("...")

cap.release()
cv2.destroyAllWindows()