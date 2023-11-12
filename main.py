import cv2
import tensorflow as tf
import numpy as np

MODEL=tf.keras.models.load_model("saved_models/1")# Loading our Model
CLASSES=["Average","Beautiful"]

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    if not ret:
        print("Cannot Open Camera !")

    img=cv2.resize(frame,(256,256))
    # cv2.imshow("Img",img)
    img_batch=np.expand_dims(img,axis=0)
    y_pred=MODEL.predict(img_batch)
    belong_class=CLASSES[round(y_pred[0][0])]

    frame=cv2.putText(frame,belong_class,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2,cv2.LINE_AA)
    cv2.imshow("Frame",frame)

    if cv2.waitKey(1)==ord("q"):
        print("...")

cap.release()
cv2.destroyAllWindows()