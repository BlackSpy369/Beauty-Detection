from fastapi import FastAPI,UploadFile,File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image

app=FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL=tf.keras.models.load_model("../saved_models/1")
CLASSES=["Average","Beautiful"]

def get_image_from_bytes(data)->np.ndarray:
    return np.resize(np.array(Image.open(BytesIO(data))),(256,256,3))
    

@app.post("/predict")
async def predict(file:UploadFile=File(...)):
    image=get_image_from_bytes(await file.read())
    image_batch=np.expand_dims(image,axis=0)
    y_pred=MODEL.predict(image_batch)
    prediction=CLASSES[round(y_pred[0][0])]
    return prediction

if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)
