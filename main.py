import os
from io import BytesIO

import uvicorn
import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


"""
Fast API backend Application
"""
current_file_path = os.path.abspath(__file__)
model_path = os.path.join(current_file_path, "saved_models/v1.keras")

app = FastAPI()

class Item(BaseModel):
    name: str
    desc: str

MODEL = tf.keras.models.load_model('saved_models/v1.keras')

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def read_file_as_image(data):
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.get("/")
async def home():
    return JSONResponse({
        'status' : 'API Started Successfully',
        'Endpoint' : '/predict',
        'Parameters' : 'Image to be passed',
        'result' : 'It will be in dictionary format providing you the resulting class along with the confidence percentage.'
    })


@app.post("/predict")
async def predict(data: Item, files:list[UploadFile]=File(...)):
    for file in files:
        contents = await file.read()
        image = read_file_as_image(contents)
        img_batch = np.expand_dims(image, 0)
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

    return JSONResponse({
        'success' : 'success',
        'class' : predicted_class,
        'confidence' : float(confidence)
    })


if __name__ == "__main__":
    uvicorn.run("main:app", host='localhost', port=8000, reload=True)
