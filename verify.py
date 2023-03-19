import cv2
import requests
import numpy as np
from io import BytesIO
from PIL import Image
import os
import urllib.request
from keras.models import load_model

url = 'https://source.unsplash.com/random?celebrity'

dir_name = 'Checking'
os.makedirs(dir_name, exist_ok=True)
os.makedirs(os.path.join(dir_name, 'male'), exist_ok=True)
os.makedirs(os.path.join(dir_name, 'female'), exist_ok=True)
os.makedirs(os.path.join(dir_name, 'cropped_images'), exist_ok=True)

model = load_model('model-gender.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


for i in range(10):
    response = requests.get(url)

    img = Image.open(BytesIO(response.content))
    img_path = os.path.join(dir_name,f'face_{i}.jpg')
    img.save(img_path)
    img = cv2.imread(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for j, (x,y,w,h) in enumerate(faces):
        face= img[y:y+h, x:x+w]
        face_path = os.path.join(dir_name,'cropped_images', f'face_{i}_{j}.jpg')
        cv2.imwrite(face_path, face)

        #from here we prepare to predict the gender
        crop_img = Image.open(face_path)
        crop_img = crop_img.resize((250,250))
        img_array = np.array(crop_img)
        img_array = img_array / 255.0

        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)
        if pred[0][0] > 0.5:
            gender = 'Male'
            color=(255,0,0)
        else:
            gender = 'Female'
            color=(0,0,255)
        
        print(f"pred: {pred} | gender: {gender}")

        confidence = round(pred[0][0] * 100, 2)
        text = f'{gender}: {confidence}%'
        cv2.rectangle(img, (x,y), (x+w, y+h), color, thickness=4)
        cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if gender == 'Male':
            img_path = os.path.join(dir_name, 'male', f'face_{i}_{j}.jpg')
            cv2.imwrite(img_path, img)
        else:
            img_path = os.path.join(dir_name, 'female', f'face_{i}_{j}.jpg')
            cv2.imwrite(img_path, img)