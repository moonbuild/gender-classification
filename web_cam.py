import cv2
import numpy as np
from tensorflow.keras.models import load_model


face_cascade = cv2.CascadeClassifier('/home/tiltedcrown/Downloads/Model-gender/haarcascade_frontalface_default.xml')
gender_model = load_model('model-gender.h5')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            resized_face = cv2.resize(face, (250, 250))

            preprocessed_face = resized_face / 255.0
            preprocessed_face = np.expand_dims(preprocessed_face, axis=0)

            # use the model to detect gender
            gender_prediction = gender_model.predict(preprocessed_face)[0][0]
            if gender_prediction > 0.4:
                gender='Male'
                color=(255,0,0)
            else:
                gender='Female'
                color=(0,0,255)

            # rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            confidence = round(gender_prediction * 100, 2)
            cv2.putText(frame, f'{gender}: {confidence}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow('Gender Detection', frame)

        # Exit---> q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
