import cv2
import os

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
print(cv2_base_dir)
face_haar_model_path = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
eye_haar_model_path = os.path.join(cv2_base_dir, 'data/haarcascade_eye.xml')


cap = cv2.VideoCapture(0)
ret, src = cap.read()
gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

face_cascade = cv2.CascadeClassifier()
face_cascade.load(face_haar_model_path)
faces = face_cascade.detectMultiScale(gray)

eye_cascade = cv2.CascadeClassifier()
eye_cascade.load(eye_haar_model_path)
eyes = eye_cascade.detectMultiScale(gray)

for (x,y,w,h) in faces:
    frame = cv2.rectangle(src, (x,y), (x+w, y+h), (255, 0, 0), 1)

    for (x2, y2, w2, h2) in eyes:
        frame = cv2.rectangle(src, (x2,y2), (x2+w2, y2+h2), (0, 255, 0), 1)

cv2.imshow("Face", frame)
cv2.waitKey(5000)