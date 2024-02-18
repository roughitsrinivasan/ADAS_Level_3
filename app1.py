from flask import Flask, render_template, Response
import cv2
import dlib
from scipy.spatial import distance as dist
import numpy as np
import pygame

app = Flask(__name__)

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load pre-trained model for drowsiness detection
# model = cv2.dnn.readNet("haarcascade_eye.xml")

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_drowsiness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = shape_to_np(shape)

        left_eye = shape[42:48]
        right_eye = shape[36:42]

        left_eye_aspect_ratio = eye_aspect_ratio(left_eye)
        right_eye_aspect_ratio = eye_aspect_ratio(right_eye)

        avg_eye_aspect_ratio = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2.0

        if avg_eye_aspect_ratio < 0.25:
            return True

    return False


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def gen():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if detect_drowsiness(frame):
            pygame.mixer.init()
            sound = pygame.mixer.Sound('static/beep.mp3')
            sound.play()
            pygame.mixer.unpause()
            cv2.putText(frame, "************************ALERT*************************", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            pygame.mixer.init()
            pygame.mixer.pause()
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
