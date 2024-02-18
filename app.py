# from flask import Flask, render_template, Response
# import cv2
# import dlib
# from scipy.spatial import distance as dist
# import numpy as np
# import pygame

# app = Flask(__name__)

# # Load face detector and landmark predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# # Load pre-trained model for drowsiness detection
# # model = cv2.dnn.readNet("haarcascade_eye.xml")

# def eye_aspect_ratio(eye):
#     A = dist.euclidean(eye[1], eye[5])
#     B = dist.euclidean(eye[2], eye[4])
#     C = dist.euclidean(eye[0], eye[3])
#     ear = (A + B) / (2.0 * C)
#     return ear

# def detect_drowsiness(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)

#     for face in faces:
#         shape = predictor(gray, face)
#         shape = shape_to_np(shape)

#         left_eye = shape[42:48]
#         right_eye = shape[36:42]

#         left_eye_aspect_ratio = eye_aspect_ratio(left_eye)
#         right_eye_aspect_ratio = eye_aspect_ratio(right_eye)

#         avg_eye_aspect_ratio = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2.0

#         if avg_eye_aspect_ratio < 0.25:
#             return True

#     return False


# def shape_to_np(shape, dtype="int"):
#     coords = np.zeros((68, 2), dtype=dtype)
#     for i in range(0, 68):
#         coords[i] = (shape.part(i).x, shape.part(i).y)
#     return coords

# def gen():
#     cap = cv2.VideoCapture(0)
#     while True:
#         ret, frame = cap.read()
#         if detect_drowsiness(frame):
#             pygame.mixer.init()
#             sound = pygame.mixer.Sound('static/beep.mp3')
#             sound.play()
#             pygame.mixer.unpause()
#             cv2.putText(frame, "************************ALERT*************************", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         else:
#             pygame.mixer.init()
#             pygame.mixer.pause()
#         ret, jpeg = cv2.imencode('.jpg', frame)
#         frame = jpeg.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(gen(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2 as cv
import numpy as np
import base64
import threading
import dlib
from scipy.spatial import distance as dist
import pygame

app = Flask(__name__)
socketio = SocketIO(app)


for i in range(10):  # Try indices 0 to 9
    cap = cv.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera at index {i} is available.")
        cap.release()



# Lane detection
cap_lane = cv.VideoCapture(0)
cap_drowsy = cv.VideoCapture(0)

def get_region(img, vertices):
    mask = np.zeros_like(img)
    match_mask = 255
    cv.fillPoly(mask, vertices, match_mask)
    masked_image = cv.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines):
    img = np.copy(img)
    line_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), thickness=4)

    img = cv.addWeighted(img, 0.8, line_image, 1, 0.0)
    return img


def proccess_img(img):

    h = img.shape[0]
    w = img.shape[1]

    region_of_interest_vertex = np.array([[
        (0, h),
        (w/2, 1.5*h/2),
        (w, h)
        # (0+w/6, h-h/10),
        # (w/2, 3*h/5),
        # (w-w/6, h-h/10)
    ]], np.int64)

    gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    canny_img = cv.Canny(gray_img, 100, 120)

    region_cropped = get_region(canny_img, region_of_interest_vertex)

    lines = cv.HoughLinesP(region_cropped,
                           rho=2,
                           theta=np.pi/180,
                           threshold=50,
                           lines=np.array([]),
                           minLineLength=40,
                           maxLineGap=100)

    if lines is not None:
        img_lines = draw_lines(img, lines)
        return img_lines
    else:
        return None

def video_stream_lane():
    while True:
        ret, frame = cap_lane.read()
        if not ret:
            break

        frame = cv.resize(frame, (1080, 720))
        processed_img = proccess_img(frame)

        if processed_img is not None:
            _, encoded_img = cv.imencode('.jpg', processed_img)
            img_data = 'data:image/jpeg;base64,' + base64.b64encode(encoded_img.tobytes()).decode('utf-8')
            socketio.emit('image_response_lane', {'image': img_data})

# Drowsiness detection
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_drowsiness(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
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
   
    while True:
        ret, frame = cap_drowsy.read()
        if detect_drowsiness(frame):
            pygame.mixer.init()
            sound = pygame.mixer.Sound('static/beep.mp3')
            sound.play()
            pygame.mixer.unpause()
            cv.putText(frame, "************************ALERT*************************", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            pygame.mixer.init()
            pygame.mixer.pause()
        ret, jpeg = cv.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed_lane')
def video_feed_lane():
    return Response(video_stream_lane(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_drowsy')
def video_feed_drowsy():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    threading.Thread(target=video_stream_lane).start()

if __name__ == '__main__':
    socketio.run(app, debug=True)

