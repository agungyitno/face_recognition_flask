from flask import Flask, render_template, Response, request, jsonify, redirect, flash
import cv2
import numpy as np
import os
from PIL import Image
from train import latihWajah

# --- Agung Prayitno ---
# --- Informatika-A ---
# --- Teknologi Informasi ---
# --- UNWAHA ---

app = Flask(__name__)
app.secret_key = '160598'


cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)


@app.route('/')
def index():
    return render_template('index.html')


def gen_frame(mode='pengenalan', id_wajah=1):
    wajah_dir = 'dataset'
    train_dir = 'trainer'
    cascadePath = "classifier/haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascadePath)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(train_dir + '/trainer.yml')
    font = cv2.FONT_HERSHEY_SIMPLEX
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)
    nomor = 0
    names = dict()
    names[0] = 'Tak dikenal'
    names[1] = 'Belum dikasih nama'
    names[10] = 'agung'
    names[11] = 'abu'
    names[12] = 'abi'
    names[13] = 'adi'
    names[14] = 'alfi'
    while True:
        success, img = cam.read()
        if not success:
            break
        else:
            img = cv2.flip(img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH)),
            )
            for(x, y, w, h) in faces:
                if mode == 'daftar':
                    nomor += 1
                    face_name = 'wajah.' + \
                        str(id_wajah) + '.' + str(nomor) + '.jpg'
                    cv2.imwrite(wajah_dir + '/' +
                                face_name, gray[y:y+h, x:x+w])
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                if mode == 'pengenalan':
                    id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                    if (confidence < 100):
                        if id in names:
                            id = names[id]
                        else:
                            id = names[1]
                        confidence = "  {0}%".format(round(100 - confidence))
                    else:
                        id = names[0]
                        confidence = "  {0}%".format(round(100 - confidence))
                    cv2.putText(img, str(id), (x+5, y-5),
                                font, 1, (255, 255, 255), 2)
                    cv2.putText(img, str(confidence), (x+5, y+h-5),
                                font, 1, (255, 255, 0), 1)
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            if nomor >= 10:
                mode = 'pengenalan'
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/pengenalan')
def pengenalan():
    return render_template('index.html')


@app.route('/daftar', methods=['GET', 'POST'])
def daftar():
    return Response(gen_frame(mode='daftar', id_wajah=int(request.form['id_wajah'])),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@ app.route('/latih')
def latih():
    x_latih = latihWajah()
    flash(str(x_latih) + ' data Wajah berhasil dilatih.')
    return redirect('/')


if __name__ == '__main__':
    app.run(debug=False)
    # app.debug = True
