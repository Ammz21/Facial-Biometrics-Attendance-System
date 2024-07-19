from flask import Flask, render_template, Response, request, url_for, redirect, current_app
import cv2
import face_recognition
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import sqlite3

app = Flask(__name__)

def create_connection():
    conn = None
    try:
        conn = sqlite3.connect('database.db')
    except sqlite3.Error as e:
        print("Error ", e)
    return conn

def create_table(conn):
    try:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS registrations (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT NOT NULL,
                            rollno TEXT NOT NULL,
                            class TEXT NOT NULL
                        )''')
    except sqlite3.Error as e:
        print("Error creating table:", e)

def insert_data(conn, name, rollno, class_):
    try:
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO registrations (name, rollno, class) VALUES (?, ?, ?)''', (name, rollno, class_))
        conn.commit()
    except sqlite3.Error as e:
        print("Error inserting data:", e)

@app.route('/register', methods=['GET', 'POST'])
def register():
    global name
    name = " "
    if request.method == 'POST':
        name = request.form['name']
        rollno = request.form['rollno']
        class_ = request.form['class']
        
        conn = create_connection()
        create_table(conn)
        insert_data(conn, name, rollno, class_)
        conn.close()
        
        return redirect(url_for('index'))
    
    return render_template('register.html')

def video():
    path = 'dataset'
    if not os.path.exists(path):
        os.makedirs(path)

    video_capture = cv2.VideoCapture(0)
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # if 'capture' in request.args:
            #     image_path = os.path.join(path, f'{name}.jpg')
            #     cv2.imwrite(image_path, frame)

                # return redirect(url_for('index'))

    video_capture.release()
    cv2.destroyAllWindows()

@app.route('/capture_frame')
def capture_frame():
    name = request.args.get('name', '')
    path = 'dataset'
    print(name)
    if not os.path.exists(path):
        os.makedirs(path)

    video_capture = cv2.VideoCapture(0)
    success, frame = video_capture.read()
    if success:
        image_path = os.path.join(path, f'{name}.jpg')
        cv2.imwrite(image_path, frame)

    video_capture.release()
    cv2.destroyAllWindows()
    return redirect(url_for('register'))

def encode_images(directory):
    encoded_faces = []
    labels = []
    label_dict = {}
    count = 0
    for filename in os.listdir(directory):
        image_path = os.path.join(directory, filename)
        image = cv2.imread(image_path)
        if image is not None:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image)
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            if face_encodings:
                encoded_faces.append(face_encodings[0])
                labels.append(count)
                label_dict[count] = filename.split('.')[0]
                count += 1
    return np.array(encoded_faces), np.array(labels), label_dict

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    path = 'dataset'
    encoded_faces, labels, label_dict = encode_images(path)
    
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(encoded_faces, labels)
    
    video_capture = cv2.VideoCapture(0)
    
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(encoded_faces, face_encoding)
                if True in matches:
                    distances, indices = knn.kneighbors([face_encoding])
                    min_distance_index = indices[0][0]
                    label = knn.predict([face_encoding])[0]
                    label_name = label_dict[label] 
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, label_name, (left + 6, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255), 2)
                else:
                    label_name = " "  

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


    video_capture.release()
    cv2.destroyAllWindows()


@app.route('/video_capture')
def video_capture():
    return Response(video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance')
def attendance():
    return render_template('attendance.html')

if __name__ == "__main__":
    app.run(debug=True)