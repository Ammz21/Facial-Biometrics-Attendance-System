import face_recognition
import cv2
import numpy as np

aaron_image = face_recognition.load_image_file("dataset/Aaron.jpeg")
aaron_face_encoding = face_recognition.face_encodings(aaron_image)[0]

aleena_kurian_image = face_recognition.load_image_file("dataset/Aleena Kurian.jpg")
aleena_kurian_face_encoding = face_recognition.face_encodings(aleena_kurian_image)[0]

anju_image = face_recognition.load_image_file("dataset/Anju.jpg")
anju_face_encoding = face_recognition.face_encodings(anju_image)[0]

annaliya_image = face_recognition.load_image_file("dataset/Annaliya.jpg")
annaliya_face_encoding = face_recognition.face_encodings(annaliya_image)[0]

aswathy_s_image = face_recognition.load_image_file("dataset/Aswathy s.JPG")
aswathy_s_face_encoding = face_recognition.face_encodings(aswathy_s_image)[0]

aswin_image = face_recognition.load_image_file("dataset/Aswin.jpg")
aswin_face_encoding = face_recognition.face_encodings(aswin_image)[0]

basil_image = face_recognition.load_image_file("dataset/Basil.jpg")
basil_face_encoding = face_recognition.face_encodings(basil_image)[0]

ben_image = face_recognition.load_image_file("dataset/ben jose.jpeg")
ben_face_encoding = face_recognition.face_encodings(ben_image)[0]



known_face_encodings = [
    aaron_face_encoding,
    aleena_kurian_face_encoding,
    anju_face_encoding,
    annaliya_face_encoding,
    aswathy_s_face_encoding,
    aswin_face_encoding,
    basil_face_encoding,
    ben_face_encoding
]

known_face_names = [
    "Aaron P Laju",
    "Aleena Kurian",
    "Anju M Kammath",
    "Annaliya V G",
    "Aswathy S",
    "Aswin Dileep",
    "Basil Aliaz",
    "Ben Jose Joseph"
]

known_faces = list(zip(known_face_encodings, known_face_names))

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = ""

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom + 10), font, 1.0, (255, 255, 255), 1)

    frame = cv2.resize(frame,(720,480))
    cv2.imshow('Face Attendance', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
