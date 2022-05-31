import face_recognition
import cv2
from os import listdir
import numpy as np
import pickle


def progress_bar(progress, total):
    percent = 100 * (progress / float(total))
    bar = "█" * int(percent) + "-" * (100 - int(percent))
    print(f"\r|{bar}| {percent:.2f}%", end="\r")


def save_variables():
    with open(
        "/home/mike/scripts/opencv/face_recognition/target_images.txt", "wb"
    ) as f:
        pickle.dump(target_images, f)
    with open(
        "/home/mike/scripts/opencv/face_recognition/target_encodings.txt", "wb"
    ) as f:
        pickle.dump(target_encodings, f)


PATH = "/home/mike/scripts/opencv/face_recognition/faces"
cap = cv2.VideoCapture(0)
image_files = listdir(PATH)
print("\nCaricamento Immagini\n")
with open("target_images.txt", "rb") as f:
    target_images = pickle.load(f)
with open("target_encodings.txt", "rb") as f:
    target_encodings = pickle.load(f)

if len(target_images) < len(image_files):

    target_images = []
    target_encodings = []
    print("\nCaricamento Immagini\n")
    for i in range(len(image_files)):
        target_images.append(
            face_recognition.load_image_file(f"{PATH}/{image_files[i]}")
        )
        target_encodings.append(face_recognition.face_encodings(target_images[i]))
        progress_bar(i + 1, len(image_files))
    save_variables()

target_names = [image_file.split(".")[0] for image_file in image_files]

process_this_frame = True

while True:
    ret, frame = cap.read()

    small_frame = cv2.resize(frame, None, fx=0.20, fy=0.20)
    rgb_small_frame = cv2.cvtColor(small_frame, 4)
    matches = []

    if process_this_frame:

        face_location = face_recognition.face_locations(rgb_small_frame)
        frame_encodings = face_recognition.face_encodings(rgb_small_frame)
        if frame_encodings:
            for i, image in enumerate(image_files):
                frame_face_encoding = frame_encodings[0]
                matches.append(
                    face_recognition.compare_faces(
                        target_encodings[i], frame_face_encoding
                    )
                )

        matches = np.asarray(matches).ravel()
        image_files = np.asarray(image_files)
        if len(matches) > 0 and True in matches:
            label = image_files[matches][0].split(".")[0]
        else:
            label = "unknown"
    process_this_frame = not process_this_frame

    if face_location:
        top, right, bottom, left = face_location[0]
        top *= 5
        right *= 5
        bottom *= 5
        left *= 5

        cv2.rectangle(frame, (left, top), (right, bottom), (128, 128, 0), 2)

        cv2.rectangle(
            frame, (left, bottom - 30), (right, bottom), (128, 128, 0), cv2.FILLED
        )
        label_font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(
            frame, label, (left + 6, bottom - 6), label_font, 0.8, (255, 255, 255), 1
        )
    cv2.imshow("VideoFeed", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
