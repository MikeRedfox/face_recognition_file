import face_recognition
import cv2

cap = cv2.VideoCapture(0)

image_file = input("Target image file > ")
target_image = face_recognition.load_image_file(image_file)
target_encoding = face_recognition.face_encodings(target_image)[0]

target_name = image_file.split(".")[0].split("/")[1]

process_this_frame = True

while True:
    ret, frame = cap.read()

    small_frame = cv2.resize(frame, None, fx=0.20, fy=0.20)
    rgb_small_frame = cv2.cvtColor(small_frame, 4)

    if process_this_frame:

        face_location = face_recognition.face_locations(rgb_small_frame)
        frame_encodings = face_recognition.face_encodings(rgb_small_frame)

        if frame_encodings:
            frame_face_encoding = frame_encodings[0]
            match = face_recognition.compare_faces(
                [target_encoding], frame_face_encoding
            )[0]
            label = target_name if match else "Unknow"

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
