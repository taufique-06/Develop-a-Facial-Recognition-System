from face_recognition import api
import cv2
import numpy as np
import time
import requests

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def send_simple_message(name):
    message_text = f"The system has recognized a person. Here are the details: Name: {name}. Please confirm to open the door."

    return requests.post(
        "https://api.mailgun.net/v3/your_domain_name/messages",
        auth=("api", "your_api_key"),
        data={
            "from": "ta5996u@gmail.com",
            "to": ["taufiqueazad999@gmail.com"],
            "subject": "Face Recognized!",
            "text": message_text
        }
    )

# Open the camera
video_capture = cv2.VideoCapture(0)

# Load sample pictures and train them to recognize the pictures in real time
M1_image = api.load_image_file("TestData/M1/M1Main.png")
M1_face_encoding = api.face_encodings(M1_image)[0]

#Train M2
M2_image = api.load_image_file("TestData/M2/M2Main.jpg")
M2_face_encoding = api.face_encodings(M2_image)[0]

#Train M3
M3_image = api.load_image_file("TestData/M3/M3Main.jpg")
M3_face_encoding = api.face_encodings(M3_image)[0]

#Train F1
F1_image = api.load_image_file("TestData/F1/F1Main.jpg")
F1_face_encoding = api.face_encodings(F1_image)[0]

#Train F2
F2_image = api.load_image_file("TestData/F2/F2Main.jpg")
F2_face_encoding = api.face_encodings(F2_image)[0]

#Train F3
F3_image = api.load_image_file("TestData/F3/F3Main.jpg")
F3_face_encoding = api.face_encodings(F3_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    M1_face_encoding,
    M2_face_encoding,
    M3_face_encoding,
    F1_face_encoding,
    F2_face_encoding,
    F3_face_encoding
]
known_face_names = [
    "M1",
    "M2",
    "M3",
    "F1",
    "F2",
    "F3",
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

frame_count = 0
start_time = time.time()

while True:
    ret, frame = video_capture.read()

    # Frame Rotate to handle different orientations
    rotated_frames = [frame]
    for angle in [90, 180, 270]: 
        rotated_frames.append(rotate_image(frame, angle))
    
    detected_faces = False
    for rotated_frame in rotated_frames:
        if process_this_frame:
            small_frame = cv2.resize(rotated_frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find all the faces and face encodings in the current frame of video
            face_locations = api.face_locations(rgb_small_frame)
            face_encodings = api.face_encodings(rgb_small_frame, face_locations)
            
            if face_encodings:
                detected_faces = True
                break
        
        if detected_faces:
            break

    process_this_frame = not process_this_frame

    # Process face recognitions for the last detected frame
    face_names = []
    for face_encoding in face_encodings:
        matches = api.compare_faces(known_face_encodings, face_encoding,0.66)
        name = "Unknown"
        face_distances = api.face_distance(known_face_encodings, face_encoding);
        # print(face_distances);
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            request = send_simple_message(name);
            print(request.status_code);
        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
