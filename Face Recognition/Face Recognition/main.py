from face_recognition import api
import cv2
import numpy as np

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

# Open the camera
video_capture = cv2.VideoCapture(0)

# Load sample pictures and train them to recognize the pictures in real time
test_image = api.load_image_file("Images/test.png")
test_image1 = face_recognition.load_image_file("Images/brown.jpg")
test_image2 = face_recognition.load_image_file("Images/black.jpg")
test_face_encoding = api.face_encodings(test_image)[0]
face_land = face_recognition.face_landmarks(test_image);
face_land1 = face_recognition.face_landmarks(test_image1);
face_land2 = face_recognition.face_landmarks(test_image2);

#print(face_land1);
#print("black");
#print(face_land2);

# Create arrays of known face encodings and their names
known_face_encodings = [
    test_face_encoding,
]
known_face_names = [
    "Mr X"
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

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
        matches = api.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = api.face_distance(known_face_encodings, face_encoding);
        # print(face_distances);
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
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

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
