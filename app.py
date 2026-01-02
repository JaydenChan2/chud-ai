import cv2
import mediapipe as mp

# Standard way to import
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Start your webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    # Convert image to RGB for MediaPipe
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        print("Face Detected! Ready for analysis.")
        # This is where we will eventually put the PSL rating logic

    cv2.imshow('Chud.ai Alpha', image)
    if cv2.waitKey(5) & 0xFF == 27: break # Press 'Esc' to quit

cap.release()
