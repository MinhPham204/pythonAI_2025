import cv2 as cv
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Read image
img = cv.imread('images/testface.jpg')

# Convert image to RGB (Mediapipe works with RGB images)
rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Initialize Face Mesh
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    # Process the image to get face landmarks
    results = face_mesh.process(rgb_img)

    # Draw face landmarks if they are found
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw landmarks on the face
            mp_drawing.draw_landmarks(img, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, 
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1))

            # Define the indexes for the left and right eyes based on the FaceMesh model
            # Left eye landmarks: 33, 133, 160, 144, 163, 249
            left_eye = [face_landmarks.landmark[33], face_landmarks.landmark[133], face_landmarks.landmark[160], 
                        face_landmarks.landmark[144], face_landmarks.landmark[163], face_landmarks.landmark[249]]
            # Right eye landmarks: 362, 263, 374, 390, 466, 469
            right_eye = [face_landmarks.landmark[362], face_landmarks.landmark[263], face_landmarks.landmark[374], 
                         face_landmarks.landmark[390], face_landmarks.landmark[466], face_landmarks.landmark[469]]

            # Convert landmark points to pixel coordinates for drawing
            h, w, _ = img.shape
            left_eye_points = [(int(point.x * w), int(point.y * h)) for point in left_eye]
            right_eye_points = [(int(point.x * w), int(point.y * h)) for point in right_eye]

            # Draw eye contours (optional)
            cv.polylines(img, [np.array(left_eye_points)], isClosed=True, color=(0, 255, 0), thickness=1)
            cv.polylines(img, [np.array(right_eye_points)], isClosed=True, color=(0, 255, 0), thickness=1)

# Show the result image
cv.imshow('Image', img)
cv.waitKey(0)
cv.destroyAllWindows()
