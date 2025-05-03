import cv2
import numpy as np
import pygame
import time
import mediapipe as mp
from scipy.spatial import distance

# Initialize Pygame and load music
pygame.mixer.init()
pygame.mixer.music.load('audio/alert.wav')

# Minimum threshold of eye aspect ratio below which alarm is triggered
EYE_ASPECT_RATIO_THRESHOLD = 0.3

# Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered
EYE_ASPECT_RATIO_CONSEC_FRAMES = 50

# Counts no. of consecutive frames below threshold value
COUNTER = 0

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Start webcam video capture
video_capture = cv2.VideoCapture(0)

# Initialize Mediapipe Face Mesh
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while True:
        ret, frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        
        # Convert the frame to RGB (Mediapipe works with RGB images)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and get face mesh landmarks
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Kiểm tra xem có đủ điểm để lấy không
                if len(face_landmarks.landmark) > 468:  # Face Mesh có 468 điểm
                    # Lấy các điểm đặc trưng của mắt trái và mắt phải
                    left_eye = [face_landmarks.landmark[33], face_landmarks.landmark[133], face_landmarks.landmark[160], 
                                face_landmarks.landmark[144], face_landmarks.landmark[163], face_landmarks.landmark[249]]
                    right_eye = [face_landmarks.landmark[362], face_landmarks.landmark[263], face_landmarks.landmark[374], 
                                 face_landmarks.landmark[390], face_landmarks.landmark[466], face_landmarks.landmark[469]]

                    # Tiến hành xử lý tiếp theo nếu các điểm này tồn tại
                    h, w, _ = frame.shape
                    left_eye_points = [(int(point.x * w), int(point.y * h)) for point in left_eye]
                    right_eye_points = [(int(point.x * w), int(point.y * h)) for point in right_eye]

                    # Vẽ đường viền mắt
                    cv2.polylines(frame, [np.array(left_eye_points)], isClosed=True, color=(0, 255, 0), thickness=1)
                    cv2.polylines(frame, [np.array(right_eye_points)], isClosed=True, color=(0, 255, 0), thickness=1)

                    # Tính toán EAR (Eye Aspect Ratio) cho mắt trái và mắt phải
                    def eye_aspect_ratio(eye_points):
                        A = distance.euclidean(eye_points[1], eye_points[5])
                        B = distance.euclidean(eye_points[2], eye_points[4])
                        C = distance.euclidean(eye_points[0], eye_points[3])
                        return (A + B) / (2.0 * C)

                    left_eye_ear = eye_aspect_ratio(left_eye_points)
                    right_eye_ear = eye_aspect_ratio(right_eye_points)
                    eye_aspect_ratio_value = (left_eye_ear + right_eye_ear) / 2.0

                    # Kiểm tra nếu EAR nhỏ hơn ngưỡng
                    if eye_aspect_ratio_value < EYE_ASPECT_RATIO_THRESHOLD:
                        COUNTER += 1
                        if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                            pygame.mixer.music.play(-1)
                            cv2.putText(frame, "You are Drowsy", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                    else:
                        pygame.mixer.music.stop()
                        COUNTER = 0
                else:
                    print("Không đủ điểm đặc trưng để nhận diện khuôn mặt hoặc mắt.")

        # Show video feed
        cv2.imshow('Video', frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture and destroy windows
video_capture.release()
cv2.destroyAllWindows()
