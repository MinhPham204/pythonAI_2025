import cv2
import mediapipe as mp

# Initialize mediapipe face detection and drawing utilities
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Set up the video capture
video_capture = cv2.VideoCapture(0)

# Initialize face detector from Mediapipe
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:  # Increase confidence
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB (Mediapipe works with RGB images)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and get face detections
        results = face_detection.process(rgb_frame)

        # Draw face detections
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                              int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Show the video feed with face detection
        cv2.imshow('Video', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
