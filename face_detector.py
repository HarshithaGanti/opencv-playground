import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection components
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
) as face_detection:
  while cap.isOpened():
    success, frame = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    # MediaPipe expects RGB images, OpenCV captures BGR
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    # Draw the face detections on the image
    if results.detections:
      for detection in results.detections:
        mp_drawing.draw_detection(frame, detection)

    # Display the resulting frame
    cv2.imshow("MediaPipe Face Detector", frame)

    if cv2.waitKey(5) & 0xFF == ord("q"):
      break

cap.release()
cv2.destroyAllWindows()