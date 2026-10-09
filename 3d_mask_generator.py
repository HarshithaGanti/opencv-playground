import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as face_mesh:

  while cap.isOpened():
    success, frame = cap.read()
    if not success:
      break

    # Mirror flip & color conversion
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        landmarks = face_landmarks.landmark

        # Loop through every connection in the tessellation mesh
        for connection in mp_face_mesh.FACEMESH_TESSELATION:
          idx1, idx2 = connection

          pt1 = landmarks[idx1]
          pt2 = landmarks[idx2]

          # Convert normalized coordinates (0.0 to 1.0) to actual pixel values
          x1, y1 = int(pt1.x * w), int(pt1.y * h)
          x2, y2 = int(pt2.x * w), int(pt2.y * h)

          # Get the average z depth of the two points
          avg_z = (pt1.z + pt2.z) / 2.0

          # Map the z value to a brightness factor and cast to native Python int
          brightness = int(np.clip((-avg_z + 0.05) * 1500, 20, 255))

          # Create a custom color tuple using standard Python ints
          color = (brightness, 255 - brightness, brightness)

          # Draw the individual line segment
          cv2.line(frame, (x1, y1), (x2, y2), color, 1)

    cv2.imshow("3D Shaded Face Mask", frame)

    if cv2.waitKey(5) & 0xFF == ord("q"):
      break

cap.release()
cv2.destroyAllWindows()