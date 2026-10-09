import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh solutions
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
# NEW: Import default drawing styles to get the grid/tesselation style
mp_drawing_styles = mp.solutions.drawing_styles

# Open webcam
cap = cv2.VideoCapture(0)

# Initialize the Face Mesh model with refined landmarks enabled
with mp_face_mesh.FaceMesh(
    static_image_mode=False,      # We are processing a video stream, not a single photo
    max_num_faces=1,              # Let's stick to one face for now
    refine_landmarks=True,        # This is crucial for detailed eye/lip outlines
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

  while cap.isOpened():
    success, frame = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    # 1. CRITICAL STEP: Flip the image horizontally for the mirror effect
    frame = cv2.flip(frame, 1)

    # MediaPipe expects RGB images, OpenCV captures BGR
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to find the face mesh
    results = face_mesh.process(rgb_frame)

    # Draw the face mesh connections (the grid/tesselation)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        # NEW: Use the predefined drawing utility specifically for the face mesh grid
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION, # This draws the triangulation grid
            # Use default styles for the landmarks and the connections
            landmark_drawing_spec=None, # Set to None to only draw connections
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        
        # OPTIONAL: If you ALSO want to draw the outline of key features (lips, eyes, irises)
        # You can add this second draw call using FACEMESH_CONTOURS
        # mp_drawing.draw_landmarks(
        #     image=frame,
        #     landmark_list=face_landmarks,
        #     connections=mp_face_mesh.FACEMESH_CONTOURS,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
        # )

    # Display the resulting frame
    cv2.imshow("Face Mesh Grid", frame)

    if cv2.waitKey(5) & 0xFF == ord("q"):
      break

cap.release()
cv2.destroyAllWindows()