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

    # Mirror flip & BGR to RGB
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)
    overlay = frame.copy()

    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        landmarks = face_landmarks.landmark
        points_2d = []
        points_z = []

        # Extract pixel coordinates and z-depth for all landmarks
        for lm in landmarks:
          px, py = int(lm.x * w), int(lm.y * h)
          points_2d.append((px, py))
          points_z.append(lm.z)

        # Set up OpenCV Subdiv2D for Delaunay triangulation over the face bounding rect
        rect = (0, 0, w, h)
        subdiv = cv2.Subdiv2D(rect)

        # Insert points into the subdivder (filter out duplicates if any)
        for p in points_2d:
          subdiv.insert(p)

        # Get triangle list
        triangle_list = subdiv.getTriangleList()

        for t in triangle_list:
          pt1 = (int(t[0]), int(t[1]))
          pt2 = (int(t[2]), int(t[3]))
          pt3 = (int(t[4]), int(t[5]))

          # Ensure the triangle is actually inside the frame boundaries
          if (
              0 <= pt1[0] < w
              and 0 <= pt1[1] < h
              and 0 <= pt2[0] < w
              and 0 <= pt2[1] < h
              and 0 <= pt3[0] < w
              and 0 <= pt3[1] < h
          ):

            triangle_vertices = np.array([pt1, pt2, pt3], np.int32)

            # Approximate depth shading by picking a default medium depth or matching nearby points
            # For a smooth look, we use a constant neon cyber tone or standard shading
            color = (150, 255, 100)  # Neon green-cyan matrix vibe

            # Fill the convex polygon triangle
            cv2.fillConvexPoly(overlay, triangle_vertices, color)

            # Optional: draw fine dark borders around triangles for definition
            cv2.polylines(overlay, [triangle_vertices], True, (0, 0, 0), 1)

    # Blend the overlay with transparency
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.imshow("Low-Poly Solid 3D Face Mask", frame)

    if cv2.waitKey(5) & 0xFF == ord("q"):
      break

cap.release()
cv2.destroyAllWindows()