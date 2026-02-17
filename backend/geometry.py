import math

def get_pixel_coords(landmarks, landmark_id, w, h):
    """Convert normalized landmark coordinates to pixel coordinates."""
    landmark = landmarks.landmark[landmark_id]
    return int(landmark.x * w), int(landmark.y * h)

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def get_interpupillary_distance(landmarks, w, h):
    """Calculate distance between pupils (IPD)."""
    left_pupil = get_pixel_coords(landmarks, 468, w, h)
    right_pupil = get_pixel_coords(landmarks, 473, w, h)
    ipd = calculate_distance(left_pupil, right_pupil)
    return ipd if ipd > 0 else 1

def normalize_face_rotation(landmarks, w, h):
    """Calculate face rotation angle to correct measurements."""
    left_eye = get_pixel_coords(landmarks, 33, w, h)
    right_eye = get_pixel_coords(landmarks, 263, w, h)
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    return math.degrees(math.atan2(dy, dx))
