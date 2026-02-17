import cv2
import time
import numpy as np
from collections import deque
from .geometry import get_pixel_coords, calculate_distance
from .analysis import (
    calculate_canthal_tilt,
    calculate_fwhr,
    calculate_midface_ratio,
    calculate_facial_symmetry,
    calculate_golden_ratio,
    assess_photo_quality,
    calculate_percentile
)

# MediaPipe Setup
try:
    from mediapipe import solutions
    from mediapipe.framework.formats import landmark_pb2
    mp_face_mesh = solutions.face_mesh
    mp_drawing = solutions.drawing_utils
    mp_drawing_styles = solutions.drawing_styles
except ImportError:
    print("Using alternative MediaPipe import...")
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Movement & Smoothing State
previous_landmarks = None
movement_threshold = 2.0
stillness_counter = 0

SMOOTHING_WINDOW = 15
measurement_buffer = {
    'canthal_tilt': deque(maxlen=SMOOTHING_WINDOW),
    'fwhr': deque(maxlen=SMOOTHING_WINDOW),
    'midface_ratio': deque(maxlen=SMOOTHING_WINDOW),
    'facial_symmetry': deque(maxlen=SMOOTHING_WINDOW),
    'golden_ratio': deque(maxlen=SMOOTHING_WINDOW),
}

def detect_movement(landmarks, w, h):
    global previous_landmarks, stillness_counter
    if previous_landmarks is None:
        previous_landmarks = landmarks
        return False, 0
    
    key_points = [1, 33, 263, 234, 454, 152]
    total_movement = 0
    for point_id in key_points:
        curr = get_pixel_coords(landmarks, point_id, w, h)
        prev = get_pixel_coords(previous_landmarks, point_id, w, h)
        total_movement += calculate_distance(curr, prev)
    
    avg_movement = total_movement / len(key_points)
    previous_landmarks = landmarks
    
    is_still = avg_movement < movement_threshold
    if is_still:
        stillness_counter += 1
    else:
        stillness_counter = 0
        
    return is_still, avg_movement

def get_smoothed_metrics(raw_metrics):
    smoothed = {}
    for key, val in raw_metrics.items():
        measurement_buffer[key].append(val)
        data = list(measurement_buffer[key])
        if len(data) > 0:
            smoothed[key] = round(np.median(data), 2)
        else:
            smoothed[key] = 0
    return smoothed

def draw_visual_guides(image, landmarks, w, h):
    overlay = image.copy()
    color_fwhr = (0, 255, 255)
    color_eye = (0, 100, 255)
    color_face = (50, 255, 50)
    
    l_cheek = get_pixel_coords(landmarks, 234, w, h)
    r_cheek = get_pixel_coords(landmarks, 454, w, h)
    brow = get_pixel_coords(landmarks, 168, w, h)
    lip = get_pixel_coords(landmarks, 0, w, h)
    
    cv2.line(overlay, l_cheek, r_cheek, color_fwhr, 2)
    center_x = (l_cheek[0] + r_cheek[0]) // 2
    cv2.line(overlay, (center_x, brow[1]), (center_x, lip[1]), color_fwhr, 2)
    
    l_in = get_pixel_coords(landmarks, 133, w, h)
    l_out = get_pixel_coords(landmarks, 33, w, h)
    r_in = get_pixel_coords(landmarks, 362, w, h)
    r_out = get_pixel_coords(landmarks, 263, w, h)
    cv2.line(overlay, l_in, l_out, color_eye, 2)
    cv2.line(overlay, r_in, r_out, color_eye, 2)
    
    nose = get_pixel_coords(landmarks, 2, w, h)
    chin = get_pixel_coords(landmarks, 152, w, h)
    cv2.line(overlay, nose, chin, color_face, 2)
    
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

def process_frame(frame, is_scanning, scan_buffer):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    
    current_metrics = {}
    quality_info = []
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        
        is_still, movement = detect_movement(landmarks, w, h)
        quality_info = assess_photo_quality(landmarks, w, h)
        
        if movement < movement_threshold * 2:
            raw_metrics = {
                'canthal_tilt': calculate_canthal_tilt(landmarks, w, h),
                'fwhr': calculate_fwhr(landmarks, w, h),
                'midface_ratio': calculate_midface_ratio(landmarks, w, h),
                'facial_symmetry': calculate_facial_symmetry(landmarks, w, h),
                'golden_ratio': calculate_golden_ratio(landmarks, w, h)
            }
            
            current_metrics = get_smoothed_metrics(raw_metrics)
            
            if is_scanning:
                for k, v in current_metrics.items():
                    scan_buffer[k].append(v)
        
        draw_visual_guides(frame, landmarks, w, h)
        
        mp_drawing.draw_landmarks(
            frame, landmarks, mp_face_mesh.FACEMESH_TESSELATION,
            None, mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        
    return frame, current_metrics, quality_info
