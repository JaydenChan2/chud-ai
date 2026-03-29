import cv2
import numpy as np
from .geometry import get_pixel_coords
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
    static_image_mode=True, # For API stateless mode
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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

def process_frame_stateless(frame):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    
    raw_metrics = {}
    quality_info = []
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        
        quality_info = assess_photo_quality(landmarks, w, h)
        
        raw_metrics = {
            'canthal_tilt': round(calculate_canthal_tilt(landmarks, w, h), 2),
            'fwhr': round(calculate_fwhr(landmarks, w, h), 2),
            'midface_ratio': round(calculate_midface_ratio(landmarks, w, h), 2),
            'facial_symmetry': round(calculate_facial_symmetry(landmarks, w, h), 2),
            'golden_ratio': round(calculate_golden_ratio(landmarks, w, h), 2)
        }
        
        draw_visual_guides(frame, landmarks, w, h)
        
        mp_drawing.draw_landmarks(
            frame, landmarks, mp_face_mesh.FACEMESH_TESSELATION,
            None, mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        
    return frame, raw_metrics, quality_info
