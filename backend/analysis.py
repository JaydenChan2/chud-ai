import math
import numpy as np
from .geometry import (
    get_pixel_coords, 
    calculate_distance, 
    get_interpupillary_distance, 
    normalize_face_rotation
)

# Percentile Reference Data
REFERENCE_STATS = {
    'canthal_tilt': {'mean': 4.0, 'std': 3.0}, # Degrees
    'fwhr': {'mean': 1.9, 'std': 0.15},        # Ratio
    'midface_ratio': {'mean': 1.0, 'std': 0.1},# Ratio
    'facial_symmetry': {'mean': 92.0, 'std': 4.0}, # Percentage
    'golden_ratio': {'mean': 85.0, 'std': 5.0}     # Percentage
}

def calculate_canthal_tilt(landmarks, w, h):
    face_rotation = normalize_face_rotation(landmarks, w, h)
    l_inner = get_pixel_coords(landmarks, 133, w, h)
    l_outer = get_pixel_coords(landmarks, 33, w, h)
    r_inner = get_pixel_coords(landmarks, 362, w, h)
    r_outer = get_pixel_coords(landmarks, 263, w, h)
    
    l_angle = math.degrees(math.atan2(l_inner[1] - l_outer[1], l_outer[0] - l_inner[0]))
    r_angle = math.degrees(math.atan2(r_inner[1] - r_outer[1], r_outer[0] - r_inner[0]))
    
    avg_tilt = (l_angle + r_angle) / 2
    return round(avg_tilt - face_rotation, 1)

def calculate_fwhr(landmarks, w, h):
    cheek_left = get_pixel_coords(landmarks, 234, w, h)
    cheek_right = get_pixel_coords(landmarks, 454, w, h)
    width = calculate_distance(cheek_left, cheek_right)
    brow = get_pixel_coords(landmarks, 168, w, h) 
    upper_lip = get_pixel_coords(landmarks, 0, w, h)
    height = calculate_distance(brow, upper_lip)
    return round(width / height, 2) if height > 0 else 0

def calculate_midface_ratio(landmarks, w, h):
    ipd = get_interpupillary_distance(landmarks, w, h)
    l_pupil = get_pixel_coords(landmarks, 468, w, h)
    r_pupil = get_pixel_coords(landmarks, 473, w, h)
    pupil_y = (l_pupil[1] + r_pupil[1]) / 2
    upper_lip = get_pixel_coords(landmarks, 0, w, h)
    midface_height = abs(upper_lip[1] - pupil_y)
    return round(midface_height / ipd, 2) if ipd > 0 else 0

def calculate_facial_symmetry(landmarks, w, h):
    pairs = [(33, 263), (133, 362), (234, 454), (61, 291), (58, 288)]
    top = get_pixel_coords(landmarks, 168, w, h)
    bottom = get_pixel_coords(landmarks, 152, w, h)
    
    axis_vec = np.array([bottom[0] - top[0], bottom[1] - top[1]])
    axis_len = np.linalg.norm(axis_vec)
    if axis_len == 0: return 0
    axis_unit = axis_vec / axis_len
    
    scores = []
    for l_id, r_id in pairs:
        l_pt = np.array(get_pixel_coords(landmarks, l_id, w, h))
        r_pt = np.array(get_pixel_coords(landmarks, r_id, w, h))
        l_vec = l_pt - top
        r_vec = r_pt - top
        l_proj = np.dot(l_vec, axis_unit)
        r_proj = np.dot(r_vec, axis_unit)
        l_dist = np.linalg.norm(l_vec - l_proj * axis_unit)
        r_dist = np.linalg.norm(r_vec - r_proj * axis_unit)
        denom = (l_dist + r_dist) / 2
        if denom > 0:
            diff = abs(l_dist - r_dist) / denom
            scores.append(max(0, 1 - diff))
            
    return round(np.mean(scores) * 100, 1) if scores else 0

def calculate_golden_ratio(landmarks, w, h):
    p1 = get_pixel_coords(landmarks, 168, w, h)
    p2 = get_pixel_coords(landmarks, 2, w, h)
    p3 = get_pixel_coords(landmarks, 152, w, h)
    mid_face = calculate_distance(p1, p2)
    lower_face = calculate_distance(p2, p3)
    if lower_face == 0: return 0
    ratio = mid_face / lower_face
    deviation = abs(1.0 - ratio)
    return round(max(0, 100 * (1 - deviation)), 1)

def assess_photo_quality(landmarks, w, h):
    warnings = []
    face_width = calculate_distance(
        get_pixel_coords(landmarks, 234, w, h),
        get_pixel_coords(landmarks, 454, w, h)
    )
    face_width_ratio = face_width / w
    if face_width_ratio < 0.20: warnings.append("Move closer")
    elif face_width_ratio > 0.70: warnings.append("Too close")
    
    face_center_x = landmarks.landmark[1].x
    if abs(face_center_x - 0.5) > 0.15: warnings.append("Center face")
    
    rotation = abs(normalize_face_rotation(landmarks, w, h))
    if rotation > 10: warnings.append("Look straight")
    
    return warnings

def calculate_percentile(metric, value):
    stats = REFERENCE_STATS.get(metric)
    if not stats: return 50
    
    z_score = (value - stats['mean']) / stats['std']
    percentile = ((1.0 + math.erf(z_score / math.sqrt(2.0))) / 2.0) * 100
    
    return int(round(percentile))
