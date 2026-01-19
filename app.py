<<<<<<< HEAD
<<<<<<< HEAD
# app.py
=======
# Placeholder AGAIN

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
=======
>>>>>>> 3c15d48 (Phase 1 v2.0: Major improvements to measurement accuracy)
import cv2
import numpy as np
import math
from collections import deque
import time

# For MediaPipe 0.10.31+ with tasks API
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

# Storage for temporal smoothing (moving average)
SMOOTHING_WINDOW = 15  # Number of frames to average
measurement_buffer = {
    'canthal_tilt': deque(maxlen=SMOOTHING_WINDOW),
    'fwhr': deque(maxlen=SMOOTHING_WINDOW),
    'midface_ratio': deque(maxlen=SMOOTHING_WINDOW),
    'facial_symmetry': deque(maxlen=SMOOTHING_WINDOW),
    'golden_ratio_score': deque(maxlen=SMOOTHING_WINDOW),
}

# Storage for 10-second scan
scan_buffer = {
    'canthal_tilt': deque(maxlen=300),
    'fwhr': deque(maxlen=300),
    'midface_ratio': deque(maxlen=300),
    'facial_symmetry': deque(maxlen=300),
    'golden_ratio_score': deque(maxlen=300),
}

# Movement detection for keyframe selection
previous_landmarks = None
movement_threshold = 2.0  # pixels
stillness_counter = 0
stillness_required = 45  # frames (~1.5 seconds at 30fps)

def get_pixel_coords(landmarks, landmark_id, w, h):
    """Convert normalized landmark to pixel coordinates"""
    landmark = landmarks.landmark[landmark_id]
    return int(landmark.x * w), int(landmark.y * h)

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def get_interpupillary_distance(landmarks, w, h):
    """
    Get interpupillary distance (IPD) as reference measurement
    This provides scale normalization
    """
    left_pupil = get_pixel_coords(landmarks, 468, w, h)
    right_pupil = get_pixel_coords(landmarks, 473, w, h)
    ipd = calculate_distance(left_pupil, right_pupil)
    return ipd if ipd > 0 else 1  # Prevent division by zero

def normalize_face_rotation(landmarks, w, h):
    """
    Calculate face rotation angle to normalize measurements
    Returns rotation angle in degrees
    """
    # Use outer eye corners to determine face horizon
    left_eye = get_pixel_coords(landmarks, 33, w, h)
    right_eye = get_pixel_coords(landmarks, 263, w, h)
    
    # Calculate angle
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    rotation_angle = math.degrees(math.atan2(dy, dx))
    
    return rotation_angle

def calculate_canthal_tilt(landmarks, w, h):
    """
    Calculate canthal tilt with rotation normalization
    Positive = upward slant (desirable)
    Negative = downward slant
    """
    # Get face rotation first
    face_rotation = normalize_face_rotation(landmarks, w, h)
    
    # Left eye: Inner 133, Outer 33
    l_inner = get_pixel_coords(landmarks, 133, w, h)
    l_outer = get_pixel_coords(landmarks, 33, w, h)
    
    # Right eye: Inner 362, Outer 263
    r_inner = get_pixel_coords(landmarks, 362, w, h)
    r_outer = get_pixel_coords(landmarks, 263, w, h)
    
    # Calculate raw tilts
    left_tilt = math.degrees(math.atan2(l_inner[1] - l_outer[1], l_outer[0] - l_inner[0]))
    right_tilt = math.degrees(math.atan2(r_inner[1] - r_outer[1], r_inner[0] - r_outer[0]))
    
    # Average both eyes
    avg_tilt = (left_tilt + right_tilt) / 2
    
    # Subtract face rotation to get normalized tilt
    normalized_tilt = avg_tilt - face_rotation
    
    # Clamp to realistic range
    normalized_tilt = np.clip(normalized_tilt, -15, 15)
    
    return round(normalized_tilt, 2)

def calculate_fwhr(landmarks, w, h):
    """
    Calculate facial Width-to-Height Ratio with IPD normalization
    """
    # Get IPD for normalization
    ipd = get_interpupillary_distance(landmarks, w, h)
    
    # Width: Bizygomatic width (cheekbones)
    left_cheek = get_pixel_coords(landmarks, 234, w, h)
    right_cheek = get_pixel_coords(landmarks, 454, w, h)
    width = calculate_distance(left_cheek, right_cheek)
    
    # Height: Upper face height
    brow = get_pixel_coords(landmarks, 10, w, h)
    upper_lip = get_pixel_coords(landmarks, 164, w, h)
    height = calculate_distance(brow, upper_lip)
    
    # Normalize by IPD
    norm_width = width / ipd
    norm_height = height / ipd
    
    ratio = norm_width / norm_height if norm_height > 0 else 0
    
    # Clamp to realistic range
    ratio = np.clip(ratio, 0.8, 3.0)
    
    return round(ratio, 3)

def calculate_midface_ratio(landmarks, w, h):
    """
    Calculate midface ratio with IPD normalization
    """
    ipd = get_interpupillary_distance(landmarks, w, h)
    
    # Pupil to nose base
    pupil = get_pixel_coords(landmarks, 168, w, h)
    nose_base = get_pixel_coords(landmarks, 2, w, h)
    chin = get_pixel_coords(landmarks, 152, w, h)
    
    upper_midface = calculate_distance(pupil, nose_base)
    lower_midface = calculate_distance(nose_base, chin)
    
    # Normalize
    norm_upper = upper_midface / ipd
    norm_lower = lower_midface / ipd
    
    ratio = norm_upper / norm_lower if norm_lower > 0 else 0
    
    # Clamp to realistic range
    ratio = np.clip(ratio, 0.4, 2.0)
    
    return round(ratio, 3)

def calculate_facial_symmetry(landmarks, w, h):
    """
    Calculate facial symmetry score (0-100)
    """
    pairs = [
        (33, 263),    # Eye outer corners
        (133, 362),   # Eye inner corners
        (234, 454),   # Cheekbones
        (61, 291),    # Mouth corners
        (127, 356),   # Lower face
    ]
    
    # Face center (nose tip)
    center_x = landmarks.landmark[1].x * w
    
    differences = []
    for left_id, right_id in pairs:
        left = get_pixel_coords(landmarks, left_id, w, h)
        right = get_pixel_coords(landmarks, right_id, w, h)
        
        left_dist = abs(left[0] - center_x)
        right_dist = abs(right[0] - center_x)
        
        avg_dist = (left_dist + right_dist) / 2
        if avg_dist > 0:
            diff = abs(left_dist - right_dist) / avg_dist
            differences.append(diff)
    
    avg_asymmetry = np.mean(differences) if differences else 0
    symmetry_score = max(0, 100 * (1 - avg_asymmetry * 2))
    
    # Clamp to realistic range
    symmetry_score = np.clip(symmetry_score, 0, 100)
    
    return round(symmetry_score, 1)

def calculate_golden_ratio(landmarks, w, h):
    """
    Calculate golden ratio score
    """
    phi = 1.618
    
    top_head = get_pixel_coords(landmarks, 10, w, h)
    eyebrow = get_pixel_coords(landmarks, 9, w, h)
    nose_base = get_pixel_coords(landmarks, 2, w, h)
    chin = get_pixel_coords(landmarks, 152, w, h)
    
    upper_third = calculate_distance(top_head, eyebrow)
    middle_third = calculate_distance(eyebrow, nose_base)
    lower_third = calculate_distance(nose_base, chin)
    
    ratios = []
    if middle_third > 0:
        ratios.append(upper_third / middle_third)
    if lower_third > 0:
        ratios.append(middle_third / lower_third)
    
    deviations = [abs(ratio - phi) / phi for ratio in ratios]
    avg_deviation = np.mean(deviations) if deviations else 1
    
    phi_score = max(0, 100 * (1 - avg_deviation))
    
    return round(phi_score, 1)

def detect_movement(landmarks, w, h):
    """
    Detect if face is moving (for keyframe selection)
    Returns: (is_still, movement_amount)
    """
    global previous_landmarks
    
    if previous_landmarks is None:
        previous_landmarks = landmarks
        return False, 0
    
    # Check movement of key landmarks
    key_points = [1, 33, 263, 234, 454, 152]  # Nose, eyes, cheeks, chin
    
    total_movement = 0
    for point_id in key_points:
        curr = get_pixel_coords(landmarks, point_id, w, h)
        prev = get_pixel_coords(previous_landmarks, point_id, w, h)
        movement = calculate_distance(curr, prev)
        total_movement += movement
    
    avg_movement = total_movement / len(key_points)
    
    previous_landmarks = landmarks
    
    is_still = avg_movement < movement_threshold
    
    return is_still, avg_movement

def get_smoothed_metrics():
    """Get temporally smoothed metrics (moving average)"""
    smoothed = {}
    
    for key, values in measurement_buffer.items():
        if len(values) >= 5:  # Need at least 5 frames
            # Use median for robustness
            smoothed[key] = round(np.median(list(values)), 2)
        elif len(values) > 0:
            smoothed[key] = round(np.mean(list(values)), 2)
        else:
            smoothed[key] = 0
    
    return smoothed

def get_scan_results():
    """Get final scan results (10-second buffer)"""
    results = {}
    
    for key, values in scan_buffer.items():
        if len(values) > 0:
            # Use median to reject outliers
            results[key] = round(np.median(list(values)), 2)
        else:
            results[key] = 0
    
    return results

def assess_photo_quality(landmarks, w, h):
    """
    Assess photo quality for analysis
    """
    warnings = []
    quality_score = 100
    
    # Face size check
    face_width = calculate_distance(
        get_pixel_coords(landmarks, 234, w, h),
        get_pixel_coords(landmarks, 454, w, h)
    )
    face_width_ratio = face_width / w
    
    if face_width_ratio < 0.25:
        warnings.append("Face too far")
        quality_score -= 30
    elif face_width_ratio < 0.35:
        warnings.append("Move closer")
        quality_score -= 15
    elif face_width_ratio > 0.70:
        warnings.append("Face too close - distortion!")
        quality_score -= 35
    elif face_width_ratio > 0.60:
        warnings.append("Slightly too close")
        quality_score -= 20
    
    # Check centering
    face_center_x = landmarks.landmark[1].x
    if abs(face_center_x - 0.5) > 0.15:
        warnings.append("Center your face")
        quality_score -= 20
    
    # Check rotation
    rotation = abs(normalize_face_rotation(landmarks, w, h))
    if rotation > 10:
        warnings.append("Face rotated - look straight")
        quality_score -= 25
    elif rotation > 5:
        warnings.append("Minor rotation detected")
        quality_score -= 10
    
    is_valid = quality_score >= 50
    
    return is_valid, quality_score, warnings

def analyze_frame(frame, scanning=False, show_quality=True):
    """Main analysis function with all improvements"""
    global stillness_counter
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape
    
    results = face_mesh.process(rgb_frame)
    
    if not results.multi_face_landmarks:
        return None, frame, False
    
    face_landmarks = results.multi_face_landmarks[0]
    
    # Detect movement (keyframe selection)
    is_still, movement = detect_movement(face_landmarks, w, h)
    
    if is_still:
        stillness_counter += 1
    else:
        stillness_counter = 0
    
    # Only calculate if relatively still (reduces jitter)
    if movement < movement_threshold * 2:
        # Calculate metrics with normalization
        metrics = {
            'canthal_tilt': calculate_canthal_tilt(face_landmarks, w, h),
            'fwhr': calculate_fwhr(face_landmarks, w, h),
            'midface_ratio': calculate_midface_ratio(face_landmarks, w, h),
            'facial_symmetry': calculate_facial_symmetry(face_landmarks, w, h),
            'golden_ratio_score': calculate_golden_ratio(face_landmarks, w, h),
        }
        
        # Add to smoothing buffer
        for key, value in metrics.items():
            measurement_buffer[key].append(value)
        
        # Add to scan buffer if scanning
        if scanning:
            for key, value in metrics.items():
                scan_buffer[key].append(value)
    
    # Get smoothed values
    display_metrics = get_smoothed_metrics()
    
    # Photo quality
    is_valid, quality_score, quality_warnings = assess_photo_quality(face_landmarks, w, h)
    
    # Draw landmarks
    try:
        mp_drawing.draw_landmarks(
            frame,
            face_landmarks,
            mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
    except:
        pass
    
    # Display quality info
    if show_quality:
        y_pos = h - 120
        cv2.rectangle(frame, (5, y_pos - 25), (w - 5, h - 5), (0, 0, 0), -1)
        
        color = (0, 255, 0) if quality_score >= 80 else (0, 255, 255) if quality_score >= 60 else (0, 0, 255)
        cv2.putText(frame, f"Quality: {quality_score}% | Movement: {movement:.1f}px", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        y_pos += 20
        for warning in quality_warnings[:3]:
            cv2.putText(frame, f"! {warning}", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
            y_pos += 18
        
        # Stillness indicator
        if stillness_counter >= stillness_required:
            cv2.putText(frame, "LOCKED - High confidence!", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    keyframe_locked = (stillness_counter >= stillness_required)
    
    return display_metrics, frame, keyframe_locked

def main():
    """Main application loop"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("=" * 70)
    print("CHUD.AI - Facial Analysis System v2.0 (IMPROVED)")
    print("=" * 70)
    print("\nIMPROVEMENTS:")
    print("✅ Temporal smoothing (15-frame moving average)")
    print("✅ Pose & rotation normalization")
    print("✅ IPD scaling for distance independence")  
    print("✅ Automatic keyframe selection (stays still 1.5s)")
    print("\nInstructions:")
    print("1. Position face at arm's length, centered")
    print("2. Press SPACE to start 10-second scan")
    print("3. Keep face STILL for best results")
    print("4. Look for 'LOCKED' indicator = high confidence")
    print("5. Press 'q' to quit")
    print("=" * 70)
    
    scanning = False
    scan_start_time = 0
    scan_duration = 10
    final_results = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check scan completion
        if scanning and (time.time() - scan_start_time >= scan_duration):
            scanning = False
            final_results = get_scan_results()
            print("\n" + "=" * 70)
            print("SCAN COMPLETE - FINAL RESULTS:")
            print("=" * 70)
            for key, value in final_results.items():
                print(f"{key.replace('_', ' ').title():.<35} {value}")
            print("=" * 70)
        
        # Analyze frame
        metrics, annotated_frame, keyframe = analyze_frame(frame, scanning, show_quality=True)
        
        # Display scan progress
        if scanning:
            elapsed = time.time() - scan_start_time
            remaining = scan_duration - elapsed
            progress = int((elapsed / scan_duration) * 100)
            
            cv2.rectangle(annotated_frame, (10, 10), (630, 50), (0, 0, 0), -1)
            cv2.rectangle(annotated_frame, (15, 15), (15 + int(progress * 6), 45), (0, 255, 0), -1)
            cv2.putText(annotated_frame, f"SCANNING... {remaining:.1f}s ({progress}%)", 
                       (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display results
        y_offset = 70 if scanning else 30
        if metrics and not scanning:
            cv2.putText(annotated_frame, "LIVE (Smoothed):", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25
        
        if final_results:
            for key, value in final_results.items():
                text = f"{key.replace('_', ' ').title()}: {value}"
                cv2.putText(annotated_frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                y_offset += 22
        elif metrics:
            for key, value in metrics.items():
                text = f"{key.replace('_', ' ').title()}: {value}"
                color = (150, 150, 150) if scanning else (0, 255, 0)
                cv2.putText(annotated_frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 20
        else:
            cv2.putText(annotated_frame, "Press SPACE to scan", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.imshow('CHUD.AI - Face Analysis', annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and not scanning:
            # Clear buffers
            for buffer in scan_buffer.values():
                buffer.clear()
            scanning = True
            scan_start_time = time.time()
            final_results = None
            print("\nScanning... Keep still!")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
<<<<<<< HEAD
    # Initialize
    engine = GeometricEngine()
    
    # Run on an actual image
    # Make sure 'test.jpg' is in the /Users/jaydenchan/CHUD AI/chud-ai/ folder
    results = engine.analyze("test.jpg")
    
    if results:
        print("\nAnalysis Complete.")
    else:
        print("\nCould not analyze the image (check file path).")
>>>>>>> 53d8858 (test Commit)
=======
    main()
>>>>>>> 3c15d48 (Phase 1 v2.0: Major improvements to measurement accuracy)
