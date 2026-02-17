
import cv2
import numpy as np
import math
from collections import deque
import time
from flask import Flask, render_template, Response, jsonify, request
import threading

# Flask Setup
app = Flask(__name__)

# Global state for web integration
current_metrics = {}
quality_info = []
is_scanning = False
scan_start_time = 0
scan_duration = 10
scan_progress = 0
final_results = {}

# For MediaPipe
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

# Storage for temporal smoothing
SMOOTHING_WINDOW = 15
measurement_buffer = {
    'canthal_tilt': deque(maxlen=SMOOTHING_WINDOW),
    'fwhr': deque(maxlen=SMOOTHING_WINDOW),
    'midface_ratio': deque(maxlen=SMOOTHING_WINDOW),
    'facial_symmetry': deque(maxlen=SMOOTHING_WINDOW),
    'golden_ratio': deque(maxlen=SMOOTHING_WINDOW),
}

scan_buffer = {
    'canthal_tilt': deque(maxlen=300),
    'fwhr': deque(maxlen=300),
    'midface_ratio': deque(maxlen=300),
    'facial_symmetry': deque(maxlen=300),
    'golden_ratio': deque(maxlen=300),
}

# Movement detection
previous_landmarks = None
movement_threshold = 2.0
stillness_counter = 0
stillness_required = 45

# --- Geometry Helpers ---
def get_pixel_coords(landmarks, landmark_id, w, h):
    landmark = landmarks.landmark[landmark_id]
    return int(landmark.x * w), int(landmark.y * h)

def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def get_interpupillary_distance(landmarks, w, h):
    left_pupil = get_pixel_coords(landmarks, 468, w, h)
    right_pupil = get_pixel_coords(landmarks, 473, w, h)
    ipd = calculate_distance(left_pupil, right_pupil)
    return ipd if ipd > 0 else 1

def normalize_face_rotation(landmarks, w, h):
    left_eye = get_pixel_coords(landmarks, 33, w, h)
    right_eye = get_pixel_coords(landmarks, 263, w, h)
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    return math.degrees(math.atan2(dy, dx))

# --- Estimators ---
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

def detect_movement(landmarks, w, h):
    global previous_landmarks
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
    return avg_movement < movement_threshold, avg_movement

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

def get_scan_results():
    results = {}
    for key, values in scan_buffer.items():
        if len(values) > 0:
            results[key] = round(np.median(list(values)), 2)
        else:
            results[key] = 0
    return results

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

from scipy.stats import norm

# --- Flask Routes ---

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/scan')
def scan_page():
    return render_template('scan.html')

# Percentile Reference Data (Means and Std Devs estimated for "Human Standards")
# These are heuristic values for demonstration.
reference_stats = {
    'canthal_tilt': {'mean': 4.0, 'std': 3.0}, # Degrees
    'fwhr': {'mean': 1.9, 'std': 0.15},        # Ratio
    'midface_ratio': {'mean': 1.0, 'std': 0.1},# Ratio
    'facial_symmetry': {'mean': 92.0, 'std': 4.0}, # Percentage
    'golden_ratio': {'mean': 85.0, 'std': 5.0}     # Percentage
}

def calculate_percentile(metric, value):
    stats = reference_stats.get(metric)
    if not stats: return 50
    
    # Calculate CDF
    z_score = (value - stats['mean']) / stats['std']
    percentile = norm.cdf(z_score) * 100
    
    # For midface ratio, lower is usually considered "better" or more compact in some aesthetics, 
    # but "percentile" usually means "score higher than X%". 
    # Let's stick to standard "higher value = higher percentile" math for consistency,
    # unless it's an error metric. 
    # Actually, let's just return the raw distribution percentile.
    
    return int(round(percentile))

def gen_frames():
    global current_metrics, quality_info, is_scanning, scan_start_time, scan_progress, final_results, stillness_counter

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # User requested inverted camera. 
        # Previously we had: frame = cv2.flip(frame, 1) # This mimics a mirror (standard for selfie cams)
        # If user wants "Inverted", they might mean TRUE webcam view (non-mirrored).
        # Let's remove the flip to make it "inverted" relative to the previous mirror state.
        # OR if they meant upside down? Unlikely. They likely meant non-mirrored.
        # But wait, usually "invert the camera" means flip it?
        # Standard webcam is non-mirrored. `cv2.flip(frame, 1)` makes it mirrored.
        # If I remove `cv2.flip(frame, 1)`, it will look like a regular video feed (not a mirror).
        # Let's assume standard behavior for "add a home page... also make it so the camera is inverted" 
        # means they want the OPPOSITE of what it was.
        # It was flipped (mirrored). So I will NOT flip it.
        
        # frame = cv2.flip(frame, 1) 
        
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            # Logic
            is_still, movement = detect_movement(landmarks, w, h)
            if is_still: stillness_counter += 1
            else: stillness_counter = 0
            
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
            
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame, landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                None, mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

        # Handle Scanning State
        if is_scanning:
            scan_progress = int(((time.time() - scan_start_time) / scan_duration) * 100)
            if time.time() - scan_start_time >= scan_duration:
                is_scanning = False
                scan_progress = 100
                final_results = get_scan_results()
                # Stop updating current_metrics with new data? No, keep it live but UI can choose.

        # Encode
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/metrics')
def metrics():
    # Calculate percentiles for the current metrics (or final results if available)
    data_to_grade = final_results if final_results else current_metrics
    percentiles = {}
    
    if data_to_grade:
        for key, value in data_to_grade.items():
            percentiles[key] = calculate_percentile(key, value)

    return jsonify({
        'metrics': current_metrics,
        'quality_warnings': quality_info,
        'is_scanning': is_scanning,
        'scan_progress': scan_progress,
        'final_results': final_results if final_results else None,
        'percentiles': percentiles
    })

@app.route('/api/toggle_scan', methods=['POST'])
def toggle_scan():
    global is_scanning, scan_start_time, final_results, scan_progress
    for q in scan_buffer.values(): q.clear()
    is_scanning = True
    scan_start_time = time.time()
    scan_progress = 0
    final_results = {}
    return jsonify({'status': 'started'})

if __name__ == "__main__":
    print("Starting Flask Server...")
    print("Go to http://127.0.0.1:5000 in your browser")
    app.run(host='0.0.0.0', port=5000, debug=True)
