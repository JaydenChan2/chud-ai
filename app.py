import cv2
import time
import numpy as np
from collections import deque
from flask import Flask, render_template, Response, jsonify, request
from backend.processor import process_frame, calculate_percentile

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

scan_buffer = {
    'canthal_tilt': deque(maxlen=300),
    'fwhr': deque(maxlen=300),
    'midface_ratio': deque(maxlen=300),
    'facial_symmetry': deque(maxlen=300),
    'golden_ratio': deque(maxlen=300),
}

def get_scan_results():
    results = {}
    for key, values in scan_buffer.items():
        if len(values) > 0:
            results[key] = round(np.median(list(values)), 2)
        else:
            results[key] = 0
    return results

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/scan')
def scan_page():
    return render_template('scan.html')

def gen_frames():
    global current_metrics, quality_info, is_scanning, scan_start_time, scan_progress, final_results

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Standard webcam view (non-mirrored) as requested
        # frame = cv2.flip(frame, 1) 
        
        frame, metrics, warnings = process_frame(frame, is_scanning, scan_buffer)
        
        if metrics:
            current_metrics = metrics
        quality_info = warnings

        # Handle Scanning State
        if is_scanning:
            scan_progress = int(((time.time() - scan_start_time) / scan_duration) * 100)
            if time.time() - scan_start_time >= scan_duration:
                is_scanning = False
                scan_progress = 100
                final_results = get_scan_results()

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
