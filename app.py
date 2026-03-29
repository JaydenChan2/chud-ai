import os
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify
from backend.processor import process_frame_stateless
from backend.analysis import calculate_percentile

# Flask Setup
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/scan')
def scan_page():
    return render_template('scan.html')

@app.route('/api/process_frame', methods=['POST'])
def process_frame_api():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
        
    try:
        # The frontend sends 'data:image/jpeg;base64,...'
        image_data = data['image'].split(',')[1] 
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Mirror the frame so it acts like a mirror (better UX)
        # frame = cv2.flip(frame, 1) 
        
        processed_frame, metrics, warnings = process_frame_stateless(frame)
        
        percentiles = {}
        if metrics:
            for key, value in metrics.items():
                percentiles[key] = calculate_percentile(key, value)
                
        # encode frame back to base64
        _, buffer = cv2.imencode('.jpg', processed_frame)
        out_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'image': 'data:image/jpeg;base64,' + out_b64,
            'metrics': metrics,
            'warnings': warnings,
            'percentiles': percentiles
        })
    except Exception as e:
        print("Error processing frame:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask Server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=True)
