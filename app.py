<<<<<<< HEAD
# app.py
=======
# Placeholder AGAIN

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2

class GeometricEngine:
    def __init__(self, model_path='face_landmarker.task'):
        # --- NEW TASKS API CONFIGURATION ---
        base_options = python.BaseOptions(model_asset_path=model_path)
        
        # We want facial blendshapes (for granular detail) and transformation matrix
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            min_face_detection_confidence=0.45



        )
        
        try:
            self.detector = vision.FaceLandmarker.create_from_options(options)
            print(f"✅ Model loaded successfully: {model_path}")
        except Exception as e:
            print(f"❌ Failed to load model. Did you download 'face_landmarker.task'?\nError: {e}")
            exit()

    def get_landmarks(self, image_path):
        """Returns numpy array of landmarks (x, y, z) scaled to image dimensions."""
        # MediaPipe Tasks requires a specific MPImage format
        cv_mat = cv2.imread(image_path)
        if cv_mat is None:
            raise ValueError(f"Image not found at {image_path}")

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_mat)
        
        # Run detection
        detection_result = self.detector.detect(image)
        
        if not detection_result.face_landmarks:
            return None, None

        # Extract the first face
        face_landmarks = detection_result.face_landmarks[0]
        
        # Convert to Numpy (x, y, z)
        h, w, c = cv_mat.shape
        coords = np.array([(lm.x * w, lm.y * h, lm.z * w) for lm in face_landmarks])
        
        return coords, cv_mat

    # --- GEOMETRIC MATH HELPER FUNCTIONS ---
    def euclidean_dist(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def calculate_2d_angle(self, p1, p2):
        """Angle relative to horizontal axis (for Canthal Tilt)."""
        delta_y = p2[1] - p1[1]
        delta_x = p2[0] - p1[0]
        return np.degrees(np.arctan2(delta_y, delta_x))

    def analyze(self, image_path):
        landmarks, img = self.get_landmarks(image_path)
        if landmarks is None:
            print("No face detected.")
            return

        # --- LANDMARK MAPPING (Standard 478 Mesh) ---
        # Eyes
        left_inner, left_outer = landmarks[133], landmarks[33]
        right_inner, right_outer = landmarks[362], landmarks[263]
        
        # Zygoma (Cheekbones)
        left_zygoma, right_zygoma = landmarks[234], landmarks[454]
        
        # Vertical Landmarks
        nasion = landmarks[168]      # Mid-eyes
        subnasale = landmarks[2]     # Base of nose
        stomion = landmarks[0]       # Lip center
        menton = landmarks[152]      # Chin bottom
        
        # --- CALCULATIONS ---
        
        # 1. Canthal Tilt
        l_tilt = self.calculate_2d_angle(left_inner, left_outer)
        r_tilt = self.calculate_2d_angle(right_inner, right_outer) * -1
        avg_tilt = (l_tilt + r_tilt) / 2

        # 2. Bizygomatic Width
        bizygo_width = self.euclidean_dist(left_zygoma, right_zygoma)

        # 3. Midface Ratio (Compactness)
        # Defined here as: Interpupillary Distance / Midface Height
        # (A higher ratio often implies a more compact, "hunter" eye area structure)
        ipd = self.euclidean_dist(landmarks[468], landmarks[473]) # 468/473 are iris centers!
        midface_height = self.euclidean_dist(nasion, subnasale)
        midface_ratio = ipd / midface_height

        # 4. fWHR (Width to Height)
        # Width (Bizygomatic) / Height (Nasion to Stomion)
        upper_face_height = self.euclidean_dist(nasion, stomion)
        fwhr = bizygo_width / upper_face_height

        # Print Report
        print(f"\n--- ANALYSIS REPORT: {image_path} ---")
        print(f"Canthal Tilt:      {avg_tilt:.2f}° {'(Positive)' if avg_tilt > 0 else '(Negative)'}")
        print(f"fWHR:              {fwhr:.2f} (Target: > 1.9 is broad)")
        print(f"Midface Ratio:     {midface_ratio:.2f}")
        print(f"Symmetry Check:    Left Eye Width vs Right: {self.euclidean_dist(left_inner, left_outer):.1f} vs {self.euclidean_dist(right_inner, right_outer):.1f}")

        return {
            "canthal_tilt": avg_tilt,
            "fwhr": fwhr,
            "midface_ratio": midface_ratio
        }

if __name__ == "__main__":
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
