
# CHUD.AI - Facial Analysis Engine

AI-powered facial geometry analysis using MediaPipe and computer vision.

## Phase 1: Geometric Foundation (Ratios & Harmony)

This phase implements the mathematical engine for facial analysis using 2D/3D landmarks from MediaPipe's 478-point face mesh.

### Implemented Metrics

#### Geometric Ratios
- **Canthal Tilt**: Eye slant angle (positive = upward slant)
- **fWHR (Facial Width-to-Height Ratio)**: Bizygomatic width vs upper face height
- **Midface Ratio**: Balance between upper and lower midface

#### Angular Analysis
- **Nasolabial Angle**: Nose to upper lip angle (ideal: 90-120°)
- **Nasofrontal Angle**: Forehead to nose bridge angle (ideal: 115-135°)

#### Harmony Metrics
- **Facial Symmetry**: Left-right facial balance (0-100 score)
- **Golden Ratio Score**: Proximity to phi (1.618) in facial proportions (0-100 score)

## Setup

### Prerequisites
- Python 3.8+
- Webcam

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/chud-ai.git
cd chud-ai
```

2. **Create a virtual environment** (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Application

```bash
python app.py
```

Press `q` to quit the application.

## How It Works

The application uses MediaPipe Face Mesh to detect 478 facial landmarks in real-time, then calculates:

1. **Geometric measurements** using Euclidean distance between key landmarks
2. **Angular measurements** using vector mathematics
3. **Symmetry analysis** by comparing bilateral landmark pairs
4. **Proportion analysis** against the golden ratio

## Project Structure

```
chud-ai/
├── app.py              # Main application
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── .gitignore         # Git ignore file
```

## MediaPipe Landmark Reference

Key landmarks used:
- Eyes: 33, 133, 263, 362
- Face width: 234, 454
- Nose: 1, 2, 4, 6, 9
- Mouth: 61, 291
- Chin: 152
- Face top: 10

Full landmark map: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

## Troubleshooting

### Virtual Environment Issues
If you accidentally started a virtual environment and closed it:
- Simply run `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux) again
- If the venv folder is corrupted, delete it and recreate: `python -m venv venv`

### MediaPipe Installation Issues
If MediaPipe fails to install:
```bash
pip install --upgrade pip
pip install mediapipe --no-cache-dir
```

### Webcam Not Found
- Check if another application is using the webcam
- Try changing the camera index in `cv2.VideoCapture(0)` to `1` or `2`

## Contributing

This is a collaborative project between the maintainers. To contribute:

1. Create a new branch for your feature
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## Roadmap

- ✅ Phase 1: Geometric Foundation (Current)
- 🔄 Phase 2: Advanced Analysis (Texture, Lighting, 3D)
- 🔄 Phase 3: ML-based Attractiveness Scoring
- 🔄 Phase 4: Personalized Improvement Recommendations

## License

[Add your license here]

## Authors

- Jayanth Vasupilli
- Jayden
