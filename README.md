# CHUD.AI - Facial Analysis Engine (Web Edition)

**Advanced AI-powered facial geometry analysis with a modern Web Interface.**

This application uses MediaPipe Face Mesh to analyze facial landmarks in real-time and provides a detailed dashboard of geometric ratios and harmony metrics directly in your browser.

## Features

- **Web-Based Interface**: Clean, dark-mode dashboard with glassmorphism design.
- **Real-Time Analysis**: Metrics update live as you move.
- **Visual Overlays**: See exactly what is being measured on your face.
- **Smart Metrics**:
    - **Canthal Tilt**: Eye angle analysis.
    - **FWHR**: Facial Width-to-Height Ratio (normalized).
    - **Midface Ratio**: Compactness of the midface.
    - **Facial Symmetry**: Bilateral comparison score.
    - **Golden Ratio**: Proportions adherence score.
- **Scanning Mode**: 10-second scan that locks the final average for easy reading.

## Setup

### Prerequisites
- Python 3.10+
- Webcam

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/chud-ai.git
   cd chud-ai
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the Application**
   ```bash
   python app.py
   ```

2. **Open the Web Interface**
   Open your browser and navigate to:
   **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

3. **Analyze**
   - Allow camera access.
   - Position your face in the center (look for the green "Excellent Conditions" status).
   - Click **START 10s SCAN**.
   - Hold still for 10 seconds.
   - The results will **LOCK** on screen for you to review.

## Metrics Explained

- **Canthal Tilt**: The angle of the eye axis. Positive tilt (outer corners higher) is often considered aesthetically desirable ("Hunter Eyes").
- **FWHR**: A higher FWHR (>1.9) is associated with robust, masculine features.
- **Midface Ratio**: Lower ratio (<0.95) indicates a compact midface.
- **Symmetry**: 100% is perfect symmetry. Most people are 85-95%.

## Troubleshooting

- **Port 5000 in use**: If the app fails to start, another program (like AirPlay Receiver on Mac) might be using port 5000. Try disabling AirPlay Receiver in System Settings -> General -> AirDrop & Handoff.

## License

[NO LICENSE]

## Authors

- Jayanth Vasupilli
- Jayden Chan
