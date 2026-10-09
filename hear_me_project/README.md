# The Hear Me Project 🤟

A real-time sign language recognition and translation engine designed to bridge communication gaps. Built with computer vision and deep learning, this system extracts hand kinematics and translates gestures into readable text via a live telemetry dashboard.

## 🚀 Features
* **Real-Time Kinematic Tracking:** Utilizes MediaPipe to extract 21 3D landmarks per hand.
* **Deep Learning Inference:** Custom Bi-LSTM (Long Short-Term Memory) neural network built with TensorFlow/Keras to classify sequences of hand movements across 8 distinct sign classes.
* **Dual-Environment Support:** Engineered to run natively on Windows via DirectShow or within Linux (Ubuntu/WSL2) using V4L2.
* **Telemetry Dashboard:** A sleek, dark-mode graphical interface built with CustomTkinter to display live confidence gauges, active states, and subtitle ribbons.

## 🛠️ Technology Stack
* **Language:** Python 3.10
* **Computer Vision:** OpenCV, MediaPipe
* **Machine Learning:** TensorFlow, Keras, NumPy
* **GUI & Visualization:** CustomTkinter, Matplotlib, Pillow

## 📦 Installation
1. Clone the repository:
   git clone https://github.com/Shreshtha731/hear_me_project.git
   cd hear_me_project

2. Create and activate a virtual environment:
   # Windows
   .\venv\Scripts\activate
   # Linux/WSL
   source venv/bin/activate

3. Install the required dependencies:
   pip install -r requirements.txt

## 🎮 Usage
To launch the primary CustomTkinter telemetry dashboard:
python app.py

To launch the lightweight Matplotlib/OpenCV HUD (optimized for headless or WSL setups):
python main.py

## 👨‍💻 Author
**Shreshtha Kumar**  
Computer Science Engineering | VIT Bhopal University
