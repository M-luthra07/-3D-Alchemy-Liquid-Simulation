# 🌊 3D Alchemy Liquid Simulation

A high-performance, real-time 3D fluid simulation powered by **MediaPipe Hand Tracking** and **OpenCV**. Experience interactive "Alchemy" with various liquid materials, all controlled through intuitive hand gestures in physical space.

![Simulation Preview](https://img.shields.io/badge/Status-Interactive-brightgreen)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-orange)

## ✨ Key Features

- **Gesture-Based Control**: Rotate the 3D simulation space simply by moving your hand.
- **Alchemy Modes**: Cycle through 4 distinct liquid materials:
  - 💧 **Deep Water**: Classic blue fluid physics.
  - 🥈 **Liquid Mercury**: Heavy, dense silver with low damping.
  - 🧪 **Toxic Sludge**: Slow, viscous green chemical flow.
  - 🏆 **Molten Gold**: Premium, high-density flowing metal.
- **Dynamic Interaction**: 
  - **Poke**: Push the liquid with your fingertips.
  - **Resize**: Pinch your fingers to scale the container in real-time.
  - **Inertia**: Flick your hand to give the simulation a spinning momentum.
- **Vectorized PBD Physics**: Hand-optimized Position Based Dynamics using NumPy for smooth performance.
- **Aesthetic Rendering**: Custom soft-gaussian splatting for a continuous fluid surface look instead of discrete particles.

---

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/M-luthra07/-3D-Alchemy-Liquid-Simulation.git
   cd 3d-liquid-simulation
   ```

2. **Install Dependencies**:
   ```bash
   pip install opencv-python mediapipe numpy
   ```

3. **Download the Model**:
   Ensure `hand_landmarker.task` is in the project root. This is the MediaPipe task file used for detection.

---

## 🎮 How to Use

Run the main application:
```bash
python app.py
```

### 🖐️ Hand Controls

| Action | Gesture | Result |
| :--- | :--- | :--- |
| **Rotate** | Open Hand | Move your hand to spin the container. |
| **Freeze** | Make a Fist | Stops rotation input (simulation continues with inertia). |
| **Resize** | Pinch | Pinch index and thumb to scale the cube size. |
| **Interact** | Point/Open | Fingertips act as "pokes" within the fluid. |

### ⌨️ Keyboard Shortcuts

- `[M]`: Cycle through Liquid Modes (Alchemy).
- `[Q]`: Quit the application.

---

## 🧪 Technical Details

The simulation uses **Vectorized PBD (Position Based Dynamics)**. 
- **Collision**: Custom wall-force functions keep particles inside the `BOUND` box.
- **Repulsion**: N-body repulsion logic ensures fluid volume consistency.
- **Rendering**: A multi-pass approach using Gaussian Blurring and Specular Highlight layers to create a "liquid" visual effect from point-cloud data.

---

## 📜 License
This project is for educational and experimental purposes. Enjoy the alchemy!
