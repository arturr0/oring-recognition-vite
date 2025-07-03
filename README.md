## O-Ring Size Detection System  
**Status: Development**  
![Tech](https://img.shields.io/badge/-React-61DAFB) ![Tech](https://img.shields.io/badge/-YOLOv5-00FFFF) ![Tech](https://img.shields.io/badge/-ONNX-005CED) ![Tech](https://img.shields.io/badge/-WebGL-990000)  

🌐 **Live Demo**: [https://oring-recognition-vite.onrender.com](https://oring-recognition-vite.onrender.com)  

<div align="center">
  <img src="https://cdn.glitch.global/79283f6f-ef1e-4285-822b-eaefe68c462e/orning.jpg?v=1751228266800" height="1000">
</div>

### ✨ Key Features  
- **Measurement**: Detects after calibration  
- **Defect Classification**: Identifies 6 defect types (SCAR, TEAR, BLOCK, etc.)  
- **Cross-Device**: Works on mobile/desktop browsers with camera access  
- **Optimized Inference**: WebWorker + ONNX Runtime for smooth 15 FPS performance  
- **Calibration System**: Reference object scaling for physical measurements  

### 🛠️ Technical Stack  
| Component               | Technology                          |
|-------------------------|-------------------------------------|
| Frontend                | React |
| Computer Vision         | YOLOv5 (custom-trained ONNX model) |
| Real-time Processing    | Web Workers |
| Camera Pipeline         | MediaDevices API                    |
| Measurement System      | Perspective calibration             |

### ⚠️ Demo Notes & Limitations  
- The detection model is **pretrained on a small dataset** – results may vary  
- The **confidence threshold is intentionally low** to showcase potential detections  
- For testing, you can try using a **coin or round object** as a substitute for O-rings  
- Make sure the object is **not too close** to the camera  
  - Ideal mobile distance: **~45 cm**  
- Ensure lighting is **not too bright or reflective**  
- Use against a **plain/uniform background** – avoid placing in front of a face or cluttered surface  
