## O-Ring Size Detection System  
![Status](https://img.shields.io/badge/status-alpha-orange)  
![Tech](https://img.shields.io/badge/-React-61DAFB) ![Tech](https://img.shields.io/badge/-YOLOv5-00FFFF) ![Tech](https://img.shields.io/badge/-ONNX-005CED) ![Tech](https://img.shields.io/badge/-WebGL-990000) 

<div align="left">
  <img src="https://cdn.glitch.global/79283f6f-ef1e-4285-822b-eaefe68c462e/orning.jpg" height="400">
</div>

🌐 **Live Demo**: [https://oring-recognition-vite.onrender.com](https://oring-recognition-vite.onrender.com)  

### ✨ Key Features  
- **±0.1mm accuracy** after calibration  
- **Defect classification** (SCAR, TEAR, BLOCK, etc.)  
- **Cross-device** browser support  
- **15 FPS inference** via WebWorker + ONNX Runtime  
- **Perspective calibration** for physical measurements  

### 🛠️ Technical Stack  
| Component               | Technology                          |
|-------------------------|-------------------------------------|
| Frontend                | ![React](https://img.shields.io/badge/-React-61DAFB) |
| Computer Vision         | ![YOLOv5](https://img.shields.io/badge/-YOLOv5-00FFFF) (ONNX) |
| Processing              | ![WebGL](https://img.shields.io/badge/-WebGL-990000) + Web Workers |
| Camera                  | MediaDevices API                    |

### ⚠️ Demo Limitations  
- **Pretrained model** (small dataset) - results may vary  
- **Low confidence threshold** for demo purposes  
- **Ideal conditions**:  
  - Distance: ~45cm from camera  
  - Lighting: Even, non-reflective  
  - Background: Plain/uniform  
- **Test objects**: Coins/round objects work as substitutes  
