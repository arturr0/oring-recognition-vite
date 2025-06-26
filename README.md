# O-Ring Detection Web Application

![Demo Screenshot](demo-screenshot.png) <!-- Add a screenshot later -->

A real-time web application that detects and measures O-rings using computer vision and machine learning. The application classifies O-rings into different quality categories and measures their physical dimensions when calibrated.

## What is an O-Ring?

An O-ring is a simple but crucial mechanical gasket in the shape of a torus (a doughnut-shaped ring). It's used to:
- Prevent leaks by creating a seal between two surfaces
- Block the passage of liquids or gases
- Commonly found in plumbing, hydraulic systems, and mechanical equipment

O-rings come in various sizes and materials, and their proper functioning depends on having the correct dimensions and being free from defects.

## Features

- 🎥 Real-time detection using device camera
- 📏 Physical size measurement after calibration
- 🏷️ Classification of O-rings into categories:
  - OK: Good condition
  - BLOCK: Blocked or obstructed
  - INNER: Inner diameter defect
  - OUTER: Outer diameter defect
  - SCAR: Surface scarring
  - TEAR: Physical tear or damage
- 🔄 Camera switching (front/rear)
- 📊 Confidence percentage display for each detection

## Technical Details

- **Model**: YOLOv5 (PyTorch) converted to ONNX format
- **Inference**: Runs client-side using ONNX Runtime Web
- **Frontend**: React.js with custom Web Worker for model inference
- **Computer Vision**: Real-time processing with Canvas API
- **Optimized** for both desktop and mobile devices
