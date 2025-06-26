import React, { useEffect, useRef, useState, useCallback } from 'react';
import './App.css';

// Constants
const MODEL_INPUT_SIZE = 640;
const classNames = ['BLOCK', 'INNER', 'OK', 'OUTER', 'SCAR', 'TEAR'];
const NMS_THRESHOLD = 0.5;
const CONFIDENCE_THRESHOLD = 0.4;
const INFERENCE_THROTTLE_MS = 500;

function App() {
  // Refs
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const workerRef = useRef(null);
  const isProcessingRef = useRef(false);
  const streamRef = useRef(null);
  const videoReadyRef = useRef(false);

  // State
  const [status, setStatus] = useState('loading');
  const [error, setError] = useState(null);
  const [boxes, setBoxes] = useState([]);
  const [calibrationMode, setCalibrationMode] = useState(false);
  const [referenceSize, setReferenceSize] = useState(10);
  const [calibrationComplete, setCalibrationComplete] = useState(false);
  const [pixelsPerMM, setPixelsPerMM] = useState(null);
  const [selectedReferenceBox, setSelectedReferenceBox] = useState(null);
  const [facingMode, setFacingMode] = useState('environment');

  // Utility functions
  const preprocess = useCallback((frame) => {
    const offscreen = document.createElement('canvas');
    offscreen.width = MODEL_INPUT_SIZE;
    offscreen.height = MODEL_INPUT_SIZE;
    const ctx = offscreen.getContext('2d', { willReadFrequently: false });
    
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(frame, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
    
    const imgData = ctx.getImageData(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE).data;
    const data = new Float32Array(3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE);
    
    for (let i = 0; i < imgData.length; i += 4) {
      const pixelIdx = i / 4;
      data[pixelIdx] = imgData[i] / 255;
      data[pixelIdx + MODEL_INPUT_SIZE * MODEL_INPUT_SIZE] = imgData[i + 1] / 255;
      data[pixelIdx + 2 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE] = imgData[i + 2] / 255;
    }
    
    return data.buffer;
  }, []);

  const parseYOLOv5Output = useCallback((data, dims) => {
    const [, num_boxes, num_attrs] = dims;
    const boxes = [];

    for (let i = 0; i < num_boxes; i++) {
      const offset = i * num_attrs;
      const slice = data.subarray(offset, offset + num_attrs);

      const x = slice[0];
      const y = slice[1];
      const w = slice[2];
      const h = slice[3];
      const conf = slice[4];
      const classConfs = slice.subarray(5);

      const classId = classConfs.indexOf(Math.max(...classConfs));
      const classConf = classConfs[classId];
      const totalConf = conf * classConf;

      if (totalConf > CONFIDENCE_THRESHOLD) {
        boxes.push({
          x1: x - w / 2,
          y1: y - h / 2,
          x2: x + w / 2,
          y2: y + h / 2,
          width: w,
          height: h,
          label: classNames[classId] || 'unknown',
          confidence: totalConf,
          classId: classId
        });
      }
    }

    return boxes;
  }, []);

  const nonMaxSuppression = useCallback((boxes) => {
    const sortedBoxes = [...boxes].sort((a, b) => b.confidence - a.confidence);
    const selectedBoxes = [];

    while (sortedBoxes.length > 0) {
      const currentBox = sortedBoxes.shift();
      selectedBoxes.push(currentBox);

      for (let i = sortedBoxes.length - 1; i >= 0; i--) {
        if (sortedBoxes[i].classId === currentBox.classId) {
          const iou = calculateIoU(currentBox, sortedBoxes[i]);
          if (iou > NMS_THRESHOLD) {
            sortedBoxes.splice(i, 1);
          }
        }
      }
    }

    return selectedBoxes;
  }, []);

  const calculateIoU = useCallback((box1, box2) => {
    const x1 = Math.max(box1.x1, box2.x1);
    const y1 = Math.max(box1.y1, box2.y1);
    const x2 = Math.min(box1.x2, box2.x2);
    const y2 = Math.min(box1.y2, box2.y2);

    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    const area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    const union = area1 + area2 - intersection;

    return intersection / union;
  }, []);

  const calculatePhysicalSize = useCallback((box, ppm) => {
    if (!ppm || isNaN(ppm)) return null;
    const widthPx = box.width * MODEL_INPUT_SIZE;
    const heightPx = box.height * MODEL_INPUT_SIZE;
    const diameterPx = (widthPx + heightPx) / 2;
    return diameterPx / ppm;
  }, []);

  // Camera handling
  const initCamera = useCallback(async () => {
    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }

      videoReadyRef.current = false;

      const constraints = {
        video: {
          facingMode,
          width: { ideal: 1280, max: 1280 },
          height: { ideal: 720, max: 720 },
          frameRate: { ideal: 15, max: 15 }
        }
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      const video = videoRef.current;
      if (video) {
        video.onloadedmetadata = null;
        video.onerror = null;
        video.srcObject = stream;
        video.playsInline = true;
        video.setAttribute('webkit-playsinline', '');
        video.setAttribute('playsinline', '');

        await new Promise((resolve, reject) => {
          video.onloadedmetadata = () => {
            videoReadyRef.current = true;
            resolve();
          };
          video.onerror = () => {
            reject(new Error('Video error'));
          };
        });

        try {
          await video.play();
        } catch (playError) {
          if (playError.name === 'NotAllowedError') {
            setError('Please allow camera access and click the page to start');
            document.body.addEventListener('click', async () => {
              try {
                await video.play();
                setError(null);
              } catch (e) {
                setError('Playback failed: ' + e.message);
              }
            }, { once: true });
          } else {
            throw playError;
          }
        }
      }
    } catch (e) {
      console.error('Camera initialization error:', e);
      setError('Camera error: ' + e.message);
      setStatus('error');
    }
  }, [facingMode]);

  // Event handlers
  const handleBoxClick = useCallback((box) => {
    if (calibrationMode) {
      setSelectedReferenceBox(box);
    }
  }, [calibrationMode]);

  const startCalibration = useCallback(() => {
    if (referenceSize > 0 && !isNaN(referenceSize)) {
      setCalibrationMode(true);
      setError(null);
    } else {
      setError('Please enter a valid reference size first');
    }
  }, [referenceSize]);

  const completeCalibration = useCallback(() => {
    if (selectedReferenceBox && referenceSize > 0 && !isNaN(referenceSize)) {
      const widthPx = selectedReferenceBox.width * MODEL_INPUT_SIZE;
      const heightPx = selectedReferenceBox.height * MODEL_INPUT_SIZE;
      const diameterPx = (widthPx + heightPx) / 2;
      const ppm = diameterPx / referenceSize;
      
      setPixelsPerMM(ppm);
      setCalibrationComplete(true);
      setCalibrationMode(false);
      setError(null);
    } else {
      setError('Please select a reference object and ensure valid size');
    }
  }, [selectedReferenceBox, referenceSize]);

  const resetCalibration = useCallback(() => {
    setCalibrationComplete(false);
    setPixelsPerMM(null);
    setSelectedReferenceBox(null);
    setCalibrationMode(false);
  }, []);

  const toggleCamera = useCallback(() => {
    setFacingMode(prev => prev === 'environment' ? 'user' : 'environment');
  }, []);

  // Effects
  useEffect(() => {
    initCamera();
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, [initCamera]);

  useEffect(() => {
    const worker = new Worker(new URL('./onnxWorker.js', import.meta.url));
    workerRef.current = worker;

    worker.onmessage = (e) => {
      const { type } = e.data;
      if (type === 'loaded') {
        setStatus('ready');
      } else if (type === 'inference') {
        const data = new Float32Array(e.data.data);
        const dims = e.data.dims;
        const parsedBoxes = parseYOLOv5Output(data, dims);
        const nmsBoxes = nonMaxSuppression(parsedBoxes);
        setBoxes(nmsBoxes);
        isProcessingRef.current = false;
      } else if (type === 'error') {
        setError('Inference error: ' + e.data.message);
        setStatus('error');
        isProcessingRef.current = false;
      }
    };

    worker.postMessage({ type: 'loadModel', modelUrl: '/best_simplified.onnx' });

    return () => worker.terminate();
  }, []);

  useEffect(() => {
    if (status !== 'ready' || !videoRef.current || !videoReadyRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d', { willReadFrequently: false });

    let animationFrameId;
    let lastInferenceTime = 0;
    let lastBoxes = [];

    const render = () => {
      if (!video || video.readyState < 2) {
        animationFrameId = requestAnimationFrame(render);
        return;
      }

      // Update canvas dimensions if needed
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      try {
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw video frame
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Always draw the latest boxes
        if (boxes.length > 0) {
          lastBoxes = [...boxes]; // Keep copy of last boxes
          drawBoxes(ctx, boxes, canvas.width, canvas.height);
        } else if (lastBoxes.length > 0) {
          // Fallback to last known boxes if current boxes are empty
          drawBoxes(ctx, lastBoxes, canvas.width, canvas.height);
        }

        // Throttled inference
        const now = performance.now();
        if (!isProcessingRef.current && now - lastInferenceTime > INFERENCE_THROTTLE_MS) {
          isProcessingRef.current = true;
          lastInferenceTime = now;
          const tensorData = preprocess(video);
          workerRef.current.postMessage({
            type: 'infer',
            tensorData,
            dims: [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
          }, [tensorData]);
        }
      } catch (e) {
        console.error('Rendering error:', e);
      }

      animationFrameId = requestAnimationFrame(render);
    };

    // Start the render loop
    animationFrameId = requestAnimationFrame(render);
    
    return () => cancelAnimationFrame(animationFrameId);
  }, [status, boxes, calibrationMode, selectedReferenceBox, pixelsPerMM, preprocess, calculatePhysicalSize]);

  // Extracted drawBoxes function
  const drawBoxes = (ctx, boxes, canvasWidth, canvasHeight) => {
    boxes.forEach(box => {
      const x = (box.x1 / MODEL_INPUT_SIZE) * canvasWidth;
      const y = (box.y1 / MODEL_INPUT_SIZE) * canvasHeight;
      const w = ((box.x2 - box.x1) / MODEL_INPUT_SIZE) * canvasWidth;
      const h = ((box.y2 - box.y1) / MODEL_INPUT_SIZE) * canvasHeight;

      ctx.strokeStyle = calibrationMode && selectedReferenceBox === box 
        ? 'yellow' 
        : box.label === 'OK' ? 'lime' : 'red';
      ctx.lineWidth = calibrationMode && selectedReferenceBox === box ? 4 : 2;
      ctx.strokeRect(x, y, w, h);

      if (box.confidence > 0.5 || selectedReferenceBox === box) {
        let label = `${box.label} (${(box.confidence * 100).toFixed(0)}%)`;
        if (box.label === 'OK' && pixelsPerMM) {
          const sizeMM = calculatePhysicalSize(box, pixelsPerMM);
          if (sizeMM) label += ` - Ø${sizeMM.toFixed(1)}mm`;
        }

        ctx.font = '12px Arial';
        const textWidth = ctx.measureText(label).width;
        ctx.fillStyle = 'rgba(255,255,255,0.7)';
        ctx.fillRect(x - 2, y - 16, textWidth + 4, 16);
        ctx.fillStyle = 'black';
        ctx.fillText(label, x, y - 4);
      }
    });
  };

  // Render
  return (
    <div className="app-container">
      {/* Loading spinner overlay */}
      {status === 'loading' && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          backgroundColor: 'rgba(0,0,0,0.5)',
          zIndex: 1000
        }}>
          <div className="loader"></div>
        </div>
      )}

      <div className="header">
        <h1>O-Ring Size Detection</h1>
      </div>
      
      <div className="content">
        {!calibrationComplete ? (
          <div className="calibration-panel">
            <h3>Calibration Required</h3>
            <p>Place a reference object of known size in view and enter its diameter:</p>
            
            <div style={{ marginBottom: '10px' }}>
              <label>Reference diameter (mm): </label>
              <input 
                type="number" 
                value={isNaN(referenceSize) ? '' : referenceSize}
                onChange={(e) => setReferenceSize(parseFloat(e.target.value) || 0)} 
                step="0.1"
                min="1"
                style={{ marginLeft: '10px' }}
              />
            </div>
            
            <div className="controls">
              {!calibrationMode ? (
                <button onClick={startCalibration}>Start Calibration</button>
              ) : (
                <>
                  <p>Click on the reference object in the video feed</p>
                  {selectedReferenceBox && (
                    <button onClick={completeCalibration}>Complete Calibration</button>
                  )}
                </>
              )}
            </div>
          </div>
        ) : (
          <div className="success">
            <p>Calibration complete: {pixelsPerMM?.toFixed(2) || 'N/A'} pixels/mm</p>
            <button onClick={resetCalibration}>Recalibrate</button>
          </div>
        )}

        <div className="camera-container">
          <div className="canvas-container">
            <video
              ref={videoRef}
              style={{ display: 'none' }}
              playsInline
              muted
            />
            <canvas
              ref={canvasRef}
              onClick={(e) => {
                if (!calibrationMode || !boxes.length) return;
                
                const rect = canvasRef.current.getBoundingClientRect();
                const scaleX = canvasRef.current.width / rect.width;
                const scaleY = canvasRef.current.height / rect.height;
                
                const x = (e.clientX - rect.left) * scaleX;
                const y = (e.clientY - rect.top) * scaleY;
                
                const clickedBox = boxes.find(box => {
                  const boxX = (box.x1 / MODEL_INPUT_SIZE) * canvasRef.current.width;
                  const boxY = (box.y1 / MODEL_INPUT_SIZE) * canvasRef.current.height;
                  const boxW = ((box.x2 - box.x1) / MODEL_INPUT_SIZE) * canvasRef.current.width;
                  const boxH = ((box.y2 - box.y1) / MODEL_INPUT_SIZE) * canvasRef.current.height;
                  
                  return x >= boxX - 10 && x <= boxX + boxW + 10 && 
                         y >= boxY - 10 && y <= boxY + boxH + 10;
                });
                
                if (clickedBox) handleBoxClick(clickedBox);
              }}
            />
          </div>
        </div>
        
        <div className="controls">
          <button className="camera-switch" onClick={toggleCamera}>
            Switch Camera ({facingMode === 'environment' ? 'Rear' : 'Front'})
          </button>
        </div>
        
        <div className="status">
          Status: {status}
          {error && <div className="error">{error}</div>}
        </div>
      </div>
    </div>
  );
}

export default App;