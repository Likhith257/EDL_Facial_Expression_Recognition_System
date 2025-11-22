import { useState, useRef, useEffect } from "react";
import Layout from "@/components/Layout";
import {
  Upload,
  Camera,
  Loader2,
  CheckCircle,
  AlertCircle,
  Download,
  Trash2,
  Settings,
  History,
  Image as ImageIcon,
  Sliders,
  FileImage,
} from "lucide-react";

type DetectionModel = "yolo" | "efficientb3";
type RecognitionModel = "yolo" | "efficientb3" | "arcface" | "swin" | "vit";

interface HistoryItem {
  id: string;
  timestamp: number;
  image: string;
  results: any;
  detectionModel: string;
  recognitionModel: string;
}

export default function Recognition() {
  const [activeTab, setActiveTab] = useState<"upload" | "webcam">("upload");
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [selectedImages, setSelectedImages] = useState<string[]>([]);
  const [batchResults, setBatchResults] = useState<any[]>([]);
  const [isBatchMode, setIsBatchMode] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const batchFileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const annotatedCanvasRef = useRef<HTMLCanvasElement>(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [imageSize, setImageSize] = useState<{ width: number, height: number } | null>(null);
  const [confidenceThreshold, setConfidenceThreshold] = useState(25);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [showHistory, setShowHistory] = useState(false);

  const [detectionModel, setDetectionModel] = useState<DetectionModel>("yolo");
  const [recognitionModel, setRecognitionModel] =
    useState<RecognitionModel>("yolo");

  // Load history from localStorage on mount
  useEffect(() => {
    const savedHistory = localStorage.getItem('emotionDetectionHistory');
    if (savedHistory) {
      setHistory(JSON.parse(savedHistory));
    }
  }, []);

  // Save to history when results are generated
  const saveToHistory = (image: string, results: any, detectionModel: string, recognitionModel: string) => {
    const newItem: HistoryItem = {
      id: Date.now().toString(),
      timestamp: Date.now(),
      image,
      results,
      detectionModel,
      recognitionModel,
    };
    const updatedHistory = [newItem, ...history].slice(0, 5); // Keep only last 5
    setHistory(updatedHistory);
    localStorage.setItem('emotionDetectionHistory', JSON.stringify(updatedHistory));
  };

  // Draw bounding boxes and labels on annotated canvas
  const drawAnnotations = (imageSrc: string, faces: any[]) => {
    if (!annotatedCanvasRef.current) return;

    const img = new Image();
    img.onload = () => {
      const canvas = annotatedCanvasRef.current!;
      const ctx = canvas.getContext('2d')!;
      canvas.width = img.width;
      canvas.height = img.height;

      // Draw original image
      ctx.drawImage(img, 0, 0);

      // Emotion colors
      const emotionColors: Record<string, string> = {
        happy: '#10b981',
        sad: '#3b82f6',
        angry: '#ef4444',
        fear: '#8b5cf6',
        disgust: '#f59e0b',
        surprised: '#eab308',
        neutral: '#6b7280',
      };

      // Draw each face box
      faces.forEach((face: any) => {
        const conf = parseFloat(face.confidence);
        if (conf < confidenceThreshold) return; // Skip if below threshold

        const { x, y, width, height } = face.position;
        const color = emotionColors[face.expression.toLowerCase()] || '#6b7280';

        // Draw bounding box
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, width, height);

        // Draw label background
        const label = `${face.expression} ${face.confidence}%`;
        ctx.font = 'bold 16px Arial';
        const textWidth = ctx.measureText(label).width;
        ctx.fillStyle = color;
        ctx.fillRect(x, y - 25, textWidth + 10, 25);

        // Draw label text
        ctx.fillStyle = 'white';
        ctx.fillText(label, x + 5, y - 7);
      });
    };
    img.src = imageSrc;
  };

  const downloadAnnotatedImage = () => {
    if (!annotatedCanvasRef.current) return;
    const link = document.createElement('a');
    link.download = `annotated-emotion-${Date.now()}.jpg`;
    link.href = annotatedCanvasRef.current.toDataURL('image/jpeg');
    link.click();
  };

  const loadFromHistory = (item: HistoryItem) => {
    setSelectedImage(item.image);
    setResults({
      ...item.results,
      detectionModel: item.detectionModel,
      recognitionModel: item.recognitionModel,
    });
    setShowHistory(false);
    drawAnnotations(item.image, item.results.faces);
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem('emotionDetectionHistory');
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (file.size > 10 * 1024 * 1024) {
        setError('Image size must be less than 10MB');
        return;
      }
      if (!file.type.startsWith('image/')) {
        setError('Please upload a valid image file');
        return;
      }
      const reader = new FileReader();
      reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
          setImageSize({ width: img.width, height: img.height });
          setSelectedImage(event.target?.result as string);
          setError(null);
          setResults(null);
        };
        img.src = event.target?.result as string;
      };
      reader.readAsDataURL(file);
    }
  };

  const handleBatchFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;

    const validFiles = files.filter(file => {
      if (file.size > 10 * 1024 * 1024) {
        setError(`${file.name} is too large (max 10MB)`);
        return false;
      }
      if (!file.type.startsWith('image/')) {
        setError(`${file.name} is not a valid image`);
        return false;
      }
      return true;
    });

    if (validFiles.length > 10) {
      setError('Maximum 10 images allowed in batch mode');
      validFiles.splice(10);
    }

    const imagePromises = validFiles.map(file => {
      return new Promise<string>((resolve) => {
        const reader = new FileReader();
        reader.onload = (event) => resolve(event.target?.result as string);
        reader.readAsDataURL(file);
      });
    });

    Promise.all(imagePromises).then(images => {
      setSelectedImages(images);
      setBatchResults([]);
      setError(null);
    });
  };

  const processBatchImages = async () => {
    if (selectedImages.length === 0) return;

    setIsProcessing(true);
    setError(null);
    const results: any[] = [];

    for (let i = 0; i < selectedImages.length; i++) {
      try {
        const response = await fetch(selectedImages[i]);
        const blob = await response.blob();

        const formData = new FormData();
        formData.append('file', blob, `image-${i}.jpg`);
        formData.append('detection_model', detectionModel);
        formData.append('recognition_model', recognitionModel);

        const apiResponse = await fetch('/api/predict', {
          method: 'POST',
          body: formData,
        });

        if (apiResponse.ok) {
          const data = await apiResponse.json();
          results.push({ success: true, data, image: selectedImages[i] });
        } else {
          results.push({ success: false, error: 'Failed to process', image: selectedImages[i] });
        }
      } catch (err) {
        results.push({ success: false, error: 'Network error', image: selectedImages[i] });
      }
    }

    setBatchResults(results);
    setIsProcessing(false);
  };

  const handleProcess = async () => {
    if (!selectedImage) return;

    setIsProcessing(true);
    setError(null);

    try {
      const startTime = Date.now();

      const response = await fetch(selectedImage);
      const blob = await response.blob();

      const formData = new FormData();
      formData.append('file', blob, 'image.jpg');
      formData.append('detection_model', detectionModel);
      formData.append('recognition_model', recognitionModel);

      const apiResponse = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!apiResponse.ok) {
        const errorData = await apiResponse.json();
        throw new Error(errorData.error || 'Prediction failed');
      }

      const data = await apiResponse.json();
      const processingTime = Date.now() - startTime;

      const modelNames: Record<DetectionModel, string> = {
        yolo: "YOLOv8",
        efficientb3: "EfficientNet-B3",
      };

      const recognitionNames: Record<RecognitionModel, string> = {
        yolo: "YOLOv8",
        efficientb3: "EfficientNet-B3",
        arcface: "ArcFace",
        swin: "Swin Transformer",
        vit: "Vision Transformer (ViT)",
      };

      const faces = data.all_detections?.map((detection: any, index: number) => ({
        id: index + 1,
        confidence: (detection.confidence * 100).toFixed(1),
        position: {
          x: detection.bbox.x1,
          y: detection.bbox.y1,
          width: detection.bbox.x2 - detection.bbox.x1,
          height: detection.bbox.y2 - detection.bbox.y1,
        },
        expression: detection.expression,
        embedding: `Detected by ${data.model}`,
      })) || [];

      const actualFramework = data.framework || 'yolo';
      const displayRecognitionModel = actualFramework === 'yolo'
        ? 'YOLOv8'
        : actualFramework === 'efficientnetb3' || actualFramework === 'efficientb3'
          ? 'EfficientNet-B3'
          : actualFramework === 'arcface'
            ? 'ArcFace (ResNet-18)'
            : recognitionNames[recognitionModel];

      setResults({
        detected: faces.length > 0,
        confidence: data.confidence ? (data.confidence * 100).toFixed(1) : 0,
        detectionModel: modelNames[detectionModel],
        recognitionModel: displayRecognitionModel,
        faces: faces,
        processingTime: processingTime,
        numFaces: data.num_faces || 0,
      });

      // Save to history
      saveToHistory(selectedImage, {
        confidence: data.confidence,
        detectionModel: modelNames[detectionModel],
        recognitionModel: displayRecognitionModel,
        faces: faces,
        numFaces: data.num_faces || 0,
      }, modelNames[detectionModel], displayRecognitionModel);

      // Draw annotations on canvas
      drawAnnotations(selectedImage, faces);

      setTimeout(() => {
        document.querySelector('.results-panel')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 100);

    } catch (err: any) {
      setError(err.message || 'Failed to process image');
      setResults(null);
    } finally {
      setIsProcessing(false);
    }
  };

  const startCamera = async () => {
    try {
      setError(null);

      // Check if getUserMedia is available
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setError("Camera access is not supported in this browser or context. Please use a modern browser and ensure the page is served over HTTPS or localhost.");
        return;
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        }
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        // Ensure video starts playing
        await videoRef.current.play();
        setIsCameraActive(true);
      }
    } catch (err: any) {
      let errorMessage = "Unable to access camera. ";

      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        errorMessage += "Camera access was denied. Please allow camera permissions in your browser settings and try again.";
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        errorMessage += "No camera found. Please connect a camera and try again.";
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        errorMessage += "Camera is already in use by another application. Please close other apps using the camera.";
      } else if (err.name === 'OverconstrainedError') {
        errorMessage += "Camera doesn't support the requested settings. Trying with default settings...";
        // Retry with basic constraints
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ video: true });
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            await videoRef.current.play();
            setIsCameraActive(true);
            return;
          }
        } catch {
          errorMessage = "Unable to access camera with any settings.";
        }
      } else {
        errorMessage += `Error: ${err.message || 'Unknown error'}`;
      }

      setError(errorMessage);
      console.error('Camera error:', err);
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach((track) => track.stop());
      setIsCameraActive(false);
    }
  };

  const captureFrame = () => {
    if (videoRef.current && canvasRef.current) {
      const context = canvasRef.current.getContext("2d");
      if (context) {
        canvasRef.current.width = videoRef.current.videoWidth;
        canvasRef.current.height = videoRef.current.videoHeight;
        context.drawImage(videoRef.current, 0, 0);
        setSelectedImage(canvasRef.current.toDataURL("image/jpeg"));
        stopCamera();
        setActiveTab("upload");
        setResults(null);
        setError(null);
      }
    }
  };

  const handleClear = () => {
    setSelectedImage(null);
    setResults(null);
    setError(null);
    setImageSize(null);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      if (file.size > 10 * 1024 * 1024) {
        setError('Image size must be less than 10MB');
        return;
      }
      const reader = new FileReader();
      reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
          setImageSize({ width: img.width, height: img.height });
          setSelectedImage(event.target?.result as string);
          setError(null);
          setResults(null);
        };
        img.src = event.target?.result as string;
      };
      reader.readAsDataURL(file);
    } else {
      setError('Please drop a valid image file');
    }
  };

  const handleTabChange = (tab: "upload" | "webcam") => {
    setActiveTab(tab);
    if (tab === "upload" && isCameraActive) {
      stopCamera();
    }
  };

  const handleDownloadReport = () => {
    if (!results) return;

    const report = {
      timestamp: new Date().toISOString(),
      detectionModel: results.detectionModel,
      recognitionModel: results.recognitionModel,
      processingTime: results.processingTime,
      overallConfidence: results.confidence,
      numberOfFaces: results.numFaces,
      faces: results.faces.map((face: any) => ({
        id: face.id,
        expression: face.expression,
        confidence: face.confidence,
        position: face.position
      }))
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `emotion-detection-report-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleKeyPress = (e: KeyboardEvent) => {
    if (e.key === 'Enter' && selectedImage && !isProcessing && !isCameraActive) {
      handleProcess();
    }
  };

  useEffect(() => {
    window.addEventListener('keydown', handleKeyPress as any);
    return () => window.removeEventListener('keydown', handleKeyPress as any);
  }, [selectedImage, isProcessing, isCameraActive]);

  return (
    <Layout>
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 py-12">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h1 className="text-4xl md:text-5xl font-bold text-slate-900 mb-4">
              Facial Expression Recognition
            </h1>
            <p className="text-xl text-slate-600 max-w-2xl mx-auto">
              Upload an image or use your webcam to detect faces and recognize emotions in real-time.
            </p>
            <div className="mt-4 flex justify-center gap-3">
              <button
                onClick={() => setShowHistory(!showHistory)}
                className="px-4 py-2 bg-white hover:bg-slate-50 border border-slate-300 text-slate-700 rounded-lg transition flex items-center gap-2"
              >
                <History className="w-4 h-4" />
                History ({history.length})
              </button>
              <button
                onClick={() => setIsBatchMode(!isBatchMode)}
                className={`px-4 py-2 border rounded-lg transition flex items-center gap-2 ${isBatchMode
                    ? 'bg-blue-600 text-white border-blue-600'
                    : 'bg-white hover:bg-slate-50 border-slate-300 text-slate-700'
                  }`}
              >
                <FileImage className="w-4 h-4" />
                Batch Mode {isBatchMode && '✓'}
              </button>
            </div>
          </div>

          {/* History Panel */}
          {showHistory && history.length > 0 && (
            <div className="mb-8 bg-white rounded-lg border border-slate-200 p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-slate-900 flex items-center gap-2">
                  <History className="w-5 h-5" />
                  Recent Analysis
                </h3>
                <button
                  onClick={clearHistory}
                  className="text-xs text-red-600 hover:text-red-700"
                >
                  Clear All
                </button>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                {history.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => loadFromHistory(item)}
                    className="relative group overflow-hidden rounded-lg border-2 border-transparent hover:border-blue-500 transition"
                  >
                    <img src={item.image} alt="History" className="w-full h-24 object-cover" />
                    <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition flex items-center justify-center">
                      <span className="text-white text-xs font-medium">
                        {item.results.numFaces} face(s)
                      </span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 -mb-px">
            <div className="lg:col-span-2 space-y-6">
              {!isBatchMode && (
                <div className="flex gap-2 bg-slate-100 p-1 rounded-lg">
                  <button
                    onClick={() => handleTabChange("upload")}
                    className={`flex-1 px-4 py-2 rounded-md font-medium transition ${activeTab === "upload"
                        ? "bg-white text-blue-600 shadow-sm"
                        : "text-slate-600 hover:text-slate-900"
                      }`}
                  >
                    <Upload className="w-4 h-4 inline mr-2" />
                    Upload Image
                  </button>
                  <button
                    onClick={() => handleTabChange("webcam")}
                    className={`flex-1 px-4 py-2 rounded-md font-medium transition ${activeTab === "webcam"
                        ? "bg-white text-blue-600 shadow-sm"
                        : "text-slate-600 hover:text-slate-900"
                      }`}
                  >
                    <Camera className="w-4 h-4 inline mr-2" />
                    Webcam
                  </button>
                </div>
              )}

              {/* Batch Upload Mode */}
              {isBatchMode && (
                <div className="space-y-4">
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <FileImage className="w-5 h-5 text-blue-600" />
                      <h3 className="font-semibold text-blue-900">Batch Processing Mode</h3>
                    </div>
                    <p className="text-sm text-blue-700">
                      Upload multiple images (up to 10) to process them all at once.
                    </p>
                  </div>

                  {selectedImages.length === 0 ? (
                    <div
                      onClick={() => batchFileInputRef.current?.click()}
                      className="rounded-lg text-center transition border-2 border-dashed border-slate-300 hover:border-blue-400 hover:bg-blue-50 p-8 md:p-32 cursor-pointer"
                    >
                      <ImageIcon className="w-12 h-12 text-slate-400 mx-auto mb-3" />
                      <p className="font-semibold text-slate-900 mb-1">
                        Click to upload multiple images
                      </p>
                      <p className="text-sm text-slate-600">
                        Up to 10 images • JPG, PNG, or WebP • Max 10MB each
                      </p>
                      <input
                        ref={batchFileInputRef}
                        type="file"
                        accept="image/*"
                        multiple
                        onChange={handleBatchFileSelect}
                        className="hidden"
                      />
                    </div>
                  ) : (
                    <>
                      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                        {selectedImages.map((img, idx) => (
                          <div key={idx} className="relative">
                            <img src={img} alt={`Batch ${idx + 1}`} className="w-full h-24 object-cover rounded-lg border-2 border-slate-200" />
                            <div className="absolute top-1 right-1 bg-black/50 text-white text-xs px-2 py-1 rounded">
                              {idx + 1}
                            </div>
                          </div>
                        ))}
                      </div>
                      <div className="flex gap-3">
                        <button
                          onClick={processBatchImages}
                          disabled={isProcessing}
                          className="flex-1 bg-gradient-to-r from-blue-600 to-purple-600 hover:shadow-lg disabled:opacity-50 text-white font-semibold py-3 rounded-lg transition flex items-center justify-center gap-2"
                        >
                          {isProcessing ? (
                            <>
                              <Loader2 className="w-5 h-5 animate-spin" />
                              Processing {batchResults.length + 1} of {selectedImages.length}...
                            </>
                          ) : (
                            <>
                              Analyze All ({selectedImages.length} images)
                            </>
                          )}
                        </button>
                        <button
                          onClick={() => { setSelectedImages([]); setBatchResults([]); }}
                          className="px-4 py-3 border border-slate-300 hover:bg-slate-50 text-slate-700 font-semibold rounded-lg transition"
                        >
                          <Trash2 className="w-5 h-5" />
                        </button>
                      </div>
                    </>
                  )}
                </div>
              )}

              {!isBatchMode && activeTab === "upload" && (
                <div className="space-y-4">
                  {!selectedImage ? (
                    <div
                      onClick={() => fileInputRef.current?.click()}
                      onDragOver={handleDragOver}
                      onDragLeave={handleDragLeave}
                      onDrop={handleDrop}
                      className={`rounded-lg text-center transition border-2 border-dashed p-8 md:p-32 cursor-pointer ${isDragging
                          ? "border-blue-500 bg-blue-100"
                          : "border-slate-300 hover:border-blue-400 hover:bg-blue-50"
                        }`}
                    >
                      <Upload className="w-12 h-12 text-slate-400 mx-auto mb-3" />
                      <p className="font-semibold text-slate-900 mb-1">
                        {isDragging ? "Drop image here" : "Click to upload an image"}
                      </p>
                      <p className="text-sm text-slate-600">
                        JPG, PNG, or WebP • Max 10MB
                      </p>
                      <input
                        ref={fileInputRef}
                        type="file"
                        accept="image/*"
                        onChange={handleFileSelect}
                        className="hidden"
                      />
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <div className="relative bg-white rounded-lg overflow-hidden border border-slate-200">
                        <img
                          src={selectedImage}
                          alt="Selected"
                          className="w-full h-auto"
                        />
                        {results && (
                          <div className="absolute inset-0 bg-black/20 flex items-center justify-center">
                            <CheckCircle className="w-16 h-16 text-green-400" />
                          </div>
                        )}
                      </div>
                      {imageSize && (
                        <div className="text-xs text-slate-500 text-center">
                          {imageSize.width} × {imageSize.height} pixels
                        </div>
                      )}
                    </div>
                  )}
                  {!selectedImage && (
                    <div className="border border-slate-200 rounded-lg p-6 text-center bg-white">
                      <div className="flex flex-col items-center justify-center h-full mb-8">
                        <div className="p-4 bg-blue-100 rounded-full mb-4">
                          <Camera className="w-8 h-8 text-blue-600" />
                        </div>
                        <div className="text-slate-600 mb-2">
                          Ready to analyze facial expressions
                        </div>
                        <div className="text-sm text-slate-500">
                          Upload an image or capture from webcam to begin
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {!isBatchMode && activeTab === "webcam" && (
                <div className="space-y-4">
                  {!isCameraActive ? (
                    <div className="space-y-4">
                      <div className="border-2 border-dashed border-slate-300 rounded-lg p-8 md:p-32 text-center bg-white">
                        <Camera className="w-12 h-12 text-slate-400 mx-auto mb-3" />
                        <p className="font-semibold text-slate-900 mb-1">
                          Use your webcam to capture
                        </p>
                        <p className="text-sm text-slate-600 mb-4">
                          Take a photo for emotion analysis
                        </p>
                      </div>
                      <button
                        onClick={startCamera}
                        className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 rounded-lg transition flex items-center justify-center gap-2"
                      >
                        <Camera className="w-5 h-5" />
                        Start Webcam
                      </button>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      <div className="relative bg-black rounded-lg overflow-hidden">
                        <video
                          ref={videoRef}
                          autoPlay
                          playsInline
                          muted
                          className="w-full rounded-lg"
                        />
                        <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 bg-black/50 text-white px-3 py-1 rounded-full text-sm">
                          Position your face in the frame
                        </div>
                      </div>
                      <canvas ref={canvasRef} className="hidden" />
                      <div className="flex gap-2">
                        <button
                          onClick={captureFrame}
                          className="flex-1 bg-green-600 hover:bg-green-700 text-white font-semibold py-3 rounded-lg transition flex items-center justify-center gap-2"
                        >
                          <Camera className="w-5 h-5" />
                          Capture Photo
                        </button>
                        <button
                          onClick={stopCamera}
                          className="px-6 bg-red-600 hover:bg-red-700 text-white font-semibold py-3 rounded-lg transition"
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex gap-3">
                  <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
                  <p className="text-red-700">{error}</p>
                </div>
              )}

              {selectedImage && !isCameraActive && (
                <div className="flex gap-3">
                  <button
                    onClick={handleProcess}
                    disabled={isProcessing}
                    className="flex-1 bg-gradient-to-r from-blue-600 to-purple-600 hover:shadow-lg disabled:opacity-50 text-white font-semibold py-3 rounded-lg transition flex items-center justify-center gap-2"
                    title="Click to analyze (or press Enter)"
                  >
                    {isProcessing ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        Processing...
                      </>
                    ) : (
                      <>
                        <span>Analyze Image</span>
                      </>
                    )}
                  </button>
                  <button
                    onClick={handleClear}
                    className="px-4 py-3 border border-slate-300 hover:bg-slate-50 text-slate-700 font-semibold rounded-lg transition flex items-center justify-center gap-2"
                    title="Clear image and results"
                  >
                    <Trash2 className="w-5 h-5" />
                  </button>
                </div>
              )}
            </div>

            <div className="mb-2 pl-px">
              <div className="bg-white rounded-lg border border-slate-200 p-6 space-y-4">
                <div className="flex items-center gap-2 mb-4">
                  <Settings className="w-5 h-5 text-blue-600" />
                  <h3 className="font-semibold text-slate-900">
                    Model Settings
                  </h3>
                </div>

                <div className="space-y-2">
                  <label className="block text-sm font-medium text-slate-700 flex items-center justify-between">
                    <span className="flex items-center gap-2">
                      <Sliders className="w-4 h-4" />
                      Confidence Threshold
                    </span>
                    <span className="text-blue-600 font-semibold">{confidenceThreshold}%</span>
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={confidenceThreshold}
                    onChange={(e) => {
                      setConfidenceThreshold(Number(e.target.value));
                      if (results && selectedImage) {
                        drawAnnotations(selectedImage, results.faces);
                      }
                    }}
                    className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                  />
                  <p className="text-xs text-slate-500">
                    Only show detections above {confidenceThreshold}% confidence
                  </p>
                </div>

                <div className="space-y-2">
                  <label className="block text-sm font-medium text-slate-700">
                    Detection Model
                  </label>
                  <select
                    value={detectionModel}
                    onChange={(e) =>
                      setDetectionModel(e.target.value as DetectionModel)
                    }
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"
                  >
                    <option value="yolo">YOLOv8 (Fast & Accurate)</option>
                    <option value="efficientb3">
                      EfficientNet-B3 (Efficient)
                    </option>
                  </select>
                  <p className="text-xs text-slate-500">
                    {detectionModel === "yolo"
                      ? "State-of-the-art real-time detection with 95%+ accuracy"
                      : "Efficient CNN with attention mechanism for mobile deployment"}
                  </p>
                </div>

                <div className="space-y-2">
                  <label className="block text-sm font-medium text-slate-700">
                    Recognition Model
                  </label>
                  <select
                    value={recognitionModel}
                    onChange={(e) =>
                      setRecognitionModel(e.target.value as RecognitionModel)
                    }
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"
                  >
                    <option value="yolo">YOLOv8 (Fast Detection) ✓</option>
                    <option value="efficientb3">
                      EfficientNet-B3 (Lightweight) ✓
                    </option>
                    <option value="arcface">ArcFace (ResNet-18) ✓</option>
                    <option value="swin" disabled>Swin Transformer (Coming Soon)</option>
                    <option value="vit" disabled>Vision Transformer (Coming Soon)</option>
                  </select>
                  <p className="text-xs text-slate-500">
                    {recognitionModel === "yolo"
                      ? "Fast detection with 7 emotion classes (angry, disgust, fear, happy, neutral, sad, surprised)"
                      : recognitionModel === "efficientb3"
                        ? "EfficientNet-B3 with CBAM attention for improved accuracy"
                        : recognitionModel === "arcface"
                          ? "ArcFace with ResNet-18 backbone and angular margin loss"
                          : recognitionModel === "swin"
                            ? "⚠️ Not yet implemented - will use YOLO"
                            : "⚠️ Not yet implemented - will use YOLO"}
                  </p>
                </div>

                <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                  <p className="text-xs text-blue-800">
                    <span className="font-semibold">Available Models:</span> YOLOv8, EfficientNet-B3, and ArcFace are fully implemented. Swin and ViT are coming soon.
                  </p>
                </div>

                {(recognitionModel === 'swin' || recognitionModel === 'vit') && (
                  <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
                    <p className="text-xs text-amber-800">
                      <span className="font-semibold">⚠️ Note:</span> Selected model is not yet implemented. The system will use YOLOv8 for inference.
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {results && (
            <div className="mt-6 results-panel">
              <div className="bg-white rounded-lg border border-slate-200 p-6">
                <div className="space-y-6">
                  <div className="flex items-center gap-2 text-green-600">
                    <CheckCircle className="w-6 h-6" />
                    <span className="font-semibold">
                      {results.numFaces === 1 ? '1 Face' : `${results.numFaces} Faces`} Detected Successfully
                    </span>
                  </div>

                  <div className="bg-slate-50 rounded-lg p-4 space-y-2 border border-slate-200">
                    <div>
                      <p className="text-xs text-slate-600 mb-1">
                        Detection Model
                      </p>
                      <p className="text-sm font-semibold text-slate-900">
                        {results.detectionModel}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-slate-600 mb-1">
                        Recognition Model
                      </p>
                      <p className="text-sm font-semibold text-slate-900">
                        {results.recognitionModel}
                      </p>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-blue-50 rounded-lg p-3">
                      <p className="text-xs text-slate-600 mb-1">
                        Overall Confidence
                      </p>
                      <p className="text-xl font-bold text-blue-600">
                        {results.confidence}%
                      </p>
                    </div>
                    <div className="bg-purple-50 rounded-lg p-3">
                      <p className="text-xs text-slate-600 mb-1">
                        Processing Time
                      </p>
                      <p className="text-xl font-bold text-purple-600">
                        {results.processingTime}ms
                      </p>
                    </div>
                  </div>

                  <div>
                    <h3 className="font-semibold text-slate-900 mb-3 text-sm">
                      Detected Faces ({results.faces.length})
                    </h3>
                    <div className="space-y-2">
                      {results.faces.map((face: any) => (
                        <div
                          key={face.id}
                          className="bg-slate-50 rounded-lg p-3 border border-slate-200 text-sm"
                        >
                          <div className="flex items-start justify-between mb-2">
                            <span className="font-medium text-slate-900">
                              Face #{face.id}
                            </span>
                            <span className="px-2 py-1 bg-green-100 text-green-700 rounded text-xs font-medium">
                              {face.confidence}%
                            </span>
                          </div>
                          <div className="grid grid-cols-1 gap-1 text-xs">
                            <div>
                              <p className="text-slate-600">Expression</p>
                              <p className="font-medium text-slate-900 capitalize">
                                {face.expression}
                              </p>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <button
                    onClick={handleDownloadReport}
                    className="w-full bg-slate-100 hover:bg-slate-200 text-slate-900 font-semibold py-2 rounded-lg transition flex items-center justify-center gap-2 text-sm">
                    <Download className="w-4 h-4" />
                    Download Report (JSON)
                  </button>

                  <button
                    onClick={downloadAnnotatedImage}
                    className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 rounded-lg transition flex items-center justify-center gap-2 text-sm">
                    <ImageIcon className="w-4 h-4" />
                    Download Annotated Image
                  </button>
                </div>
              </div>

              {/* Hidden canvas for annotations */}
              <canvas ref={annotatedCanvasRef} className="hidden" />
            </div>
          )}

          {/* Batch Results */}
          {isBatchMode && batchResults.length > 0 && (
            <div className="mt-6 space-y-4">
              <div className="bg-white rounded-lg border border-slate-200 p-6">
                <h3 className="font-semibold text-slate-900 mb-4 flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-green-600" />
                  Batch Processing Results ({batchResults.length} images)
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {batchResults.map((result, idx) => (
                    <div key={idx} className="border border-slate-200 rounded-lg p-4">
                      <div className="flex gap-3">
                        <img src={result.image} alt={`Result ${idx + 1}`} className="w-24 h-24 object-cover rounded-lg" />
                        <div className="flex-1">
                          <div className="flex items-center justify-between mb-2">
                            <span className="font-medium text-slate-900">Image #{idx + 1}</span>
                            {result.success ? (
                              <CheckCircle className="w-5 h-5 text-green-600" />
                            ) : (
                              <AlertCircle className="w-5 h-5 text-red-600" />
                            )}
                          </div>
                          {result.success ? (
                            <>
                              <p className="text-sm text-slate-600">
                                Faces: <span className="font-semibold text-slate-900">{result.data.num_faces || 0}</span>
                              </p>
                              <p className="text-sm text-slate-600">
                                Primary: <span className="font-semibold text-slate-900 capitalize">{result.data.expression}</span> ({(result.data.confidence * 100).toFixed(1)}%)
                              </p>
                            </>
                          ) : (
                            <p className="text-sm text-red-600">{result.error}</p>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
                <button
                  onClick={() => {
                    const report = {
                      timestamp: new Date().toISOString(),
                      totalImages: batchResults.length,
                      successful: batchResults.filter(r => r.success).length,
                      failed: batchResults.filter(r => !r.success).length,
                      results: batchResults.map((r, idx) => ({
                        imageNumber: idx + 1,
                        success: r.success,
                        faces: r.success ? r.data.num_faces : 0,
                        expression: r.success ? r.data.expression : null,
                        confidence: r.success ? r.data.confidence : null,
                      }))
                    };
                    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `batch-emotion-report-${Date.now()}.json`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                  }}
                  className="w-full mt-4 bg-slate-100 hover:bg-slate-200 text-slate-900 font-semibold py-2 rounded-lg transition flex items-center justify-center gap-2 text-sm"
                >
                  <Download className="w-4 h-4" />
                  Download Batch Report
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
}
