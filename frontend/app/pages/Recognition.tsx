import { useState, useRef } from "react";
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
} from "lucide-react";

type DetectionModel = "yolo" | "efficientb3";
type RecognitionModel = "yolo" | "efficientb3" | "arcface" | "swin" | "vit";

export default function Recognition() {
  const [activeTab, setActiveTab] = useState<"upload" | "webcam">("upload");
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isCameraActive, setIsCameraActive] = useState(false);

  // Model selection state
  const [detectionModel, setDetectionModel] = useState<DetectionModel>("yolo");
  const [recognitionModel, setRecognitionModel] =
    useState<RecognitionModel>("arcface");

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setSelectedImage(event.target?.result as string);
        setError(null);
        setResults(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleProcess = async () => {
    if (!selectedImage) return;

    setIsProcessing(true);
    setError(null);

    try {
      const startTime = Date.now();
      
      // Convert base64 to blob
      const response = await fetch(selectedImage);
      const blob = await response.blob();
      
      // Create form data
      const formData = new FormData();
      formData.append('file', blob, 'image.jpg');
      
      // Call backend API
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
      
      // Map API response to frontend format
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
        age: "Unknown",
        gender: "Unknown",
        embedding: `Detected by ${data.model}`,
      })) || [];

      setResults({
        detected: faces.length > 0,
        confidence: data.confidence ? (data.confidence * 100).toFixed(1) : 0,
        detectionModel: modelNames[detectionModel],
        recognitionModel: data.framework === 'yolo' ? 'YOLOv8' : recognitionNames[recognitionModel],
        faces: faces,
        processingTime: processingTime,
        numFaces: data.num_faces || 0,
      });
      
    } catch (err: any) {
      setError(err.message || 'Failed to process image');
      setResults(null);
    } finally {
      setIsProcessing(false);
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsCameraActive(true);
        setError(null);
      }
    } catch (err) {
      setError("Unable to access camera. Please check permissions.");
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
      }
    }
  };

  const handleClear = () => {
    setSelectedImage(null);
    setResults(null);
    setError(null);
  };

  return (
    <Layout>
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 py-12">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* Header */}
          <div className="text-center mb-12">
            <h1 className="text-4xl md:text-5xl font-bold text-slate-900 mb-4">
              Facial Recognition
            </h1>
            <p className="text-xl text-slate-600 max-w-2xl mx-auto">
              Upload an image or use your webcam to detect and analyze faces in
              real-time.
            </p>
          </div>

          {/* Main Content */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 -mb-px">
            {/* Input Section - spans 2 columns */}
            <div className="lg:col-span-2 space-y-6">
              {/* Tab Selection */}
              <div className="flex gap-2 bg-slate-100 p-1 rounded-lg">
                <button
                  onClick={() => setActiveTab("upload")}
                  className={`flex-1 px-4 py-2 rounded-md font-medium transition ${
                    activeTab === "upload"
                      ? "bg-white text-blue-600 shadow-sm"
                      : "text-slate-600 hover:text-slate-900"
                  }`}
                >
                  <Upload className="w-4 h-4 inline mr-2" />
                  Upload Image
                </button>
                <button
                  onClick={() => setActiveTab("webcam")}
                  className={`flex-1 px-4 py-2 rounded-md font-medium transition ${
                    activeTab === "webcam"
                      ? "bg-white text-blue-600 shadow-sm"
                      : "text-slate-600 hover:text-slate-900"
                  }`}
                >
                  <Camera className="w-4 h-4 inline mr-2" />
                  Webcam
                </button>
              </div>

              {/* Upload Tab */}
              {activeTab === "upload" && (
                <div className="space-y-4">
                  {!selectedImage ? (
                    <div
                      onClick={() => fileInputRef.current?.click()}
                      className="rounded-lg text-center transition border-2 border-dashed border-slate-300 p-8 md:p-32 cursor-pointer hover:border-blue-400 hover:bg-blue-50"
                    >
                      <Upload className="w-12 h-12 text-slate-400 mx-auto mb-3" />
                      <p className="font-semibold text-slate-900 mb-1">
                        Click to upload an image
                      </p>
                      <p className="text-sm text-slate-600">
                        or drag and drop (JPG, PNG, WebP)
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
                  )}
                  {!selectedImage && (
                    <div className="border border-slate-200 rounded-lg p-6 text-center bg-white">
                      <div className="flex flex-col items-center justify-center h-full mb-8">
                        <div className="p-4 bg-blue-100 rounded-full mb-4">
                          <Camera className="w-8 h-8 text-blue-600" />
                        </div>
                        <div className="text-slate-600 mb-2">
                          Upload an image or use your webcam to get started
                        </div>
                        <div className="text-sm text-slate-500">
                          Results will appear here
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Webcam Tab */}
              {activeTab === "webcam" && (
                <div className="space-y-4">
                  {!isCameraActive ? (
                    <button
                      onClick={startCamera}
                      className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 rounded-lg transition flex items-center justify-center gap-2"
                    >
                      <Camera className="w-5 h-5" />
                      Start Webcam
                    </button>
                  ) : (
                    <div className="space-y-3">
                      <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        className="w-full rounded-lg bg-black"
                      />
                      <canvas ref={canvasRef} className="hidden" />
                      <div className="flex gap-2">
                        <button
                          onClick={captureFrame}
                          className="flex-1 bg-green-600 hover:bg-green-700 text-white font-semibold py-2 rounded-lg transition"
                        >
                          Capture Photo
                        </button>
                        <button
                          onClick={stopCamera}
                          className="flex-1 bg-red-600 hover:bg-red-700 text-white font-semibold py-2 rounded-lg transition"
                        >
                          Close Camera
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Error Message */}
              {error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex gap-3">
                  <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
                  <p className="text-red-700">{error}</p>
                </div>
              )}

              {/* Action Buttons */}
              {selectedImage && !isCameraActive && (
                <div className="flex gap-3">
                  <button
                    onClick={handleProcess}
                    disabled={isProcessing}
                    className="flex-1 bg-gradient-to-r from-blue-600 to-purple-600 hover:shadow-lg disabled:opacity-50 text-white font-semibold py-3 rounded-lg transition flex items-center justify-center gap-2"
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
                  >
                    <Trash2 className="w-5 h-5" />
                  </button>
                </div>
              )}
            </div>

            {/* Model Selection Panel - right column */}
            <div className="mb-2 pl-px">
              {/* Model Selection Panel */}
              <div className="bg-white rounded-lg border border-slate-200 p-6 space-y-4">
                <div className="flex items-center gap-2 mb-4">
                  <Settings className="w-5 h-5 text-blue-600" />
                  <h3 className="font-semibold text-slate-900">
                    Model Settings
                  </h3>
                </div>

                {/* Detection Model Selection */}
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
                      ? "Real-time object detection with high accuracy"
                      : "Lightweight model optimized for efficiency"}
                  </p>
                </div>

                {/* Recognition Model Selection */}
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
                    <option value="yolo">YOLOv8 (Fast Detection)</option>
                    <option value="efficientb3">
                      EfficientNet-B3 (Lightweight)
                    </option>
                    <option value="arcface">ArcFace (Industry Standard)</option>
                    <option value="swin">Swin Transformer</option>
                    <option value="vit">Vision Transformer (ViT)</option>
                  </select>
                  <p className="text-xs text-slate-500">
                    {recognitionModel === "yolo"
                      ? "Real-time face detection and embedding extraction"
                      : recognitionModel === "efficientb3"
                        ? "Efficient embedding model for mobile and edge devices"
                        : recognitionModel === "arcface"
                          ? "Industry-standard face embedding model"
                          : recognitionModel === "swin"
                            ? "Transformer-based with hierarchical structure"
                            : "Pure vision transformer architecture"}
                  </p>
                </div>

                {/* Info Box */}
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                  <p className="text-xs text-blue-800">
                    <span className="font-semibold">Tip:</span> Different models
                    provide different accuracy/speed tradeoffs. Try different
                    combinations to find what works best for your use case.
                  </p>
                </div>

                {/* Start Button */}
                <button className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:shadow-lg text-white font-semibold py-2 px-4 rounded-lg transition flex items-center justify-center gap-2">
                  Start
                </button>
              </div>
            </div>
          </div>

          {/* Results Panel - below upload section */}
          {results && (
            <div className="mt-6">
              <div className="bg-white rounded-lg border border-slate-200 p-6">
                <div className="space-y-6">
                  <div className="flex items-center gap-2 text-green-600">
                    <CheckCircle className="w-6 h-6" />
                    <span className="font-semibold">
                      Face(s) Detected Successfully
                    </span>
                  </div>

                  {/* Model Info */}
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

                  {/* Overall Stats */}
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

                  {/* Detected Faces */}
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
                          <div className="grid grid-cols-2 gap-1 text-xs">
                            <div>
                              <p className="text-slate-600">Expression</p>
                              <p className="font-medium text-slate-900 capitalize">
                                {face.expression}
                              </p>
                            </div>
                            <div>
                              <p className="text-slate-600">Age</p>
                              <p className="font-medium text-slate-900">
                                {face.age}
                              </p>
                            </div>
                            <div className="col-span-2">
                              <p className="text-slate-600">Gender</p>
                              <p className="font-medium text-slate-900">
                                {face.gender}
                              </p>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Export Button */}
                  <button className="w-full bg-slate-100 hover:bg-slate-200 text-slate-900 font-semibold py-2 rounded-lg transition flex items-center justify-center gap-2 text-sm">
                    <Download className="w-4 h-4" />
                    Download Report
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
}
