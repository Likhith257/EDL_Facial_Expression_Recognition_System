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

  const [detectionModel, setDetectionModel] = useState<DetectionModel>("yolo");
  const [recognitionModel, setRecognitionModel] =
    useState<RecognitionModel>("yolo");

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
  };

  const handleTabChange = (tab: "upload" | "webcam") => {
    setActiveTab(tab);
    if (tab === "upload" && isCameraActive) {
      stopCamera();
    }
  };

  const handleKeyPress = (e: KeyboardEvent) => {
    if (e.key === 'Enter' && selectedImage && !isProcessing && !isCameraActive) {
      handleProcess();
    }
  };

  useState(() => {
    window.addEventListener('keydown', handleKeyPress as any);
    return () => window.removeEventListener('keydown', handleKeyPress as any);
  });

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
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 -mb-px">
            <div className="lg:col-span-2 space-y-6">
              <div className="flex gap-2 bg-slate-100 p-1 rounded-lg">
                <button
                  onClick={() => handleTabChange("upload")}
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
                  onClick={() => handleTabChange("webcam")}
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

              {activeTab === "webcam" && (
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
                    <option value="arcface" disabled>ArcFace (Coming Soon)</option>
                    <option value="swin" disabled>Swin Transformer (Coming Soon)</option>
                    <option value="vit" disabled>Vision Transformer (Coming Soon)</option>
                  </select>
                  <p className="text-xs text-slate-500">
                    {recognitionModel === "yolo"
                      ? "Fast detection with 7 emotion classes (angry, disgust, fear, happy, neutral, sad, surprised)"
                      : recognitionModel === "efficientb3"
                        ? "EfficientNet-B3 with CBAM attention for improved accuracy"
                        : recognitionModel === "arcface"
                          ? "⚠️ Not yet implemented - will use YOLO"
                          : recognitionModel === "swin"
                            ? "⚠️ Not yet implemented - will use YOLO"
                            : "⚠️ Not yet implemented - will use YOLO"}
                  </p>
                </div>

                <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                  <p className="text-xs text-blue-800">
                    <span className="font-semibold">Available Models:</span> YOLOv8 and EfficientNet-B3 are fully implemented. ArcFace, Swin, and ViT are coming soon.
                  </p>
                </div>

                {(recognitionModel === 'arcface' || recognitionModel === 'swin' || recognitionModel === 'vit') && (
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
