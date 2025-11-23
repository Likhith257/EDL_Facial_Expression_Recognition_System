import { useState, useRef } from "react";
import Layout from "@/components/Layout";
import {
  Upload,
  Loader2,
  GitCompare,
  CheckCircle2,
  XCircle,
  TrendingUp,
} from "lucide-react";

interface ModelResult {
  expression?: string;
  emotion?: string;
  confidence: number;
  processing_time: number;
  framework: string;
  error?: string;
}

interface ComparisonResult {
  comparisons: Record<string, ModelResult>;
  consensus: {
    emotion: string;
    agreement_percentage: number;
    total_models: number;
  };
}

export default function CompareModels() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState<ComparisonResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setSelectedImage(event.target?.result as string);
        setResults(null);
        setError(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const compareModels = async () => {
    if (!selectedImage) return;

    setIsProcessing(true);
    setError(null);

    try {
      const formData = new FormData();
      const response = await fetch(selectedImage);
      const blob = await response.blob();
      formData.append("file", blob, "image.jpg");

      const apiResponse = await fetch("/api/compare", {
        method: "POST",
        body: formData,
      });

      if (!apiResponse.ok) {
        throw new Error("Comparison failed");
      }

      const data = await apiResponse.json();
      setResults(data);
    } catch (err: any) {
      setError(err.message || "Failed to compare models");
    } finally {
      setIsProcessing(false);
    }
  };

  const getEmotionColor = (emotion: string) => {
    const colors: Record<string, string> = {
      happy: "text-green-600 bg-green-50",
      sad: "text-blue-600 bg-blue-50",
      angry: "text-red-600 bg-red-50",
      fear: "text-purple-600 bg-purple-50",
      disgust: "text-orange-600 bg-orange-50",
      surprised: "text-yellow-600 bg-yellow-50",
      neutral: "text-gray-600 bg-gray-50",
    };
    return colors[emotion.toLowerCase()] || "text-gray-600 bg-gray-50";
  };

  return (
    <Layout>
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 dark:from-slate-900 dark:via-slate-900 dark:to-slate-800 py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent flex items-center gap-3">
              <GitCompare className="w-10 h-10 text-blue-600" />
              Model Comparison
            </h1>
            <p className="text-slate-600 dark:text-slate-400 mt-2">
              Compare predictions from multiple models on the same image
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Upload Section */}
            <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-semibold mb-4">Upload Image</h2>

              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                className="hidden"
              />

              <div
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-slate-300 dark:border-slate-600 rounded-lg p-8 text-center cursor-pointer hover:border-blue-500 transition"
              >
                <Upload className="w-16 h-16 mx-auto mb-4 text-slate-400" />
                <p className="text-slate-600 dark:text-slate-400">
                  Click to upload an image
                </p>
              </div>

              {selectedImage && (
                <div className="mt-6">
                  <img
                    src={selectedImage}
                    alt="Selected"
                    className="w-full rounded-lg shadow-md"
                  />
                  <button
                    onClick={compareModels}
                    disabled={isProcessing}
                    className="w-full mt-4 px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg font-semibold hover:shadow-lg transition disabled:opacity-50 flex items-center justify-center gap-2"
                  >
                    {isProcessing ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        Comparing...
                      </>
                    ) : (
                      <>
                        <GitCompare className="w-5 h-5" />
                        Compare All Models
                      </>
                    )}
                  </button>
                </div>
              )}

              {error && (
                <div className="mt-4 p-4 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 rounded-lg flex items-center gap-2">
                  <XCircle className="w-5 h-5 flex-shrink-0" />
                  <span>{error}</span>
                </div>
              )}
            </div>

            {/* Results Section */}
            <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-semibold mb-4">Comparison Results</h2>

              {!results && !isProcessing && (
                <div className="text-center text-slate-400 py-12">
                  <GitCompare className="w-16 h-16 mx-auto mb-4 opacity-50" />
                  <p>Upload an image and compare models to see results</p>
                </div>
              )}

              {results && (
                <div className="space-y-6">
                  {/* Consensus */}
                  <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg p-6 border border-green-200 dark:border-green-800">
                    <div className="flex items-center gap-3 mb-2">
                      <CheckCircle2 className="w-6 h-6 text-green-600" />
                      <h3 className="text-lg font-semibold text-green-900 dark:text-green-100">
                        Consensus
                      </h3>
                    </div>
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-3xl font-bold text-green-600 capitalize">
                          {results.consensus.emotion}
                        </p>
                        <p className="text-sm text-green-700 dark:text-green-300">
                          {results.consensus.agreement_percentage.toFixed(1)}%
                          agreement
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="text-sm text-green-700 dark:text-green-300">
                          {results.consensus.total_models} models
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Individual Model Results */}
                  <div className="space-y-4">
                    <h3 className="font-semibold text-lg">
                      Individual Model Results
                    </h3>

                    {Object.entries(results.comparisons).map(
                      ([modelName, result]) => (
                        <div
                          key={modelName}
                          className="border border-slate-200 dark:border-slate-700 rounded-lg p-4"
                        >
                          {result.error ? (
                            <div className="flex items-center justify-between">
                              <div>
                                <h4 className="font-semibold uppercase text-slate-900 dark:text-white">
                                  {modelName}
                                </h4>
                                <p className="text-red-600 text-sm">
                                  {result.error}
                                </p>
                              </div>
                              <XCircle className="w-6 h-6 text-red-500" />
                            </div>
                          ) : (
                            <div className="space-y-3">
                              <div className="flex items-center justify-between">
                                <h4 className="font-semibold uppercase text-slate-900 dark:text-white">
                                  {modelName}
                                </h4>
                                <span
                                  className={`px-3 py-1 rounded-full text-sm font-medium capitalize ${getEmotionColor(
                                    result.expression || result.emotion || ""
                                  )}`}
                                >
                                  {result.expression || result.emotion}
                                </span>
                              </div>

                              <div className="grid grid-cols-2 gap-4 text-sm">
                                <div>
                                  <p className="text-slate-600 dark:text-slate-400">
                                    Confidence
                                  </p>
                                  <p className="font-semibold text-slate-900 dark:text-white">
                                    {(result.confidence * 100).toFixed(1)}%
                                  </p>
                                </div>
                                <div>
                                  <p className="text-slate-600 dark:text-slate-400">
                                    Processing Time
                                  </p>
                                  <p className="font-semibold text-slate-900 dark:text-white">
                                    {result.processing_time.toFixed(3)}s
                                  </p>
                                </div>
                              </div>

                              {/* Confidence Bar */}
                              <div className="relative pt-1">
                                <div className="flex mb-2 items-center justify-between">
                                  <div>
                                    <span className="text-xs font-semibold inline-block text-blue-600">
                                      Confidence Level
                                    </span>
                                  </div>
                                </div>
                                <div className="overflow-hidden h-2 text-xs flex rounded bg-blue-200">
                                  <div
                                    style={{
                                      width: `${result.confidence * 100}%`,
                                    }}
                                    className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-blue-600"
                                  ></div>
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      )
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}
