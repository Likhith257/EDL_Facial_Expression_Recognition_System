import { useEffect, useState } from "react";
import Layout from "@/components/Layout";
import {
  BarChart,
  Bar,
  PieChart,
  Pie,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from "recharts";
import {
  TrendingUp,
  Activity,
  Users,
  Clock,
  Download,
  RefreshCw,
} from "lucide-react";

interface Statistics {
  total_predictions: number;
  unique_sessions: number;
  emotion_distribution: Record<string, { count: number; percentage: number }>;
  model_usage: Record<string, { count: number; percentage: number }>;
  average_processing_time: number;
  average_confidence: number;
  time_range: string;
}

export default function Analytics() {
  const [statistics, setStatistics] = useState<Statistics | null>(null);
  const [trends, setTrends] = useState<any[]>([]);
  const [timeRange, setTimeRange] = useState<string>("day");
  const [loading, setLoading] = useState(true);

  const EMOTION_COLORS: Record<string, string> = {
    happy: "#10b981",
    sad: "#3b82f6",
    angry: "#ef4444",
    fear: "#8b5cf6",
    disgust: "#f59e0b",
    surprised: "#eab308",
    neutral: "#6b7280",
  };

  useEffect(() => {
    fetchAnalytics();
  }, [timeRange]);

  const fetchAnalytics = async () => {
    setLoading(true);
    try {
      // Fetch statistics
      const statsRes = await fetch(
        `/api/analytics/statistics?time_range=${timeRange}`
      );
      const statsData = await statsRes.json();
      setStatistics(statsData);

      // Fetch trends
      const trendsRes = await fetch(`/api/analytics/trends?interval=hour`);
      const trendsData = await trendsRes.json();
      setTrends(trendsData.trends || []);
    } catch (error) {
      console.error("Error fetching analytics:", error);
    } finally {
      setLoading(false);
    }
  };

  const exportData = async (format: string) => {
    try {
      const response = await fetch(`/api/analytics/export?format=${format}`);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `analytics_${new Date().toISOString().split("T")[0]}.${format}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (error) {
      console.error("Error exporting data:", error);
    }
  };

  if (loading) {
    return (
      <Layout>
        <div className="min-h-screen flex items-center justify-center">
          <div className="flex items-center gap-3">
            <RefreshCw className="w-6 h-6 animate-spin text-blue-600" />
            <span className="text-lg">Loading analytics...</span>
          </div>
        </div>
      </Layout>
    );
  }

  if (!statistics) {
    return (
      <Layout>
        <div className="min-h-screen flex items-center justify-center">
          <div className="text-center">
            <Activity className="w-16 h-16 mx-auto mb-4 text-slate-400" />
            <h2 className="text-2xl font-semibold mb-2">No Data Available</h2>
            <p className="text-slate-600 dark:text-slate-400">
              Start making predictions to see analytics
            </p>
          </div>
        </div>
      </Layout>
    );
  }

  // Prepare chart data
  const emotionChartData = Object.entries(statistics.emotion_distribution).map(
    ([emotion, data]) => ({
      emotion: emotion.charAt(0).toUpperCase() + emotion.slice(1),
      count: data.count,
      percentage: data.percentage,
    })
  );

  const modelChartData = Object.entries(statistics.model_usage).map(
    ([model, data]) => ({
      model: model.toUpperCase(),
      count: data.count,
      percentage: data.percentage,
    })
  );

  return (
    <Layout>
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 dark:from-slate-900 dark:via-slate-900 dark:to-slate-800 py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* Header */}
          <div className="flex justify-between items-center mb-8">
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Analytics Dashboard
              </h1>
              <p className="text-slate-600 dark:text-slate-400 mt-2">
                Real-time insights and statistics
              </p>
            </div>

            <div className="flex gap-3">
              {/* Time Range Selector */}
              <select
                value={timeRange}
                onChange={(e) => setTimeRange(e.target.value)}
                className="px-4 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-lg"
              >
                <option value="hour">Last Hour</option>
                <option value="day">Last 24 Hours</option>
                <option value="week">Last Week</option>
                <option value="month">Last Month</option>
              </select>

              {/* Export Buttons */}
              <button
                onClick={() => exportData("csv")}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 flex items-center gap-2"
              >
                <Download className="w-4 h-4" />
                CSV
              </button>
              <button
                onClick={() => exportData("json")}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2"
              >
                <Download className="w-4 h-4" />
                JSON
              </button>
              <button
                onClick={fetchAnalytics}
                className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 flex items-center gap-2"
              >
                <RefreshCw className="w-4 h-4" />
                Refresh
              </button>
            </div>
          </div>

          {/* Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    Total Predictions
                  </p>
                  <p className="text-3xl font-bold text-blue-600">
                    {statistics.total_predictions}
                  </p>
                </div>
                <TrendingUp className="w-12 h-12 text-blue-600 opacity-20" />
              </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    Unique Sessions
                  </p>
                  <p className="text-3xl font-bold text-purple-600">
                    {statistics.unique_sessions}
                  </p>
                </div>
                <Users className="w-12 h-12 text-purple-600 opacity-20" />
              </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    Avg Confidence
                  </p>
                  <p className="text-3xl font-bold text-green-600">
                    {(statistics.average_confidence * 100).toFixed(1)}%
                  </p>
                </div>
                <Activity className="w-12 h-12 text-green-600 opacity-20" />
              </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    Avg Processing
                  </p>
                  <p className="text-3xl font-bold text-orange-600">
                    {statistics.average_processing_time.toFixed(2)}s
                  </p>
                </div>
                <Clock className="w-12 h-12 text-orange-600 opacity-20" />
              </div>
            </div>
          </div>

          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            {/* Emotion Distribution Bar Chart */}
            <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-semibold mb-4">
                Emotion Distribution
              </h2>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={emotionChartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="emotion" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" radius={[8, 8, 0, 0]}>
                    {emotionChartData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={EMOTION_COLORS[entry.emotion.toLowerCase()]}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Emotion Distribution Pie Chart */}
            <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-semibold mb-4">
                Emotion Percentages
              </h2>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={emotionChartData}
                    dataKey="percentage"
                    nameKey="emotion"
                    cx="50%"
                    cy="50%"
                    outerRadius={100}
                    label={({ emotion, percentage }) =>
                      `${emotion}: ${percentage.toFixed(1)}%`
                    }
                  >
                    {emotionChartData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={EMOTION_COLORS[entry.emotion.toLowerCase()]}
                      />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Model Usage */}
          {modelChartData.length > 0 && (
            <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6 mb-8">
              <h2 className="text-xl font-semibold mb-4">Model Usage</h2>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={modelChartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="model" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar
                    dataKey="count"
                    fill="#8b5cf6"
                    radius={[8, 8, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Trends Over Time */}
          {trends.length > 0 && (
            <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-semibold mb-4">
                Prediction Trends Over Time
              </h2>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={trends}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timestamp" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="total"
                    stroke="#3b82f6"
                    strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
}
