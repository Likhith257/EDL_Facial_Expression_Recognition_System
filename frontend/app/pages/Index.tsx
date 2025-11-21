import { Link } from "react-router-dom";
import Layout from "@/components/Layout";
import { ArrowRight, Zap, Eye } from "lucide-react";

export default function Index() {
  return (
    <Layout>
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-blue-50 via-white to-purple-50" />
        <div className="absolute top-0 right-0 -translate-y-1/4 translate-x-1/4 w-96 h-96 bg-blue-200 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob" />
        <div className="absolute bottom-0 left-0 translate-y-1/4 -translate-x-1/4 w-96 h-96 bg-purple-200 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-2000" />

        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 md:py-32 -mb-5px">
          <div className="text-center">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-blue-50 border border-blue-200 rounded-full mb-6">
              <Zap className="w-4 h-4 text-blue-600" />
              <span className="text-sm font-medium text-blue-600">
                Powered by Advanced Models
              </span>
            </div>

            <h1 className="text-5xl md:text-7xl font-bold mb-6 bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent">
              Facial Expression
              <br />
              Recognition
            </h1>

            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-500 rounded-2xl blur opacity-30" />
              <div className="relative bg-white rounded-2xl border border-slate-200 p-1 overflow-hidden" />
            </div>
          </div>
        </div>

        <div className="relative max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <div className="text-4xl md:text-5xl font-bold text-slate-900 mb-6">
            Ready to Get Started?
          </div>
          <div className="text-xl text-slate-600 mb-8">
            Experience the power of our facial expression recognition technology
            today.
          </div>
          <a
            href="https://53443c32a5fe4b468bbf0ac30e6d6b70-e2dc4119e9b446c2b9753ca20.fly.dev/recognition"
            className="inline-flex items-center justify-center px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg font-semibold hover:shadow-2xl hover:scale-105 transition transform"
          >
            Start Recognizing
            <ArrowRight className="ml-2 w-5 h-5" />
          </a>
        </div>
      </section>
    </Layout>
  );
}
