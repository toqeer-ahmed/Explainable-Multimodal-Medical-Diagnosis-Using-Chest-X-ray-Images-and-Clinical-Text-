"use client";

import { useState } from "react";
import { UploadCard } from "@/components/UploadCard";
import { ImageHeatmap } from "@/components/ImageHeatmap";
import { PredictionCard } from "@/components/PredictionCard";
import { TextExplanation } from "@/components/TextExplanation";

export default function Home() {
  const [isLoading, setIsLoading] = useState(false);
  const [originalImage, setOriginalImage] = useState<string | null>(null);
  
  // State for backend results
  const [heatmapImage, setHeatmapImage] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<{ label: string; confidence: number }[]>([]);
  const [textTokens, setTextTokens] = useState<{ word: string; score: number }[]>([]);

  const handlePredict = async (file: File, text: string) => {
    setIsLoading(true);
    
    // Create an object URL for the uploaded image to display immediately
    const objectUrl = URL.createObjectURL(file);
    setOriginalImage(objectUrl);

    try {
      const formData = new FormData();
      formData.append("image", file);
      formData.append("text", text);
      
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error("Backend API returned an error.");
      }

      const data = await response.json();
      setPredictions(data.predictions);
      setHeatmapImage(data.heatmap_url);
      setTextTokens(data.highlighted_text || []);

    } catch (error) {
      console.error("Prediction failed:", error);
      alert("Failed to connect to backend AI server.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-50 text-slate-900 font-sans selection:bg-blue-100">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-5">
          <div className="flex items-center gap-3">
            <div className="bg-blue-600 p-2 rounded-xl shadow-sm">
              <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
              </svg>
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-slate-900">AI Medical Diagnosis System</h1>
              <p className="text-sm text-slate-500 font-medium">Explainable Multimodal Analysis (EfficientNet-B0 + Bio_ClinicalBERT)</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        
        {/* Input Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-3">
            <UploadCard onSubmit={handlePredict} isLoading={isLoading} />
          </div>
        </div>

        {/* Results Section */}
        {isLoading && (
          <div className="space-y-6 animate-in fade-in duration-500">
            <h2 className="text-xl font-semibold text-slate-800 border-b border-slate-200 pb-2 flex items-center gap-2">
              <svg className="animate-spin h-5 w-5 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Analyzing Model...
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {[1, 2, 3].map((i) => (
                <div key={i} className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6 h-[400px] flex flex-col gap-4 animate-pulse">
                  <div className="h-6 bg-slate-200 rounded w-1/3"></div>
                  <div className="flex-1 bg-slate-100 rounded-xl"></div>
                </div>
              ))}
            </div>
          </div>
        )}

        {!isLoading && predictions.length > 0 && (
          <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-700 ease-out">
            <h2 className="text-xl font-semibold text-slate-800 border-b border-slate-200 pb-2">Analysis Results</h2>
            
            <div className={`grid grid-cols-1 md:grid-cols-2 ${textTokens.length > 0 ? 'lg:grid-cols-3' : ''} gap-6`}>
              <div className="h-full">
                <ImageHeatmap 
                  originalImage={originalImage} 
                  heatmapImage={heatmapImage} 
                />
              </div>
              
              <div className="h-full">
                <PredictionCard predictions={predictions} />
              </div>
              
              {textTokens.length > 0 && (
                <div className="h-full">
                  <TextExplanation tokens={textTokens} />
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
