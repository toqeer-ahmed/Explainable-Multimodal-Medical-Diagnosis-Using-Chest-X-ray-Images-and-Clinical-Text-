"use client";

import { useState } from "react";
import { Image as ImageIcon, Layers } from "lucide-react";

interface ImageHeatmapProps {
  originalImage: string | null;
  heatmapImage: string | null;
}

export function ImageHeatmap({ originalImage, heatmapImage }: ImageHeatmapProps) {
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [opacity, setOpacity] = useState(0.6);

  if (!originalImage) return null;

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6 flex flex-col h-full">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <ImageIcon className="w-5 h-5 text-indigo-500" />
          <h3 className="text-lg font-semibold text-slate-800">Grad-CAM Heatmap</h3>
        </div>
        
        {heatmapImage && (
          <button
            onClick={() => setShowHeatmap(!showHeatmap)}
            className="flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 bg-slate-100 text-slate-700 hover:bg-slate-200 rounded-lg transition-colors"
          >
            <Layers className="w-3 h-3" />
            {showHeatmap ? "Hide Heatmap" : "Show Heatmap"}
          </button>
        )}
      </div>

      <div className="relative aspect-square w-full rounded-xl overflow-hidden bg-slate-50 border border-slate-100 flex-1">
        {/* Original Image (Always at bottom) */}
        <img 
          src={originalImage} 
          alt="Original X-Ray" 
          className="absolute inset-0 w-full h-full object-contain mix-blend-normal"
        />
        
        {/* Heatmap Overlay */}
        {heatmapImage && showHeatmap && (
          <img 
            src={heatmapImage} 
            alt="Grad-CAM Overlay" 
            className="absolute inset-0 w-full h-full object-contain transition-opacity duration-300"
            style={{ opacity: opacity }}
          />
        )}
      </div>

      {heatmapImage && showHeatmap && (
        <div className="mt-4 space-y-2">
          <div className="flex justify-between text-xs text-slate-500 font-medium">
            <span>Heatmap Opacity</span>
            <span>{Math.round(opacity * 100)}%</span>
          </div>
          <input 
            type="range" 
            min="0" 
            max="1" 
            step="0.05" 
            value={opacity}
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
            className="w-full accent-indigo-500"
          />
        </div>
      )}
    </div>
  );
}
