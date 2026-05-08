import { Activity } from "lucide-react";
import { cn } from "@/lib/utils";

interface Prediction {
  label: string;
  confidence: number;
}

interface PredictionCardProps {
  predictions: Prediction[];
}

export function PredictionCard({ predictions }: PredictionCardProps) {
  // Sort by confidence
  const sortedPreds = [...predictions].sort((a, b) => b.confidence - a.confidence);
  
  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6 flex flex-col h-full">
      <div className="flex items-center gap-2 mb-6">
        <Activity className="w-5 h-5 text-blue-500" />
        <h3 className="text-lg font-semibold text-slate-800">Diagnostic Predictions</h3>
      </div>
      
      <div className="space-y-5 flex-1">
        {sortedPreds.length > 0 ? (
          sortedPreds.map((pred, i) => (
            <div key={i} className="space-y-2">
              <div className="flex justify-between items-center text-sm">
                <span className={cn("font-medium", i === 0 ? "text-slate-900" : "text-slate-600")}>
                  {pred.label.charAt(0).toUpperCase() + pred.label.slice(1)}
                </span>
                <span className={cn("font-semibold", i === 0 ? "text-blue-600" : "text-slate-500")}>
                  {(pred.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                <div 
                  className={cn("h-full rounded-full transition-all duration-1000", 
                    i === 0 ? "bg-blue-500" : "bg-slate-300"
                  )}
                  style={{ width: `${pred.confidence * 100}%` }}
                />
              </div>
            </div>
          ))
        ) : (
          <div className="flex items-center justify-center h-full text-slate-400 text-sm">
            No predictions generated yet.
          </div>
        )}
      </div>
    </div>
  );
}
