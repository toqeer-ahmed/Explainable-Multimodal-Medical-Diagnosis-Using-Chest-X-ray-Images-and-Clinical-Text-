import { BookOpen } from "lucide-react";

interface HighlightToken {
  word: string;
  score: number; // 0 to 1
}

interface TextExplanationProps {
  tokens: HighlightToken[];
}

export function TextExplanation({ tokens }: TextExplanationProps) {
  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6 flex flex-col h-full">
      <div className="flex items-center gap-2 mb-6">
        <BookOpen className="w-5 h-5 text-emerald-500" />
        <h3 className="text-lg font-semibold text-slate-800">BERT Self-Attention</h3>
      </div>
      
      <div className="bg-slate-50 p-5 rounded-xl border border-slate-100 leading-relaxed text-slate-700 text-sm flex-1 overflow-y-auto">
        {tokens.length > 0 ? (
          tokens.map((token, idx) => {
            // Map score [0, 1] to a background color intensity.
            // Using emerald/green to indicate importance.
            const alpha = Math.min(0.8, token.score * 1.5); // scale up slightly for visibility
            const isImportant = token.score > 0.3;
            
            return (
              <span 
                key={idx} 
                className="inline-block transition-colors rounded-sm px-[2px] mx-[1px]"
                style={{ 
                  backgroundColor: `rgba(16, 185, 129, ${alpha})`,
                  fontWeight: isImportant ? 600 : 400,
                  color: isImportant ? '#064e3b' : 'inherit'
                }}
                title={`Attention score: ${(token.score * 100).toFixed(1)}%`}
              >
                {token.word}
              </span>
            );
          })
        ) : (
          <span className="text-slate-400">Analysis results will appear here...</span>
        )}
      </div>
      
      {tokens.length > 0 && (
        <div className="mt-4 flex items-center gap-2 text-xs text-slate-500">
          <span className="font-medium">Legend:</span>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-emerald-100 rounded-sm"></div>
            <span>Low</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-emerald-300 rounded-sm"></div>
            <span>Medium</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-emerald-500 rounded-sm"></div>
            <span className="text-emerald-900 font-semibold">High</span>
          </div>
        </div>
      )}
    </div>
  );
}
