"use client";

import { useState, useRef } from "react";
import { UploadCloud, FileImage, X } from "lucide-react";
import { cn } from "@/lib/utils";

interface UploadCardProps {
  onSubmit: (file: File, text: string) => void;
  isLoading: boolean;
}

export function UploadCard({ onSubmit, isLoading }: UploadCardProps) {
  const [file, setFile] = useState<File | null>(null);
  const [text, setText] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (file && text) {
      onSubmit(file, text);
    }
  };

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6 transition-all">
      <h2 className="text-xl font-semibold text-slate-800 mb-6 flex items-center gap-2">
        Patient Data Input
      </h2>
      
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Image Upload Area */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-slate-700">Chest X-Ray Image</label>
          <div
            className={cn(
              "relative border-2 border-dashed rounded-xl p-8 text-center transition-colors cursor-pointer",
              dragActive ? "border-blue-500 bg-blue-50" : "border-slate-200 hover:bg-slate-50",
              file ? "border-green-500 bg-green-50" : ""
            )}
            onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
            onDragLeave={() => setDragActive(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              type="file"
              accept="image/*"
              className="hidden"
              ref={fileInputRef}
              onChange={handleFileChange}
            />
            {file ? (
              <div className="flex flex-col items-center gap-2 relative z-10">
                <FileImage className="w-10 h-10 text-green-500" />
                <span className="text-sm font-medium text-slate-700">{file.name}</span>
                <button
                  type="button"
                  onClick={(e) => { e.stopPropagation(); setFile(null); }}
                  className="mt-2 text-xs text-red-500 hover:text-red-600 flex items-center"
                >
                  <X className="w-3 h-3 mr-1" /> Remove
                </button>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-3 relative z-10">
                <div className="bg-blue-100 p-3 rounded-full">
                  <UploadCloud className="w-6 h-6 text-blue-600" />
                </div>
                <div>
                  <p className="text-sm font-medium text-slate-700">Click to upload or drag & drop</p>
                  <p className="text-xs text-slate-500 mt-1">PNG, JPG up to 10MB</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Text Area */}
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <label className="text-sm font-medium text-slate-700">Clinical Report</label>
            <div className="relative">
              <input
                type="file"
                accept=".txt,.pdf,.docx"
                className="hidden"
                id="doc-upload"
                onChange={async (e) => {
                  const file = e.target.files?.[0];
                  if (!file) return;
                  
                  const formData = new FormData();
                  formData.append('file', file);
                  
                  try {
                    const res = await fetch("http://127.0.0.1:5000/extract_text", {
                      method: "POST",
                      body: formData,
                    });
                    const data = await res.json();
                    if (data.text) {
                      setText(data.text);
                    } else if (data.error) {
                      alert("Error extracting text: " + data.error);
                    }
                  } catch (err) {
                    alert("Failed to extract text from document.");
                  }
                  e.target.value = '';
                }}
              />
              <label htmlFor="doc-upload" className="cursor-pointer flex items-center gap-1 text-xs font-medium text-blue-600 hover:text-blue-700 bg-blue-50 px-3 py-1.5 rounded-full transition-colors border border-blue-100">
                <UploadCloud className="w-3.5 h-3.5" />
                Upload PDF/TXT/DOCX
              </label>
            </div>
          </div>
          <textarea
            className="w-full min-h-[120px] p-4 rounded-xl border border-slate-200 bg-slate-50 focus:bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all resize-none text-sm text-slate-700 placeholder:text-slate-400"
            placeholder="E.g., Heart size is normal. Lungs are clear. No focal consolidation..."
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
        </div>

        <button
          type="submit"
          disabled={!file || isLoading}
          className="w-full bg-slate-900 text-white rounded-xl py-3 px-4 font-medium hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-slate-900 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
        >
          {isLoading ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Analyzing Multimodal Data...
            </span>
          ) : (
            "Generate Diagnosis & XAI Explanation"
          )}
        </button>
      </form>
    </div>
  );
}
