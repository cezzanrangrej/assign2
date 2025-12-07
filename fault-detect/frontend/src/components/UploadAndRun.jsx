/**
 * UploadAndRun.jsx
 * File upload, signal analysis, and diagnosis interface
 * Features: Drag-drop upload, preview, analysis, confidence gauge, SHAP explanation, PDF download
 */
import React, { useState, useCallback, useRef } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload,
  FileUp,
  Play,
  Download,
  Loader2,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  FileText,
  Trash2,
  Sparkles,
  Activity,
  Zap,
  TrendingUp,
  BarChart3,
  Gauge,
} from 'lucide-react';
import SignalChart from './SignalChart';
import FFTChart, { computeFFT } from './FFTChart';
import FeatureContributionCard from './FeatureContributionCard';

const API_URL = 'http://localhost:8001';

// Confidence gauge component (PowerBI style)
function ConfidenceGauge({ value, color }) {
  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  const progress = (value / 100) * circumference;

  return (
    <div className="relative w-24 h-24">
      <svg className="w-full h-full transform -rotate-90">
        <circle
          cx="48"
          cy="48"
          r={radius}
          stroke="#2a2a38"
          strokeWidth="6"
          fill="none"
        />
        <motion.circle
          cx="48"
          cy="48"
          r={radius}
          stroke={color}
          strokeWidth="6"
          fill="none"
          strokeLinecap="round"
          initial={{ strokeDasharray: circumference, strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: circumference - progress }}
          transition={{ duration: 1, ease: 'easeOut' }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-xl font-semibold text-white">{value.toFixed(0)}%</span>
        <span className="text-xs text-slate-500">Confidence</span>
      </div>
    </div>
  );
}

// Feature card component (PowerBI style)
function FeatureCard({ icon: Icon, label, value, unit, color }) {
  return (
    <div className="flex items-center gap-3 px-4 py-3 rounded bg-slate-800/50 border border-slate-700/50">
      <div className="p-2 rounded" style={{ background: `${color}20` }}>
        <Icon size={14} style={{ color }} />
      </div>
      <div>
        <p className="text-xs text-slate-500">{label}</p>
        <p className="text-sm font-semibold text-white font-mono">
          {typeof value === 'number' ? value.toFixed(4) : value}
          {unit && <span className="text-slate-500 text-xs ml-1">{unit}</span>}
        </p>
      </div>
    </div>
  );
}

export default function UploadAndRun({ onDiagnosisComplete, className = '' }) {
  const [signal, setSignal] = useState([]);
  const [fileName, setFileName] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [clientFFT, setClientFFT] = useState(null);
  
  const fileInputRef = useRef(null);

  // Parse file content
  const parseSignalFile = useCallback(async (file) => {
    try {
      const text = await file.text();
      
      // Try to parse as JSON first
      try {
        const json = JSON.parse(text);
        if (Array.isArray(json)) {
          return json.filter(x => typeof x === 'number' && !isNaN(x));
        } else if (json.signal && Array.isArray(json.signal)) {
          return json.signal.filter(x => typeof x === 'number' && !isNaN(x));
        }
      } catch {
        // Not JSON, try CSV/text format
      }
      
      // Parse as newline or comma separated values
      const nums = text
        .trim()
        .split(/[\n,\r\t]+/)
        .map(s => parseFloat(s.trim()))
        .filter(x => !isNaN(x));
      
      return nums;
    } catch (err) {
      throw new Error('Failed to parse file: ' + err.message);
    }
  }, []);

  // Handle file selection
  const handleFile = useCallback(async (file) => {
    if (!file) return;
    
    setError(null);
    setResult(null);
    setFileName(file.name);
    
    try {
      const nums = await parseSignalFile(file);
      
      if (nums.length < 16) {
        throw new Error('Signal too short (minimum 16 samples required)');
      }
      
      setSignal(nums);
      
      // Compute client-side FFT immediately
      const fft = computeFFT(nums, 2000);
      setClientFFT(fft);
      
    } catch (err) {
      setError(err.message);
      setSignal([]);
      setClientFFT(null);
    }
  }, [parseSignalFile]);

  // Handle file input change
  const handleFileChange = useCallback((e) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  }, [handleFile]);

  // Handle drag and drop
  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  // Run diagnosis
  const runDiagnosis = useCallback(async () => {
    if (signal.length === 0) return;
    
    setIsAnalyzing(true);
    setError(null);
    
    try {
      const response = await axios.post(`${API_URL}/predict`, {
        signal: signal,
        sample_rate: 2000,
      });
      
      setResult(response.data);
      onDiagnosisComplete?.(response.data, signal, fileName);
      
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Analysis failed');
    } finally {
      setIsAnalyzing(false);
    }
  }, [signal, fileName, onDiagnosisComplete]);

  // Download PDF report
  const downloadReport = useCallback(async () => {
    if (signal.length === 0) return;
    
    try {
      const response = await axios.post(
        `${API_URL}/diagnostic-report`,
        { signal: signal, sample_rate: 2000 },
        { responseType: 'blob' }
      );
      
      const url = window.URL.createObjectURL(new Blob([response.data], { type: 'application/pdf' }));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `diagnostic_report_${Date.now()}.pdf`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      
    } catch (err) {
      setError('Failed to generate report: ' + (err.message || 'Unknown error'));
    }
  }, [signal]);

  // Clear all
  const clearAll = useCallback(() => {
    setSignal([]);
    setFileName('');
    setResult(null);
    setError(null);
    setClientFFT(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, []);

  // Use captured signal from stream
  const useStreamCapture = useCallback((capturedValues, capturedTimes) => {
    setSignal(capturedValues);
    setFileName('stream_capture');
    setResult(null);
    setError(null);
    
    const fft = computeFFT(capturedValues, 2000);
    setClientFFT(fft);
  }, []);

  // Expose method for parent to inject captured signal
  React.useImperativeHandle(
    React.useRef(),
    () => ({ useStreamCapture }),
    [useStreamCapture]
  );

  // Get severity styling
  const getSeverityStyle = (severity) => {
    switch (severity) {
      case 0: return { bg: 'bg-green-500', text: 'text-green-400', ring: 'border-green-500/30' };
      case 1: return { bg: 'bg-amber-500', text: 'text-amber-400', ring: 'border-amber-500/30' };
      case 2: return { bg: 'bg-orange-500', text: 'text-orange-400', ring: 'border-orange-500/30' };
      case 3: return { bg: 'bg-red-500', text: 'text-red-400', ring: 'border-red-500/30' };
      default: return { bg: 'bg-slate-500', text: 'text-slate-400', ring: 'border-slate-500/30' };
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.3 }}
      className={`space-y-5 ${className}`}
    >
      {/* Upload Area */}
      <div className="glass-card p-5">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-sm" style={{ background: '#118DFF' }} />
          <h3 className="text-sm font-semibold text-white uppercase tracking-wide">Upload Signal Data</h3>
        </div>

        {/* Drag and Drop Zone */}
        <div
          onClick={() => fileInputRef.current?.click()}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          className={`relative border-2 border-dashed rounded p-8 text-center cursor-pointer transition-all duration-200 ${
            isDragging
              ? 'border-[#118DFF] bg-[#118DFF]/5'
              : signal.length > 0
              ? 'border-green-500/50 bg-green-500/5'
              : 'border-slate-600 hover:border-slate-500 hover:bg-slate-800/30'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,.txt,.json"
            onChange={handleFileChange}
            className="hidden"
          />

          <AnimatePresence mode="wait">
            {signal.length > 0 ? (
              <motion.div
                key="loaded"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="flex flex-col items-center"
              >
                <CheckCircle2 size={36} className="text-green-400 mb-3" />
                <p className="text-white font-medium">{fileName}</p>
                <p className="text-sm text-slate-500 mt-1">
                  {signal.length.toLocaleString()} samples loaded
                </p>
              </motion.div>
            ) : (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex flex-col items-center"
              >
                <Upload size={36} className={`mb-3 ${isDragging ? 'text-[#118DFF]' : 'text-slate-600'}`} />
                <p className={`font-medium ${isDragging ? 'text-[#118DFF]' : 'text-slate-400'}`}>
                  Drop your signal file here
                </p>
                <p className="text-sm text-slate-600 mt-1">
                  or click to browse • CSV, TXT, JSON
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Error Message */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-4 p-3 rounded bg-red-500/10 border border-red-500/20 flex items-center gap-3"
            >
              <XCircle size={18} className="text-red-400" />
              <p className="text-sm text-red-300">{error}</p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Action Buttons */}
        {signal.length > 0 && (
          <div className="flex gap-3 mt-4">
            <button
              onClick={runDiagnosis}
              disabled={isAnalyzing}
              className="flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded text-white font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed btn-primary"
            >
              {isAnalyzing ? (
                <>
                  <Loader2 size={18} className="animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Sparkles size={18} />
                  Run Diagnosis
                </>
              )}
            </button>
            <button
              onClick={downloadReport}
              disabled={signal.length === 0}
              className="px-4 py-3 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white font-medium transition-colors flex items-center gap-2"
            >
              <Download size={18} />
              PDF Report
            </button>
            <button
              onClick={clearAll}
              className="px-4 py-3 rounded bg-slate-800 hover:bg-red-500/20 text-slate-400 hover:text-red-400 transition-colors"
            >
              <Trash2 size={18} />
            </button>
          </div>
        )}
      </div>

      {/* Signal Preview */}
      {signal.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          <SignalChart
            data={signal.slice(0, 500)}
            title="Signal Preview"
            height={220}
          />
          <FFTChart
            signal={signal}
            title="Frequency Spectrum (Client FFT)"
            height={220}
            maxFrequency={500}
          />
        </div>
      )}

      {/* Diagnosis Result */}
      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="glass-card p-6"
          >
            <div className="flex items-center gap-2 mb-5">
              <div className="w-1 h-5 rounded-sm bg-green-500" />
              <h3 className="text-sm font-semibold text-white uppercase tracking-wide">Diagnosis Result</h3>
            </div>

            <div className="flex flex-col lg:flex-row gap-6">
              {/* Main Result */}
              <div className="flex items-center gap-6">
                <ConfidenceGauge value={result.confidence * 100} color={result.color} />
                
                <div>
                  <div className={`inline-flex items-center gap-2 px-4 py-2 rounded border ${getSeverityStyle(result.severity).ring}`}>
                    <div className={`w-2.5 h-2.5 rounded-full ${getSeverityStyle(result.severity).bg}`} />
                    <span className="text-lg font-semibold text-white">{result.prediction}</span>
                  </div>
                  <p className="mt-2 text-sm text-slate-500 max-w-xs">
                    {result.description}
                  </p>
                  <div className="flex items-center gap-2 mt-3">
                    {[...Array(3)].map((_, i) => (
                      <div
                        key={i}
                        className={`w-8 h-1 rounded ${
                          i < result.severity ? getSeverityStyle(result.severity).bg : 'bg-slate-700'
                        }`}
                      />
                    ))}
                    <span className="text-xs text-slate-500 ml-2">Severity {result.severity}/3</span>
                  </div>
                </div>
              </div>

              {/* Features Grid */}
              <div className="flex-1 grid grid-cols-2 md:grid-cols-3 gap-3">
                <FeatureCard icon={Activity} label="RMS" value={result.features.rms} color="#118DFF" />
                <FeatureCard icon={Zap} label="Kurtosis" value={result.features.kurtosis} color="#E66C37" />
                <FeatureCard icon={TrendingUp} label="Crest Factor" value={result.features.crest_factor} color="#22c55e" />
                <FeatureCard icon={BarChart3} label="Dominant Freq" value={result.features.dominant_freq} unit="Hz" color="#6B007B" />
                <FeatureCard icon={Activity} label="Spectral Entropy" value={result.features.spectral_entropy} color="#E044A7" />
                <FeatureCard icon={Gauge} label="Shape Factor" value={result.features.shape_factor} color="#12239E" />
              </div>
            </div>

            {/* Timestamp */}
            <div className="mt-4 pt-4 border-t border-slate-700/50 flex items-center justify-between">
              <span className="text-xs text-slate-500">
                Analysis completed at {new Date(result.timestamp).toLocaleString()}
              </span>
              <button
                onClick={downloadReport}
                className="text-xs flex items-center gap-1 transition-colors"
                style={{ color: '#118DFF' }}
              >
                <FileText size={14} />
                Download Full Report
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* SHAP Explanation Card - Shown after diagnosis */}
      <AnimatePresence>
        {result && result.explanation && (
          <FeatureContributionCard
            explanation={result.explanation}
            predictedClass={result.prediction}
          />
        )}
      </AnimatePresence>
    </motion.div>
  );
}
