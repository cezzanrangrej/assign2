/**
 * FFTChart.jsx
 * Interactive frequency spectrum visualization with client-side FFT computation
 * Features: Real-time FFT, log scale, peak detection, frequency annotations
 */
import React, { useRef, useMemo, useCallback, useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  LogarithmicScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import zoomPlugin from 'chartjs-plugin-zoom';
import annotationPlugin from 'chartjs-plugin-annotation';
import { Activity, Zap, TrendingUp, Info, Maximize2, ZoomIn, ZoomOut, RotateCcw, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import FFT from 'fft.js';

// Register Chart.js plugins
ChartJS.register(
  CategoryScale,
  LinearScale,
  LogarithmicScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  zoomPlugin,
  annotationPlugin
);

/**
 * Compute FFT using fft.js library
 * @param {number[]} signal - Input signal array
 * @param {number} sampleRate - Sample rate in Hz
 * @returns {Object} - { frequencies: number[], magnitudes: number[], peaks: Object[] }
 */
export function computeFFT(signal, sampleRate = 2000) {
  if (!signal || signal.length < 4) {
    return { frequencies: [], magnitudes: [], peaks: [] };
  }

  // Pad to nearest power of 2 for FFT
  const n = Math.pow(2, Math.ceil(Math.log2(signal.length)));
  const paddedSignal = new Array(n).fill(0);
  
  // Apply Hanning window and copy signal
  for (let i = 0; i < signal.length; i++) {
    const window = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (signal.length - 1)));
    paddedSignal[i] = signal[i] * window;
  }

  // Create FFT instance
  const fft = new FFT(n);
  const out = fft.createComplexArray();
  const data = fft.toComplexArray(paddedSignal, null);
  
  // Perform FFT
  fft.transform(out, data);

  // Compute magnitudes (only positive frequencies)
  const numBins = n / 2;
  const frequencies = [];
  const magnitudes = [];
  const freqResolution = sampleRate / n;

  for (let i = 0; i < numBins; i++) {
    const re = out[2 * i];
    const im = out[2 * i + 1];
    const mag = Math.sqrt(re * re + im * im) * (2 / n); // Normalize
    frequencies.push(i * freqResolution);
    magnitudes.push(mag);
  }

  // Find peaks (local maxima above threshold)
  const peaks = findPeaks(frequencies, magnitudes, 5);

  return { frequencies, magnitudes, peaks };
}

/**
 * Find spectral peaks
 */
function findPeaks(freqs, mags, numPeaks = 5) {
  const peaks = [];
  const threshold = Math.max(...mags) * 0.1; // 10% of max

  for (let i = 1; i < mags.length - 1; i++) {
    if (mags[i] > mags[i - 1] && mags[i] > mags[i + 1] && mags[i] > threshold) {
      peaks.push({
        frequency: freqs[i],
        magnitude: mags[i],
        index: i,
      });
    }
  }

  // Sort by magnitude and return top N
  return peaks.sort((a, b) => b.magnitude - a.magnitude).slice(0, numPeaks);
}

export default function FFTChart({
  signal = [],
  sampleRate = 2000,
  title = 'Frequency Spectrum (FFT)',
  precomputedFFT = null, // Can pass pre-computed FFT from backend
  showPeaks = true,
  logScale = true,
  maxFrequency = null,
  height = 280,
  className = '',
}) {
  const chartRef = useRef(null);
  const fullscreenChartRef = useRef(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Handle ESC key to close fullscreen
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && isFullscreen) {
        setIsFullscreen(false);
      }
    };
    if (isFullscreen) {
      document.addEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'hidden';
    }
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = '';
    };
  }, [isFullscreen]);

  const toggleFullscreen = useCallback(() => {
    setIsFullscreen(prev => !prev);
  }, []);

  // Compute FFT (either use precomputed or calculate client-side)
  const fftData = useMemo(() => {
    if (precomputedFFT) {
      return {
        frequencies: precomputedFFT.frequencies,
        magnitudes: precomputedFFT.magnitudes,
        peaks: findPeaks(precomputedFFT.frequencies, precomputedFFT.magnitudes, 5),
      };
    }
    return computeFFT(signal, sampleRate);
  }, [signal, sampleRate, precomputedFFT]);

  // Filter to max frequency if specified
  const displayData = useMemo(() => {
    if (!maxFrequency) return fftData;
    
    const maxIdx = fftData.frequencies.findIndex(f => f > maxFrequency);
    if (maxIdx === -1) return fftData;
    
    return {
      frequencies: fftData.frequencies.slice(0, maxIdx),
      magnitudes: fftData.magnitudes.slice(0, maxIdx),
      peaks: fftData.peaks.filter(p => p.frequency <= maxFrequency),
    };
  }, [fftData, maxFrequency]);

  // Generate gradient
  const getGradient = (ctx, chartArea) => {
    if (!chartArea) return '#12239E';
    const gradient = ctx.createLinearGradient(0, chartArea.bottom, 0, chartArea.top);
    gradient.addColorStop(0, 'rgba(18, 35, 158, 0)');
    gradient.addColorStop(0.5, 'rgba(18, 35, 158, 0.12)');
    gradient.addColorStop(1, 'rgba(18, 35, 158, 0.3)');
    return gradient;
  };

  // Chart data
  const chartData = useMemo(() => ({
    labels: displayData.frequencies.map(f => f.toFixed(1)),
    datasets: [
      {
        label: 'Magnitude',
        data: displayData.magnitudes.map(m => Math.max(m, 1e-10)), // Avoid log(0)
        borderColor: '#12239E',
        backgroundColor: (context) => {
          const chart = context.chart;
          const { ctx, chartArea } = chart;
          return getGradient(ctx, chartArea);
        },
        fill: true,
        tension: 0.1,
        pointRadius: 0,
        pointHoverRadius: 5,
        pointHoverBackgroundColor: '#12239E',
        pointHoverBorderColor: '#fff',
        pointHoverBorderWidth: 2,
        borderWidth: 1.5,
      },
    ],
  }), [displayData]);

  // Create peak annotations
  const peakAnnotations = useMemo(() => {
    if (!showPeaks) return {};
    
    return displayData.peaks.reduce((acc, peak, idx) => {
      acc[`peak${idx}`] = {
        type: 'point',
        xValue: peak.frequency.toFixed(1),
        yValue: peak.magnitude,
        backgroundColor: idx === 0 ? '#E66C37' : '#6B007B',
        borderColor: '#fff',
        borderWidth: 2,
        radius: idx === 0 ? 8 : 6,
      };
      acc[`peakLabel${idx}`] = {
        type: 'label',
        xValue: peak.frequency.toFixed(1),
        yValue: peak.magnitude,
        yAdjust: -20,
        content: `${peak.frequency.toFixed(0)} Hz`,
        font: { size: 10, weight: 'bold' },
        color: idx === 0 ? '#E66C37' : '#6B007B',
        backgroundColor: '#1a1a24',
        padding: 4,
      };
      return acc;
    }, {});
  }, [displayData.peaks, showPeaks]);

  // Chart options
  const options = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: { display: false },
      tooltip: {
        enabled: true,
        backgroundColor: '#1a1a24',
        titleColor: '#e5e7eb',
        bodyColor: '#9ca3af',
        borderColor: 'rgba(18, 35, 158, 0.3)',
        borderWidth: 1,
        padding: 12,
        cornerRadius: 4,
        displayColors: false,
        callbacks: {
          title: (items) => `Frequency: ${items[0]?.label ?? ''} Hz`,
          label: (item) => `Magnitude: ${Number(item.raw).toExponential(2)}`,
        },
      },
      zoom: {
        pan: { enabled: true, mode: 'x', modifierKey: 'ctrl' },
        zoom: {
          wheel: { enabled: true, speed: 0.1 },
          pinch: { enabled: true },
          drag: {
            enabled: true,
            backgroundColor: 'rgba(18, 35, 158, 0.1)',
            borderColor: 'rgba(18, 35, 158, 0.5)',
            borderWidth: 1,
          },
          mode: 'x',
        },
      },
      annotation: { annotations: peakAnnotations },
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Frequency (Hz)',
          color: '#6b7280',
          font: { size: 11, weight: '500' },
        },
        grid: { color: 'rgba(55, 65, 81, 0.3)', drawBorder: false },
        ticks: {
          color: '#6b7280',
          font: { size: 10 },
          maxTicksLimit: 12,
          callback: function(value, index) {
            const freq = parseFloat(this.getLabelForValue(value));
            return freq % 100 === 0 ? freq : '';
          },
        },
      },
      y: {
        type: logScale ? 'logarithmic' : 'linear',
        display: true,
        title: {
          display: true,
          text: 'Magnitude',
          color: '#6b7280',
          font: { size: 11, weight: '500' },
        },
        grid: { color: 'rgba(55, 65, 81, 0.3)', drawBorder: false },
        ticks: {
          color: '#6b7280',
          font: { size: 10 },
          callback: (value) => value.toExponential(0),
        },
        min: logScale ? 1e-6 : 0,
      },
    },
    animation: { duration: 400 },
  }), [logScale, peakAnnotations]);

  // Stats
  const stats = useMemo(() => {
    if (displayData.peaks.length === 0) return null;
    const dominant = displayData.peaks[0];
    const totalEnergy = displayData.magnitudes.reduce((sum, m) => sum + m * m, 0);
    return {
      dominantFreq: dominant?.frequency.toFixed(1) ?? 'N/A',
      dominantMag: dominant?.magnitude.toExponential(2) ?? 'N/A',
      totalEnergy: totalEnergy.toExponential(2),
      numPeaks: displayData.peaks.length,
    };
  }, [displayData]);

  const handleResetZoom = () => {
    chartRef.current?.resetZoom();
  };

  const handleFullscreenZoomIn = () => {
    fullscreenChartRef.current?.zoom(1.2);
  };

  const handleFullscreenZoomOut = () => {
    fullscreenChartRef.current?.zoom(0.8);
  };

  const handleFullscreenResetZoom = () => {
    fullscreenChartRef.current?.resetZoom();
  };

  // Fullscreen modal content
  const renderFullscreenModal = () => (
    <AnimatePresence>
      {isFullscreen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="fixed inset-0 z-50 flex items-center justify-center"
          style={{ backgroundColor: 'rgba(0, 0, 0, 0.9)' }}
          onClick={(e) => e.target === e.currentTarget && setIsFullscreen(false)}
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="w-[95vw] h-[90vh] glass-card p-6 relative"
          >
            {/* Fullscreen Header */}
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="w-1 h-6 rounded-sm" style={{ background: '#12239E' }} />
                <h3 className="text-lg font-semibold text-white uppercase tracking-wide">{title}</h3>
                <span className="px-3 py-1 text-sm font-medium rounded" style={{ background: 'rgba(18, 35, 158, 0.15)', color: '#12239E' }}>
                  {logScale ? 'Log Scale' : 'Linear'}
                </span>
              </div>

              {/* Fullscreen Controls */}
              <div className="flex items-center gap-2">
                <button
                  onClick={handleFullscreenZoomIn}
                  className="p-2 rounded-lg hover:bg-white/10 text-slate-400 hover:text-white transition-colors"
                  title="Zoom In"
                >
                  <ZoomIn size={20} />
                </button>
                <button
                  onClick={handleFullscreenZoomOut}
                  className="p-2 rounded-lg hover:bg-white/10 text-slate-400 hover:text-white transition-colors"
                  title="Zoom Out"
                >
                  <ZoomOut size={20} />
                </button>
                <button
                  onClick={handleFullscreenResetZoom}
                  className="p-2 rounded-lg hover:bg-white/10 text-slate-400 hover:text-white transition-colors"
                  title="Reset Zoom"
                >
                  <RotateCcw size={20} />
                </button>
                <div className="w-px h-6 bg-slate-700 mx-2" />
                <button
                  onClick={() => setIsFullscreen(false)}
                  className="p-2 rounded-lg bg-red-500/20 hover:bg-red-500/30 text-red-400 hover:text-red-300 transition-colors"
                  title="Close Fullscreen (ESC)"
                >
                  <X size={20} />
                </button>
              </div>
            </div>

            {/* Fullscreen Stats */}
            {stats && (
              <div className="grid grid-cols-4 gap-4 mb-4">
                <div className="flex items-center gap-3 px-4 py-3 rounded-lg bg-slate-800/50">
                  <Zap size={18} className="text-[#E66C37]" />
                  <div>
                    <p className="text-xs text-slate-500">Dominant Frequency</p>
                    <p className="text-lg font-semibold text-white">{stats.dominantFreq} Hz</p>
                  </div>
                </div>
                <div className="flex items-center gap-3 px-4 py-3 rounded-lg bg-slate-800/50">
                  <Activity size={18} className="text-[#12239E]" />
                  <div>
                    <p className="text-xs text-slate-500">Peak Magnitude</p>
                    <p className="text-lg font-semibold text-white">{stats.dominantMag}</p>
                  </div>
                </div>
                <div className="flex items-center gap-3 px-4 py-3 rounded-lg bg-slate-800/50">
                  <TrendingUp size={18} className="text-green-500" />
                  <div>
                    <p className="text-xs text-slate-500">Total Energy</p>
                    <p className="text-lg font-semibold text-white">{stats.totalEnergy}</p>
                  </div>
                </div>
                <div className="flex items-center gap-3 px-4 py-3 rounded-lg bg-slate-800/50">
                  <Info size={18} className="text-[#6B007B]" />
                  <div>
                    <p className="text-xs text-slate-500">Detected Peaks</p>
                    <p className="text-lg font-semibold text-white">{stats.numPeaks}</p>
                  </div>
                </div>
              </div>
            )}

            {/* Fullscreen Chart */}
            <div className="h-[calc(100%-160px)]">
              <Line ref={fullscreenChartRef} data={chartData} options={options} />
            </div>

            {/* Peak Legend in Fullscreen */}
            {showPeaks && displayData.peaks.length > 0 && (
              <div className="absolute bottom-16 left-6 flex flex-wrap gap-2">
                {displayData.peaks.map((peak, idx) => (
                  <span
                    key={idx}
                    className="px-3 py-1.5 text-sm rounded"
                    style={{
                      background: idx === 0 ? 'rgba(230, 108, 55, 0.15)' : 'rgba(107, 0, 123, 0.15)',
                      color: idx === 0 ? '#E66C37' : '#6B007B'
                    }}
                  >
                    {peak.frequency.toFixed(0)} Hz ({peak.magnitude.toExponential(1)})
                  </span>
                ))}
              </div>
            )}

            {/* Fullscreen Help text */}
            <p className="absolute bottom-4 left-1/2 -translate-x-1/2 text-sm text-slate-500">
              Drag to select region • Scroll to zoom • Ctrl+drag to pan • Press ESC to close
            </p>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );

  return (
    <>
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: 0.1 }}
      className={`glass-card p-5 ${className}`}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="w-1 h-5 rounded-sm" style={{ background: '#12239E' }} />
          <h3 className="text-sm font-semibold text-white uppercase tracking-wide">{title}</h3>
          <span className="px-2 py-0.5 text-xs font-medium rounded" style={{ background: 'rgba(18, 35, 158, 0.15)', color: '#12239E' }}>
            {logScale ? 'Log Scale' : 'Linear'}
          </span>
        </div>
        <button
          onClick={handleResetZoom}
          className="px-3 py-1.5 text-xs font-medium rounded bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white transition-colors"
        >
          Reset Zoom
        </button>
        <button
          onClick={toggleFullscreen}
          className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white transition-colors"
          title="Fullscreen"
        >
          <Maximize2 size={16} />
        </button>
      </div>

      {/* Quick Stats */}
      {stats && (
        <div className="grid grid-cols-4 gap-3 mb-4">
          <div className="flex items-center gap-2 px-3 py-2 rounded bg-slate-800/50">
            <Zap size={14} className="text-[#E66C37]" />
            <div>
              <p className="text-xs text-slate-500">Dominant</p>
              <p className="text-sm font-semibold text-white">{stats.dominantFreq} Hz</p>
            </div>
          </div>
          <div className="flex items-center gap-2 px-3 py-2 rounded bg-slate-800/50">
            <Activity size={14} className="text-[#12239E]" />
            <div>
              <p className="text-xs text-slate-500">Magnitude</p>
              <p className="text-sm font-semibold text-white">{stats.dominantMag}</p>
            </div>
          </div>
          <div className="flex items-center gap-2 px-3 py-2 rounded bg-slate-800/50">
            <TrendingUp size={14} className="text-green-500" />
            <div>
              <p className="text-xs text-slate-500">Energy</p>
              <p className="text-sm font-semibold text-white">{stats.totalEnergy}</p>
            </div>
          </div>
          <div className="flex items-center gap-2 px-3 py-2 rounded bg-slate-800/50">
            <Info size={14} className="text-[#6B007B]" />
            <div>
              <p className="text-xs text-slate-500">Peaks</p>
              <p className="text-sm font-semibold text-white">{stats.numPeaks}</p>
            </div>
          </div>
        </div>
      )}

      {/* Chart */}
      <div className="chart-container" style={{ height }}>
        {displayData.magnitudes.length > 0 ? (
          <Line ref={chartRef} data={chartData} options={options} />
        ) : (
          <div className="flex items-center justify-center h-full">
            <p className="text-slate-500">Upload a signal to view frequency spectrum</p>
          </div>
        )}
      </div>

      {/* Peak Legend */}
      {showPeaks && displayData.peaks.length > 0 && (
        <div className="mt-4 flex flex-wrap gap-2">
          {displayData.peaks.slice(0, 3).map((peak, idx) => (
            <span
              key={idx}
              className="px-2 py-1 text-xs rounded"
              style={{
                background: idx === 0 ? 'rgba(230, 108, 55, 0.15)' : 'rgba(107, 0, 123, 0.15)',
                color: idx === 0 ? '#E66C37' : '#6B007B'
              }}
            >
              {peak.frequency.toFixed(0)} Hz ({peak.magnitude.toExponential(1)})
            </span>
          ))}
        </div>
      )}
    </motion.div>

    {/* Fullscreen Modal */}
    {renderFullscreenModal()}
    </>
  );
}
