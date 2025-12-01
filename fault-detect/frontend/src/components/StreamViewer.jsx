/**
 * StreamViewer.jsx
 * Real-time SSE streaming signal visualization with status indicators
 * Features: Live data, rolling buffer, connection status, pause/resume, statistics
 */
import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Filler,
} from 'chart.js';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Radio,
  Pause,
  Play,
  Trash2,
  Download,
  Save,
  Wifi,
  WifiOff,
  Activity,
  TrendingUp,
  AlertCircle,
} from 'lucide-react';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Filler);

const API_URL = 'http://localhost:8000';
const BUFFER_SIZE = 300; // Show last 300 samples (~15 seconds at 20Hz)

export default function StreamViewer({ onCapture, onDownload, className = '' }) {
  const [values, setValues] = useState([]);
  const [times, setTimes] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [status, setStatus] = useState('normal');
  const [stats, setStats] = useState({ min: 0, max: 0, mean: 0, rms: 0 });
  const [sampleCount, setSampleCount] = useState(0);
  
  const eventSourceRef = useRef(null);
  const valuesBufferRef = useRef([]);
  const timesBufferRef = useRef([]);

  // Calculate real-time statistics
  const calculateStats = useCallback((data) => {
    if (data.length === 0) return { min: 0, max: 0, mean: 0, rms: 0 };
    const min = Math.min(...data);
    const max = Math.max(...data);
    const mean = data.reduce((a, b) => a + b, 0) / data.length;
    const rms = Math.sqrt(data.reduce((a, b) => a + b * b, 0) / data.length);
    return { min, max, mean, rms };
  }, []);

  // Connect to SSE stream
  const connect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    try {
      const es = new EventSource(`${API_URL}/stream-signal`);
      
      es.addEventListener('signal', (e) => {
        if (isPaused) return;
        
        try {
          const data = JSON.parse(e.data);
          
          valuesBufferRef.current = [...valuesBufferRef.current, data.value].slice(-BUFFER_SIZE);
          timesBufferRef.current = [...timesBufferRef.current, data.t].slice(-BUFFER_SIZE);
          
          setValues([...valuesBufferRef.current]);
          setTimes([...timesBufferRef.current]);
          setStatus(data.status);
          setSampleCount(prev => prev + 1);
          
          // Update stats every 10 samples
          if (valuesBufferRef.current.length % 10 === 0) {
            setStats(calculateStats(valuesBufferRef.current));
          }
        } catch (err) {
          console.error('Parse error:', err);
        }
      });

      es.onopen = () => {
        setIsConnected(true);
        console.log('SSE Connected');
      };

      es.onerror = (err) => {
        console.error('SSE Error:', err);
        setIsConnected(false);
        es.close();
        // Attempt reconnection after 3 seconds
        setTimeout(() => {
          if (!isPaused) connect();
        }, 3000);
      };

      eventSourceRef.current = es;
    } catch (err) {
      console.error('Connection error:', err);
      setIsConnected(false);
    }
  }, [isPaused, calculateStats]);

  // Initial connection
  useEffect(() => {
    connect();
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  // Handle pause/resume
  const togglePause = useCallback(() => {
    setIsPaused(prev => {
      if (prev) {
        // Resuming
        connect();
      } else {
        // Pausing
        if (eventSourceRef.current) {
          eventSourceRef.current.close();
        }
      }
      return !prev;
    });
  }, [connect]);

  // Clear buffer
  const clearBuffer = useCallback(() => {
    valuesBufferRef.current = [];
    timesBufferRef.current = [];
    setValues([]);
    setTimes([]);
    setSampleCount(0);
    setStats({ min: 0, max: 0, mean: 0, rms: 0 });
  }, []);

  // Capture current buffer
  const captureSignal = useCallback(() => {
    if (values.length > 0 && onCapture) {
      onCapture(values, times);
    }
  }, [values, times, onCapture]);

  // Download as CSV and update parent stats
  const downloadCSV = useCallback(() => {
    if (values.length === 0) return;

    // Create CSV content
    const csvContent = values.map((v, i) => `${times[i].toFixed(6)},${v.toFixed(6)}`).join('\n');
    const header = 'time,amplitude\n';
    const blob = new Blob([header + csvContent], { type: 'text/csv;charset=utf-8;' });
    
    // Create filename with timestamp
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const filename = `live_stream_${timestamp}.csv`;
    
    // Download file
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    // Notify parent to update stats with this data
    if (onDownload) {
      onDownload(values, times, filename, stats);
    }
  }, [values, times, stats, onDownload]);

  // Status colors and labels
  const statusConfig = useMemo(() => ({
    normal: { color: '#10B981', label: 'Normal', bgColor: 'bg-green-500/20', textColor: 'text-green-400' },
    unbalance: { color: '#F59E0B', label: 'Unbalance', bgColor: 'bg-amber-500/20', textColor: 'text-amber-400' },
    bearing: { color: '#EF4444', label: 'Bearing', bgColor: 'bg-red-500/20', textColor: 'text-red-400' },
  }), []);

  const currentStatus = statusConfig[status] || statusConfig.normal;

  // Chart data
  const chartData = useMemo(() => ({
    labels: times.map(t => t.toFixed(2)),
    datasets: [{
      label: 'Signal',
      data: values,
      borderColor: currentStatus.color,
      backgroundColor: `${currentStatus.color}20`,
      fill: true,
      tension: 0.3,
      pointRadius: 0,
      borderWidth: 1.5,
    }],
  }), [values, times, currentStatus.color]);

  // Chart options
  const options = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 0 },
    plugins: {
      legend: { display: false },
      tooltip: { enabled: false },
    },
    scales: {
      x: {
        display: true,
        grid: { color: 'rgba(55, 65, 81, 0.3)', drawBorder: false },
        ticks: { color: '#6b7280', font: { size: 9 }, maxTicksLimit: 8 },
        title: { display: true, text: 'Time (s)', color: '#6b7280', font: { size: 10 } },
      },
      y: {
        display: true,
        grid: { color: 'rgba(55, 65, 81, 0.3)', drawBorder: false },
        ticks: { color: '#6b7280', font: { size: 9 } },
        title: { display: true, text: 'Amplitude', color: '#6b7280', font: { size: 10 } },
        suggestedMin: -3,
        suggestedMax: 3,
      },
    },
  }), []);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: 0.2 }}
      className={`glass-card p-5 ${className}`}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className={`w-2.5 h-2.5 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}>
              {isConnected && !isPaused && (
                <span className="absolute inset-0 rounded-full bg-green-500 animate-ping opacity-75" />
              )}
            </div>
          </div>
          <div className="w-1 h-5 rounded-sm" style={{ background: '#118DFF' }} />
          <h3 className="text-sm font-semibold text-white uppercase tracking-wide">Live Signal Stream</h3>
          
          {/* Connection Status */}
          <span className={`flex items-center gap-1.5 px-2 py-0.5 text-xs font-medium rounded ${
            isConnected ? 'bg-green-500/10 text-green-400' : 'bg-red-500/10 text-red-400'
          }`}>
            {isConnected ? <Wifi size={12} /> : <WifiOff size={12} />}
            {isConnected ? (isPaused ? 'Paused' : 'Live') : 'Disconnected'}
          </span>
          
          {/* Fault Status */}
          <AnimatePresence mode="wait">
            <motion.span
              key={status}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className={`flex items-center gap-1.5 px-2 py-0.5 text-xs font-medium rounded ${currentStatus.bgColor} ${currentStatus.textColor}`}
            >
              <AlertCircle size={12} />
              {currentStatus.label}
            </motion.span>
          </AnimatePresence>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={togglePause}
            className={`p-2 rounded transition-colors ${
              isPaused
                ? 'bg-green-500/10 text-green-400 hover:bg-green-500/20'
                : 'bg-amber-500/10 text-amber-400 hover:bg-amber-500/20'
            }`}
            title={isPaused ? 'Resume' : 'Pause'}
          >
            {isPaused ? <Play size={16} /> : <Pause size={16} />}
          </button>
          <button
            onClick={clearBuffer}
            className="p-2 rounded bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-white transition-colors"
            title="Clear Buffer"
          >
            <Trash2 size={16} />
          </button>
          <button
            onClick={captureSignal}
            disabled={values.length === 0}
            className="p-2 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            style={{ background: 'rgba(17, 141, 255, 0.15)', color: '#118DFF' }}
            title="Capture for Analysis"
          >
            <Download size={16} />
          </button>
          <button
            onClick={downloadCSV}
            disabled={values.length === 0}
            className="flex items-center gap-1.5 px-3 py-2 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            style={{ background: 'rgba(16, 185, 129, 0.15)', color: '#10B981' }}
            title="Download as CSV & Update Stats"
          >
            <Save size={16} />
            <span className="text-xs font-medium">Save CSV</span>
          </button>
        </div>
      </div>

      {/* Stats Bar */}
      <div className="grid grid-cols-5 gap-2 mb-4">
        <div className="px-3 py-2 rounded bg-slate-800/50 text-center">
          <p className="text-xs text-slate-500">Samples</p>
          <p className="text-sm font-mono font-semibold text-white">{sampleCount.toLocaleString()}</p>
        </div>
        <div className="px-3 py-2 rounded bg-slate-800/50 text-center">
          <p className="text-xs text-slate-500">Min</p>
          <p className="text-sm font-mono font-semibold" style={{ color: '#118DFF' }}>{stats.min.toFixed(3)}</p>
        </div>
        <div className="px-3 py-2 rounded bg-slate-800/50 text-center">
          <p className="text-xs text-slate-500">Max</p>
          <p className="text-sm font-mono font-semibold" style={{ color: '#E66C37' }}>{stats.max.toFixed(3)}</p>
        </div>
        <div className="px-3 py-2 rounded bg-slate-800/50 text-center">
          <p className="text-xs text-slate-500">Mean</p>
          <p className="text-sm font-mono font-semibold text-white">{stats.mean.toFixed(3)}</p>
        </div>
        <div className="px-3 py-2 rounded bg-slate-800/50 text-center">
          <p className="text-xs text-slate-500">RMS</p>
          <p className="text-sm font-mono font-semibold" style={{ color: '#6B007B' }}>{stats.rms.toFixed(3)}</p>
        </div>
      </div>

      {/* Chart */}
      <div className="chart-container relative" style={{ height: 220 }}>
        {values.length > 0 ? (
          <Line data={chartData} options={options} />
        ) : (
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <Radio size={32} className="text-slate-600 mb-2 animate-pulse" />
            <p className="text-slate-500 text-sm">
              {isConnected ? 'Waiting for data...' : 'Connecting to stream...'}
            </p>
          </div>
        )}
        
        {/* Live indicator overlay */}
        {isConnected && !isPaused && values.length > 0 && (
          <div className="absolute top-2 right-2 flex items-center gap-1.5 px-2 py-1 rounded-full bg-black/50 backdrop-blur-sm">
            <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
            <span className="text-xs text-white font-medium">LIVE</span>
          </div>
        )}
      </div>

      {/* Help text */}
      <div className="mt-3 flex items-center justify-between text-xs text-slate-500">
        <span>Streaming at ~20 Hz • Buffer: {values.length}/{BUFFER_SIZE} samples</span>
        <span>Click capture to analyze this signal</span>
      </div>
    </motion.div>
  );
}
