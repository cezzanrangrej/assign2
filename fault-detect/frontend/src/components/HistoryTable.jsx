/**
 * HistoryTable.jsx
 * Diagnosis history with localStorage persistence, filtering, and sparklines
 * Features: Sortable table, status badges, mini charts, export options
 */
import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  History,
  Trash2,
  Download,
  ChevronDown,
  ChevronUp,
  Filter,
  Search,
  FileText,
  Activity,
  X,
  Calendar,
  Clock,
} from 'lucide-react';

const STORAGE_KEY = 'vibration_diagnosis_history';

// Mini sparkline component
function Sparkline({ data, color = '#118DFF', width = 80, height = 24 }) {
  if (!data || data.length < 2) return null;

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;

  const points = data.slice(-50).map((v, i, arr) => {
    const x = (i / (arr.length - 1)) * width;
    const y = height - ((v - min) / range) * height;
    return `${x},${y}`;
  }).join(' ');

  return (
    <svg width={width} height={height} className="opacity-70">
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

// Status badge component
function StatusBadge({ prediction, severity }) {
  const styles = {
    0: 'bg-green-500/15 text-green-400 border-green-500/20',
    1: 'bg-amber-500/15 text-amber-400 border-amber-500/20',
    2: 'bg-orange-500/15 text-orange-400 border-orange-500/20',
    3: 'bg-red-500/15 text-red-400 border-red-500/20',
  };

  return (
    <span className={`px-2.5 py-1 text-xs font-medium rounded border ${styles[severity] || styles[0]}`}>
      {prediction}
    </span>
  );
}

export default function HistoryTable({ className = '' }) {
  const [history, setHistory] = useState([]);
  const [sortField, setSortField] = useState('timestamp');
  const [sortDirection, setSortDirection] = useState('desc');
  const [filterPrediction, setFilterPrediction] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [expandedRow, setExpandedRow] = useState(null);

  // Load history from localStorage
  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        setHistory(JSON.parse(stored));
      }
    } catch (err) {
      console.error('Failed to load history:', err);
    }
  }, []);

  // Save history to localStorage
  const saveHistory = useCallback((newHistory) => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(newHistory));
      setHistory(newHistory);
    } catch (err) {
      console.error('Failed to save history:', err);
    }
  }, []);

  // Add new entry (called from parent component)
  const addEntry = useCallback((result, signal, fileName) => {
    const entry = {
      id: Date.now(),
      timestamp: new Date().toISOString(),
      fileName: fileName || 'Unknown',
      prediction: result.prediction,
      confidence: result.confidence,
      severity: result.severity,
      features: result.features,
      signalPreview: signal.slice(0, 100), // Store first 100 samples for sparkline
      sampleCount: signal.length,
    };

    const newHistory = [entry, ...history].slice(0, 50); // Keep last 50 entries
    saveHistory(newHistory);
  }, [history, saveHistory]);

  // Expose addEntry method to parent
  React.useImperativeHandle(
    React.useRef(),
    () => ({ addEntry }),
    [addEntry]
  );

  // Delete entry
  const deleteEntry = useCallback((id) => {
    const newHistory = history.filter(entry => entry.id !== id);
    saveHistory(newHistory);
  }, [history, saveHistory]);

  // Clear all history
  const clearHistory = useCallback(() => {
    if (window.confirm('Are you sure you want to clear all history?')) {
      saveHistory([]);
    }
  }, [saveHistory]);

  // Export to CSV
  const exportToCSV = useCallback(() => {
    const headers = ['Timestamp', 'File', 'Prediction', 'Confidence', 'Severity', 'Samples', 'RMS', 'Kurtosis', 'Dominant Freq'];
    const rows = history.map(entry => [
      new Date(entry.timestamp).toLocaleString(),
      entry.fileName,
      entry.prediction,
      (entry.confidence * 100).toFixed(1) + '%',
      entry.severity,
      entry.sampleCount,
      entry.features?.rms?.toFixed(4) || '',
      entry.features?.kurtosis?.toFixed(4) || '',
      entry.features?.dominant_freq?.toFixed(2) || '',
    ]);

    const csv = [headers, ...rows].map(row => row.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `diagnosis_history_${Date.now()}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  }, [history]);

  // Sort handler
  const handleSort = useCallback((field) => {
    if (sortField === field) {
      setSortDirection(prev => prev === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  }, [sortField]);

  // Get unique predictions for filter
  const uniquePredictions = useMemo(() => {
    const predictions = new Set(history.map(e => e.prediction));
    return ['all', ...Array.from(predictions)];
  }, [history]);

  // Filtered and sorted history
  const displayHistory = useMemo(() => {
    let filtered = history;

    // Apply prediction filter
    if (filterPrediction !== 'all') {
      filtered = filtered.filter(e => e.prediction === filterPrediction);
    }

    // Apply search filter
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(e =>
        e.fileName.toLowerCase().includes(term) ||
        e.prediction.toLowerCase().includes(term)
      );
    }

    // Apply sorting
    return filtered.sort((a, b) => {
      let comparison = 0;
      switch (sortField) {
        case 'timestamp':
          comparison = new Date(a.timestamp) - new Date(b.timestamp);
          break;
        case 'confidence':
          comparison = a.confidence - b.confidence;
          break;
        case 'severity':
          comparison = a.severity - b.severity;
          break;
        case 'prediction':
          comparison = a.prediction.localeCompare(b.prediction);
          break;
        default:
          comparison = 0;
      }
      return sortDirection === 'asc' ? comparison : -comparison;
    });
  }, [history, filterPrediction, searchTerm, sortField, sortDirection]);

  // Sort header component
  const SortHeader = ({ field, children }) => (
    <th
      onClick={() => handleSort(field)}
      className="px-4 py-3 text-left text-xs font-semibold text-slate-500 uppercase tracking-wider cursor-pointer hover:text-slate-300 transition-colors bg-slate-800/50"
    >
      <div className="flex items-center gap-1">
        {children}
        {sortField === field && (
          sortDirection === 'asc' ? <ChevronUp size={14} /> : <ChevronDown size={14} />
        )}
      </div>
    </th>
  );

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: 0.4 }}
      className={`glass-card p-5 ${className}`}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="w-1 h-5 rounded-sm" style={{ background: '#E66C37' }} />
          <h3 className="text-sm font-semibold text-white uppercase tracking-wide">Diagnosis History</h3>
          <span className="px-2 py-0.5 text-xs font-medium bg-slate-800 text-slate-400 rounded">
            {history.length} entries
          </span>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={exportToCSV}
            disabled={history.length === 0}
            className="p-2 rounded bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-white transition-colors disabled:opacity-50"
            title="Export to CSV"
          >
            <Download size={16} />
          </button>
          <button
            onClick={clearHistory}
            disabled={history.length === 0}
            className="p-2 rounded bg-slate-800 text-slate-400 hover:bg-red-500/20 hover:text-red-400 transition-colors disabled:opacity-50"
            title="Clear History"
          >
            <Trash2 size={16} />
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3 mb-4">
        {/* Search */}
        <div className="relative flex-1 min-w-[200px]">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <input
            type="text"
            placeholder="Search by filename or prediction..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 rounded bg-slate-800 border border-slate-700 text-white placeholder-slate-500 text-sm focus:outline-none focus:border-[#118DFF] transition-colors"
          />
          {searchTerm && (
            <button
              onClick={() => setSearchTerm('')}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-white"
            >
              <X size={14} />
            </button>
          )}
        </div>

        {/* Prediction Filter */}
        <div className="relative">
          <Filter size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <select
            value={filterPrediction}
            onChange={(e) => setFilterPrediction(e.target.value)}
            className="pl-10 pr-8 py-2 rounded bg-slate-800 border border-slate-700 text-white text-sm focus:outline-none focus:border-[#118DFF] appearance-none cursor-pointer"
          >
            {uniquePredictions.map(pred => (
              <option key={pred} value={pred} className="bg-slate-800">
                {pred === 'all' ? 'All Predictions' : pred}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Table */}
      {history.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-12 text-slate-500">
          <History size={40} className="mb-3 opacity-50" />
          <p className="font-medium">No diagnosis history</p>
          <p className="text-sm mt-1">Run a diagnosis to see it here</p>
        </div>
      ) : displayHistory.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-12 text-slate-500">
          <Search size={40} className="mb-3 opacity-50" />
          <p className="font-medium">No matching results</p>
          <p className="text-sm mt-1">Try adjusting your filters</p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="border-b border-slate-700">
              <tr>
                <SortHeader field="timestamp">Date</SortHeader>
                <th className="px-4 py-3 text-left text-xs font-semibold text-slate-500 uppercase tracking-wider bg-slate-800/50">Signal</th>
                <SortHeader field="prediction">Prediction</SortHeader>
                <SortHeader field="confidence">Confidence</SortHeader>
                <SortHeader field="severity">Severity</SortHeader>
                <th className="px-4 py-3 text-left text-xs font-semibold text-slate-500 uppercase tracking-wider bg-slate-800/50">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700/50">
              <AnimatePresence>
                {displayHistory.map((entry) => (
                  <React.Fragment key={entry.id}>
                    <motion.tr
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="hover:bg-slate-800/30 transition-colors cursor-pointer"
                      onClick={() => setExpandedRow(expandedRow === entry.id ? null : entry.id)}
                    >
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <Calendar size={14} className="text-slate-500" />
                          <div>
                            <p className="text-sm text-white">{new Date(entry.timestamp).toLocaleDateString()}</p>
                            <p className="text-xs text-slate-500">{new Date(entry.timestamp).toLocaleTimeString()}</p>
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-3">
                          <Sparkline data={entry.signalPreview} color={entry.severity > 1 ? '#E66C37' : '#118DFF'} />
                          <div>
                            <p className="text-sm text-white truncate max-w-[120px]">{entry.fileName}</p>
                            <p className="text-xs text-slate-500">{entry.sampleCount?.toLocaleString()} samples</p>
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <StatusBadge prediction={entry.prediction} severity={entry.severity} />
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <div className="w-16 h-1.5 bg-slate-700 rounded overflow-hidden">
                            <div
                              className="h-full rounded transition-all duration-500"
                              style={{
                                width: `${entry.confidence * 100}%`,
                                backgroundColor: entry.severity > 1 ? '#E66C37' : '#118DFF',
                              }}
                            />
                          </div>
                          <span className="text-sm text-white font-mono">{(entry.confidence * 100).toFixed(0)}%</span>
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center">
                          {[...Array(3)].map((_, i) => (
                            <div
                              key={i}
                              className={`w-2 h-2 rounded-full mr-1 ${
                                i < entry.severity
                                  ? entry.severity === 3
                                    ? 'bg-red-500'
                                    : entry.severity === 2
                                    ? 'bg-orange-500'
                                    : 'bg-amber-500'
                                  : 'bg-slate-700'
                              }`}
                            />
                          ))}
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            deleteEntry(entry.id);
                          }}
                          className="p-1.5 rounded text-slate-500 hover:bg-red-500/20 hover:text-red-400 transition-colors"
                        >
                          <Trash2 size={14} />
                        </button>
                      </td>
                    </motion.tr>

                    {/* Expanded details */}
                    <AnimatePresence>
                      {expandedRow === entry.id && (
                        <motion.tr
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          exit={{ opacity: 0, height: 0 }}
                        >
                          <td colSpan={6} className="px-4 py-4 bg-slate-800/30">
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                              <div className="px-3 py-2 rounded bg-slate-800/50">
                                <p className="text-xs text-slate-500">RMS</p>
                                <p className="text-sm font-mono text-white">{entry.features?.rms?.toFixed(4) || 'N/A'}</p>
                              </div>
                              <div className="px-3 py-2 rounded bg-slate-800/50">
                                <p className="text-xs text-slate-500">Kurtosis</p>
                                <p className="text-sm font-mono text-white">{entry.features?.kurtosis?.toFixed(4) || 'N/A'}</p>
                              </div>
                              <div className="px-3 py-2 rounded bg-slate-800/50">
                                <p className="text-xs text-slate-500">Crest Factor</p>
                                <p className="text-sm font-mono text-white">{entry.features?.crest_factor?.toFixed(4) || 'N/A'}</p>
                              </div>
                              <div className="px-3 py-2 rounded bg-slate-800/50">
                                <p className="text-xs text-slate-500">Dominant Freq</p>
                                <p className="text-sm font-mono text-white">{entry.features?.dominant_freq?.toFixed(2) || 'N/A'} Hz</p>
                              </div>
                            </div>
                          </td>
                        </motion.tr>
                      )}
                    </AnimatePresence>
                  </React.Fragment>
                ))}
              </AnimatePresence>
            </tbody>
          </table>
        </div>
      )}
    </motion.div>
  );
}

// Export addEntry hook for parent components
export function useHistoryTable() {
  const ref = React.useRef(null);

  const addEntry = useCallback((result, signal, fileName) => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      const history = stored ? JSON.parse(stored) : [];
      
      const entry = {
        id: Date.now(),
        timestamp: new Date().toISOString(),
        fileName: fileName || 'Unknown',
        prediction: result.prediction,
        confidence: result.confidence,
        severity: result.severity,
        features: result.features,
        signalPreview: signal.slice(0, 100),
        sampleCount: signal.length,
      };

      const newHistory = [entry, ...history].slice(0, 50);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(newHistory));
      
      // Force re-render by dispatching storage event
      window.dispatchEvent(new Event('storage'));
      
    } catch (err) {
      console.error('Failed to add history entry:', err);
    }
  }, []);

  return { addEntry };
}
