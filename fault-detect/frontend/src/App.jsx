/**
 * App.jsx
 * Main application dashboard with professional PowerBI-style design
 * Features: Navigation tabs, KPI metrics, data visualizations, responsive layout
 */
import React, { useState, useCallback, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Activity,
  Radio,
  Upload,
  History,
  BarChart3,
  Cpu,
  Zap,
  AlertTriangle,
  CheckCircle2,
  TrendingUp,
  Clock,
  Server,
  ExternalLink,
  ChevronRight,
  ChevronDown,
  Sparkles,
  Play,
  FileText,
  Heart,
  List,
  Download,
  Loader2,
  Filter,
} from 'lucide-react';
import { Toaster, toast } from 'react-hot-toast';

// Components
import StreamViewer from './components/StreamViewer';
import UploadAndRun from './components/UploadAndRun';
import HistoryTable, { useHistoryTable } from './components/HistoryTable';
import FaultDistributionChart from './components/FaultDistributionChart';
import SeverityDonut from './components/SeverityDonut';
import AnalyticsTrendChart from './components/AnalyticsTrendChart';

// Navigation tabs configuration
const TABS = [
  { id: 'stream', label: 'Live Stream', icon: Radio },
  { id: 'analyze', label: 'Analyze', icon: Upload },
  { id: 'history', label: 'History', icon: History },
];

// PowerBI color palette
const PBI_BLUE = '#118DFF';
const PBI_COLORS = ['#118DFF', '#12239E', '#E66C37', '#6B007B', '#E044A7'];

// Time filter options
const TIME_FILTERS = [
  { id: '10min', label: 'Last 10 minutes', ms: 10 * 60 * 1000 },
  { id: '1hour', label: 'Last 1 hour', ms: 60 * 60 * 1000 },
  { id: 'today', label: 'Today', ms: null }, // Special case - start of day
  { id: 'all', label: 'All Time', ms: null },
];

// Utility function to filter records by time window
function filterByTimeWindow(records, filterId) {
  if (!records || records.length === 0) return [];
  if (filterId === 'all') return records;
  
  const now = new Date();
  const filter = TIME_FILTERS.find(f => f.id === filterId);
  
  if (filterId === 'today') {
    const startOfDay = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    return records.filter(record => {
      const recordDate = new Date(record.timestamp);
      return recordDate >= startOfDay;
    });
  }
  
  if (filter?.ms) {
    const cutoff = new Date(now.getTime() - filter.ms);
    return records.filter(record => {
      const recordDate = new Date(record.timestamp);
      return recordDate >= cutoff;
    });
  }
  
  return records;
}

// Compute stats from filtered data
function computeStatsFromHistory(history) {
  if (!history || history.length === 0) {
    return { totalAnalyses: 0, faultsDetected: 0, avgConfidence: 0 };
  }
  
  const totalAnalyses = history.length;
  const faultsDetected = history.filter(e => e.severity > 0).length;
  const avgConfidence = history.reduce((sum, e) => sum + e.confidence, 0) / history.length;
  
  return {
    totalAnalyses,
    faultsDetected,
    avgConfidence: Math.round(avgConfidence * 100),
  };
}

// Time Filter Dropdown Component
function TimeFilterDropdown({ value, onChange }) {
  const [isOpen, setIsOpen] = useState(false);
  const selectedFilter = TIME_FILTERS.find(f => f.id === value) || TIME_FILTERS[3];

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-slate-800 border border-slate-700 hover:border-slate-600 transition-colors"
      >
        <Filter size={14} className="text-slate-400" />
        <span className="text-xs font-medium text-slate-300">{selectedFilter.label}</span>
        <ChevronDown size={14} className={`text-slate-400 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>
      
      <AnimatePresence>
        {isOpen && (
          <>
            {/* Backdrop */}
            <div 
              className="fixed inset-0 z-10" 
              onClick={() => setIsOpen(false)}
            />
            
            {/* Dropdown */}
            <motion.div
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.15 }}
              className="absolute right-0 mt-1 z-20 min-w-[160px] py-1 rounded-lg bg-slate-800 border border-slate-700 shadow-xl"
            >
              {TIME_FILTERS.map((filter) => (
                <button
                  key={filter.id}
                  onClick={() => {
                    onChange(filter.id);
                    setIsOpen(false);
                  }}
                  className={`w-full px-3 py-2 text-left text-xs font-medium transition-colors ${
                    value === filter.id
                      ? 'bg-blue-500/20 text-blue-400'
                      : 'text-slate-300 hover:bg-slate-700/50'
                  }`}
                >
                  {filter.label}
                </button>
              ))}
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}

// KPI Card component (PowerBI style)
function KPICard({ icon: Icon, label, value, unit, trend, delay = 0 }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay }}
      className="glass-card p-4"
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="w-1 h-4 rounded-sm" style={{ background: PBI_BLUE }} />
          <p className="text-xs text-slate-400 uppercase tracking-wide font-medium">{label}</p>
        </div>
        {trend !== undefined && trend !== 0 && (
          <span className={`flex items-center gap-1 text-xs font-medium ${
            trend > 0 ? 'text-green-400' : 'text-red-400'
          }`}>
            <TrendingUp size={12} className={trend < 0 ? 'rotate-180' : ''} />
            {Math.abs(trend)}%
          </span>
        )}
      </div>
      <div className="flex items-end justify-between">
        <p className="text-3xl font-semibold text-white">
          {value}
          {unit && <span className="text-base text-slate-500 ml-1">{unit}</span>}
        </p>
        <Icon size={24} className="text-slate-600" />
      </div>
    </motion.div>
  );
}

// Architecture diagram component (PowerBI style)
function ArchitectureDiagram() {
  const steps = [
    { icon: Radio, label: 'Sensor Input', subtitle: 'Live / Upload', color: PBI_COLORS[0] },
    { icon: Activity, label: 'Preprocessing', subtitle: 'Filter & Normalize', color: PBI_COLORS[1] },
    { icon: BarChart3, label: 'FFT & Features', subtitle: 'Time + Freq Domain', color: PBI_COLORS[2] },
    { icon: Cpu, label: 'ML Engine', subtitle: 'RF + CNN', color: PBI_COLORS[3] },
    { icon: Server, label: 'FastAPI', subtitle: 'API & Logic', color: PBI_COLORS[0] },
    { icon: Zap, label: 'React UI', subtitle: 'Dashboard', color: PBI_COLORS[1] },
    { icon: AlertTriangle, label: 'Alerts', subtitle: 'Reports & PDF', color: PBI_COLORS[4] },
  ];

  const tags = {
    'Streaming': ['Live Stream', 'SSE/WebSocket', 'Signal Buffering'],
    'Analytics': ['Real-time FFT', 'Feature Extraction', 'Severity Logic'],
    'Reliability': ['Health Checks', 'History Tracking', 'PDF Reports'],
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.4 }}
      className="glass-card p-5 mb-5"
    >
      <div className="flex items-center gap-2 mb-4">
        <div className="w-1 h-5 rounded-sm" style={{ background: PBI_BLUE }} />
        <h3 className="text-sm font-semibold text-white uppercase tracking-wide">System Pipeline</h3>
      </div>
      <div className="flex items-center justify-between overflow-x-auto pb-2">
        {steps.map((step, idx) => (
          <React.Fragment key={step.label}>
            <div className="flex flex-col items-center min-w-[70px]">
              <div 
                className="w-12 h-12 rounded-md flex items-center justify-center" 
                style={{ background: `${step.color}20`, border: `1px solid ${step.color}40` }}
              >
                <step.icon size={22} style={{ color: step.color }} />
              </div>
              <span className="text-xs text-white mt-2 font-medium text-center">{step.label}</span>
              <span className="text-[10px] text-slate-500 text-center">{step.subtitle}</span>
            </div>
            {idx < steps.length - 1 && (
              <div className="flex-1 h-px bg-slate-700 mx-1 min-w-[10px]" />
            )}
          </React.Fragment>
        ))}
      </div>
      
      {/* Feature badges grouped by category */}
      <div className="flex flex-wrap gap-4 mt-5 pt-4 border-t border-slate-700/50">
        {Object.entries(tags).map(([category, items]) => (
          <div key={category} className="flex items-center gap-2">
            <span className="text-[10px] text-slate-500 uppercase tracking-wide">{category}:</span>
            <div className="flex flex-wrap gap-1.5">
              {items.map((tag) => (
                <span
                  key={tag}
                  className="px-2 py-1 text-xs font-medium rounded bg-slate-800 text-slate-400"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </motion.div>
  );
}

// Beautiful Response Card Component
function ResponseCard({ endpoint, result, dataSource, onClose }) {
  if (!result) return null;
  
  const { success, data, error } = result;
  
  // Format different response types beautifully
  const renderBeautifulResponse = () => {
    if (!success) {
      return (
        <div className="flex items-center gap-3 p-4 bg-red-500/10 border border-red-500/20 rounded-lg">
          <AlertTriangle className="text-red-400" size={20} />
          <div>
            <p className="text-sm font-medium text-red-400">Error</p>
            <p className="text-xs text-red-300/70">{error}</p>
          </div>
        </div>
      );
    }

    switch (endpoint.id) {
      case 'health':
        return (
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <div className={`w-3 h-3 rounded-full ${data.status === 'healthy' ? 'bg-green-500' : 'bg-red-500'} animate-pulse`} />
              <span className="text-lg font-semibold text-white capitalize">{data.status}</span>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 bg-slate-800/50 rounded-lg">
                <p className="text-xs text-slate-500 mb-1">API Status</p>
                <p className="text-sm font-medium text-green-400">Online</p>
              </div>
              <div className="p-3 bg-slate-800/50 rounded-lg">
                <p className="text-xs text-slate-500 mb-1">ML Model</p>
                <p className={`text-sm font-medium ${data.model_loaded ? 'text-green-400' : 'text-amber-400'}`}>
                  {data.model_loaded ? 'Loaded' : 'Demo Mode'}
                </p>
              </div>
            </div>
          </div>
        );

      case 'root':
        return (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-lg font-semibold text-white">{data.message}</span>
              <span className="px-2 py-1 bg-blue-500/20 text-blue-400 text-xs font-mono rounded">v{data.version}</span>
            </div>
            <div className="space-y-2">
              <p className="text-xs text-slate-500 uppercase tracking-wide">Available Endpoints</p>
              <div className="flex flex-wrap gap-2">
                {data.endpoints?.map((ep, i) => (
                  <span key={i} className="px-2 py-1 bg-slate-800 text-slate-300 text-xs font-mono rounded">{ep}</span>
                ))}
              </div>
            </div>
          </div>
        );

      case 'fault-types':
        return (
          <div className="space-y-3">
            <p className="text-xs text-slate-500 uppercase tracking-wide mb-2">Detectable Fault Categories</p>
            <div className="grid gap-2">
              {Object.entries(data).map(([name, info]) => (
                <div key={name} className="flex items-center gap-3 p-3 bg-slate-800/50 rounded-lg">
                  <div className="w-3 h-3 rounded-full" style={{ background: info.color }} />
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-white">{name}</span>
                      <span className="px-1.5 py-0.5 bg-slate-700 text-slate-400 text-[10px] rounded">
                        Severity: {info.severity}
                      </span>
                    </div>
                    <p className="text-xs text-slate-500 mt-0.5">{info.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        );

      case 'stream':
        return (
          <div className="flex items-center gap-4 p-4 bg-orange-500/10 border border-orange-500/20 rounded-lg">
            <div className="w-12 h-12 rounded-lg bg-orange-500/20 flex items-center justify-center">
              <Radio className="text-orange-400" size={24} />
            </div>
            <div>
              <p className="text-sm font-medium text-orange-400">{data.status}</p>
              <p className="text-xs text-slate-500">{data.message}</p>
              <p className="text-xs text-slate-600 mt-1">Protocol: Server-Sent Events (SSE)</p>
            </div>
          </div>
        );

      case 'predict':
        return (
          <div className="space-y-4">
            {/* Prediction Header */}
            <div className="flex items-center gap-4 p-4 rounded-lg" style={{ background: `${data.color}15`, border: `1px solid ${data.color}30` }}>
              <div className="w-14 h-14 rounded-lg flex items-center justify-center" style={{ background: `${data.color}25` }}>
                <Cpu size={28} style={{ color: data.color }} />
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <span className="text-xl font-bold text-white">{data.prediction}</span>
                  <span className="px-2 py-0.5 rounded text-xs font-medium" style={{ background: `${data.color}30`, color: data.color }}>
                    Severity {data.severity}/3
                  </span>
                </div>
                <p className="text-sm text-slate-400 mt-1">{data.description}</p>
              </div>
              <div className="text-right">
                <p className="text-3xl font-bold text-white">{(data.confidence * 100).toFixed(1)}%</p>
                <p className="text-xs text-slate-500">Confidence</p>
              </div>
            </div>

            {/* SHAP Explanation */}
            {data.explanation && data.explanation.top_features && data.explanation.top_features.length > 0 && (
              <div className="p-4 bg-amber-500/5 border border-amber-500/20 rounded-lg">
                <div className="flex items-center gap-2 mb-3">
                  <Sparkles size={14} className="text-amber-400" />
                  <span className="text-xs font-semibold text-amber-400 uppercase tracking-wide">Why this diagnosis?</span>
                  <span className="text-[10px] text-slate-500 ml-auto">{data.explanation.method || 'SHAP Analysis'}</span>
                </div>
                <div className="space-y-2">
                  {data.explanation.top_features.slice(0, 5).map((feat, idx) => {
                    const isPositive = feat.direction === 'positive' || feat.shap_value >= 0;
                    const barWidth = Math.min((feat.abs_importance / (data.explanation.top_features[0]?.abs_importance || 1)) * 100, 100);
                    const barColor = isPositive ? '#10B981' : '#EF4444';
                    
                    return (
                      <div key={feat.name} className="flex items-center gap-3">
                        <span className="text-xs text-slate-400 w-28 truncate" title={feat.name}>
                          {feat.name.replace(/_/g, ' ')}
                        </span>
                        <div className="flex-1 h-2 rounded-full bg-slate-800 overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all duration-500"
                            style={{ width: `${barWidth}%`, backgroundColor: barColor }}
                          />
                        </div>
                        <span className={`text-xs font-mono w-16 text-right ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                          {feat.shap_value >= 0 ? '+' : ''}{feat.shap_value.toFixed(3)}
                        </span>
                        <span className={`text-[10px] ${isPositive ? 'text-green-400/70' : 'text-red-400/70'}`}>
                          {isPositive ? '↑' : '↓'}
                        </span>
                      </div>
                    );
                  })}
                </div>
                <div className="flex items-center gap-4 mt-3 pt-2 border-t border-slate-800 text-[10px] text-slate-500">
                  <span className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-green-500" />
                    Supports prediction
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-red-500" />
                    Opposes prediction
                  </span>
                  {data.explanation.note && (
                    <span className="ml-auto text-amber-400/70">{data.explanation.note}</span>
                  )}
                </div>
              </div>
            )}

            {/* Features Grid */}
            <div>
              <p className="text-xs text-slate-500 uppercase tracking-wide mb-2">Extracted Features</p>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                {Object.entries(data.features || {}).slice(0, 8).map(([key, value]) => (
                  <div key={key} className="p-2 bg-slate-800/50 rounded">
                    <p className="text-[10px] text-slate-500 uppercase">{key.replace(/_/g, ' ')}</p>
                    <p className="text-sm font-mono text-white">{typeof value === 'number' ? value.toFixed(4) : value}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* FFT Info */}
            {data.fft && (
              <div className="p-3 bg-slate-800/30 rounded-lg">
                <p className="text-xs text-slate-500">FFT Analysis: {data.fft.frequencies?.length || 0} frequency bins computed</p>
              </div>
            )}
          </div>
        );

      case 'report':
        return (
          <div className="flex items-center gap-4 p-4 bg-indigo-500/10 border border-indigo-500/20 rounded-lg">
            <div className="w-12 h-12 rounded-lg bg-indigo-500/20 flex items-center justify-center">
              <Download className="text-indigo-400" size={24} />
            </div>
            <div>
              <p className="text-sm font-medium text-indigo-400">PDF Report Generated</p>
              <p className="text-xs text-slate-500">{data.message}</p>
              <p className="text-xs text-slate-600 mt-1">Contains: Signal plot, FFT, PSD, Features table, SHAP explanation</p>
            </div>
          </div>
        );

      default:
        return (
          <pre className="text-xs text-slate-400 bg-slate-900/50 p-3 rounded-lg overflow-x-auto font-mono">
            {JSON.stringify(data, null, 2)}
          </pre>
        );
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: 'auto' }}
      exit={{ opacity: 0, height: 0 }}
      className="mt-4 pt-4 border-t border-slate-700/50"
    >
      {/* Response Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full" style={{ background: endpoint.color }} />
          <span className="text-sm font-medium text-white">{endpoint.label} Response</span>
          <span className={`px-1.5 py-0.5 text-[10px] font-bold rounded ${endpoint.method === 'GET' ? 'bg-green-500/20 text-green-400' : 'bg-amber-500/20 text-amber-400'}`}>
            {endpoint.method}
          </span>
          <span className="text-xs text-slate-600 font-mono">{endpoint.path}</span>
        </div>
        <button onClick={onClose} className="text-slate-500 hover:text-white transition-colors">
          <ChevronRight size={16} className="rotate-90" />
        </button>
      </div>

      {/* Beautiful Response Content */}
      {renderBeautifulResponse()}

      {/* Data Source Footer */}
      <div className="mt-4 pt-3 border-t border-slate-800">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-4 text-xs text-slate-600">
            <span className="flex items-center gap-1">
              <Server size={12} />
              <span>API: FastAPI Backend</span>
            </span>
            <span>•</span>
            <span className="font-mono">http://localhost:8001{endpoint.path}</span>
          </div>
          <span className="text-xs text-slate-600">
            {new Date().toLocaleTimeString()}
          </span>
        </div>
        {/* Show actual data source for POST endpoints */}
        {endpoint.method === 'POST' && dataSource?.hasData && (
          <div className="flex items-center gap-3 p-2 bg-slate-800/50 rounded text-xs">
            <span className="text-slate-500">Data Source:</span>
            <span className="flex items-center gap-1.5 text-blue-400">
              {dataSource.isLive ? <Radio size={12} /> : <FileText size={12} />}
              {dataSource.source}
            </span>
            <span className="text-slate-600">•</span>
            <span className="text-slate-400">{dataSource.samples?.toLocaleString()} samples</span>
            {dataSource.isLive && (
              <span className="px-1.5 py-0.5 bg-green-500/20 text-green-400 text-[10px] rounded animate-pulse">LIVE</span>
            )}
          </div>
        )}
      </div>
    </motion.div>
  );
}

// API Endpoints Panel - Test all backend endpoints
function APIEndpointsPanel({ signalData, signalSource, isLive = false }) {
  const [loading, setLoading] = useState({});
  const [results, setResults] = useState({});
  const [activeEndpoint, setActiveEndpoint] = useState(null);
  const [liveResults, setLiveResults] = useState(null);
  
  const API_BASE = 'http://localhost:8001';
  
  // Data source info based on what's available
  const getDataSourceInfo = () => {
    if (signalData && signalData.length > 0) {
      return {
        source: signalSource || 'Uploaded CSV',
        samples: signalData.length,
        isLive: isLive,
        hasData: true
      };
    }
    return {
      source: 'No data loaded',
      samples: 0,
      isLive: false,
      hasData: false
    };
  };
  
  const dataSourceInfo = getDataSourceInfo();
  
  const endpoints = [
    { 
      id: 'health', 
      method: 'GET', 
      path: '/health', 
      label: 'Health Check',
      icon: Heart,
      color: '#10B981',
      description: 'Check API status and model availability',
      requiresData: false
    },
    { 
      id: 'root', 
      method: 'GET', 
      path: '/', 
      label: 'API Info',
      icon: Server,
      color: '#118DFF',
      description: 'Get API version and available endpoints',
      requiresData: false
    },
    { 
      id: 'fault-types', 
      method: 'GET', 
      path: '/fault-types', 
      label: 'Fault Types',
      icon: List,
      color: '#6B007B',
      description: 'List all detectable fault categories',
      requiresData: false
    },
    { 
      id: 'stream', 
      method: 'GET', 
      path: '/stream-signal', 
      label: 'Stream Signal',
      icon: Radio,
      color: '#E66C37',
      description: 'SSE endpoint for real-time signal data',
      requiresData: false
    },
    { 
      id: 'predict', 
      method: 'POST', 
      path: '/predict', 
      label: 'Predict',
      icon: Cpu,
      color: '#E044A7',
      description: 'Analyze signal and predict fault type',
      requiresData: true
    },
    { 
      id: 'report', 
      method: 'POST', 
      path: '/diagnostic-report', 
      label: 'Generate Report',
      icon: FileText,
      color: '#12239E',
      description: 'Generate PDF diagnostic report',
      requiresData: true
    },
  ];

  const handleEndpointClick = async (endpoint) => {
    // Check if endpoint requires data but none is available
    if (endpoint.requiresData && !dataSourceInfo.hasData) {
      toast.error(`${endpoint.label} requires signal data. Please upload a CSV or capture from Live Stream first.`, {
        style: {
          background: 'rgba(30, 41, 59, 0.95)',
          color: '#fff',
          border: '1px solid rgba(239, 68, 68, 0.4)',
          borderRadius: '8px',
        },
        duration: 4000
      });
      return;
    }
    
    setLoading(prev => ({ ...prev, [endpoint.id]: true }));
    setActiveEndpoint(endpoint);
    
    try {
      let response;
      let data;
      
      if (endpoint.id === 'stream') {
        // For SSE, just show it's available
        const testResponse = await fetch(`${API_BASE}${endpoint.path}`, { 
          method: 'GET',
          headers: { 'Accept': 'text/event-stream' }
        });
        if (testResponse.ok) {
          data = { 
            status: 'SSE Stream Available', 
            message: 'Use Live Stream tab to view real-time data',
            dataSource: 'Backend synthetic generator (60Hz + harmonics)'
          };
        }
        testResponse.body?.cancel();
      } else if (endpoint.method === 'POST') {
        // Use actual signal data from props
        response = await fetch(`${API_BASE}${endpoint.path}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ signal: signalData, sample_rate: 2000 })
        });
        
        if (endpoint.id === 'report') {
          // Handle PDF download
          if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `diagnostic_report_${Date.now()}.pdf`;
            a.click();
            window.URL.revokeObjectURL(url);
            data = { 
              status: 'success', 
              message: 'PDF report downloaded successfully!',
              dataSource: dataSourceInfo.source,
              samplesAnalyzed: signalData.length
            };
          } else {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || 'Failed to generate report');
          }
        } else {
          if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Request failed with status ${response.status}`);
          }
          data = await response.json();
          // Add data source info to response
          data._dataSource = {
            source: dataSourceInfo.source,
            samples: signalData.length,
            isLive: dataSourceInfo.isLive
          };
        }
      } else {
        response = await fetch(`${API_BASE}${endpoint.path}`);
        data = await response.json();
      }
      
      setResults(prev => ({ ...prev, [endpoint.id]: { success: true, data } }));
      toast.success(`${endpoint.label}: Success!`, {
        style: {
          background: 'rgba(30, 41, 59, 0.95)',
          color: '#fff',
          border: `1px solid ${endpoint.color}40`,
          borderRadius: '8px',
        },
      });
    } catch (error) {
      setResults(prev => ({ 
        ...prev, 
        [endpoint.id]: { success: false, error: error.message } 
      }));
      toast.error(`${endpoint.label}: ${error.message}`, {
        style: {
          background: 'rgba(30, 41, 59, 0.95)',
          color: '#fff',
          border: '1px solid rgba(239, 68, 68, 0.4)',
          borderRadius: '8px',
        },
      });
    } finally {
      setLoading(prev => ({ ...prev, [endpoint.id]: false }));
    }
  };

  const currentEndpoint = endpoints.find(e => e.id === activeEndpoint?.id);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="glass-card p-5 mb-5"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="w-1 h-5 rounded-sm" style={{ background: PBI_BLUE }} />
          <h3 className="text-sm font-semibold text-white uppercase tracking-wide">API Endpoints</h3>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-slate-600">Backend:</span>
          <span className="flex items-center gap-1.5 px-2 py-1 bg-slate-800 rounded text-xs">
            <span className="w-1.5 h-1.5 rounded-full bg-green-500" />
            <span className="text-slate-400 font-mono">{API_BASE}</span>
          </span>
        </div>
      </div>
      
      {/* Data Source Banner */}
      <div className={`mb-4 p-3 rounded-lg border ${dataSourceInfo.hasData ? 'bg-blue-500/10 border-blue-500/20' : 'bg-amber-500/10 border-amber-500/20'}`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={`w-8 h-8 rounded-md flex items-center justify-center ${dataSourceInfo.hasData ? 'bg-blue-500/20' : 'bg-amber-500/20'}`}>
              {dataSourceInfo.isLive ? (
                <Radio size={16} className="text-blue-400" />
              ) : dataSourceInfo.hasData ? (
                <FileText size={16} className="text-blue-400" />
              ) : (
                <AlertTriangle size={16} className="text-amber-400" />
              )}
            </div>
            <div>
              <p className="text-xs text-slate-500 uppercase tracking-wide">Current Data Source</p>
              <p className={`text-sm font-medium ${dataSourceInfo.hasData ? 'text-blue-400' : 'text-amber-400'}`}>
                {dataSourceInfo.source}
                {dataSourceInfo.isLive && <span className="ml-2 px-1.5 py-0.5 bg-green-500/20 text-green-400 text-[10px] rounded animate-pulse">LIVE</span>}
              </p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-xs text-slate-500">Samples</p>
            <p className="text-lg font-mono text-white">{dataSourceInfo.samples.toLocaleString()}</p>
          </div>
        </div>
        {!dataSourceInfo.hasData && (
          <p className="text-xs text-amber-400/70 mt-2">
            ⚠️ Upload a CSV file in the Analyze tab or capture signal from Live Stream to use POST endpoints
          </p>
        )}
      </div>
      
      {/* Endpoint Buttons Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        {endpoints.map((endpoint) => (
          <button
            key={endpoint.id}
            onClick={() => handleEndpointClick(endpoint)}
            disabled={loading[endpoint.id] || (endpoint.requiresData && !dataSourceInfo.hasData)}
            className={`group relative flex flex-col items-center p-4 rounded-lg border transition-all disabled:opacity-50 disabled:cursor-not-allowed ${
              activeEndpoint?.id === endpoint.id 
                ? 'bg-slate-800 border-slate-600' 
                : 'bg-slate-800/50 border-slate-700/50 hover:border-slate-600 hover:bg-slate-800'
            }`}
          >
            {/* Method badge */}
            <span 
              className={`absolute top-2 right-2 text-[10px] font-bold px-1.5 py-0.5 rounded ${
                endpoint.method === 'GET' 
                  ? 'bg-green-500/20 text-green-400' 
                  : 'bg-amber-500/20 text-amber-400'
              }`}
            >
              {endpoint.method}
            </span>
            
            {/* Requires data indicator */}
            {endpoint.requiresData && (
              <span className="absolute top-2 left-2 text-[8px] text-slate-500">
                📊
              </span>
            )}
            
            {/* Icon */}
            <div 
              className="w-10 h-10 rounded-md flex items-center justify-center mb-2 transition-transform group-hover:scale-110"
              style={{ background: `${endpoint.color}20`, border: `1px solid ${endpoint.color}40` }}
            >
              {loading[endpoint.id] ? (
                <Loader2 size={20} className="animate-spin" style={{ color: endpoint.color }} />
              ) : (
                <endpoint.icon size={20} style={{ color: endpoint.color }} />
              )}
            </div>
            
            {/* Label */}
            <span className="text-xs font-medium text-white mb-1">{endpoint.label}</span>
            <span className="text-[10px] text-slate-500 font-mono">{endpoint.path}</span>
            
            {/* Result indicator */}
            {results[endpoint.id] && (
              <div className="absolute -bottom-1 left-1/2 -translate-x-1/2">
                {results[endpoint.id].success ? (
                  <CheckCircle2 size={14} className="text-green-400" />
                ) : (
                  <AlertTriangle size={14} className="text-red-400" />
                )}
              </div>
            )}
          </button>
        ))}
      </div>
      
      {/* Beautiful Response Display */}
      <AnimatePresence>
        {currentEndpoint && results[currentEndpoint.id] && (
          <ResponseCard 
            endpoint={currentEndpoint} 
            result={results[currentEndpoint.id]}
            dataSource={dataSourceInfo}
            onClose={() => setActiveEndpoint(null)}
          />
        )}
      </AnimatePresence>
    </motion.div>
  );
}

// Header component (PowerBI style)
function Header() {
  return (
    <header className="relative z-10 mb-6">
      <div className="flex items-center justify-between">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4 }}
          className="flex items-center gap-3"
        >
          <div className="relative">
            <div 
              className="w-10 h-10 rounded-md flex items-center justify-center" 
              style={{ background: PBI_BLUE }}
            >
              <Zap size={22} className="text-white" />
            </div>
            <span className="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-green-500 border-2 border-[#0d0d12]" />
          </div>
          <div>
            <h1 className="text-xl font-semibold text-white">VibrationAI</h1>
            <p className="text-xs text-slate-500">Fault Detection Dashboard</p>
          </div>
        </motion.div>
        
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4 }}
          className="flex items-center gap-3"
        >
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-green-500/10 border border-green-500/20">
            <span className="w-2 h-2 rounded-full bg-green-500" />
            <span className="text-xs font-medium text-green-400">Online</span>
          </div>
        </motion.div>
      </div>
    </header>
  );
}

// Tab navigation component (PowerBI style)
function TabNav({ activeTab, onTabChange }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.1 }}
      className="flex gap-1 mb-5 border-b border-slate-700/50"
    >
      {TABS.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onTabChange(tab.id)}
          className={`relative flex items-center gap-2 px-5 py-3 font-medium text-sm transition-colors ${
            activeTab === tab.id
              ? 'text-white'
              : 'text-slate-500 hover:text-slate-300'
          }`}
        >
          <tab.icon size={16} />
          {tab.label}
          {activeTab === tab.id && (
            <motion.div
              layoutId="activeTabIndicator"
              className="absolute bottom-0 left-0 right-0 h-0.5"
              style={{ background: PBI_BLUE }}
              transition={{ type: 'spring', bounce: 0.2, duration: 0.5 }}
            />
          )}
        </button>
      ))}
    </motion.div>
  );
}

// Main App component
export default function App() {
  const [activeTab, setActiveTab] = useState('analyze');
  const [timeFilter, setTimeFilter] = useState('all');
  const [stats, setStats] = useState({
    totalAnalyses: 0,
    faultsDetected: 0,
    avgConfidence: 0,
    uptime: '00:00:00',
  });
  const [capturedSignal, setCapturedSignal] = useState(null);
  const [historyData, setHistoryData] = useState([]);
  
  // State for current signal data (for API endpoints panel)
  const [currentSignal, setCurrentSignal] = useState({
    data: [],
    source: null,
    isLive: false
  });
  
  const { addEntry } = useHistoryTable();

  // Filter history data based on time filter (memoized for performance)
  const filteredHistory = useMemo(() => {
    return filterByTimeWindow(historyData, timeFilter);
  }, [historyData, timeFilter]);

  // Compute filtered stats (memoized for performance)
  const filteredStats = useMemo(() => {
    return computeStatsFromHistory(filteredHistory);
  }, [filteredHistory]);

  // Update uptime
  useEffect(() => {
    const startTime = Date.now();
    const interval = setInterval(() => {
      const elapsed = Math.floor((Date.now() - startTime) / 1000);
      const hours = String(Math.floor(elapsed / 3600)).padStart(2, '0');
      const minutes = String(Math.floor((elapsed % 3600) / 60)).padStart(2, '0');
      const seconds = String(elapsed % 60).padStart(2, '0');
      setStats(prev => ({ ...prev, uptime: `${hours}:${minutes}:${seconds}` }));
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  // Load stats from localStorage
  useEffect(() => {
    try {
      const stored = localStorage.getItem('vibration_diagnosis_history');
      if (stored) {
        const history = JSON.parse(stored);
        setHistoryData(history);
        const totalAnalyses = history.length;
        const faultsDetected = history.filter(e => e.severity > 0).length;
        const avgConfidence = history.length > 0
          ? history.reduce((sum, e) => sum + e.confidence, 0) / history.length
          : 0;
        setStats(prev => ({
          ...prev,
          totalAnalyses,
          faultsDetected,
          avgConfidence: Math.round(avgConfidence * 100),
        }));
      }
    } catch (err) {
      console.error('Failed to load stats:', err);
    }
  }, [activeTab]);

  // Handle diagnosis completion
  const handleDiagnosisComplete = useCallback((result, signal, fileName) => {
    addEntry(result, signal, fileName);
    
    // Update current signal for API endpoints panel
    setCurrentSignal({
      data: signal,
      source: fileName || 'Uploaded CSV',
      isLive: false
    });
    
    // Update historyData for charts
    const newEntry = {
      id: Date.now(),
      timestamp: new Date().toISOString(),
      prediction: result.prediction,
      confidence: result.confidence,
      severity: result.severity,
      color: result.color,
      fileName: fileName || 'upload',
    };
    setHistoryData(prev => [newEntry, ...prev]);
    
    setStats(prev => ({
      ...prev,
      totalAnalyses: prev.totalAnalyses + 1,
      faultsDetected: result.severity > 0 ? prev.faultsDetected + 1 : prev.faultsDetected,
    }));

    // Show toast notification
    const toastStyle = {
      background: 'rgba(30, 41, 59, 0.95)',
      color: '#fff',
      border: `1px solid ${result.color}40`,
      borderRadius: '12px',
    };

    if (result.severity === 0) {
      toast.success(`No faults detected! Confidence: ${(result.confidence * 100).toFixed(0)}%`, {
        style: toastStyle,
        icon: <CheckCircle2 className="text-green-400" />,
      });
    } else {
      toast(`${result.prediction} detected! Confidence: ${(result.confidence * 100).toFixed(0)}%`, {
        style: toastStyle,
        icon: <AlertTriangle className="text-amber-400" />,
      });
    }
  }, [addEntry]);

  // Handle stream capture
  const handleStreamCapture = useCallback((values, times) => {
    setCapturedSignal({ values, times });
    
    // Update current signal for API endpoints panel
    setCurrentSignal({
      data: values,
      source: 'Live Stream Capture',
      isLive: true
    });
    
    setActiveTab('analyze');
    toast.success('Signal captured! Ready for analysis.', {
      style: {
        background: 'rgba(30, 41, 59, 0.95)',
        color: '#fff',
        border: '1px solid rgba(99, 102, 241, 0.3)',
        borderRadius: '12px',
      },
    });
  }, []);

  // Handle stream download - updates stats when CSV is saved
  const handleStreamDownload = useCallback((values, times, filename, streamStats) => {
    // Update current signal for API endpoints panel
    setCurrentSignal({
      data: values,
      source: filename,
      isLive: true
    });
    
    toast.success(`Saved ${filename} (${values.length} samples). Stats updated!`, {
      style: {
        background: 'rgba(30, 41, 59, 0.95)',
        color: '#fff',
        border: '1px solid rgba(16, 185, 129, 0.3)',
        borderRadius: '12px',
      },
      icon: '💾',
    });
  }, []);

  return (
    <div className="min-h-screen p-6 lg:p-8">
      <Toaster position="top-right" />
      
      <div className="max-w-7xl mx-auto">
        <Header />

        {/* API Endpoints Panel */}
        <APIEndpointsPanel 
          signalData={currentSignal.data}
          signalSource={currentSignal.source}
          isLive={currentSignal.isLive}
        />

        {/* KPI Stats Grid with Time Filter */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <div className="w-1 h-5 rounded-sm" style={{ background: PBI_BLUE }} />
            <h3 className="text-sm font-semibold text-white uppercase tracking-wide">Analytics Overview</h3>
            {timeFilter !== 'all' && (
              <span className="px-2 py-0.5 text-[10px] font-medium rounded bg-blue-500/20 text-blue-400">
                Filtered
              </span>
            )}
          </div>
          <TimeFilterDropdown value={timeFilter} onChange={setTimeFilter} />
        </div>

        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-5">
          <KPICard
            icon={BarChart3}
            label="Total Analyses"
            value={filteredStats.totalAnalyses}
            delay={0.1}
          />
          <KPICard
            icon={AlertTriangle}
            label="Faults Detected"
            value={filteredStats.faultsDetected}
            trend={filteredStats.totalAnalyses > 0 ? Math.round((filteredStats.faultsDetected / filteredStats.totalAnalyses) * 100) : 0}
            delay={0.15}
          />
          <KPICard
            icon={Sparkles}
            label="Avg Confidence"
            value={filteredStats.avgConfidence}
            unit="%"
            delay={0.2}
          />
          <KPICard
            icon={Clock}
            label="Session Uptime"
            value={stats.uptime}
            delay={0.25}
          />
        </div>

        {/* Analytics Dashboard Grid - PowerBI Style */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.2 }}
          className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-5"
        >
          <FaultDistributionChart history={filteredHistory} />
          <SeverityDonut history={filteredHistory} />
          <AnalyticsTrendChart history={filteredHistory} />
        </motion.div>

        {/* Architecture Diagram (shown on analyze tab) */}
        {activeTab === 'analyze' && <ArchitectureDiagram />}

        {/* Tab Navigation */}
        <TabNav activeTab={activeTab} onTabChange={setActiveTab} />

        {/* Tab Content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
          >
            {activeTab === 'stream' && (
              <StreamViewer 
                onCapture={handleStreamCapture} 
                onDownload={handleStreamDownload}
              />
            )}
            
            {activeTab === 'analyze' && (
              <UploadAndRun
                onDiagnosisComplete={handleDiagnosisComplete}
                initialSignal={capturedSignal?.values}
              />
            )}
            
            {activeTab === 'history' && (
              <HistoryTable />
            )}
          </motion.div>
        </AnimatePresence>

        {/* Footer */}
        <motion.footer
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="mt-10 pt-6 border-t border-slate-800 text-center"
        >
          <div className="flex items-center justify-center gap-4">
            {['React', 'FastAPI', 'Chart.js', 'TailwindCSS'].map((tech, idx) => (
              <React.Fragment key={tech}>
                {idx > 0 && <span className="text-slate-700">•</span>}
                <span className="text-xs text-slate-600">{tech}</span>
              </React.Fragment>
            ))}
          </div>
        </motion.footer>
      </div>
    </div>
  );
}
