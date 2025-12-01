/**
 * App.jsx
 * Main application dashboard with professional PowerBI-style design
 * Features: Navigation tabs, KPI metrics, data visualizations, responsive layout
 */
import React, { useState, useCallback, useEffect } from 'react';
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
  Github,
  ExternalLink,
  ChevronRight,
  Sparkles,
  Play,
  FileText,
  Heart,
  List,
  Download,
  Loader2,
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
    { icon: Activity, label: 'Sensor', color: PBI_COLORS[0] },
    { icon: BarChart3, label: 'React UI', color: PBI_COLORS[1] },
    { icon: Server, label: 'FastAPI', color: PBI_COLORS[2] },
    { icon: Cpu, label: 'ML Model', color: PBI_COLORS[3] },
    { icon: AlertTriangle, label: 'Alert', color: PBI_COLORS[4] },
  ];

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
      <div className="flex items-center justify-between">
        {steps.map((step, idx) => (
          <React.Fragment key={step.label}>
            <div className="flex flex-col items-center">
              <div 
                className="w-12 h-12 rounded-md flex items-center justify-center" 
                style={{ background: `${step.color}20`, border: `1px solid ${step.color}40` }}
              >
                <step.icon size={22} style={{ color: step.color }} />
              </div>
              <span className="text-xs text-slate-500 mt-2 font-medium">{step.label}</span>
            </div>
            {idx < steps.length - 1 && (
              <div className="flex-1 h-px bg-slate-700 mx-2" />
            )}
          </React.Fragment>
        ))}
      </div>
      
      {/* Feature badges */}
      <div className="flex flex-wrap gap-2 mt-5 pt-4 border-t border-slate-700/50">
        {['Real-time FFT', 'SSE Streaming', 'Feature Extraction', 'PDF Reports', 'History Tracking'].map((feature) => (
          <span
            key={feature}
            className="px-3 py-1.5 text-xs font-medium rounded bg-slate-800 text-slate-400"
          >
            {feature}
          </span>
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
              <p className="text-xs text-slate-600 mt-1">Contains: Signal plot, FFT, PSD, Features table</p>
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
            <span className="font-mono">http://localhost:8000{endpoint.path}</span>
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
  
  const API_BASE = 'http://localhost:8000';
  
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
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="p-2 rounded-md bg-slate-800 text-slate-400 hover:text-white hover:bg-slate-700 transition-all"
          >
            <Github size={18} />
          </a>
          <div className="h-6 w-px bg-slate-700" />
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

        {/* KPI Stats Grid */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-5">
          <KPICard
            icon={BarChart3}
            label="Total Analyses"
            value={stats.totalAnalyses}
            delay={0.1}
          />
          <KPICard
            icon={AlertTriangle}
            label="Faults Detected"
            value={stats.faultsDetected}
            trend={stats.totalAnalyses > 0 ? Math.round((stats.faultsDetected / stats.totalAnalyses) * 100) : 0}
            delay={0.15}
          />
          <KPICard
            icon={Sparkles}
            label="Avg Confidence"
            value={stats.avgConfidence}
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
          <FaultDistributionChart history={historyData} />
          <SeverityDonut history={historyData} />
          <AnalyticsTrendChart history={historyData} />
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
