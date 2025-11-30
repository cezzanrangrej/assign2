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

  return (
    <div className="min-h-screen p-6 lg:p-8">
      <Toaster position="top-right" />
      
      <div className="max-w-7xl mx-auto">
        <Header />

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
              <StreamViewer onCapture={handleStreamCapture} />
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
