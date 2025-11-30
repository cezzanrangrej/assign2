/**
 * SeverityDonut.jsx
 * PowerBI-style donut chart showing severity breakdown
 */
import React, { useMemo } from 'react';
import { Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
} from 'chart.js';
import { motion } from 'framer-motion';

ChartJS.register(ArcElement, Tooltip);

const SEVERITY_COLORS = {
  0: '#22c55e', // Healthy - Green
  1: '#f59e0b', // Low - Amber
  2: '#f97316', // Medium - Orange
  3: '#ef4444', // High - Red
};

const SEVERITY_LABELS = {
  0: 'Healthy',
  1: 'Low',
  2: 'Medium',
  3: 'High',
};

export default function SeverityDonut({ history = [], className = '' }) {
  // Calculate severity distribution
  const distribution = useMemo(() => {
    const counts = { 0: 0, 1: 0, 2: 0, 3: 0 };
    history.forEach(entry => {
      const sev = entry.severity ?? 0;
      counts[sev] = (counts[sev] || 0) + 1;
    });
    return counts;
  }, [history]);

  const total = Object.values(distribution).reduce((a, b) => a + b, 0);
  const healthyPercent = total > 0 ? Math.round((distribution[0] / total) * 100) : 0;

  const chartData = {
    labels: Object.keys(distribution).map(k => SEVERITY_LABELS[k]),
    datasets: [{
      data: Object.values(distribution),
      backgroundColor: Object.keys(distribution).map(k => SEVERITY_COLORS[k]),
      borderWidth: 0,
      cutout: '70%',
    }],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#1a1a24',
        titleColor: '#e5e7eb',
        bodyColor: '#9ca3af',
        borderColor: '#2a2a38',
        borderWidth: 1,
        padding: 10,
        cornerRadius: 4,
        callbacks: {
          label: (ctx) => ` ${ctx.raw} (${total > 0 ? Math.round((ctx.raw / total) * 100) : 0}%)`,
        },
      },
    },
  };

  if (history.length === 0) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className={`card p-5 ${className}`}
      >
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-5 rounded-sm" style={{ background: '#6B007B' }} />
          <h3 className="text-sm font-semibold text-white uppercase tracking-wide">Severity Breakdown</h3>
        </div>
        <div className="h-40 flex items-center justify-center text-slate-500 text-sm">
          No data available
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`card p-5 ${className}`}
    >
      <div className="flex items-center gap-2 mb-4">
        <div className="w-1 h-5 rounded-sm" style={{ background: '#6B007B' }} />
        <h3 className="text-sm font-semibold text-white uppercase tracking-wide">Severity Breakdown</h3>
      </div>
      
      <div className="flex items-center gap-4">
        {/* Donut */}
        <div className="relative" style={{ width: 120, height: 120 }}>
          <Doughnut data={chartData} options={options} />
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-2xl font-semibold text-white">{healthyPercent}%</span>
            <span className="text-xs text-slate-500">Healthy</span>
          </div>
        </div>
        
        {/* Legend */}
        <div className="flex-1 space-y-2">
          {Object.entries(distribution).map(([sev, count]) => (
            <div key={sev} className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div 
                  className="w-3 h-3 rounded-sm" 
                  style={{ background: SEVERITY_COLORS[sev] }}
                />
                <span className="text-xs text-slate-400">{SEVERITY_LABELS[sev]}</span>
              </div>
              <span className="text-xs font-mono text-white">{count}</span>
            </div>
          ))}
        </div>
      </div>
    </motion.div>
  );
}
