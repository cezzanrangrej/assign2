/**
 * AnalyticsTrendChart.jsx
 * PowerBI-style area chart showing analysis trend over time
 */
import React, { useMemo } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
} from 'chart.js';
import { motion } from 'framer-motion';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Filler, Tooltip);

export default function AnalyticsTrendChart({ history = [], className = '' }) {
  // Group analyses by date
  const trendData = useMemo(() => {
    const byDate = {};
    history.forEach(entry => {
      const date = new Date(entry.timestamp).toLocaleDateString('en-US', { 
        month: 'short', 
        day: 'numeric' 
      });
      byDate[date] = (byDate[date] || 0) + 1;
    });
    
    // Get last 7 days
    const dates = Object.keys(byDate).slice(-7);
    const counts = dates.map(d => byDate[d]);
    
    return { dates, counts };
  }, [history]);

  const chartData = {
    labels: trendData.dates,
    datasets: [{
      data: trendData.counts,
      borderColor: '#118DFF',
      backgroundColor: 'rgba(17, 141, 255, 0.1)',
      fill: true,
      tension: 0.4,
      pointRadius: 4,
      pointBackgroundColor: '#118DFF',
      pointBorderColor: '#0d0d12',
      pointBorderWidth: 2,
      borderWidth: 2,
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
          label: (ctx) => ` ${ctx.raw} analyses`,
        },
      },
    },
    scales: {
      x: {
        grid: { display: false },
        ticks: { color: '#6b7280', font: { size: 10 } },
      },
      y: {
        grid: { color: 'rgba(55, 65, 81, 0.3)', drawBorder: false },
        ticks: { 
          color: '#6b7280', 
          font: { size: 10 },
          stepSize: 1,
        },
        beginAtZero: true,
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
          <div className="w-1 h-5 rounded-sm" style={{ background: '#E66C37' }} />
          <h3 className="text-sm font-semibold text-white uppercase tracking-wide">Analysis Trend</h3>
        </div>
        <div className="h-32 flex items-center justify-center text-slate-500 text-sm">
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
        <div className="w-1 h-5 rounded-sm" style={{ background: '#E66C37' }} />
        <h3 className="text-sm font-semibold text-white uppercase tracking-wide">Analysis Trend</h3>
      </div>
      <div style={{ height: 120 }}>
        <Line data={chartData} options={options} />
      </div>
    </motion.div>
  );
}
