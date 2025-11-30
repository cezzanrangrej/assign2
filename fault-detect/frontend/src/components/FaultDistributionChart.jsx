/**
 * FaultDistributionChart.jsx
 * PowerBI-style bar chart showing fault type distribution
 */
import React, { useMemo } from 'react';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
} from 'chart.js';
import { motion } from 'framer-motion';

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip);

// PowerBI color palette
const PBI_COLORS = ['#118DFF', '#12239E', '#E66C37', '#6B007B', '#E044A7'];

export default function FaultDistributionChart({ history = [], className = '' }) {
  // Calculate fault distribution
  const distribution = useMemo(() => {
    const counts = {};
    history.forEach(entry => {
      const pred = entry.prediction || 'Unknown';
      counts[pred] = (counts[pred] || 0) + 1;
    });
    
    const sorted = Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);
    
    return {
      labels: sorted.map(([label]) => label),
      values: sorted.map(([, count]) => count),
    };
  }, [history]);

  const chartData = {
    labels: distribution.labels,
    datasets: [{
      data: distribution.values,
      backgroundColor: PBI_COLORS.slice(0, distribution.labels.length),
      borderRadius: 3,
      barThickness: 28,
    }],
  };

  const options = {
    indexAxis: 'y',
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
        displayColors: true,
        callbacks: {
          label: (ctx) => ` ${ctx.raw} analyses`,
        },
      },
    },
    scales: {
      x: {
        grid: { color: 'rgba(55, 65, 81, 0.3)', drawBorder: false },
        ticks: { color: '#6b7280', font: { size: 10 } },
        beginAtZero: true,
      },
      y: {
        grid: { display: false },
        ticks: { color: '#e5e7eb', font: { size: 11, weight: '500' } },
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
          <div className="w-1 h-5 rounded-sm" style={{ background: '#118DFF' }} />
          <h3 className="text-sm font-semibold text-white uppercase tracking-wide">Fault Distribution</h3>
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
        <div className="w-1 h-5 rounded-sm" style={{ background: '#118DFF' }} />
        <h3 className="text-sm font-semibold text-white uppercase tracking-wide">Fault Distribution</h3>
      </div>
      <div style={{ height: 160 }}>
        <Bar data={chartData} options={options} />
      </div>
    </motion.div>
  );
}
