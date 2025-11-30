/**
 * SignalChart.jsx
 * Interactive time-series visualization with pan/zoom, crosshair, and annotations
 * Features: Real-time updates, gradient fills, tooltips, zoom controls
 */
import React, { useRef, useEffect, useMemo } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import zoomPlugin from 'chartjs-plugin-zoom';
import annotationPlugin from 'chartjs-plugin-annotation';
import { ZoomIn, ZoomOut, RotateCcw, Maximize2 } from 'lucide-react';
import { motion } from 'framer-motion';

// Register Chart.js plugins
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  zoomPlugin,
  annotationPlugin
);

export default function SignalChart({
  data = [],
  labels = [],
  title = 'Vibration Signal',
  color = '#118DFF',
  showControls = true,
  height = 280,
  annotations = [],
  onZoomChange,
  className = '',
}) {
  const chartRef = useRef(null);

  // Generate gradient for the line fill
  const getGradient = (ctx, chartArea) => {
    if (!chartArea) return color;
    const gradient = ctx.createLinearGradient(0, chartArea.bottom, 0, chartArea.top);
    gradient.addColorStop(0, 'rgba(17, 141, 255, 0)');
    gradient.addColorStop(0.5, 'rgba(17, 141, 255, 0.08)');
    gradient.addColorStop(1, 'rgba(17, 141, 255, 0.2)');
    return gradient;
  };

  // Chart data configuration
  const chartData = useMemo(() => ({
    labels: labels.length > 0 ? labels : data.map((_, i) => i),
    datasets: [
      {
        label: 'Amplitude',
        data: data,
        borderColor: color,
        backgroundColor: (context) => {
          const chart = context.chart;
          const { ctx, chartArea } = chart;
          return getGradient(ctx, chartArea);
        },
        fill: true,
        tension: 0.2,
        pointRadius: 0,
        pointHoverRadius: 6,
        pointHoverBackgroundColor: color,
        pointHoverBorderColor: '#fff',
        pointHoverBorderWidth: 2,
        borderWidth: 2,
      },
    ],
  }), [data, labels, color]);

  // Chart options with zoom/pan
  const options = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: false,
      },
      tooltip: {
        enabled: true,
        backgroundColor: '#1a1a24',
        titleColor: '#e5e7eb',
        bodyColor: '#9ca3af',
        borderColor: 'rgba(17, 141, 255, 0.3)',
        borderWidth: 1,
        padding: 12,
        cornerRadius: 4,
        displayColors: false,
        callbacks: {
          title: (items) => `Time: ${items[0]?.label ?? ''}`,
          label: (item) => `Amplitude: ${item.raw?.toFixed(4) ?? item.raw}`,
        },
      },
      zoom: {
        pan: {
          enabled: true,
          mode: 'x',
          modifierKey: 'ctrl',
        },
        zoom: {
          wheel: {
            enabled: true,
            speed: 0.1,
          },
          pinch: {
            enabled: true,
          },
          drag: {
            enabled: true,
            backgroundColor: 'rgba(17, 141, 255, 0.1)',
            borderColor: 'rgba(17, 141, 255, 0.5)',
            borderWidth: 1,
          },
          mode: 'x',
          onZoomComplete: ({ chart }) => {
            onZoomChange?.({
              min: chart.scales.x.min,
              max: chart.scales.x.max,
            });
          },
        },
      },
      annotation: {
        annotations: annotations.reduce((acc, ann, idx) => {
          acc[`annotation${idx}`] = {
            type: 'line',
            xMin: ann.x,
            xMax: ann.x,
            borderColor: ann.color || '#f59e0b',
            borderWidth: 2,
            borderDash: [5, 5],
            label: {
              display: true,
              content: ann.label,
              position: 'start',
              backgroundColor: 'rgba(15, 23, 42, 0.9)',
              color: '#fff',
              font: { size: 11 },
              padding: 4,
            },
          };
          return acc;
        }, {}),
      },
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Time (samples)',
          color: '#6b7280',
          font: { size: 11, weight: '500' },
        },
        grid: {
          color: 'rgba(55, 65, 81, 0.3)',
          drawBorder: false,
        },
        ticks: {
          color: '#6b7280',
          font: { size: 10 },
          maxTicksLimit: 10,
        },
      },
      y: {
        display: true,
        title: {
          display: true,
          text: 'Amplitude',
          color: '#6b7280',
          font: { size: 11, weight: '500' },
        },
        grid: {
          color: 'rgba(55, 65, 81, 0.3)',
          drawBorder: false,
        },
        ticks: {
          color: '#6b7280',
          font: { size: 10 },
        },
      },
    },
    animation: {
      duration: 300,
    },
  }), [annotations, onZoomChange]);

  // Zoom control handlers
  const handleZoomIn = () => {
    chartRef.current?.zoom(1.2);
  };

  const handleZoomOut = () => {
    chartRef.current?.zoom(0.8);
  };

  const handleResetZoom = () => {
    chartRef.current?.resetZoom();
  };

  const handleFitData = () => {
    chartRef.current?.resetZoom();
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={`glass-card p-5 ${className}`}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="w-1 h-5 rounded-sm" style={{ background: '#118DFF' }} />
          <h3 className="text-sm font-semibold text-white uppercase tracking-wide">{title}</h3>
          <span className="px-2 py-0.5 text-xs font-medium rounded" style={{ background: 'rgba(17, 141, 255, 0.15)', color: '#118DFF' }}>
            {data.length} samples
          </span>
        </div>

        {/* Zoom Controls */}
        {showControls && (
          <div className="flex items-center gap-1">
            <button
              onClick={handleZoomIn}
              className="p-2 rounded-lg hover:bg-white/5 text-slate-400 hover:text-white transition-colors"
              title="Zoom In"
            >
              <ZoomIn size={18} />
            </button>
            <button
              onClick={handleZoomOut}
              className="p-2 rounded-lg hover:bg-white/5 text-slate-400 hover:text-white transition-colors"
              title="Zoom Out"
            >
              <ZoomOut size={18} />
            </button>
            <button
              onClick={handleResetZoom}
              className="p-2 rounded-lg hover:bg-white/5 text-slate-400 hover:text-white transition-colors"
              title="Reset Zoom"
            >
              <RotateCcw size={18} />
            </button>
            <button
              onClick={handleFitData}
              className="p-2 rounded-lg hover:bg-white/5 text-slate-400 hover:text-white transition-colors"
              title="Fit to Data"
            >
              <Maximize2 size={18} />
            </button>
          </div>
        )}
      </div>

      {/* Chart */}
      <div className="chart-container" style={{ height }}>
        {data.length > 0 ? (
          <Line ref={chartRef} data={chartData} options={options} />
        ) : (
          <div className="flex items-center justify-center h-full">
            <p className="text-slate-500">No data available</p>
          </div>
        )}
      </div>

      {/* Help text */}
      <p className="mt-3 text-xs text-slate-500 text-center">
        Drag to select region • Scroll to zoom • Ctrl+drag to pan
      </p>
    </motion.div>
  );
}
