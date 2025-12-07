/**
 * FeatureContributionCard.jsx
 * SHAP-based feature contribution visualization
 * Shows which features contributed most to the model's prediction
 */
import React from 'react';
import { motion } from 'framer-motion';
import { 
  Lightbulb, 
  TrendingUp, 
  TrendingDown,
  Info,
  Sparkles
} from 'lucide-react';

// PowerBI color palette
const PBI_BLUE = '#118DFF';

// Feature descriptions for tooltips
const FEATURE_DESCRIPTIONS = {
  rms: "Root Mean Square - Overall vibration amplitude",
  peak: "Peak amplitude value",
  crest_factor: "Peak-to-RMS ratio - Indicates impulsiveness",
  kurtosis: "Signal peakedness - High values indicate impacts",
  skewness: "Signal asymmetry",
  shape_factor: "RMS-to-mean-absolute ratio",
  dominant_freq: "Primary frequency component (Hz)",
  spectral_centroid: "Center of spectral mass",
  spectral_entropy: "Spectral flatness measure",
  spectral_kurtosis: "Frequency distribution peakedness",
  low_freq_ratio: "Low frequency band energy ratio",
  mid_freq_ratio: "Mid frequency band energy ratio",
  high_freq_ratio: "High frequency band energy ratio"
};

/**
 * Single horizontal bar representing a feature's contribution
 */
function ContributionBar({ feature, maxAbsValue, index }) {
  const { name, shap_value, abs_importance, direction } = feature;
  
  // Calculate bar width as percentage of max
  const barWidth = Math.min((abs_importance / (maxAbsValue || 1)) * 100, 100);
  
  // Color based on direction
  const isPositive = direction === 'positive' || shap_value >= 0;
  const barColor = isPositive ? '#10B981' : '#EF4444'; // green for positive, red for negative
  const bgColor = isPositive ? 'bg-green-500/10' : 'bg-red-500/10';
  const textColor = isPositive ? 'text-green-400' : 'text-red-400';
  
  // Human-readable name
  const displayName = name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  const description = FEATURE_DESCRIPTIONS[name] || name;
  
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3, delay: index * 0.1 }}
      className="group"
    >
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2">
          {isPositive ? (
            <TrendingUp size={12} className="text-green-400" />
          ) : (
            <TrendingDown size={12} className="text-red-400" />
          )}
          <span className="text-xs font-medium text-slate-300">{displayName}</span>
          
          {/* Tooltip on hover */}
          <div className="relative">
            <Info size={10} className="text-slate-600 cursor-help" />
            <div className="absolute left-full ml-2 top-1/2 -translate-y-1/2 z-50 hidden group-hover:block">
              <div className="bg-slate-900 border border-slate-700 rounded-md px-2 py-1 text-[10px] text-slate-400 whitespace-nowrap shadow-lg">
                {description}
              </div>
            </div>
          </div>
        </div>
        
        <span className={`text-xs font-mono ${textColor}`}>
          {shap_value >= 0 ? '+' : ''}{shap_value.toFixed(4)}
        </span>
      </div>
      
      {/* Bar */}
      <div className={`h-2 rounded-full ${bgColor} overflow-hidden`}>
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${barWidth}%` }}
          transition={{ duration: 0.5, delay: index * 0.1 + 0.2 }}
          className="h-full rounded-full"
          style={{ backgroundColor: barColor }}
        />
      </div>
    </motion.div>
  );
}

/**
 * Legend component explaining colors
 */
function ContributionLegend() {
  return (
    <div className="flex items-center gap-4 text-[10px] text-slate-500">
      <div className="flex items-center gap-1">
        <div className="w-2 h-2 rounded-full bg-green-500" />
        <span>Supports prediction</span>
      </div>
      <div className="flex items-center gap-1">
        <div className="w-2 h-2 rounded-full bg-red-500" />
        <span>Opposes prediction</span>
      </div>
    </div>
  );
}

/**
 * Main FeatureContributionCard component
 * Renders a card with horizontal bar chart showing feature contributions
 */
export default function FeatureContributionCard({ 
  explanation, 
  predictedClass,
  className = '' 
}) {
  // Early return if no explanation data
  if (!explanation || !explanation.top_features || explanation.top_features.length === 0) {
    return null;
  }
  
  const { model, method, top_features, note } = explanation;
  
  // Find max absolute value for scaling bars
  const maxAbsValue = Math.max(...top_features.map(f => f.abs_importance));
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className={`glass-card p-5 ${className}`}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="w-1 h-5 rounded-sm" style={{ background: '#E66C37' }} />
          <div>
            <h3 className="text-sm font-semibold text-white uppercase tracking-wide">
              Why this diagnosis?
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Top contributing features from the {model || 'ML'} model
            </p>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          <Lightbulb size={14} className="text-amber-400" />
          <span className="text-xs text-slate-500">{method || 'SHAP Analysis'}</span>
        </div>
      </div>
      
      {/* Feature bars */}
      <div className="space-y-3 mb-4">
        {top_features.map((feature, index) => (
          <ContributionBar
            key={feature.name}
            feature={feature}
            maxAbsValue={maxAbsValue}
            index={index}
          />
        ))}
      </div>
      
      {/* Legend */}
      <div className="pt-3 border-t border-slate-700/50 flex items-center justify-between">
        <ContributionLegend />
        
        {predictedClass && (
          <span className="text-xs text-slate-500">
            Predicting: <span className="text-white font-medium">{predictedClass}</span>
          </span>
        )}
      </div>
      
      {/* Note (for fallback mode) */}
      {note && (
        <div className="mt-3 p-2 rounded bg-amber-500/10 border border-amber-500/20">
          <p className="text-[10px] text-amber-400 flex items-center gap-1">
            <Sparkles size={10} />
            {note}
          </p>
        </div>
      )}
    </motion.div>
  );
}

/**
 * Compact version for embedding in other cards
 */
export function FeatureContributionCompact({ explanation, className = '' }) {
  if (!explanation || !explanation.top_features || explanation.top_features.length === 0) {
    return null;
  }
  
  const { top_features } = explanation;
  const maxAbsValue = Math.max(...top_features.map(f => f.abs_importance));
  
  return (
    <div className={`space-y-2 ${className}`}>
      <div className="flex items-center gap-2 mb-2">
        <Lightbulb size={12} className="text-amber-400" />
        <span className="text-xs text-slate-400 font-medium">Key Contributors</span>
      </div>
      
      {top_features.slice(0, 3).map((feature, index) => {
        const isPositive = feature.direction === 'positive' || feature.shap_value >= 0;
        const barWidth = Math.min((feature.abs_importance / (maxAbsValue || 1)) * 100, 100);
        const barColor = isPositive ? '#10B981' : '#EF4444';
        const displayName = feature.name.replace(/_/g, ' ');
        
        return (
          <div key={feature.name} className="flex items-center gap-2">
            <span className="text-[10px] text-slate-500 w-20 truncate">{displayName}</span>
            <div className="flex-1 h-1.5 rounded-full bg-slate-800 overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${barWidth}%` }}
                transition={{ duration: 0.4, delay: index * 0.1 }}
                className="h-full rounded-full"
                style={{ backgroundColor: barColor }}
              />
            </div>
            <span className={`text-[10px] font-mono ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
              {feature.shap_value >= 0 ? '+' : ''}{feature.shap_value.toFixed(2)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
