# backend/app/main.py
"""
Vibration Fault Detection API
FastAPI backend with SSE streaming, prediction, and PDF report generation
"""
import asyncio
import io
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from scipy.fft import rfft, rfftfreq
from scipy.signal import detrend, windows, welch
from sse_starlette.sse import EventSourceResponse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

app = FastAPI(
    title="Vibration Fault Detection API",
    description="Real-time vibration signal analysis and fault diagnosis",
    version="1.0.0"
)

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
SAMPLE_RATE = 2000  # Hz
MODEL_PATH = Path(__file__).parent.parent / "models" / "demo_model.pkl"

# Fault type labels and descriptions
FAULT_TYPES = {
    "Normal": {"color": "#10B981", "severity": 0, "description": "Machine operating within normal parameters"},
    "Unbalance": {"color": "#F59E0B", "severity": 1, "description": "Rotor mass imbalance detected - schedule maintenance"},
    "Misalignment": {"color": "#F97316", "severity": 2, "description": "Shaft misalignment detected - check coupling"},
    "Bearing Fault": {"color": "#EF4444", "severity": 3, "description": "Bearing degradation detected - immediate attention required"},
    "Looseness": {"color": "#8B5CF6", "severity": 2, "description": "Mechanical looseness detected - check mounting"}
}

# ============== Helper Functions ==============

def moving_average(x: np.ndarray, w: int = 5) -> np.ndarray:
    """Apply moving average smoothing"""
    return np.convolve(x, np.ones(w) / w, mode='same')

def preprocess(signal: List[float]) -> np.ndarray:
    """Preprocess raw signal: smooth, detrend, window, normalize"""
    x = np.array(signal, dtype=np.float64)
    
    # Handle edge cases
    if len(x) < 16:
        raise ValueError("Signal too short (minimum 16 samples required)")
    
    # Remove outliers using IQR method
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    mask = (x >= q1 - 3 * iqr) & (x <= q3 + 3 * iqr)
    x = np.where(mask, x, np.median(x))
    
    # Smoothing
    x = moving_average(x, w=5)
    
    # Detrend (remove DC offset and linear trend)
    x = detrend(x)
    
    # Apply Hanning window to reduce spectral leakage
    win = windows.hann(len(x))
    x = x * win
    
    # Normalize (z-score)
    x = (x - x.mean()) / (np.std(x) + 1e-9)
    
    return x

def compute_fft(x: np.ndarray, sample_rate: int = SAMPLE_RATE) -> tuple:
    """Compute FFT and return frequencies and magnitudes"""
    N = len(x)
    yf = np.abs(rfft(x)) * 2 / N  # Normalize amplitude
    xf = rfftfreq(N, 1 / sample_rate)
    return xf.tolist(), yf.tolist()

def compute_psd(x: np.ndarray, sample_rate: int = SAMPLE_RATE) -> tuple:
    """Compute Power Spectral Density using Welch's method"""
    nperseg = min(256, len(x) // 4)
    if nperseg < 16:
        nperseg = len(x)
    freqs, psd = welch(x, fs=sample_rate, nperseg=nperseg)
    return freqs.tolist(), psd.tolist()

def extract_features(x: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict:
    """Extract time-domain and frequency-domain features"""
    xf, yf = compute_fft(x, sample_rate)
    xf = np.array(xf)
    yf = np.array(yf)
    
    # Time-domain features
    rms = np.sqrt(np.mean(x ** 2))
    peak = np.max(np.abs(x))
    crest_factor = peak / (rms + 1e-9)
    
    # Kurtosis (4th moment - indicates impulsiveness)
    mean_x = x.mean()
    std_x = np.std(x) + 1e-9
    kurtosis = np.mean(((x - mean_x) / std_x) ** 4)
    
    # Skewness (3rd moment - indicates asymmetry)
    skewness = np.mean(((x - mean_x) / std_x) ** 3)
    
    # Shape factor
    shape_factor = rms / (np.mean(np.abs(x)) + 1e-9)
    
    # Frequency-domain features
    yf_norm = yf / (np.sum(yf) + 1e-12)
    
    # Dominant frequency
    dominant_freq = xf[np.argmax(yf)] if len(yf) > 0 else 0
    
    # Spectral centroid (center of mass of spectrum)
    spectral_centroid = np.sum(xf * yf) / (np.sum(yf) + 1e-12)
    
    # Spectral entropy (measure of spectral flatness)
    spectral_entropy = -np.sum(yf_norm * np.log(yf_norm + 1e-12))
    
    # Spectral kurtosis
    spectral_mean = spectral_centroid
    spectral_std = np.sqrt(np.sum(((xf - spectral_mean) ** 2) * yf_norm))
    spectral_kurtosis = np.sum((((xf - spectral_mean) / (spectral_std + 1e-9)) ** 4) * yf_norm)
    
    # Band energies (for bearing fault detection)
    low_freq_energy = np.sum(yf[(xf >= 10) & (xf < 100)] ** 2)
    mid_freq_energy = np.sum(yf[(xf >= 100) & (xf < 500)] ** 2)
    high_freq_energy = np.sum(yf[(xf >= 500)] ** 2)
    total_energy = low_freq_energy + mid_freq_energy + high_freq_energy + 1e-12
    
    return {
        "rms": float(rms),
        "peak": float(peak),
        "crest_factor": float(crest_factor),
        "kurtosis": float(kurtosis),
        "skewness": float(skewness),
        "shape_factor": float(shape_factor),
        "dominant_freq": float(dominant_freq),
        "spectral_centroid": float(spectral_centroid),
        "spectral_entropy": float(spectral_entropy),
        "spectral_kurtosis": float(spectral_kurtosis),
        "low_freq_ratio": float(low_freq_energy / total_energy),
        "mid_freq_ratio": float(mid_freq_energy / total_energy),
        "high_freq_ratio": float(high_freq_energy / total_energy)
    }

def demo_predict(features: dict) -> tuple:
    """Rule-based classifier for demo (when model not available)"""
    k = features['kurtosis']
    crest = features['crest_factor']
    dominant = features['dominant_freq']
    high_ratio = features['high_freq_ratio']
    spectral_entropy = features['spectral_entropy']
    
    # Decision rules based on vibration analysis theory
    if k > 8 and high_ratio > 0.3:
        return "Bearing Fault", 0.89 + np.random.uniform(0, 0.08)
    elif k > 5 and crest > 4:
        return "Looseness", 0.78 + np.random.uniform(0, 0.12)
    elif 40 < dominant < 80 and spectral_entropy < 3:
        return "Unbalance", 0.82 + np.random.uniform(0, 0.10)
    elif 2 < k < 5 and 100 < dominant < 300:
        return "Misalignment", 0.76 + np.random.uniform(0, 0.14)
    else:
        return "Normal", 0.91 + np.random.uniform(0, 0.07)

# Load model if available
try:
    model = joblib.load(MODEL_PATH)
    print(f"✓ Loaded model from {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"⚠ Model not found at {MODEL_PATH}, using rule-based prediction")

# ============== Pydantic Models ==============

class SignalPayload(BaseModel):
    signal: List[float]
    sample_rate: Optional[int] = SAMPLE_RATE

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    severity: int
    description: str
    color: str
    features: dict
    fft: dict
    timestamp: str

# ============== API Endpoints ==============

@app.get("/")
async def root():
    return {
        "message": "Vibration Fault Detection API",
        "version": "1.0.0",
        "endpoints": ["/stream-signal", "/predict", "/diagnostic-report", "/health"]
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/stream-signal")
async def stream_signal():
    """
    Server-Sent Events streaming synthetic vibration signal at ~20 Hz
    Simulates real sensor data with varying fault conditions
    """
    async def generator():
        t = 0.0
        dt = 0.05  # 20 samples per second for smooth visualization
        fault_mode = 0  # 0=normal, 1=unbalance, 2=bearing
        mode_duration = 0
        
        while True:
            # Periodically switch fault modes for demo
            mode_duration += 1
            if mode_duration > 200:  # Switch every ~10 seconds
                fault_mode = (fault_mode + 1) % 3
                mode_duration = 0
            
            # Base signal: 60 Hz fundamental (typical motor speed)
            val = 0.5 * np.sin(2 * np.pi * 60 * t)
            
            # Add harmonics based on fault mode
            if fault_mode == 0:  # Normal operation
                val += 0.1 * np.sin(2 * np.pi * 120 * t)  # 2x harmonic
                val += 0.05 * np.random.randn()
                status = "normal"
            elif fault_mode == 1:  # Unbalance
                val += 0.4 * np.sin(2 * np.pi * 60 * t + 0.5)  # Strong 1x
                val += 0.15 * np.sin(2 * np.pi * 120 * t)
                val += 0.08 * np.random.randn()
                status = "unbalance"
            else:  # Bearing fault - high frequency impacts
                val += 0.2 * np.sin(2 * np.pi * 120 * t)
                val += 0.3 * np.sin(2 * np.pi * 450 * t)  # BPFO-like frequency
                # Random impulses
                if np.random.random() > 0.92:
                    val += 1.5 * np.random.choice([-1, 1])
                val += 0.12 * np.random.randn()
                status = "bearing"
            
            event_data = json.dumps({
                "t": round(t, 4),
                "value": round(float(val), 6),
                "status": status,
                "timestamp": datetime.now().isoformat()
            })
            
            yield {"event": "signal", "data": event_data}
            t += dt
            await asyncio.sleep(dt)
    
    return EventSourceResponse(generator())

@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: SignalPayload):
    """
    Analyze vibration signal and predict fault type
    Returns prediction with confidence, features, and FFT data
    """
    try:
        # Preprocess signal
        x = preprocess(payload.signal)
        
        # Extract features
        features = extract_features(x, payload.sample_rate)
        
        # Compute FFT for visualization
        freqs, mags = compute_fft(x, payload.sample_rate)
        
        # Make prediction
        if model is not None:
            feature_vector = np.array([list(features.values())[:10]])  # Use first 10 features
            pred_label = model.predict(feature_vector)[0]
            proba = model.predict_proba(feature_vector)[0]
            conf = float(max(proba))
        else:
            pred_label, conf = demo_predict(features)
        
        fault_info = FAULT_TYPES.get(pred_label, FAULT_TYPES["Normal"])
        
        return PredictionResponse(
            prediction=pred_label,
            confidence=round(conf, 3),
            severity=fault_info["severity"],
            description=fault_info["description"],
            color=fault_info["color"],
            features=features,
            fft={"frequencies": freqs[:500], "magnitudes": mags[:500]},  # Limit for response size
            timestamp=datetime.now().isoformat()
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/diagnostic-report")
async def diagnostic_report(payload: SignalPayload):
    """
    Generate comprehensive PDF diagnostic report
    Includes raw signal, FFT, PSD, features, and prediction
    """
    try:
        signal = np.array(payload.signal)
        x = preprocess(payload.signal)
        
        # Compute all analyses
        freqs, mags = compute_fft(x, payload.sample_rate)
        psd_freqs, psd_vals = compute_psd(x, payload.sample_rate)
        features = extract_features(x, payload.sample_rate)
        
        if model is not None:
            feature_vector = np.array([list(features.values())[:10]])
            pred_label = model.predict(feature_vector)[0]
            conf = float(max(model.predict_proba(feature_vector)[0]))
        else:
            pred_label, conf = demo_predict(features)
        
        fault_info = FAULT_TYPES.get(pred_label, FAULT_TYPES["Normal"])
        
        # Generate PDF
        buf = io.BytesIO()
        with PdfPages(buf) as pdf:
            # Page 1: Header and Summary
            fig = plt.figure(figsize=(11, 8.5))
            fig.suptitle("Vibration Diagnostic Report", fontsize=20, fontweight='bold', y=0.98)
            
            # Summary box
            ax = fig.add_axes([0.1, 0.7, 0.8, 0.2])
            ax.axis('off')
            summary_text = f"""
            Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            Sample Rate: {payload.sample_rate} Hz
            Signal Length: {len(signal)} samples ({len(signal)/payload.sample_rate:.2f} seconds)
            
            DIAGNOSIS: {pred_label}
            Confidence: {conf*100:.1f}%
            Severity: {'●' * fault_info['severity']}{'○' * (3 - fault_info['severity'])}
            
            {fault_info['description']}
            """
            ax.text(0.5, 0.5, summary_text, transform=ax.transAxes, fontsize=12,
                   verticalalignment='center', horizontalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
            
            # Raw signal plot
            ax1 = fig.add_axes([0.1, 0.35, 0.8, 0.28])
            t_axis = np.arange(len(signal)) / payload.sample_rate
            ax1.plot(t_axis, signal, 'b-', linewidth=0.5, alpha=0.8)
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Amplitude')
            ax1.set_title('Raw Vibration Signal', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 2: Frequency Analysis
            fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
            fig.suptitle("Frequency Domain Analysis", fontsize=16, fontweight='bold')
            
            # FFT plot
            ax1 = axes[0]
            ax1.semilogy(freqs[:len(freqs)//2], np.array(mags[:len(mags)//2]) + 1e-12, 'b-', linewidth=0.8)
            ax1.set_xlabel('Frequency (Hz)')
            ax1.set_ylabel('Magnitude (log)')
            ax1.set_title('FFT Spectrum')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, min(1000, payload.sample_rate / 2))
            
            # PSD plot
            ax2 = axes[1]
            ax2.semilogy(psd_freqs, np.array(psd_vals) + 1e-12, 'r-', linewidth=0.8)
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('Power/Frequency (dB/Hz)')
            ax2.set_title('Power Spectral Density (Welch)')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, min(1000, payload.sample_rate / 2))
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 3: Feature Summary
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('off')
            
            # Create feature table
            feature_names = list(features.keys())
            feature_values = [f"{v:.4f}" for v in features.values()]
            
            table_data = [[name, value] for name, value in zip(feature_names, feature_values)]
            table = ax.table(cellText=table_data, colLabels=['Feature', 'Value'],
                           loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1.2, 1.8)
            
            ax.set_title('Extracted Features', fontsize=16, fontweight='bold', pad=20)
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        buf.seek(0)
        filename = f"diagnostic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        return StreamingResponse(
            buf,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@app.get("/fault-types")
async def get_fault_types():
    """Return available fault types and their metadata"""
    return FAULT_TYPES

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
