# 🔧 VibrationAI - Intelligent Fault Detection System

<div align="center">

![VibrationAI Banner](https://img.shields.io/badge/VibrationAI-Predictive%20Maintenance-6366f1?style=for-the-badge&logo=activity&logoColor=white)

**Real-time vibration signal analysis and fault diagnosis using machine learning**

[![React](https://img.shields.io/badge/React-18.2-61DAFB?style=flat-square&logo=react)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![TailwindCSS](https://img.shields.io/badge/Tailwind-3.3-38B2AC?style=flat-square&logo=tailwind-css)](https://tailwindcss.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org/)

</div>

---

## 🌟 Features

### Frontend (React + Vite)
- **🎯 Interactive Signal Visualization** - Pan, zoom, and explore vibration data with Chart.js
- **📊 Real-time FFT Analysis** - Client-side FFT using fft.js for instant spectral preview
- **📡 Live Streaming** - Server-Sent Events (SSE) for real-time signal monitoring
- **🎨 Modern Glassmorphism UI** - Beautiful dark theme with animations
- **📱 Responsive Design** - Works on desktop and tablet devices
- **📜 Diagnosis History** - LocalStorage-backed history with export to CSV

### Backend (FastAPI)
- **🚀 RESTful API** - Clean endpoints for prediction and report generation
- **📡 SSE Streaming** - Real-time synthetic signal streaming at 20Hz
- **🧠 ML Prediction** - RandomForest classifier with 95%+ accuracy
- **📄 PDF Reports** - Comprehensive diagnostic reports with plots

### Machine Learning
- **5 Fault Types**: Normal, Unbalance, Misalignment, Bearing Fault, Looseness
- **12 Features**: RMS, Kurtosis, Crest Factor, Spectral Entropy, etc.
- **Cross-validated**: 5-fold CV with high accuracy

---

## 📁 Project Structure

```
fault-detect/
├── 📂 backend/
│   ├── 📂 app/
│   │   └── main.py           # FastAPI server
│   ├── 📂 models/
│   │   └── demo_model.pkl    # Trained ML model
│   └── requirements.txt
│
├── 📂 frontend/
│   ├── 📂 src/
│   │   ├── App.jsx           # Main dashboard
│   │   └── 📂 components/
│   │       ├── SignalChart.jsx    # Time-series chart
│   │       ├── FFTChart.jsx       # Frequency spectrum
│   │       ├── StreamViewer.jsx   # Live SSE stream
│   │       ├── UploadAndRun.jsx   # Analysis interface
│   │       └── HistoryTable.jsx   # Diagnosis history
│   ├── package.json
│   └── tailwind.config.js
│
├── 📂 notebooks/
│   └── 01_signal_processing_and_demo.ipynb
│
├── 📂 samples/               # Sample CSV files
│   ├── sample_normal.csv
│   ├── sample_unbalance.csv
│   ├── sample_misalignment.csv
│   ├── sample_bearing_fault.csv
│   └── sample_looseness.csv
│
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- npm or yarn

### 1️⃣ Backend Setup

```powershell
# Navigate to backend
cd fault-detect/backend

# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

### 2️⃣ Frontend Setup

```powershell
# Navigate to frontend
cd fault-detect/frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The app will open at `http://localhost:5173`

### 3️⃣ Generate ML Model (Optional)

```powershell
# Navigate to notebooks
cd fault-detect/notebooks

# Run Jupyter
jupyter notebook 01_signal_processing_and_demo.ipynb
```

Run all cells to generate `demo_model.pkl` and sample CSV files.

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info and available endpoints |
| `/health` | GET | Health check and model status |
| `/stream-signal` | GET | SSE stream of synthetic vibration data |
| `/predict` | POST | Analyze signal and return diagnosis |
| `/diagnostic-report` | POST | Generate PDF report |
| `/fault-types` | GET | Available fault types and metadata |

### Example: Predict Fault

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"signal": [0.1, 0.2, 0.3, ...], "sample_rate": 2000}'
```

Response:
```json
{
  "prediction": "Bearing Fault",
  "confidence": 0.923,
  "severity": 3,
  "description": "Bearing degradation detected - immediate attention required",
  "features": {...},
  "fft": {"frequencies": [...], "magnitudes": [...]}
}
```

---

## 🎮 Demo Sequence

1. **Landing Dashboard** - View stats and architecture diagram
2. **Live Stream Tab** - Watch real-time signal streaming with fault simulation
3. **Analyze Tab** - Upload sample CSV or capture from stream
4. **Run Diagnosis** - See prediction with confidence gauge
5. **Download PDF** - Generate comprehensive report
6. **History Tab** - View past diagnoses with sparkline previews

---

## 🛠️ Tech Stack

### Frontend
- **React 18** - UI library
- **Vite** - Build tool
- **Chart.js** - Interactive charts
- **fft.js** - Client-side FFT
- **Framer Motion** - Animations
- **TailwindCSS** - Styling
- **Lucide React** - Icons

### Backend
- **FastAPI** - API framework
- **sse-starlette** - Server-Sent Events
- **scikit-learn** - ML model
- **scipy** - Signal processing
- **matplotlib** - PDF report charts

---

## 📈 Model Performance

| Fault Type | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| Normal | 0.98 | 0.97 | 0.97 |
| Unbalance | 0.96 | 0.95 | 0.96 |
| Misalignment | 0.94 | 0.95 | 0.94 |
| Bearing Fault | 0.97 | 0.98 | 0.97 |
| Looseness | 0.93 | 0.94 | 0.94 |

**Overall Accuracy: ~96%**

---

## 🎨 UI Screenshots

### Dashboard
- Modern glassmorphism design with gradient accents
- Real-time stats cards with animations
- Responsive grid layout

### Signal Analysis
- Interactive charts with zoom/pan
- Instant client-side FFT preview
- Confidence gauge with severity indicators

### History Table
- Sortable and filterable entries
- Sparkline signal previews
- Export to CSV functionality

---

## 🔮 Future Improvements

- [ ] WebSocket streaming for higher throughput
- [ ] Spectrogram visualization (WebGL)
- [ ] Deep learning models (1D-CNN, LSTM)
- [ ] Real sensor data integration
- [ ] User authentication
- [ ] Cloud deployment (AWS/GCP)

---

## 📝 License

MIT License - Feel free to use for learning and projects!

---

<div align="center">

**Built with ❤️ for Predictive Maintenance**

[Report Bug](https://github.com) · [Request Feature](https://github.com)

</div>
