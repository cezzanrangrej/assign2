"""
Quick script to train a compatible RandomForest model for vibration fault detection
This creates a demo model that works with the current scikit-learn version
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Feature names (must match those used in main.py)
FEATURE_NAMES = [
    "rms", "peak", "crest_factor", "kurtosis", "skewness", "shape_factor",
    "dominant_freq", "spectral_centroid", "spectral_entropy", "spectral_kurtosis",
    "low_freq_ratio", "mid_freq_ratio", "high_freq_ratio"
]

# Class labels
CLASSES = ["Normal", "Unbalance", "Misalignment", "Bearing Fault", "Looseness"]

def generate_synthetic_features(label: str, n_samples: int = 100) -> np.ndarray:
    """Generate synthetic feature vectors for each fault type based on domain knowledge"""
    features = np.zeros((n_samples, len(FEATURE_NAMES)))
    
    if label == "Normal":
        # Normal operation: low amplitude, low kurtosis, smooth spectrum
        features[:, 0] = np.random.uniform(0.1, 0.4, n_samples)    # rms
        features[:, 1] = np.random.uniform(0.3, 1.0, n_samples)    # peak
        features[:, 2] = np.random.uniform(2.5, 4.0, n_samples)    # crest_factor
        features[:, 3] = np.random.uniform(2.5, 4.0, n_samples)    # kurtosis (Gaussian ~3)
        features[:, 4] = np.random.uniform(-0.3, 0.3, n_samples)   # skewness
        features[:, 5] = np.random.uniform(1.2, 1.4, n_samples)    # shape_factor
        features[:, 6] = np.random.uniform(50, 70, n_samples)      # dominant_freq (1x speed)
        features[:, 7] = np.random.uniform(100, 200, n_samples)    # spectral_centroid
        features[:, 8] = np.random.uniform(3.5, 5.0, n_samples)    # spectral_entropy (broadband)
        features[:, 9] = np.random.uniform(3.0, 5.0, n_samples)    # spectral_kurtosis
        for i in range(n_samples):
            ratios = np.random.dirichlet([2, 3, 1])
            features[i, 10:13] = ratios
            
    elif label == "Unbalance":
        # Unbalance: 1x running speed dominant, sinusoidal pattern
        features[:, 0] = np.random.uniform(0.4, 1.0, n_samples)    # rms (elevated)
        features[:, 1] = np.random.uniform(1.0, 3.0, n_samples)    # peak
        features[:, 2] = np.random.uniform(2.8, 4.5, n_samples)    # crest_factor
        features[:, 3] = np.random.uniform(2.5, 4.0, n_samples)    # kurtosis (still sinusoidal)
        features[:, 4] = np.random.uniform(-0.2, 0.2, n_samples)   # skewness
        features[:, 5] = np.random.uniform(1.3, 1.5, n_samples)    # shape_factor
        features[:, 6] = np.random.uniform(55, 65, n_samples)      # dominant_freq (1x speed)
        features[:, 7] = np.random.uniform(60, 120, n_samples)     # spectral_centroid (low)
        features[:, 8] = np.random.uniform(1.5, 3.0, n_samples)    # spectral_entropy (tonal)
        features[:, 9] = np.random.uniform(5.0, 10.0, n_samples)   # spectral_kurtosis
        for i in range(n_samples):
            ratios = np.random.dirichlet([5, 2, 0.5])  # High low-freq
            features[i, 10:13] = ratios
            
    elif label == "Misalignment":
        # Misalignment: 2x harmonics, elevated kurtosis
        features[:, 0] = np.random.uniform(0.5, 1.2, n_samples)    # rms
        features[:, 1] = np.random.uniform(1.5, 4.0, n_samples)    # peak
        features[:, 2] = np.random.uniform(3.5, 5.5, n_samples)    # crest_factor (elevated)
        features[:, 3] = np.random.uniform(4.0, 7.0, n_samples)    # kurtosis (elevated)
        features[:, 4] = np.random.uniform(-0.5, 0.5, n_samples)   # skewness
        features[:, 5] = np.random.uniform(1.3, 1.6, n_samples)    # shape_factor
        features[:, 6] = np.random.uniform(110, 140, n_samples)    # dominant_freq (2x)
        features[:, 7] = np.random.uniform(150, 250, n_samples)    # spectral_centroid
        features[:, 8] = np.random.uniform(2.0, 3.5, n_samples)    # spectral_entropy
        features[:, 9] = np.random.uniform(4.0, 8.0, n_samples)    # spectral_kurtosis
        for i in range(n_samples):
            ratios = np.random.dirichlet([2, 4, 1])  # High mid-freq
            features[i, 10:13] = ratios
            
    elif label == "Bearing Fault":
        # Bearing fault: high kurtosis (impacts), high frequency content
        features[:, 0] = np.random.uniform(0.3, 0.8, n_samples)    # rms
        features[:, 1] = np.random.uniform(2.0, 5.0, n_samples)    # peak (impacts)
        features[:, 2] = np.random.uniform(5.0, 8.0, n_samples)    # crest_factor (high)
        features[:, 3] = np.random.uniform(8.0, 15.0, n_samples)   # kurtosis (very high)
        features[:, 4] = np.random.uniform(-1.0, 1.0, n_samples)   # skewness
        features[:, 5] = np.random.uniform(1.4, 1.8, n_samples)    # shape_factor
        features[:, 6] = np.random.uniform(200, 500, n_samples)    # dominant_freq (BPFO, BPFI)
        features[:, 7] = np.random.uniform(300, 500, n_samples)    # spectral_centroid (high)
        features[:, 8] = np.random.uniform(2.5, 4.0, n_samples)    # spectral_entropy
        features[:, 9] = np.random.uniform(6.0, 12.0, n_samples)   # spectral_kurtosis
        for i in range(n_samples):
            ratios = np.random.dirichlet([1, 2, 4])  # High high-freq
            features[i, 10:13] = ratios
            
    elif label == "Looseness":
        # Looseness: subharmonics, rattling, elevated crest and kurtosis
        features[:, 0] = np.random.uniform(0.3, 0.9, n_samples)    # rms
        features[:, 1] = np.random.uniform(1.5, 4.5, n_samples)    # peak
        features[:, 2] = np.random.uniform(4.0, 6.5, n_samples)    # crest_factor
        features[:, 3] = np.random.uniform(5.0, 10.0, n_samples)   # kurtosis
        features[:, 4] = np.random.uniform(-0.8, 0.8, n_samples)   # skewness
        features[:, 5] = np.random.uniform(1.3, 1.7, n_samples)    # shape_factor
        features[:, 6] = np.random.uniform(30, 80, n_samples)      # dominant_freq (subharmonics)
        features[:, 7] = np.random.uniform(100, 250, n_samples)    # spectral_centroid
        features[:, 8] = np.random.uniform(3.5, 5.5, n_samples)    # spectral_entropy (broadband)
        features[:, 9] = np.random.uniform(4.0, 8.0, n_samples)    # spectral_kurtosis
        for i in range(n_samples):
            ratios = np.random.dirichlet([2, 3, 2])  # Spread across bands
            features[i, 10:13] = ratios
    
    return features


def main():
    print("Generating synthetic training data...")
    
    # Generate training data
    X_list = []
    y_list = []
    
    samples_per_class = 200
    
    for idx, label in enumerate(CLASSES):
        features = generate_synthetic_features(label, samples_per_class)
        X_list.append(features)
        y_list.extend([idx] * samples_per_class)
    
    X = np.vstack(X_list)
    y = np.array(y_list)
    
    print(f"Training data shape: {X.shape}")
    print(f"Labels distribution: {np.bincount(y)}")
    
    # Train RandomForest model
    print("\nTraining RandomForest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X, y)
    
    # Show feature importance
    print("\nFeature Importance:")
    importance = list(zip(FEATURE_NAMES, model.feature_importances_))
    importance.sort(key=lambda x: x[1], reverse=True)
    for name, imp in importance:
        print(f"  {name:20s}: {imp:.4f}")
    
    # Save model
    model_path = Path(__file__).parent / "models" / "demo_model.pkl"
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\n✓ Model saved to {model_path}")
    
    # Also save class labels for reference
    metadata = {
        "classes": CLASSES,
        "feature_names": FEATURE_NAMES,
        "sklearn_version": "1.7.2"
    }
    metadata_path = Path(__file__).parent / "models" / "model_metadata.pkl"
    joblib.dump(metadata, metadata_path)
    print(f"✓ Metadata saved to {metadata_path}")
    
    # Quick test
    print("\nQuick validation test:")
    test_idx = np.random.choice(len(X), 5, replace=False)
    predictions = model.predict(X[test_idx])
    for i, (pred, actual) in enumerate(zip(predictions, y[test_idx])):
        print(f"  Sample {i+1}: Predicted={CLASSES[pred]}, Actual={CLASSES[actual]}")
    
    print("\n✓ Model training complete!")


if __name__ == "__main__":
    main()
