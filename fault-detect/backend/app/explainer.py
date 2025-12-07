# backend/app/explainer.py
"""
Model Explainer for Vibration Fault Detection
Provides feature contribution explanations using:
1. SHAP TreeExplainer (if available)
2. Permutation Importance fallback (scikit-learn built-in)
3. Model Feature Importances (for tree-based models)

This ensures explainability works even without SHAP installed.
"""
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Try to import SHAP - graceful fallback if not available
try:
    import shap
    SHAP_AVAILABLE = True
    logger.info("✓ SHAP library loaded successfully")
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("⚠ SHAP library not available - using sklearn feature importance fallback")

# Import sklearn for permutation importance (always available)
try:
    from sklearn.inspection import permutation_importance
    SKLEARN_INSPECTION_AVAILABLE = True
except ImportError:
    SKLEARN_INSPECTION_AVAILABLE = False


@dataclass
class FeatureContribution:
    """Represents a single feature's contribution to the prediction"""
    name: str
    shap_value: float
    abs_importance: float
    direction: str  # 'positive' or 'negative'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "shap_value": round(self.shap_value, 4),
            "abs_importance": round(self.abs_importance, 4),
            "direction": self.direction
        }


class ModelExplainer:
    """
    Multi-method explainer for ML models
    Uses SHAP if available, otherwise falls back to sklearn's feature importance
    """
    
    # Feature names in the order used by the model
    FEATURE_NAMES = [
        "rms",
        "peak", 
        "crest_factor",
        "kurtosis",
        "skewness",
        "shape_factor",
        "dominant_freq",
        "spectral_centroid",
        "spectral_entropy",
        "spectral_kurtosis",
        "low_freq_ratio",
        "mid_freq_ratio",
        "high_freq_ratio"
    ]
    
    # Human-readable feature descriptions for reports
    FEATURE_DESCRIPTIONS = {
        "rms": "Root Mean Square amplitude",
        "peak": "Peak amplitude value",
        "crest_factor": "Peak-to-RMS ratio (impulsiveness indicator)",
        "kurtosis": "Signal impulsiveness (4th moment) - high = impacts",
        "skewness": "Signal asymmetry (3rd moment)",
        "shape_factor": "RMS-to-mean-absolute ratio",
        "dominant_freq": "Primary frequency component (Hz)",
        "spectral_centroid": "Center of spectral mass (Hz)",
        "spectral_entropy": "Spectral flatness - low = tonal",
        "spectral_kurtosis": "Frequency distribution peakedness",
        "low_freq_ratio": "Low frequency band (10-100Hz) energy ratio",
        "mid_freq_ratio": "Mid frequency band (100-500Hz) energy ratio",
        "high_freq_ratio": "High frequency band (>500Hz) energy ratio"
    }
    
    # Class labels for the model
    CLASS_LABELS = ["Normal", "Unbalance", "Misalignment", "Bearing Fault", "Looseness"]
    
    def __init__(self, model=None, background_data: Optional[np.ndarray] = None):
        """
        Initialize the explainer with the trained model
        
        Args:
            model: Trained sklearn model (RandomForest, GradientBoosting, etc.)
            background_data: Optional background dataset for SHAP explainer
        """
        self.model = model
        self.shap_explainer = None
        self.use_shap = False
        self.is_initialized = False
        self.method = "Not Initialized"
        
        if model is None:
            logger.warning("No model provided - using domain knowledge fallback")
            self.method = "Domain Knowledge"
            return
        
        # Try to initialize SHAP TreeExplainer first
        if SHAP_AVAILABLE:
            try:
                if background_data is None:
                    background_data = self._generate_synthetic_background()
                
                self.shap_explainer = shap.TreeExplainer(
                    model,
                    data=background_data,
                    feature_perturbation='interventional'
                )
                self.use_shap = True
                self.is_initialized = True
                self.method = "SHAP TreeExplainer"
                logger.info(f"✓ SHAP TreeExplainer initialized with {len(background_data)} background samples")
                return
            except Exception as e:
                logger.warning(f"SHAP initialization failed: {e}, falling back to feature importance")
        
        # Fallback: Use model's built-in feature importance
        if hasattr(model, 'feature_importances_'):
            self.is_initialized = True
            self.method = "RandomForest Feature Importance"
            logger.info("✓ Using sklearn feature_importances_ for explanations")
        else:
            self.method = "Domain Knowledge"
            logger.info("✓ Using domain knowledge fallback for explanations")
    
    def _generate_synthetic_background(self, n_samples: int = 100) -> np.ndarray:
        """Generate synthetic background data for SHAP explainer"""
        np.random.seed(42)
        background = np.zeros((n_samples, len(self.FEATURE_NAMES)))
        
        # Time-domain features
        background[:, 0] = np.random.uniform(0.1, 2.0, n_samples)    # rms
        background[:, 1] = np.random.uniform(0.5, 5.0, n_samples)    # peak
        background[:, 2] = np.random.uniform(2.0, 8.0, n_samples)    # crest_factor
        background[:, 3] = np.random.uniform(2.0, 15.0, n_samples)   # kurtosis
        background[:, 4] = np.random.uniform(-1.0, 1.0, n_samples)   # skewness
        background[:, 5] = np.random.uniform(1.0, 2.0, n_samples)    # shape_factor
        
        # Frequency-domain features
        background[:, 6] = np.random.uniform(20, 500, n_samples)     # dominant_freq
        background[:, 7] = np.random.uniform(50, 400, n_samples)     # spectral_centroid
        background[:, 8] = np.random.uniform(1.0, 6.0, n_samples)    # spectral_entropy
        background[:, 9] = np.random.uniform(2.0, 10.0, n_samples)   # spectral_kurtosis
        
        # Band energy ratios
        for i in range(n_samples):
            ratios = np.random.dirichlet([2, 3, 1])
            background[i, 10] = ratios[0]  # low_freq_ratio
            background[i, 11] = ratios[1]  # mid_freq_ratio
            background[i, 12] = ratios[2]  # high_freq_ratio
        
        return background
    
    def explain(
        self, 
        features: Dict[str, float], 
        predicted_class: str,
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Compute explanation for a single prediction
        
        Args:
            features: Dictionary of feature name -> value
            predicted_class: The predicted class label
            top_n: Number of top contributing features to return
            
        Returns:
            Dictionary with explanation data
        """
        feature_vector = self._build_feature_vector(features)
        
        # Method 1: SHAP (if available and initialized)
        if self.use_shap and self.shap_explainer is not None:
            return self._explain_with_shap(feature_vector, features, predicted_class, top_n)
        
        # Method 2: RandomForest feature importance with local contribution estimation
        if self.model is not None and hasattr(self.model, 'feature_importances_'):
            return self._explain_with_feature_importance(feature_vector, features, predicted_class, top_n)
        
        # Method 3: Domain knowledge fallback
        return self._explain_with_domain_knowledge(features, predicted_class, top_n)
    
    def _explain_with_shap(
        self,
        feature_vector: np.ndarray,
        features: Dict[str, float],
        predicted_class: str,
        top_n: int
    ) -> Dict[str, Any]:
        """Explain using SHAP values"""
        try:
            shap_values = self.shap_explainer.shap_values(feature_vector.reshape(1, -1))
            class_idx = self._get_class_index(predicted_class)
            
            if isinstance(shap_values, list):
                values = shap_values[class_idx][0]
            else:
                values = shap_values[0]
            
            contributions = []
            for i, (name, shap_val) in enumerate(zip(self.FEATURE_NAMES[:len(values)], values)):
                contributions.append(FeatureContribution(
                    name=name,
                    shap_value=float(shap_val),
                    abs_importance=float(abs(shap_val)),
                    direction="positive" if shap_val >= 0 else "negative"
                ))
            
            contributions.sort(key=lambda x: x.abs_importance, reverse=True)
            
            return {
                "model": "RandomForest",
                "method": "SHAP TreeExplainer",
                "predicted_class": predicted_class,
                "top_features": [c.to_dict() for c in contributions[:top_n]],
                "all_contributions": [c.to_dict() for c in contributions]
            }
        except Exception as e:
            logger.error(f"SHAP computation failed: {e}")
            return self._explain_with_feature_importance(feature_vector, features, predicted_class, top_n)
    
    def _explain_with_feature_importance(
        self,
        feature_vector: np.ndarray,
        features: Dict[str, float],
        predicted_class: str,
        top_n: int
    ) -> Dict[str, Any]:
        """
        Explain using model's feature_importances_ combined with local feature values
        
        This computes a "local importance" by:
        1. Getting the global feature importance from the model
        2. Weighting it by how much the feature deviates from typical values
        3. Determining direction based on the feature's relationship to the predicted class
        """
        global_importance = self.model.feature_importances_
        
        # Get prediction probabilities to understand which features push towards the prediction
        try:
            proba = self.model.predict_proba(feature_vector.reshape(1, -1))[0]
            class_idx = self._get_class_index(predicted_class)
            predicted_prob = proba[class_idx]
        except:
            predicted_prob = 0.5
        
        # Typical feature values (approximate means from vibration analysis)
        typical_values = {
            "rms": 0.5, "peak": 1.5, "crest_factor": 4.0, "kurtosis": 4.0,
            "skewness": 0.0, "shape_factor": 1.3, "dominant_freq": 60.0,
            "spectral_centroid": 150.0, "spectral_entropy": 3.5, "spectral_kurtosis": 5.0,
            "low_freq_ratio": 0.3, "mid_freq_ratio": 0.5, "high_freq_ratio": 0.2
        }
        
        # Feature direction indicators per fault type
        # Positive means "high value supports this fault"
        fault_feature_direction = {
            "Normal": {"rms": -1, "kurtosis": -1, "crest_factor": -1, "spectral_entropy": 1},
            "Unbalance": {"dominant_freq": 1, "low_freq_ratio": 1, "spectral_entropy": -1, "rms": 1},
            "Misalignment": {"kurtosis": 1, "dominant_freq": 1, "mid_freq_ratio": 1, "crest_factor": 1},
            "Bearing Fault": {"kurtosis": 1, "high_freq_ratio": 1, "crest_factor": 1, "spectral_kurtosis": 1},
            "Looseness": {"kurtosis": 1, "crest_factor": 1, "spectral_entropy": 1, "mid_freq_ratio": 1}
        }
        
        directions = fault_feature_direction.get(predicted_class, {})
        
        contributions = []
        for i, name in enumerate(self.FEATURE_NAMES[:len(global_importance)]):
            importance = global_importance[i]
            feature_val = features.get(name, 0)
            typical_val = typical_values.get(name, 1.0)
            
            # Calculate deviation from typical
            deviation = (feature_val - typical_val) / (typical_val + 1e-9)
            
            # Local importance = global importance * deviation magnitude
            local_importance = importance * (1 + min(abs(deviation), 2.0))
            
            # Determine direction based on fault-feature relationship
            default_direction = 1 if deviation > 0 else -1
            fault_direction = directions.get(name, default_direction)
            
            # If feature deviation aligns with what the fault expects, it's positive
            if (deviation > 0 and fault_direction > 0) or (deviation < 0 and fault_direction < 0):
                direction = "positive"
                shap_like_value = local_importance
            else:
                direction = "negative" if deviation != 0 else "positive"
                shap_like_value = -local_importance if deviation != 0 else local_importance
            
            contributions.append(FeatureContribution(
                name=name,
                shap_value=float(shap_like_value),
                abs_importance=float(abs(local_importance)),
                direction=direction
            ))
        
        contributions.sort(key=lambda x: x.abs_importance, reverse=True)
        
        return {
            "model": "RandomForest",
            "method": "Feature Importance + Local Analysis",
            "predicted_class": predicted_class,
            "top_features": [c.to_dict() for c in contributions[:top_n]],
            "all_contributions": [c.to_dict() for c in contributions]
        }
    
    def _explain_with_domain_knowledge(
        self,
        features: Dict[str, float],
        predicted_class: str,
        top_n: int
    ) -> Dict[str, Any]:
        """
        Domain knowledge-based explanation when no model is available
        Uses vibration analysis expertise to explain predictions
        """
        # Domain knowledge: which features matter most for each fault type
        fault_feature_weights = {
            "Normal": [
                ("kurtosis", 0.30, "Low kurtosis indicates no impacts"),
                ("crest_factor", 0.25, "Low crest factor = smooth operation"),
                ("rms", 0.20, "Low RMS = low vibration amplitude"),
                ("spectral_entropy", 0.15, "High entropy = broadband noise"),
                ("dominant_freq", 0.10, "Stable dominant frequency")
            ],
            "Unbalance": [
                ("dominant_freq", 0.35, "1x running speed dominance"),
                ("low_freq_ratio", 0.25, "High low-frequency energy"),
                ("rms", 0.20, "Elevated overall amplitude"),
                ("spectral_entropy", 0.12, "Low entropy = tonal"),
                ("spectral_centroid", 0.08, "Low spectral centroid")
            ],
            "Misalignment": [
                ("kurtosis", 0.30, "Moderate kurtosis from harmonics"),
                ("dominant_freq", 0.25, "2x running speed presence"),
                ("crest_factor", 0.20, "Elevated crest factor"),
                ("mid_freq_ratio", 0.15, "Higher mid-frequency energy"),
                ("spectral_kurtosis", 0.10, "Spectral peakedness")
            ],
            "Bearing Fault": [
                ("kurtosis", 0.35, "High kurtosis from impacts"),
                ("high_freq_ratio", 0.25, "High-frequency energy from defects"),
                ("crest_factor", 0.20, "High crest from impulses"),
                ("spectral_kurtosis", 0.12, "Sharp spectral peaks"),
                ("rms", 0.08, "Possibly elevated RMS")
            ],
            "Looseness": [
                ("kurtosis", 0.30, "Elevated kurtosis"),
                ("crest_factor", 0.25, "High crest from rattling"),
                ("spectral_entropy", 0.20, "Higher entropy from subharmonics"),
                ("mid_freq_ratio", 0.15, "Broad mid-frequency content"),
                ("dominant_freq", 0.10, "Multiple frequency components")
            ]
        }
        
        weights = fault_feature_weights.get(predicted_class, fault_feature_weights["Normal"])
        
        contributions = []
        for name, weight, description in weights:
            value = features.get(name, 0)
            # Positive contribution for the predicted class
            simulated_shap = weight * 0.5  # Scale to reasonable range
            contributions.append(FeatureContribution(
                name=name,
                shap_value=simulated_shap,
                abs_importance=weight,
                direction="positive"
            ))
        
        return {
            "model": "RuleBased",
            "method": "Domain Knowledge Analysis",
            "predicted_class": predicted_class,
            "top_features": [c.to_dict() for c in contributions[:top_n]],
            "note": "Based on vibration analysis domain expertise"
        }
    
    def _build_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Build numpy array from feature dictionary"""
        return np.array([features.get(name, 0.0) for name in self.FEATURE_NAMES])
    
    def _get_class_index(self, class_name: str) -> int:
        """Map class name to index"""
        try:
            return self.CLASS_LABELS.index(class_name)
        except ValueError:
            return 0
    
    def get_feature_description(self, feature_name: str) -> str:
        """Get human-readable description for a feature"""
        return self.FEATURE_DESCRIPTIONS.get(feature_name, feature_name)
    
    def format_for_report(
        self, 
        explanation: Dict[str, Any],
        predicted_class: str
    ) -> List[str]:
        """Format explanation for PDF report"""
        lines = []
        top_features = explanation.get("top_features", [])
        
        for feat in top_features[:5]:
            name = feat["name"]
            direction = "↑" if feat["direction"] == "positive" else "↓"
            importance = feat["abs_importance"]
            description = self.FEATURE_DESCRIPTIONS.get(name, "")
            
            line = f"• {name} ({direction}) - importance: {importance:.3f}"
            if description:
                line += f"\n  [{description}]"
            line += f" → supports {predicted_class}"
            lines.append(line)
        
        return lines


# Alias for backward compatibility
ShapExplainer = ModelExplainer

# Global explainer instance
_global_explainer: Optional[ModelExplainer] = None


def init_explainer(model=None, background_data: Optional[np.ndarray] = None) -> ModelExplainer:
    """Initialize the global explainer"""
    global _global_explainer
    _global_explainer = ModelExplainer(model=model, background_data=background_data)
    return _global_explainer


def get_explainer() -> Optional[ModelExplainer]:
    """Get the global explainer instance"""
    return _global_explainer


def explain_prediction(
    features: Dict[str, float],
    predicted_class: str,
    top_n: int = 5
) -> Dict[str, Any]:
    """Convenience function to get explanation for a prediction"""
    if _global_explainer is None:
        return {
            "model": "Unknown",
            "method": "Not Initialized",
            "top_features": [],
            "error": "Explainer not initialized"
        }
    
    return _global_explainer.explain(features, predicted_class, top_n)
