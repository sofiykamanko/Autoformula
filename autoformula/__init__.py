"""
AutoFeatures — lightweight feature engineering utilities.

Public API:
- data sanitization
- feature type detection
- statistical feature analysis
- feature mixing / generation
"""

# Sanitization
from .sanitization import SanitizationEngine
preprocess_dataset = SanitizationEngine.preprocess_dataset

# Feature type detection
from .feature_types import FeatureTypeDetection

detect_feature_types = FeatureTypeDetection.detect_feature_types
type_mismatch = FeatureTypeDetection.type_mismatch
handle_selected_missmatch = FeatureTypeDetection.handle_selected_missmatch
feature_summary = FeatureTypeDetection.feature_summary

# Feature analysis
from .analysis import FeatureAnalysis

check_correlations = FeatureAnalysis.check_correlations
plot_correlation_heatmaps = FeatureAnalysis.plot_correlation_heatmaps
cramers_v_matrix = FeatureAnalysis.cramers_v_matrix

compute_overlap = FeatureAnalysis.compute_overlap
compute_separability = FeatureAnalysis.compute_separability
compute_impact = FeatureAnalysis.compute_impact
compute_feature_quality = FeatureAnalysis.compute_feature_quality


# Feature mixing
from .feature_generation import FeatureMixing

generate_feature_combinations = FeatureMixing.generate_feature_combinations
generate_custom_feature_combinations = FeatureMixing.generate_custom_feature_combinations
