"""
EEG preprocessing module including filters and ICA analysis.
"""

from .filters import apply_bandpass_filter, apply_notch_filter, apply_all_filters
from .ica_analysis import apply_global_ica, reconstruct_signals, suggest_components_to_remove

__all__ = [
    'apply_bandpass_filter', 
    'apply_notch_filter', 
    'apply_all_filters',
    'apply_global_ica', 
    'reconstruct_signals', 
    'suggest_components_to_remove'
]
