"""
Visualization utilities for EEG data and analysis results.
"""

from .time_plots import plot_eeg_channels, compare_preprocessing_stages
from .ica_plots import visualize_component_detailed, visualize_all_components_summary

__all__ = [
    'plot_eeg_channels', 
    'compare_preprocessing_stages',
    'visualize_component_detailed', 
    'visualize_all_components_summary'
]
