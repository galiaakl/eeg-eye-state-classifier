"""
EEG Eye State Detection Package

A comprehensive pipeline for preprocessing EEG data and detecting eye states
using ICA analysis and machine learning classification.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .data.loader import load_eeg_data, validate_eeg_data, EEGDataLoader

__all__ = ['load_eeg_data', 'validate_eeg_data', 'EEGDataLoader']
