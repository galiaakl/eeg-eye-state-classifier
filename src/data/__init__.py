"""
Data loading and validation module for EEG data.
"""

from .loader import load_eeg_data, validate_eeg_data, EEGDataLoader

__all__ = ['load_eeg_data', 'validate_eeg_data', 'EEGDataLoader']
