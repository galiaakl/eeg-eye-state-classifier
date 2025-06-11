"""
Signal filtering functions for EEG preprocessing.

This module contains functions for applying various filters to EEG data,
including bandpass and notch filters.
"""

import numpy as np
from scipy import signal
import logging
from typing import Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.config import get_config

logger = logging.getLogger(__name__)


def apply_bandpass_filter(eeg_data: np.ndarray, 
                         fs: Optional[float] = None,
                         lowcut: Optional[float] = None, 
                         highcut: Optional[float] = None, 
                         order: Optional[int] = None) -> np.ndarray:
    """
    Apply bandpass filter to EEG data.
    
    Parameters:
    -----------
    eeg_data : np.ndarray
        EEG data with shape (samples, channels)
    fs : float, optional
        Sampling frequency. If None, uses config value.
    lowcut : float, optional
        Low cutoff frequency. If None, uses config value.
    highcut : float, optional
        High cutoff frequency. If None, uses config value.
    order : int, optional
        Filter order. If None, uses config value.
        
    Returns:
    --------
    np.ndarray : Filtered EEG data
    """
    # Get config values if not provided
    config = get_config()
    if fs is None:
        fs = config.get('data.sampling_frequency', 127.95)
    if lowcut is None:
        lowcut = config.get('preprocessing.filters.bandpass.low_cut', 4.0)
    if highcut is None:
        highcut = config.get('preprocessing.filters.bandpass.high_cut', 40.0)
    if order is None:
        order = config.get('preprocessing.filters.bandpass.order', 6)
    
    logger.info(f"Applying bandpass filter: {lowcut}-{highcut} Hz, order {order}")
    
    # Calculate normalized frequencies
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = min(highcut / nyquist, 0.99)
    
    # Design filter
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply filter to each channel
    filtered_data = np.zeros_like(eeg_data)
    for i in range(eeg_data.shape[1]):
        filtered_data[:, i] = signal.filtfilt(b, a, eeg_data[:, i])
    
    logger.info(f"Bandpass filter applied to {eeg_data.shape[1]} channels")
    return filtered_data


def apply_notch_filter(eeg_data: np.ndarray, 
                      fs: Optional[float] = None,
                      notch_freq: Optional[float] = None, 
                      quality_factor: Optional[float] = None) -> np.ndarray:
    """
    Apply notch filter to remove power line interference.
    
    Parameters:
    -----------
    eeg_data : np.ndarray
        EEG data with shape (samples, channels)
    fs : float, optional
        Sampling frequency. If None, uses config value.
    notch_freq : float, optional
        Notch frequency. If None, uses config value.
    quality_factor : float, optional
        Quality factor. If None, uses config value.
        
    Returns:
    --------
    np.ndarray : Filtered EEG data
    """
    # Get config values if not provided
    config = get_config()
    if fs is None:
        fs = config.get('data.sampling_frequency', 127.95)
    if notch_freq is None:
        notch_freq = config.get('preprocessing.filters.notch.frequency', 50.0)
    if quality_factor is None:
        quality_factor = config.get('preprocessing.filters.notch.quality_factor', 35)
    
    logger.info(f"Applying notch filter: {notch_freq} Hz, Q={quality_factor}")
    
    # Check if notch frequency is valid
    nyquist = 0.5 * fs
    if notch_freq >= nyquist:
        logger.warning(f"Notch frequency {notch_freq} Hz is above Nyquist frequency ({nyquist:.2f} Hz)")
        return eeg_data
    
    # Calculate normalized frequency
    freq = notch_freq / nyquist
    
    # Design notch filter
    b, a = signal.iirnotch(freq, quality_factor)
    
    # Apply filter to each channel
    filtered_data = np.zeros_like(eeg_data)
    for i in range(eeg_data.shape[1]):
        filtered_data[:, i] = signal.filtfilt(b, a, eeg_data[:, i])
    
    logger.info(f"Notch filter applied to {eeg_data.shape[1]} channels")
    return filtered_data


def apply_all_filters(eeg_data: np.ndarray, 
                     fs: Optional[float] = None,
                     apply_notch: bool = True,
                     apply_bandpass: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply all preprocessing filters in the correct order.
    
    Parameters:
    -----------
    eeg_data : np.ndarray
        Raw EEG data with shape (samples, channels)
    fs : float, optional
        Sampling frequency. If None, uses config value.
    apply_notch : bool
        Whether to apply notch filter
    apply_bandpass : bool
        Whether to apply bandpass filter
        
    Returns:
    --------
    tuple : (raw_data, notch_filtered, bandpass_filtered)
        - raw_data: Original data
        - notch_filtered: Data after notch filter
        - bandpass_filtered: Data after both filters
    """
    logger.info("Starting filter pipeline...")
    
    # Step 1: Original data
    data_notch = eeg_data.copy()
    
    # Step 2: Apply notch filter first
    if apply_notch:
        data_notch = apply_notch_filter(data_notch, fs=fs)
    else:
        logger.info("Skipping notch filter")
    
    # Step 3: Apply bandpass filter
    data_bp = data_notch.copy()
    if apply_bandpass:
        data_bp = apply_bandpass_filter(data_notch, fs=fs)
    else:
        logger.info("Skipping bandpass filter")
    
    logger.info("Filter pipeline completed")
    return eeg_data, data_notch, data_bp

