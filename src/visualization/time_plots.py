# src/visualization/time_plots.py
"""
Time domain visualization for EEG data.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def plot_eeg_channels(eeg_data: np.ndarray, 
                     channel_names: List[str],
                     fs: float = 127.95,
                     time_window: float = 10.0,
                     start_time: float = 0.0,
                     title: str = "EEG Channels") -> plt.Figure:
    """
    Plot multiple EEG channels in time domain.
    
    Parameters:
    -----------
    eeg_data : np.ndarray
        EEG data with shape (samples, channels)
    channel_names : List[str]
        List of channel names
    fs : float
        Sampling frequency
    time_window : float
        Time window to display in seconds
    start_time : float
        Start time in seconds
    title : str
        Plot title
        
    Returns:
    --------
    plt.Figure : Figure object
    """
    # Calculate sample indices
    start_sample = int(start_time * fs)
    end_sample = min(start_sample + int(time_window * fs), eeg_data.shape[0])
    
    # Extract data segment
    data_segment = eeg_data[start_sample:end_sample, :]
    time = np.arange(data_segment.shape[0]) / fs + start_time
    
    # Create figure
    n_channels = min(len(channel_names), 8)  # Limit to 8 channels for readability
    fig, axes = plt.subplots(n_channels, 1, figsize=(15, 2*n_channels), sharex=True)
    
    if n_channels == 1:
        axes = [axes]
    
    # Plot each channel
    for i in range(n_channels):
        axes[i].plot(time, data_segment[:, i], 'b-', linewidth=0.8)
        axes[i].set_ylabel(f'{channel_names[i]}\n(μV)')
        axes[i].grid(True, alpha=0.3)
        
        # Add some basic statistics as text
        mean_val = np.mean(data_segment[:, i])
        std_val = np.std(data_segment[:, i])
        axes[i].text(0.02, 0.95, f'μ={mean_val:.1f}, σ={std_val:.1f}', 
                    transform=axes[i].transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    fontsize=8)
    
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    logger.info(f"Plotted {n_channels} EEG channels for {time_window}s")
    return fig


def compare_preprocessing_stages(raw_data: np.ndarray,
                               filtered_data: np.ndarray,
                               cleaned_data: np.ndarray,
                               channel_names: List[str],
                               fs: float = 127.95,
                               channel_idx: int = 0,
                               time_window: float = 5.0) -> plt.Figure:
    """
    Compare different preprocessing stages for a single channel.
    
    Parameters:
    -----------
    raw_data : np.ndarray
        Raw EEG data
    filtered_data : np.ndarray
        Filtered EEG data
    cleaned_data : np.ndarray
        ICA-cleaned EEG data
    channel_names : List[str]
        Channel names
    fs : float
        Sampling frequency
    channel_idx : int
        Channel index to display
    time_window : float
        Time window in seconds
        
    Returns:
    --------
    plt.Figure : Figure object
    """
    # Extract time segment
    n_samples = min(int(time_window * fs), raw_data.shape[0])
    time = np.arange(n_samples) / fs
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    # Plot raw data
    axes[0].plot(time, raw_data[:n_samples, channel_idx], 'k-', linewidth=1)
    axes[0].set_title(f'Raw Data - {channel_names[channel_idx]}')
    axes[0].set_ylabel('Amplitude (μV)')
    axes[0].grid(True, alpha=0.3)
    
    # Plot filtered data
    axes[1].plot(time, filtered_data[:n_samples, channel_idx], 'b-', linewidth=1)
    axes[1].set_title(f'Filtered Data - {channel_names[channel_idx]}')
    axes[1].set_ylabel('Amplitude (μV)')
    axes[1].grid(True, alpha=0.3)
    
    # Plot cleaned data
    axes[2].plot(time, cleaned_data[:n_samples, channel_idx], 'g-', linewidth=1)
    axes[2].set_title(f'ICA Cleaned Data - {channel_names[channel_idx]}')
    axes[2].set_ylabel('Amplitude (μV)')
    axes[2].set_xlabel('Time (s)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
