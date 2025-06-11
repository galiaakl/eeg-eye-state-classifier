# src/visualization/ica_plots.py
"""
ICA Visualization Module

This module contains functions for visualizing ICA components,
including detailed component analysis and summary plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pywt
from typing import Dict, List, Any, Optional
from scipy import signal
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.config import get_config

logger = logging.getLogger(__name__)


def visualize_component_detailed(ica, components: np.ndarray, y: np.ndarray, 
                                component_idx: int, component_info: Dict[int, Dict[str, Any]], 
                                channel_names: List[str], fs: float = 127.95) -> plt.Figure:
    """
    Create detailed visualization of a single ICA component.
    
    Parameters:
    -----------
    ica : FastICA object
        Fitted ICA object
    components : np.ndarray
        ICA components array
    y : np.ndarray
        Labels (0=open, 1=closed)
    component_idx : int
        Index of component to visualize
    component_info : Dict[int, Dict[str, Any]]
        Component analysis information
    channel_names : List[str]
        Channel names
    fs : float
        Sampling frequency
        
    Returns:
    --------
    plt.Figure : Figure object
    """
    # Get component data
    comp_data = components[:, component_idx]
    
    # Get component info
    info = component_info[component_idx]
    primary_region = info['primary_region']
    disc_score = info['discriminative_score']
    comp_type = info['component_type']
    
    # Determine rank
    sorted_scores = sorted([info['discriminative_score'] for info in component_info.values()], reverse=True)
    rank = sorted_scores.index(disc_score) + 1
    
    # Separate data by eye state
    eyes_open_indices = np.where(y == 0)[0]
    eyes_closed_indices = np.where(y == 1)[0]
    
    comp_open = comp_data[eyes_open_indices]
    comp_closed = comp_data[eyes_closed_indices]
    
    # Create figure with comprehensive layout
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig)
    
    # Determine recommendation color
    if disc_score > np.percentile([info['discriminative_score'] for info in component_info.values()], 75):
        recommendation = "KEEP (high discrimination)"
        color = 'lightgreen'
    elif disc_score > np.percentile([info['discriminative_score'] for info in component_info.values()], 50):
        recommendation = "KEEP (moderate discrimination)"
        color = 'palegoldenrod'
    elif disc_score > np.percentile([info['discriminative_score'] for info in component_info.values()], 25):
        recommendation = "CONSIDER (low discrimination)"
        color = 'lightsalmon'
    else:
        recommendation = "CONSIDER REMOVING (very low discrimination)"
        color = 'lightcoral'
    
    # Set title
    fig.suptitle(f'ICA Component {component_idx+1} - {primary_region.upper()} Region, {comp_type}',
                 fontsize=16)
    
    # 1. Time domain - First 3 seconds
    ax_time = fig.add_subplot(gs[0, 0])
    samples_to_show = min(int(fs * 3), comp_data.shape[0])
    time = np.arange(samples_to_show) / fs
    
    ax_time.plot(time, comp_data[:samples_to_show], color='purple', linewidth=1.5)
    ax_time.set_title(f'Component {component_idx+1}: Score {disc_score:.4f} - Rank {rank}/{len(component_info)}\n{recommendation}')
    ax_time.set_xlabel('Time (s)')
    ax_time.set_ylabel('Amplitude')
    ax_time.grid(True, alpha=0.3)
    ax_time.set_facecolor(color)
    
    # Add band power labels
    y_pos = 0.85
    bands = ['alpha', 'beta', 'theta', 'delta', 'gamma']
    for band in bands:
        ratio = info['power_ratios'][f'{band}_ratio']
        ax_time.text(0.02, y_pos, f"{band}: {ratio:.3f}x (closed/open)",
                    transform=ax_time.transAxes, fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.7))
        y_pos -= 0.07
    
    # 2. Eyes open PSD
    ax_open = fig.add_subplot(gs[0, 1])
    f_open, psd_open = info['f_open'], info['psd_open']
    ax_open.semilogy(f_open, psd_open, 'b-', linewidth=1.5)
    
    # Add band power values
    bands_freq = {'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
    for band_name, (low, high) in bands_freq.items():
        band_mask = (f_open >= low) & (f_open <= high)
        if np.any(band_mask):
            band_power = np.mean(psd_open[band_mask])
            ax_open.text(0.02, 0.95 - list(bands_freq.keys()).index(band_name) * 0.07,
                        f"{band_name}: {band_power:.2e}",
                        transform=ax_open.transAxes, fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.7))
    
    ax_open.set_title('Eyes Open PSD')
    ax_open.set_xlabel('Frequency (Hz)')
    ax_open.set_ylabel('PSD (log scale)')
    ax_open.set_xlim([0, min(fs/2, 60)])
    ax_open.grid(True, alpha=0.3)
    
    # Add frequency band markers
    ax_open.axvspan(4, 8, alpha=0.2, color='lightgreen')  # Theta
    ax_open.axvspan(8, 13, alpha=0.2, color='yellow')     # Alpha
    ax_open.axvspan(13, 30, alpha=0.2, color='orange')    # Beta
    
    # 3. Eyes closed PSD
    ax_closed = fig.add_subplot(gs[0, 2])
    f_closed, psd_closed = info['f_closed'], info['psd_closed']
    ax_closed.semilogy(f_closed, psd_closed, 'r-', linewidth=1.5)
    
    # Add band power values
    for band_name, (low, high) in bands_freq.items():
        band_mask = (f_closed >= low) & (f_closed <= high)
        if np.any(band_mask):
            band_power = np.mean(psd_closed[band_mask])
            ax_closed.text(0.02, 0.95 - list(bands_freq.keys()).index(band_name) * 0.07,
                          f"{band_name}: {band_power:.2e}",
                          transform=ax_closed.transAxes, fontsize=9,
                          bbox=dict(facecolor='white', alpha=0.7))
    
    ax_closed.set_title('Eyes Closed PSD')
    ax_closed.set_xlabel('Frequency (Hz)')
    ax_closed.set_ylabel('PSD (log scale)')
    ax_closed.set_xlim([0, min(fs/2, 60)])
    ax_closed.grid(True, alpha=0.3)
    
    # Add frequency band markers
    ax_closed.axvspan(4, 8, alpha=0.2, color='lightgreen')   # Theta
    ax_closed.axvspan(8, 13, alpha=0.2, color='yellow')      # Alpha
    ax_closed.axvspan(13, 30, alpha=0.2, color='orange')     # Beta
    
    # 4. Channel weights
    ax_weights = fig.add_subplot(gs[0, 3])
    channel_weights = info['channel_weights']
    
    # Sort weights by magnitude
    weight_indices = np.argsort(np.abs(channel_weights))[::-1]
    sorted_weights = channel_weights[weight_indices]
    sorted_channels = [channel_names[i] for i in weight_indices]
    
    # Show top 10 weights for readability
    if len(sorted_channels) > 10:
        sorted_weights = sorted_weights[:10]
        sorted_channels = sorted_channels[:10]
    
    # Plot weights as bar chart
    bars = ax_weights.barh(range(len(sorted_channels)), sorted_weights, color='skyblue')
    ax_weights.set_title('Channel Weights')
    ax_weights.set_xlabel('Weight')
    ax_weights.set_yticks(range(len(sorted_channels)))
    ax_weights.set_yticklabels(sorted_channels)
    
    # Color code bars by weight magnitude
    for j, bar in enumerate(bars):
        if abs(sorted_weights[j]) > 0.5:
            bar.set_color('red')
        elif abs(sorted_weights[j]) > 0.3:
            bar.set_color('orange')
        else:
            bar.set_color('blue')
    
    ax_weights.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax_weights.grid(True, alpha=0.3)
    
    # 5. Frequency domain comparison
    ax_freq_comp = fig.add_subplot(gs[1, :2])
    
    # Plot both PSDs for comparison
    ax_freq_comp.semilogy(f_open, psd_open, 'b-', linewidth=1.5, label='Eyes Open')
    ax_freq_comp.semilogy(f_closed, psd_closed, 'r-', linewidth=1.5, label='Eyes Closed')
    
    # Add frequency band markers
    ax_freq_comp.axvspan(1, 4, alpha=0.1, color='gray', label='Delta')
    ax_freq_comp.axvspan(4, 8, alpha=0.1, color='lightgreen', label='Theta')
    ax_freq_comp.axvspan(8, 13, alpha=0.1, color='yellow', label='Alpha')
    ax_freq_comp.axvspan(13, 30, alpha=0.1, color='orange', label='Beta')
    ax_freq_comp.axvspan(30, 45, alpha=0.1, color='red', label='Gamma')
    
    # Highlight significant differences
    for band_name, (low_freq, high_freq) in bands_freq.items():
        band_mask_open = (f_open >= low_freq) & (f_open <= high_freq)
        band_mask_closed = (f_closed >= low_freq) & (f_closed <= high_freq)
        
        if np.any(band_mask_open) and np.any(band_mask_closed):
            power_open = np.mean(psd_open[band_mask_open])
            power_closed = np.mean(psd_closed[band_mask_closed])
            ratio = power_closed / power_open if power_open > 0 else 1
            
            # Add marker for significant differences
            if ratio > 1.5 or ratio < 0.6:
                mid_freq = (low_freq + high_freq) / 2
                max_psd = max(np.max(psd_open), np.max(psd_closed))
                
                if ratio > 1.5:  # Closed > Open
                    ax_freq_comp.annotate(f"{ratio:.1f}x", xy=(mid_freq, max_psd/10),
                                         xytext=(mid_freq, max_psd/100),
                                         arrowprops=dict(facecolor='red', shrink=0.05, width=1.5),
                                         ha='center', va='bottom', color='red')
                else:  # Open > Closed
                    ax_freq_comp.annotate(f"{1/ratio:.1f}x", xy=(mid_freq, max_psd/10),
                                         xytext=(mid_freq, max_psd/100),
                                         arrowprops=dict(facecolor='blue', shrink=0.05, width=1.5),
                                         ha='center', va='bottom', color='blue')
    
    ax_freq_comp.set_title('Frequency Domain Comparison (Eyes Open vs. Closed)')
    ax_freq_comp.set_xlabel('Frequency (Hz)')
    ax_freq_comp.set_ylabel('Power Spectral Density (log scale)')
    ax_freq_comp.set_xlim([0, 45])
    ax_freq_comp.legend(loc='upper right')
    ax_freq_comp.grid(True, alpha=0.3)
    
    # 6. Component topography
    ax_topo = fig.add_subplot(gs[1, 2:])
    
    # Create simple topographic map
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    ax_topo.add_patch(circle)
    
    # Define channel locations
    channel_locations = {
        'FP1': (-0.3, 0.8), 'FP2': (0.3, 0.8),
        'AF3': (-0.5, 0.6), 'AF4': (0.5, 0.6),
        'F7': (-0.8, 0.4), 'F3': (-0.4, 0.4), 'Fz': (0, 0.4), 'F4': (0.4, 0.4), 'F8': (0.8, 0.4),
        'FC5': (-0.6, 0.2), 'FC1': (-0.2, 0.2), 'FC2': (0.2, 0.2), 'FC6': (0.6, 0.2),
        'T7': (-0.9, 0), 'C3': (-0.4, 0), 'Cz': (0, 0), 'C4': (0.4, 0), 'T8': (0.9, 0),
        'CP5': (-0.6, -0.2), 'CP1': (-0.2, -0.2), 'CP2': (0.2, -0.2), 'CP6': (0.6, -0.2),
        'P7': (-0.8, -0.4), 'P3': (-0.4, -0.4), 'Pz': (0, -0.4), 'P4': (0.4, -0.4), 'P8': (0.8, -0.4),
        'PO9': (-0.6, -0.6), 'O1': (-0.3, -0.8), 'Oz': (0, -0.8), 'O2': (0.3, -0.8), 'PO10': (0.6, -0.6)
    }
    
    # Add default locations for missing channels
    for channel in channel_names:
        if channel not in channel_locations:
            channel_locations[channel] = (0, 0)
    
    # Find strongest weight for normalization
    max_weight = max(abs(channel_weights))
    
    # Plot each channel
    for i, channel in enumerate(channel_names):
        if channel in channel_locations:
            x, y = channel_locations[channel]
            weight = channel_weights[i]
            
            # Normalize weight
            norm_weight = weight / max_weight if max_weight > 0 else 0
            
            # Color mapping
            if norm_weight >= 0:
                color = (1, 0, 0, min(abs(norm_weight), 1.0))  # Red for positive
            else:
                color = (0, 0, 1, min(abs(norm_weight), 1.0))  # Blue for negative
            
            # Draw circle
            circle_size = 0.1 + 0.1 * abs(norm_weight)
            topo_circle = plt.Circle((x, y), circle_size, color=color, alpha=0.7)
            ax_topo.add_patch(topo_circle)
            
            # Add label
            ax_topo.text(x, y, channel, ha='center', va='center', fontsize=8)
    
    ax_topo.set_xlim(-1.2, 1.2)
    ax_topo.set_ylim(-1.2, 1.2)
    ax_topo.set_aspect('equal')
    ax_topo.axis('off')
    ax_topo.set_title('Component Topography (Channel Weights)')
    
    # 7. Wavelet time-frequency analysis
    ax_wavelet = fig.add_subplot(gs[2, :2])
    
    # Select short segment for wavelet analysis
    segment_length = min(int(fs * 3), len(comp_data))
    comp_segment = comp_data[:segment_length]
    wavelet_time = np.arange(segment_length) / fs
    
    # Compute continuous wavelet transform
    scales = np.arange(1, 100)
    wavelet = 'cmor1.5-1.0'
    
    # Convert scales to frequencies
    frequencies = pywt.scale2frequency(wavelet, scales) * fs
    
    # Limit to 1-45 Hz range
    freq_mask = (frequencies >= 1) & (frequencies <= 45)
    frequencies = frequencies[freq_mask]
    scales = scales[freq_mask]
    
    coeffs, _ = pywt.cwt(comp_segment, scales, wavelet, 1.0 / fs)
    power = np.abs(coeffs) ** 2
    
    # Plot time-frequency map
    im = ax_wavelet.pcolormesh(wavelet_time, frequencies, power, cmap='jet', shading='gouraud')
    plt.colorbar(im, ax=ax_wavelet, label='Power')
    
    # Add frequency band markers
    ax_wavelet.axhline(y=4, color='white', linestyle='--', alpha=0.7)   # Delta-Theta
    ax_wavelet.axhline(y=8, color='white', linestyle='--', alpha=0.7)   # Theta-Alpha
    ax_wavelet.axhline(y=13, color='white', linestyle='--', alpha=0.7)  # Alpha-Beta
    ax_wavelet.axhline(y=30, color='white', linestyle='--', alpha=0.7)  # Beta-Gamma
    
    # Add band labels
    ax_wavelet.text(wavelet_time[-1]*1.02, 2, 'Delta', fontsize=8, color='white')
    ax_wavelet.text(wavelet_time[-1]*1.02, 6, 'Theta', fontsize=8, color='white')
    ax_wavelet.text(wavelet_time[-1]*1.02, 10, 'Alpha', fontsize=8, color='white')
    ax_wavelet.text(wavelet_time[-1]*1.02, 20, 'Beta', fontsize=8, color='white')
    ax_wavelet.text(wavelet_time[-1]*1.02, 38, 'Gamma', fontsize=8, color='white')
    
    ax_wavelet.set_title('Time-Frequency Analysis')
    ax_wavelet.set_xlabel('Time (s)')
    ax_wavelet.set_ylabel('Frequency (Hz)')
    ax_wavelet.set_ylim([1, 45])
    
    # 8. Band power distribution
    ax_energy = fig.add_subplot(gs[2, 2:])
    
    # Get band powers
    band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    open_powers = [info['open_powers'][band] for band in band_names]
    closed_powers = [info['closed_powers'][band] for band in band_names]
    
    # Create grouped bar chart
    x = np.arange(len(band_names))
    width = 0.35
    
    bars1 = ax_energy.bar(x - width/2, open_powers, width, label='Eyes Open', color='blue', alpha=0.7)
    bars2 = ax_energy.bar(x + width/2, closed_powers, width, label='Eyes Closed', color='red', alpha=0.7)
    
    # Add ratio labels for significant differences
    for i, (v1, v2) in enumerate(zip(open_powers, closed_powers)):
        ratio = v2/v1 if v1 > 0 else float('inf')
        if ratio != float('inf'):
            if ratio > 1.2 or ratio < 0.8:
                y_pos = max(v1, v2) * 1.05
                color = 'red' if v2 > v1 else 'blue'
                ax_energy.text(i, y_pos, f"{ratio:.1f}x", ha='center', va='bottom', 
                              color=color, fontweight='bold', fontsize=9)
    
    ax_energy.set_title('Band Power Distribution')
    ax_energy.set_xlabel('Frequency Band')
    ax_energy.set_ylabel('Power')
    ax_energy.set_xticks(x)
    ax_energy.set_xticklabels([band.capitalize() for band in band_names])
    ax_energy.legend()
    ax_energy.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig


def visualize_all_components_summary(ica, components: np.ndarray, y: np.ndarray, 
                                   component_info: Dict[int, Dict[str, Any]], 
                                   sorted_indices: List[int], channel_names: List[str], 
                                   fs: float = 127.95) -> plt.Figure:
    """
    Create summary visualization of all ICA components.
    
    Parameters:
    -----------
    ica : FastICA object
        Fitted ICA object
    components : np.ndarray
        ICA components array
    y : np.ndarray
        Labels (0=open, 1=closed)
    component_info : Dict[int, Dict[str, Any]]
        Component analysis information
    sorted_indices : List[int]
        Component indices sorted by discriminative score
    channel_names : List[str]
        Channel names
    fs : float
        Sampling frequency
        
    Returns:
    --------
    plt.Figure : Figure object
    """
    n_components = len(sorted_indices)
    
    # Create grid layout
    n_cols = min(4, n_components)
    n_rows = int(np.ceil(n_components / n_cols))
    
    # Create figure
    fig = plt.figure(figsize=(16, n_rows * 4))
    fig.suptitle('ICA Components Summary', fontsize=16)
    
    # Plot each component
    for i, comp_idx in enumerate(sorted_indices):
        info = component_info[comp_idx]
        primary_region = info['primary_region']
        disc_score = info['discriminative_score']
        comp_type = info['component_type']
        
        # Create subplot
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        
        # Plot time domain
        component_data = components[:, comp_idx]
        samples_to_show = min(int(fs * 2), component_data.shape[0])
        time = np.arange(samples_to_show) / fs
        
        # Get background color based on score
        if disc_score > np.percentile([info['discriminative_score'] for info in component_info.values()], 75):
            color = 'lightgreen'
        elif disc_score > np.percentile([info['discriminative_score'] for info in component_info.values()], 50):
            color = 'palegoldenrod' 
        elif disc_score > np.percentile([info['discriminative_score'] for info in component_info.values()], 25):
            color = 'lightsalmon'
        else:
            color = 'lightcoral'
            
        # Plot component
        ax.plot(time, component_data[:samples_to_show], color='purple', linewidth=1.0)
        ax.set_title(f'Component {comp_idx+1}: {comp_type}', fontsize=10)
        ax.text(0.5, 0.02, f"Score: {disc_score:.2f}, Region: {primary_region}", 
                transform=ax.transAxes, ha='center', fontsize=8)
        
        # Add alpha and beta ratios
        alpha_ratio = info['power_ratios']['alpha_ratio']
        beta_ratio = info['power_ratios']['beta_ratio']
        ax.text(0.5, -0.08, f"Alpha: {alpha_ratio:.2f}x, Beta: {beta_ratio:.2f}x", 
                transform=ax.transAxes, ha='center', fontsize=8)
        
        # Set background color
        ax.set_facecolor(color)
        
        # Labels only for edge subplots
        if i % n_cols == 0:
            ax.set_ylabel('Amplitude')
        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel('Time (s)')
            
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    return fig
