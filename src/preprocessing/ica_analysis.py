# src/preprocessing/ica_analysis.py
"""
ICA Analysis Module for EEG Preprocessing

This module handles Independent Component Analysis (ICA) for EEG data,
including component analysis, scoring, and signal reconstruction.
"""

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.decomposition import FastICA
import logging
from typing import Tuple, List, Dict, Any, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.config import get_config

logger = logging.getLogger(__name__)


def define_channel_groups(channel_names: List[str]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Define channel groups based on brain regions.
    
    Parameters:
    -----------
    channel_names : List[str]
        List of EEG channel names
        
    Returns:
    --------
    tuple : (channel_groups, channel_to_group)
        - channel_groups: Dictionary mapping region names to channel lists
        - channel_to_group: Dictionary mapping channel names to regions
    """
    channel_groups = {
        'posterior': ['O1', 'O2', 'P7', 'P8'],
        'frontal': ['F3', 'F4', 'F7', 'F8', 'AF3', 'AF4', 'FP1', 'FP2'],
        'central': [],
    }
    
    channel_to_group = {}
    
    for channel in channel_names:
        if channel in channel_groups['posterior']:
            channel_to_group[channel] = 'posterior'
        elif channel in channel_groups['frontal']:
            channel_to_group[channel] = 'frontal'
        else:
            channel_to_group[channel] = 'central'
            channel_groups['central'].append(channel)
    
    logger.info("Channel to functional group mapping:")
    for group, channels in channel_groups.items():
        if channels:
            logger.info(f"  {group}: {channels}")
    
    return channel_groups, channel_to_group


def calculate_band_powers(data: np.ndarray, fs: float, bands: Optional[Dict[str, Tuple[float, float]]] = None) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Calculate power in different frequency bands.
    
    Parameters:
    -----------
    data : np.ndarray
        1D time series data
    fs : float
        Sampling frequency
    bands : dict, optional
        Dictionary of frequency bands. If None, uses default bands.
        
    Returns:
    --------
    tuple : (powers, frequencies, psd)
        - powers: Dictionary of band powers
        - frequencies: Frequency array
        - psd: Power spectral density
    """
    if bands is None:
        config = get_config()
        bands = config.get('features.bands', {
            'delta': [1, 4],
            'theta': [4, 8],
            'alpha': [8, 13],
            'beta': [13, 30],
            'gamma': [30, 45]
        })
    
    nperseg = min(1024, len(data))
    f, psd = signal.welch(data, fs, nperseg=nperseg)
    
    powers = {}
    for band_name, (low_freq, high_freq) in bands.items():
        band_mask = (f >= low_freq) & (f <= high_freq)
        if np.any(band_mask):
            powers[band_name] = np.mean(psd[band_mask])
        else:
            powers[band_name] = 0
    
    return powers, f, psd


def apply_global_ica(eeg_data: np.ndarray, y: np.ndarray, channel_names: List[str], 
                    fs: Optional[float] = None, n_components: Optional[int] = None) -> Tuple[FastICA, np.ndarray, Dict[int, Dict[str, Any]], List[int]]:
    """
    Apply ICA to all channels and analyze components.
    """
    # Get config values
    config = get_config()
    if fs is None:
        fs = config.get('data.sampling_frequency', 127.95)
    if n_components is None:
        n_components = min(len(channel_names), config.get('preprocessing.ica.n_components', 14))
    
    logger.info(f"Applying ICA to all {len(channel_names)} channels...")
    
    # Store original data statistics for proper reconstruction
    original_mean = np.mean(eeg_data, axis=0)
    original_std = np.std(eeg_data, axis=0)
    
    # Standardize the data for ICA
    eeg_data_std = (eeg_data - original_mean) / original_std
    
    logger.info(f"Extracting {n_components} components...")
    
    # Apply ICA
    ica = FastICA(
        n_components=n_components, 
        random_state=config.get('preprocessing.ica.random_state', 42),
        max_iter=config.get('preprocessing.ica.max_iter', 2000),
        tol=config.get('preprocessing.ica.tolerance', 0.0001)
    )
    
    # Store the standardization parameters in the ICA object for reconstruction
    ica.original_mean_ = original_mean
    ica.original_std_ = original_std
    
    components = ica.fit_transform(eeg_data_std)
    
    # Get mixing matrix
    mixing_matrix = ica.mixing_
    
    # Separate data by eye state
    eyes_open_indices = np.where(y == 0)[0]
    eyes_closed_indices = np.where(y == 1)[0]
    
    # Calculate metrics for each component
    component_info = {}
    
    # Calculate peak-to-peak values for all components first
    all_peak_to_peak = []
    for comp_idx in range(components.shape[1]):
        comp_data = components[:, comp_idx]
        comp_pp = np.max(comp_data) - np.min(comp_data)
        all_peak_to_peak.append(comp_pp)
    avg_peak_to_peak = np.mean(all_peak_to_peak)
    
    logger.info("Analyzing component characteristics:")
    for comp_idx in range(components.shape[1]):
        component_data = components[:, comp_idx]
        
        # Extract component data for each state
        comp_open = component_data[eyes_open_indices]
        comp_closed = component_data[eyes_closed_indices]
        
        # Calculate band powers for each state
        open_powers, f_open, psd_open = calculate_band_powers(comp_open, fs)
        closed_powers, f_closed, psd_closed = calculate_band_powers(comp_closed, fs)
        
        # Calculate band power ratios (closed/open)
        power_ratios = {}
        for band in open_powers:
            if open_powers[band] > 0:
                ratio = closed_powers[band] / open_powers[band]
            else:
                ratio = float('inf') if closed_powers[band] > 0 else 1.0
            power_ratios[f"{band}_ratio"] = ratio
        
        # Calculate standard deviation ratios
        std_open = np.std(comp_open)
        std_closed = np.std(comp_closed)
        std_ratio = std_closed / std_open if std_open > 0 else float('inf')
        
        # Calculate peak-to-peak
        peak_to_peak = all_peak_to_peak[comp_idx]
        
        # Analyze channel contributions
        channel_weights = mixing_matrix[:, comp_idx]
        max_weight_idx = np.argmax(np.abs(channel_weights))
        max_weight_channel = channel_names[max_weight_idx]
        
        # Calculate weighted channel contributions
        abs_weights = np.abs(channel_weights)
        total_weight = np.sum(abs_weights)
        
        # Group influence
        influence = {}
        channel_groups, channel_to_group = define_channel_groups(channel_names)
        
        for group, group_channels in channel_groups.items():
            group_indices = [i for i, ch in enumerate(channel_names) if ch in group_channels]
            if group_indices:
                group_contribution = np.sum(abs_weights[group_indices]) / total_weight
                influence[group] = group_contribution
            else:
                influence[group] = 0.0
        
        # Main region this component belongs to
        primary_region = max(influence, key=influence.get)
        region_score = influence[primary_region]
        
        # Calculate discriminative score based on band power differences
        if primary_region == 'posterior':
            # For posterior: prioritize alpha differences
            alpha_diff = np.abs(closed_powers['alpha'] - open_powers['alpha'])
            alpha_avg = (closed_powers['alpha'] + open_powers['alpha']) / 2 if (closed_powers['alpha'] + open_powers['alpha']) > 0 else 1
            alpha_score = alpha_diff / alpha_avg if alpha_avg > 0 else 0
            
            # Check if alpha is higher in closed eyes (expected for posterior)
            correct_pattern = closed_powers['alpha'] > open_powers['alpha']
            
            # Discriminative score
            disc_score = alpha_score * 3.0 + std_ratio * 0.5
            disc_score = disc_score * (2.0 if correct_pattern else 0.5)
            
        elif primary_region == 'frontal':
            # For frontal: prioritize beta and eye movement patterns
            beta_diff = np.abs(open_powers['beta'] - closed_powers['beta'])
            beta_avg = (open_powers['beta'] + closed_powers['beta']) / 2 if (open_powers['beta'] + closed_powers['beta']) > 0 else 1
            beta_score = beta_diff / beta_avg if beta_avg > 0 else 0
            
            # Check if beta is higher in open eyes (expected for frontal)
            correct_pattern = open_powers['beta'] > closed_powers['beta']
            
            # Discriminative score
            disc_score = beta_score * 2.0 + std_ratio * 0.8 + peak_to_peak * 0.5
            disc_score = disc_score * (1.5 if correct_pattern else 0.7)
            
        else:  # central or other
            # Generic scoring focusing on overall differences
            disc_score = std_ratio * 1.5 + sum([np.abs(closed_powers[b] - open_powers[b]) for b in ['alpha', 'beta', 'theta']]) * 0.5
        
        # Classify the component type
        if std_ratio > 10 or any(ratio > 10 for band, ratio in power_ratios.items()):
            component_type = "Likely artifact"
        elif primary_region == 'posterior' and power_ratios['alpha_ratio'] > 1.5:
            component_type = "Posterior alpha"
        elif primary_region == 'frontal' and power_ratios['beta_ratio'] < 0.8:
            component_type = "Frontal beta"
        elif primary_region == 'frontal' and peak_to_peak > avg_peak_to_peak:
            component_type = "Possible eye movement"
        else:
            component_type = "Mixed neural"
        
        # Store component information
        component_info[comp_idx] = {
            'open_powers': open_powers,
            'closed_powers': closed_powers,
            'power_ratios': power_ratios,
            'std_ratio': std_ratio,
            'peak_to_peak': peak_to_peak,
            'max_weight_channel': max_weight_channel,
            'channel_weights': channel_weights,
            'influence': influence,
            'primary_region': primary_region,
            'region_score': region_score,
            'discriminative_score': disc_score,
            'component_type': component_type,
            'f_open': f_open,
            'psd_open': psd_open, 
            'f_closed': f_closed,
            'psd_closed': psd_closed
        }
        
        # Print component summary
        logger.info(f"Component {comp_idx+1}: Score {disc_score:.4f}, "
                   f"Type: {component_type}, Region: {primary_region} ({region_score*100:.1f}%), "
                   f"Alpha ratio: {power_ratios['alpha_ratio']:.2f}, "
                   f"Beta ratio: {power_ratios['beta_ratio']:.2f}, "
                   f"Max channel: {max_weight_channel}")
    
    # Sort components by discriminative score
    sorted_components = sorted(component_info.items(), 
                             key=lambda x: x[1]['discriminative_score'], 
                             reverse=True)
    
    # Format the sorted list
    sorted_indices = [idx for idx, _ in sorted_components]
    
    # Print sorted components
    logger.info("Components sorted by discriminative score:")
    for rank, (comp_idx, info) in enumerate(sorted_components):
        logger.info(f"Rank {rank+1}: Component {comp_idx+1}, Score {info['discriminative_score']:.4f}, " 
                   f"Type: {info['component_type']}, Region: {info['primary_region']}")
    
    return ica, components, component_info, sorted_indices


def reconstruct_signals(ica: FastICA, components: np.ndarray, components_to_remove: Optional[List[int]] = None) -> np.ndarray:
    """
    Reconstruct signal with selected components removed.
    """
    # Create a clean version with selected components removed
    components_clean = components.copy()
    
    if components_to_remove:
        logger.info(f"Removing components: {[i+1 for i in components_to_remove]}")
        # Zero out the components to remove
        for comp_idx in components_to_remove:
            components_clean[:, comp_idx] = 0
    
    # Reconstruct to standardized space
    reconstructed_std = ica.inverse_transform(components_clean)
    
    # Convert back to original scale using stored parameters
    if hasattr(ica, 'original_mean_') and hasattr(ica, 'original_std_'):
        reconstructed_data = reconstructed_std * ica.original_std_ + ica.original_mean_
    else:
        logger.warning("No original scaling parameters found, returning standardized reconstruction")
        reconstructed_data = reconstructed_std
    
    logger.info(f"Signal reconstructed with {len(components_to_remove) if components_to_remove else 0} components removed")
    return reconstructed_data


def suggest_components_to_remove(component_info: Dict[int, Dict[str, Any]], 
                               threshold_score: float = 1.0) -> List[int]:
    """
    Suggest components to remove based on balanced analysis.
    """
    suggested_removals = []
    
    for comp_idx, info in component_info.items():
        # NEVER remove high-scoring components (score > 5.0)
        if info['discriminative_score'] > 5.0:
            continue
        
        # Remove artifacts
        if "artifact" in info['component_type'].lower():
            suggested_removals.append(comp_idx)
            logger.info(f"Component {comp_idx+1} suggested for removal: artifact (Score: {info['discriminative_score']:.3f})")
            
        # Remove low scoring components
        elif info['discriminative_score'] < threshold_score:
            suggested_removals.append(comp_idx)
            logger.info(f"Component {comp_idx+1} suggested for removal: low score (Score: {info['discriminative_score']:.3f})")
    
    return suggested_removals
