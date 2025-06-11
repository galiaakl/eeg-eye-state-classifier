# src/features/extraction.py
"""
Focused Feature Extraction for ICA-cleaned EEG data.
"""

import numpy as np
from scipy import signal
from typing import Tuple, List, Dict, Any, Optional
import logging
import sys
import os
sys.path.append(os.getcwd())

logger = logging.getLogger(__name__)


def calculate_band_powers(data: np.ndarray, fs: float) -> Dict[str, float]:
    """Calculate power in different frequency bands for a single channel."""
    bands = {
        'delta': [1, 4],
        'theta': [4, 8], 
        'alpha': [8, 13],
        'beta': [13, 30],
        'gamma': [30, 45]
    }
    
    # Calculate PSD using Welch method
    nperseg = min(256, len(data))
    f, psd = signal.welch(data, fs, nperseg=nperseg)
    
    # Calculate band powers
    powers = {}
    for band_name, (low_freq, high_freq) in bands.items():
        band_mask = (f >= low_freq) & (f <= high_freq)
        if np.any(band_mask):
            powers[band_name] = np.mean(psd[band_mask])
        else:
            powers[band_name] = 0.0
    
    return powers


def calculate_statistical_features(data: np.ndarray) -> Dict[str, float]:
    """Calculate statistical features for a single channel."""
    features = {}
    
    # Basic statistics
    features['mean'] = np.mean(data)
    features['std'] = np.std(data)
    features['var'] = np.var(data)
    
    # Higher order moments (with safety checks)
    if features['std'] > 1e-10:
        normalized = (data - features['mean']) / features['std']
        features['skewness'] = np.mean(normalized ** 3)
        features['kurtosis'] = np.mean(normalized ** 4) - 3
    else:
        features['skewness'] = 0.0
        features['kurtosis'] = 0.0
    
    # Hjorth parameters
    diff1 = np.diff(data)
    diff2 = np.diff(diff1)
    
    if features['std'] > 1e-10:
        features['mobility'] = np.std(diff1) / features['std']
        if features['mobility'] > 1e-10 and np.std(diff1) > 1e-10:
            features['complexity'] = (np.std(diff2) / np.std(diff1)) / features['mobility']
        else:
            features['complexity'] = 0.0
    else:
        features['mobility'] = 0.0
        features['complexity'] = 0.0
    
    return features


def extract_focused_features(eeg_data: np.ndarray, y: np.ndarray, channel_names: List[str],
                           fs: float = 127.95, window_size: float = 2.0, 
                           overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract focused features from ICA-cleaned EEG data.
    
    Focus on features most relevant for eye state detection:
    - Alpha power (key for eyes closed detection)
    - Beta power (key for eyes open detection) 
    - Regional differences (posterior vs frontal)
    - Key statistical measures
    
    Parameters:
    -----------
    eeg_data : np.ndarray
        ICA-cleaned EEG data with shape (samples, channels)
    y : np.ndarray
        Labels (0=eyes open, 1=eyes closed)
    channel_names : List[str]
        Channel names
    fs : float
        Sampling frequency
    window_size : float
        Window size in seconds
    overlap : float
        Overlap between windows (0-1)
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, List[str]]
        - X_features: Feature matrix
        - y_windows: Labels for each window
        - feature_names: List of feature names
    """
    logger.info(f"Extracting focused features with {window_size}s windows, {overlap*100}% overlap")
    
    # Calculate window parameters
    window_samples = int(window_size * fs)
    step_size = int(window_samples * (1 - overlap))
    
    # Define key channel regions based on available channels
    posterior_channels = ['O1', 'O2', 'P7', 'P8']
    frontal_channels = ['F3', 'F4', 'F7', 'F8', 'AF3', 'AF4']
    
    # Find available channels
    posterior_indices = [i for i, ch in enumerate(channel_names) if ch in posterior_channels]
    frontal_indices = [i for i, ch in enumerate(channel_names) if ch in frontal_channels]
    
    logger.info(f"Available channels: {channel_names}")
    logger.info(f"Posterior channels: {[channel_names[i] for i in posterior_indices]}")
    logger.info(f"Frontal channels: {[channel_names[i] for i in frontal_indices]}")
    
    # Initialize feature storage
    X_features = []
    y_windows = []
    feature_names = []
    
    # Build feature names systematically
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    
    # 1. Individual channel features (focus on most informative channels)
    key_channels = []
    if posterior_indices:
        key_channels.extend([channel_names[i] for i in posterior_indices[:2]])  # Top 2 posterior
    if frontal_indices:
        key_channels.extend([channel_names[i] for i in frontal_indices[:2]])    # Top 2 frontal
    
    # If we don't have posterior/frontal, use first few channels
    if not key_channels:
        key_channels = channel_names[:min(4, len(channel_names))]
    
    logger.info(f"Key channels for individual features: {key_channels}")
    
    for ch_name in key_channels:
        # Band powers
        for band in bands:
            feature_names.append(f"{ch_name}_{band}")
        
        # Key ratios for eye state detection
        feature_names.append(f"{ch_name}_alpha_beta_ratio")
        feature_names.append(f"{ch_name}_theta_beta_ratio")
        
        # Important statistical features
        feature_names.append(f"{ch_name}_std")
        feature_names.append(f"{ch_name}_var")
        feature_names.append(f"{ch_name}_mobility")
        feature_names.append(f"{ch_name}_skewness")
    
    # 2. Regional average features
    if posterior_indices:
        for band in bands:
            feature_names.append(f"posterior_avg_{band}")
        feature_names.append("posterior_alpha_beta_ratio")
        feature_names.append("posterior_avg_std")
    
    if frontal_indices:
        for band in bands:
            feature_names.append(f"frontal_avg_{band}")
        feature_names.append("frontal_alpha_beta_ratio")
        feature_names.append("frontal_avg_std")
    
    # 3. Cross-regional features (key for eye state detection)
    if posterior_indices and frontal_indices:
        feature_names.append("posterior_frontal_alpha_ratio")
        feature_names.append("posterior_frontal_beta_ratio")
        feature_names.append("posterior_frontal_theta_ratio")
    
    # 4. Global features (all channels)
    for band in bands:
        feature_names.append(f"global_avg_{band}")
    feature_names.append("global_alpha_beta_ratio")
    feature_names.append("global_avg_std")
    
    logger.info(f"Total features to extract: {len(feature_names)}")
    
    # Extract features for each window
    n_valid_windows = 0
    
    for start_idx in range(0, len(eeg_data) - window_samples + 1, step_size):
        end_idx = start_idx + window_samples
        
        # Extract window data
        window_data = eeg_data[start_idx:end_idx, :]
        window_labels = y[start_idx:end_idx]
        
        # Skip windows with mixed labels (transition periods)
        if len(np.unique(window_labels)) != 1:
            continue
            
        window_label = window_labels[0]
        window_features = []
        
        # 1. Individual channel features
        for ch_name in key_channels:
            ch_idx = channel_names.index(ch_name)
            ch_data = window_data[:, ch_idx]
            
            # Band powers
            band_powers = calculate_band_powers(ch_data, fs)
            for band in bands:
                window_features.append(band_powers.get(band, 0.0))
            
            # Key ratios
            alpha = band_powers.get('alpha', 0.0)
            beta = band_powers.get('beta', 0.0)
            theta = band_powers.get('theta', 0.0)
            
            alpha_beta_ratio = alpha / beta if beta > 1e-10 else 0.0
            theta_beta_ratio = theta / beta if beta > 1e-10 else 0.0
            
            window_features.extend([alpha_beta_ratio, theta_beta_ratio])
            
            # Statistical features
            stats = calculate_statistical_features(ch_data)
            window_features.extend([
                stats['std'], 
                stats['var'], 
                stats['mobility'], 
                stats['skewness']
            ])
        
        # 2. Posterior regional features
        if posterior_indices:
            posterior_powers = {band: [] for band in bands}
            posterior_stats = []
            
            for ch_idx in posterior_indices:
                ch_data = window_data[:, ch_idx]
                
                # Band powers
                band_powers = calculate_band_powers(ch_data, fs)
                for band in bands:
                    posterior_powers[band].append(band_powers.get(band, 0.0))
                
                # Statistics
                stats = calculate_statistical_features(ch_data)
                posterior_stats.append(stats['std'])
            
            # Regional averages
            for band in bands:
                avg_power = np.mean(posterior_powers[band]) if posterior_powers[band] else 0.0
                window_features.append(avg_power)
            
            # Regional ratios and stats
            avg_alpha = np.mean(posterior_powers['alpha']) if posterior_powers['alpha'] else 0.0
            avg_beta = np.mean(posterior_powers['beta']) if posterior_powers['beta'] else 0.0
            post_alpha_beta = avg_alpha / avg_beta if avg_beta > 1e-10 else 0.0
            avg_std = np.mean(posterior_stats) if posterior_stats else 0.0
            
            window_features.extend([post_alpha_beta, avg_std])
        
        # 3. Frontal regional features
        if frontal_indices:
            frontal_powers = {band: [] for band in bands}
            frontal_stats = []
            
            for ch_idx in frontal_indices:
                ch_data = window_data[:, ch_idx]
                
                # Band powers
                band_powers = calculate_band_powers(ch_data, fs)
                for band in bands:
                    frontal_powers[band].append(band_powers.get(band, 0.0))
                
                # Statistics
                stats = calculate_statistical_features(ch_data)
                frontal_stats.append(stats['std'])
            
            # Regional averages
            for band in bands:
                avg_power = np.mean(frontal_powers[band]) if frontal_powers[band] else 0.0
                window_features.append(avg_power)
            
            # Regional ratios and stats
            avg_alpha = np.mean(frontal_powers['alpha']) if frontal_powers['alpha'] else 0.0
            avg_beta = np.mean(frontal_powers['beta']) if frontal_powers['beta'] else 0.0
            front_alpha_beta = avg_alpha / avg_beta if avg_beta > 1e-10 else 0.0
            avg_std = np.mean(frontal_stats) if frontal_stats else 0.0
            
            window_features.extend([front_alpha_beta, avg_std])
        
        # 4. Cross-regional features
        if posterior_indices and frontal_indices:
            # Calculate regional averages for comparison
            post_alpha = np.mean([calculate_band_powers(window_data[:, i], fs)['alpha'] for i in posterior_indices])
            post_beta = np.mean([calculate_band_powers(window_data[:, i], fs)['beta'] for i in posterior_indices])
            post_theta = np.mean([calculate_band_powers(window_data[:, i], fs)['theta'] for i in posterior_indices])
            
            front_alpha = np.mean([calculate_band_powers(window_data[:, i], fs)['alpha'] for i in frontal_indices])
            front_beta = np.mean([calculate_band_powers(window_data[:, i], fs)['beta'] for i in frontal_indices])
            front_theta = np.mean([calculate_band_powers(window_data[:, i], fs)['theta'] for i in frontal_indices])
            
            # Cross-regional ratios
            post_front_alpha = post_alpha / front_alpha if front_alpha > 1e-10 else 0.0
            post_front_beta = post_beta / front_beta if front_beta > 1e-10 else 0.0
            post_front_theta = post_theta / front_theta if front_theta > 1e-10 else 0.0
            
            window_features.extend([post_front_alpha, post_front_beta, post_front_theta])
        
        # 5. Global features (all channels)
        global_powers = {band: [] for band in bands}
        global_stats = []
        
        for ch_idx in range(len(channel_names)):
            ch_data = window_data[:, ch_idx]
            
            # Band powers
            band_powers = calculate_band_powers(ch_data, fs)
            for band in bands:
                global_powers[band].append(band_powers.get(band, 0.0))
            
            # Statistics
            stats = calculate_statistical_features(ch_data)
            global_stats.append(stats['std'])
        
        # Global averages
        for band in bands:
            avg_power = np.mean(global_powers[band]) if global_powers[band] else 0.0
            window_features.append(avg_power)
        
        # Global ratios and stats
        global_alpha = np.mean(global_powers['alpha']) if global_powers['alpha'] else 0.0
        global_beta = np.mean(global_powers['beta']) if global_powers['beta'] else 0.0
        global_alpha_beta = global_alpha / global_beta if global_beta > 1e-10 else 0.0
        global_avg_std = np.mean(global_stats) if global_stats else 0.0
        
        window_features.extend([global_alpha_beta, global_avg_std])
        
        # Add window to dataset
        X_features.append(window_features)
        y_windows.append(window_label)
        n_valid_windows += 1
    
    # Convert to arrays
    X_features = np.array(X_features)
    y_windows = np.array(y_windows)
    
    logger.info(f"Feature extraction completed:")
    logger.info(f"  Valid windows: {n_valid_windows}")
    logger.info(f"  Feature matrix shape: {X_features.shape}")
    logger.info(f"  Features per window: {X_features.shape[1]}")
    logger.info(f"  Class distribution: Eyes open: {np.sum(y_windows == 0)}, Eyes closed: {np.sum(y_windows == 1)}")
    
    return X_features, y_windows, feature_names
