# src/data/loader.py
"""
EEG Data Loading Module

This module handles loading and initial processing of EEG data from various formats.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Union
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EEGDataLoader:
    """
    A class to handle loading EEG data from different file formats.
    
    Supported formats:
    - ARFF files
    - CSV files
    - NPZ files (preprocessed data)
    """
    
    def __init__(self, sampling_rate: float = 127.95):
        """
        Initialize the EEG data loader.
        
        Parameters:
        -----------
        sampling_rate : float
            Sampling rate of the EEG data in Hz
        """
        self.sampling_rate = sampling_rate
        
    def load_arff_data(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load ARFF file format data (same as your original load_eeg_data function).
        """
        try:
            logger.info(f"Loading ARFF data from {file_path}")
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Find data section and attribute names
            data_start = next(i for i, line in enumerate(lines) 
                            if line.strip().startswith('@DATA'))
            attribute_lines = [line for line in lines 
                             if line.strip().startswith('@ATTRIBUTE')]
            attribute_names = [line.split()[1] for line in attribute_lines]
            
            # Parse data
            data = []
            for line in lines[data_start+1:]:
                if line.strip():
                    values = line.strip().split(',')
                    try:
                        data.append([float(val) for val in values])
                    except ValueError as e:
                        logger.warning(f"Skipping invalid line: {line.strip()}")
                        continue
            
            # Create DataFrame and separate features/labels
            df = pd.DataFrame(data, columns=attribute_names)
            X = df.iloc[:, :-1].values  # EEG channels
            y = df.iloc[:, -1].values   # Eye state
            channel_names = attribute_names[:-1]
            
            logger.info(f"Successfully loaded {X.shape[0]} samples with {X.shape[1]} channels")
            return X, y, channel_names
            
        except Exception as e:
            logger.error(f"Error loading ARFF dataset: {e}")
            return None, None, None
    
    def load_csv_data(self, file_path: str, 
                     label_column: str = 'eyeDetection') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load CSV file format data.
        """
        try:
            logger.info(f"Loading CSV data from {file_path}")
            
            # Load CSV file
            df = pd.read_csv(file_path)
            
            # Separate features and labels
            if label_column in df.columns:
                X = df.drop(columns=[label_column]).values
                y = df[label_column].values
                channel_names = df.drop(columns=[label_column]).columns.tolist()
            else:
                # Assume last column is the label
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
                channel_names = df.columns[:-1].tolist()
            
            logger.info(f"Successfully loaded {X.shape[0]} samples with {X.shape[1]} channels")
            return X, y, channel_names
            
        except Exception as e:
            logger.error(f"Error loading CSV dataset: {e}")
            return None, None, None
    
    def auto_load(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Automatically detect file format and load data.
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None, None, None
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.arff':
            return self.load_arff_data(file_path)
        elif file_extension == '.csv':
            return self.load_csv_data(file_path)
        else:
            logger.error(f"Unsupported file format: {file_extension}")
            return None, None, None
    
    def validate_data(self, X: np.ndarray, y: np.ndarray, 
                     channel_names: List[str]) -> bool:
        """
        Validate the loaded EEG data.
        """
        try:
            # Check if data is not None
            if X is None or y is None or channel_names is None:
                logger.error("Data contains None values")
                return False
            
            # Check dimensions
            if X.ndim != 2:
                logger.error(f"X should be 2D array, got {X.ndim}D")
                return False
            
            if y.ndim != 1:
                logger.error(f"y should be 1D array, got {y.ndim}D")
                return False
            
            # Check shapes match
            if X.shape[0] != len(y):
                logger.error(f"Number of samples mismatch: X has {X.shape[0]}, y has {len(y)}")
                return False
            
            if X.shape[1] != len(channel_names):
                logger.error(f"Number of channels mismatch: X has {X.shape[1]}, channel_names has {len(channel_names)}")
                return False
            
            # Check for valid labels (should be 0 or 1)
            unique_labels = np.unique(y)
            if not all(label in [0, 1] for label in unique_labels):
                logger.error(f"Invalid labels found: {unique_labels}. Expected only 0 and 1")
                return False
            
            # Check for NaN or infinite values
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                logger.error("X contains NaN or infinite values")
                return False
            
            if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                logger.error("y contains NaN or infinite values")
                return False
            
            logger.info("Data validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error during data validation: {e}")
            return False


# Convenience functions
def load_eeg_data(file_path: str, sampling_rate: float = 127.95) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Convenience function to load EEG data (matches your original function name).
    """
    loader = EEGDataLoader(sampling_rate=sampling_rate)
    return loader.auto_load(file_path)


def validate_eeg_data(X: np.ndarray, y: np.ndarray, 
                     channel_names: List[str], 
                     sampling_rate: float = 127.95) -> bool:
    """
    Convenience function to validate EEG data.
    """
    loader = EEGDataLoader(sampling_rate=sampling_rate)
    return loader.validate_data(X, y, channel_names)
