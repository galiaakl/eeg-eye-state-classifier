"""
Configuration management for EEG analysis pipeline.
"""

import yaml
import os
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for EEG analysis pipeline."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration manager.
        
        Parameters:
        -----------
        config_path : str, optional
            Path to configuration file. If None, uses default config.yaml
        """
        if config_path is None:
            # Get the project root directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            config_path = os.path.join(project_root, 'config', 'config.yaml')
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            return self._get_default_config()
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file loading fails."""
        return {
            'data': {
                'sampling_frequency': 127.95,
                'label_column': 'eyeDetection'
            },
            'preprocessing': {
                'filters': {
                    'notch': {'frequency': 50.0, 'quality_factor': 35},
                    'bandpass': {'low_cut': 1.0, 'high_cut': 45.0, 'order': 6}
                },
                'ica': {
                    'n_components': 14,
                    'max_iter': 2000,
                    'tolerance': 0.0001,
                    'random_state': 42
                }
            },
            'features': {
                'window_size': 2.0,
                'overlap': 0.5,
                'bands': {
                    'delta': [1, 4],
                    'theta': [4, 8],
                    'alpha': [8, 13],
                    'beta': [13, 30],
                    'gamma': [30, 45]
                }
            },
            'classification': {
                'test_size': 0.2,
                'validation_size': 0.25,
                'cv_folds': 5,
                'random_state': 42,
                'n_selected_features': 20
            },
            'visualization': {
                'time_window': 3.0,
                'figure_size': [20, 12],
                'dpi': 300
            }
        }
    
    def get(self, key_path: str, default=None):
        """
        Get configuration value using dot notation.
        
        Parameters:
        -----------
        key_path : str
            Dot-separated path to configuration value (e.g., 'data.sampling_frequency')
        default : Any
            Default value if key not found
            
        Returns:
        --------
        Any : Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            logger.warning(f"Configuration key '{key_path}' not found, using default: {default}")
            return default
    
    def update(self, key_path: str, value: Any):
        """
        Update configuration value using dot notation.
        
        Parameters:
        -----------
        key_path : str
            Dot-separated path to configuration value
        value : Any
            New value to set
        """
        keys = key_path.split('.')
        config_section = self.config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config_section:
                config_section[key] = {}
            config_section = config_section[key]
        
        # Set the value
        config_section[keys[-1]] = value
        logger.info(f"Updated configuration: {key_path} = {value}")
    
    def save(self, output_path: str = None):
        """
        Save current configuration to file.
        
        Parameters:
        -----------
        output_path : str, optional
            Path to save configuration. If None, overwrites original file.
        """
        if output_path is None:
            output_path = self.config_path
        
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")


# Global configuration instance
_config = None

def get_config(config_path: str = None) -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config

def reload_config(config_path: str = None):
    """Reload configuration from file."""
    global _config
    _config = Config(config_path)
