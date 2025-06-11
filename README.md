# EEG Eye State Classifier with Advanced Preprocessing and Model Evaulation
## Overview
This project focuses on the implementation of an EEG signal processing pipeline that automatically detects the open and closed eye states of a person from their EEG recordings. The system merges the conventional approaches to processing signals with machine learning methods for high accuracy classification of the eye states.

## Features:
### Signal Preprocessing
- **Notch Filtering:** Gets rid of 50/60 Hz power line noise.
- **Bandpass Filtering:** Extracts necessary EEG frequencies (1-45 Hz).
- **Independent Component Analysis (ICA)**: Disentangles neural signals from noise.
- **Component Classification:** Automagically distinguishes between artifact and neural components.


### Feature Extraction and Feature Selection
- Power spectral bands
- Some statistical moments
- Hjorth parameters
- A subset of features determined by multiple algorithms which is optimal.

### Model Comparison
- Five different classification algorithms.
- Cross-Validation: Performance assessment using stratified k-fold CV is more robust.
- Performance Metrics: Accuracy, AUC, sensitivity, specificity.

### Visualization
- **Component Analysis:** Time, frequency, and topographic domain plots.
- **Time and frequency analysis:** Wavelet based spectrograms.
- **Results of classification:** ROC curves, confusion matrices, and feature importance.
- **User Driven (interactive component):** Visual component suggestion system.

## Dataset
The pipeline is meant to process the EEG Eye State Dataset from UCI Machine Learning Repository.

### Dataset Specifications
- Format: .arff or .csv
- Channels: 14 EEG channels: AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
- Sample Rate: 128Hz
- Duration: Length of recording is continuous.
- Labels: Binarized into 0 and 1 (0 = eyes open, 1 = eyes closed).
- Size: Approximately 15,000 samples.



