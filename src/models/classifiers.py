# src/models/classifiers.py
"""
Classification models for EEG eye state detection.
"""

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, List
import logging
import sys
import os
sys.path.append(os.getcwd())

logger = logging.getLogger(__name__)


def train_and_evaluate_classifiers(X: np.ndarray, y: np.ndarray, 
                                 feature_names: List[str] = None) -> Dict[str, Any]:
    """
    Train and evaluate multiple classifiers for eye state detection.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels (0=open, 1=closed)
    feature_names : List[str], optional
        Feature names for importance analysis
        
    Returns:
    --------
    Dict[str, Any] : Results dictionary with classifier performance
    """
    logger.info(f"Training classifiers on {X.shape[0]} samples with {X.shape[1]} features")
    
    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    logger.info(f"Class distribution: {class_distribution}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42)
    }
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Results storage
    results = {}
    
    # Train and evaluate each classifier
    for name, clf in classifiers.items():
        logger.info(f"Training {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        # Train on full training set
        clf.fit(X_train_scaled, y_train)
        
        # Predictions
        y_train_pred = clf.predict(X_train_scaled)
        y_test_pred = clf.predict(X_test_scaled)
        
        # Probabilities for ROC curve
        if hasattr(clf, 'predict_proba'):
            y_test_proba = clf.predict_proba(X_test_scaled)[:, 1]
        else:
            y_test_proba = clf.decision_function(X_test_scaled)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(y_test, y_test_proba)
            fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        except:
            roc_auc = 0.0
            fpr, tpr = [0, 1], [0, 1]
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        
        # Classification report
        class_report = classification_report(y_test, y_test_pred, output_dict=True)
        
        # Calculate sensitivity and specificity
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            sensitivity = specificity = 0
        
        # Feature importance (if available)
        feature_importance = None
        if hasattr(clf, 'feature_importances_') and feature_names is not None:
            feature_importance = dict(zip(feature_names, clf.feature_importances_))
        elif hasattr(clf, 'coef_') and feature_names is not None:
            # For linear models, use absolute coefficients
            feature_importance = dict(zip(feature_names, np.abs(clf.coef_[0])))
        
        # Store results
        results[name] = {
            'classifier': clf,
            'scaler': scaler,
            'cv_scores': cv_scores,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'fpr': fpr,
            'tpr': tpr,
            'feature_importance': feature_importance,
            'y_test': y_test,
            'y_test_pred': y_test_pred,
            'y_test_proba': y_test_proba
        }
        
        # Print results
        print(f"\n{name} Results:")
        print(f"  CV Accuracy: {cv_mean:.3f} ± {cv_std:.3f}")
        print(f"  Train Accuracy: {train_accuracy:.3f}")
        print(f"  Test Accuracy: {test_accuracy:.3f}")
        print(f"  ROC AUC: {roc_auc:.3f}")
        print(f"  Sensitivity: {sensitivity:.3f}")
        print(f"  Specificity: {specificity:.3f}")
    
    return results


def visualize_classification_results(results: Dict[str, Any], 
                                   feature_names: List[str] = None,
                                   save_path: str = 'results/figures/classification_results.png'):
    """
    Create comprehensive visualization of classification results.
    
    Parameters:
    -----------
    results : Dict[str, Any]
        Results from train_and_evaluate_classifiers
    feature_names : List[str], optional
        Feature names
    save_path : str
        Path to save the visualization
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('EEG Eye State Classification Results', fontsize=16)
    
    # 1. Accuracy comparison
    ax1 = axes[0, 0]
    classifier_names = list(results.keys())
    test_accuracies = [results[name]['test_accuracy'] for name in classifier_names]
    cv_means = [results[name]['cv_mean'] for name in classifier_names]
    
    x_pos = np.arange(len(classifier_names))
    width = 0.35
    
    ax1.bar(x_pos - width/2, test_accuracies, width, label='Test Accuracy', alpha=0.7)
    ax1.bar(x_pos + width/2, cv_means, width, label='CV Accuracy', alpha=0.7)
    
    ax1.set_xlabel('Classifier')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Classification Accuracy Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(classifier_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (test_acc, cv_acc) in enumerate(zip(test_accuracies, cv_means)):
        ax1.text(i - width/2, test_acc + 0.01, f'{test_acc:.3f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i + width/2, cv_acc + 0.01, f'{cv_acc:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. ROC curves
    ax2 = axes[0, 1]
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    for i, (name, result) in enumerate(results.items()):
        ax2.plot(result['fpr'], result['tpr'], 
                color=colors[i % len(colors)], 
                label=f"{name} (AUC={result['roc_auc']:.3f})",
                linewidth=2)
    
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Best classifier confusion matrix
    best_classifier = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
    best_result = results[best_classifier]
    
    ax3 = axes[0, 2]
    sns.heatmap(best_result['confusion_matrix'], 
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['Open', 'Closed'],
                yticklabels=['Open', 'Closed'],
                ax=ax3)
    ax3.set_title(f'Confusion Matrix - {best_classifier}')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    
    # 4. Feature importance (if available)
    ax4 = axes[1, 0]
    if best_result['feature_importance'] is not None and feature_names is not None:
        # Get top 10 features
        importance_items = list(best_result['feature_importance'].items())
        importance_items.sort(key=lambda x: x[1], reverse=True)
        top_features = importance_items[:10]
        
        feature_names_short = [name[:20] + '...' if len(name) > 20 else name for name, _ in top_features]
        importances = [importance for _, importance in top_features]
        
        y_pos = np.arange(len(feature_names_short))
        ax4.barh(y_pos, importances)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(feature_names_short)
        ax4.set_xlabel('Importance')
        ax4.set_title(f'Top 10 Features - {best_classifier}')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Feature importance\nnot available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Feature Importance')
    
    # 5. Performance metrics comparison
    ax5 = axes[1, 1]
    metrics = ['test_accuracy', 'roc_auc', 'sensitivity', 'specificity']
    metric_labels = ['Test Accuracy', 'ROC AUC', 'Sensitivity', 'Specificity']
    
    x_pos = np.arange(len(metrics))
    width = 0.15
    
    for i, name in enumerate(classifier_names):
        values = [results[name][metric] for metric in metrics]
        ax5.bar(x_pos + i*width, values, width, label=name, alpha=0.7)
    
    ax5.set_xlabel('Metrics')
    ax5.set_ylabel('Score')
    ax5.set_title('Performance Metrics Comparison')
    ax5.set_xticks(x_pos + width * (len(classifier_names)-1) / 2)
    ax5.set_xticklabels(metric_labels, rotation=45, ha='right')
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 1.1)
    
    # 6. Cross-validation scores
    ax6 = axes[1, 2]
    cv_data = []
    cv_labels = []
    
    for name in classifier_names:
        cv_scores = results[name]['cv_scores']
        cv_data.extend(cv_scores)
        cv_labels.extend([name] * len(cv_scores))
    
    # Create box plot
    unique_labels = list(classifier_names)
    cv_scores_by_classifier = [results[name]['cv_scores'] for name in unique_labels]
    
    bp = ax6.boxplot(cv_scores_by_classifier, labels=unique_labels)
    ax6.set_xlabel('Classifier')
    ax6.set_ylabel('CV Accuracy')
    ax6.set_title('Cross-Validation Score Distribution')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Classification results saved to: {save_path}")
    
    return fig


def get_classification_summary(results: Dict[str, Any]) -> str:
    """
    Generate a text summary of classification results.
    
    Parameters:
    -----------
    results : Dict[str, Any]
        Results from train_and_evaluate_classifiers
        
    Returns:
    --------
    str : Summary text
    """
    # Find best classifier
    best_name = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
    best_result = results[best_name]
    
    summary = []
    summary.append("="*60)
    summary.append("EEG EYE STATE CLASSIFICATION SUMMARY")
    summary.append("="*60)
    summary.append("")
    summary.append(f"Best Classifier: {best_name}")
    summary.append(f"  Test Accuracy: {best_result['test_accuracy']:.3f}")
    summary.append(f"  ROC AUC: {best_result['roc_auc']:.3f}")
    summary.append(f"  Sensitivity (Eyes Closed): {best_result['sensitivity']:.3f}")
    summary.append(f"  Specificity (Eyes Open): {best_result['specificity']:.3f}")
    summary.append(f"  CV Accuracy: {best_result['cv_mean']:.3f} ± {best_result['cv_std']:.3f}")
    summary.append("")
    
    summary.append("All Classifiers Performance:")
    summary.append("-" * 40)
    for name, result in results.items():
        summary.append(f"{name:20s}: {result['test_accuracy']:.3f} (AUC: {result['roc_auc']:.3f})")
    
    summary.append("")
    summary.append("Classification Report (Best Classifier):")
    summary.append("-" * 40)
    
    class_report = best_result['classification_report']
    for class_name in ['0', '1']:  # Eyes open, Eyes closed
        if class_name in class_report:
            precision = class_report[class_name]['precision']
            recall = class_report[class_name]['recall']
            f1 = class_report[class_name]['f1-score']
            support = class_report[class_name]['support']
            
            class_label = "Eyes Open" if class_name == '0' else "Eyes Closed"
            summary.append(f"{class_label:12s}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f} (n={support})")
    
    return "\n".join(summary)
