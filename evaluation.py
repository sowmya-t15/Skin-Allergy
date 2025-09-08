"""
Model Evaluation utilities for Skin Allergy Risk Prediction
Comprehensive evaluation metrics and reporting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve
import logging
from typing import Dict, List, Tuple, Any
import config

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL),
                   format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluation class with detailed metrics and visualizations
    """
    
    def __init__(self):
        """Initialize the evaluator"""
        self.evaluation_results = {}
        logger.info("Model Evaluator initialized")
    
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate basic classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0)
        }
        
        return metrics
    
    def calculate_multiclass_roc_auc(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate ROC AUC for multiclass classification"""
        try:
            roc_metrics = {
                'roc_auc_ovr_macro': roc_auc_score(y_true, y_pred_proba, 
                                                  multi_class='ovr', average='macro'),
                'roc_auc_ovr_weighted': roc_auc_score(y_true, y_pred_proba, 
                                                     multi_class='ovr', average='weighted'),
                'roc_auc_ovo_macro': roc_auc_score(y_true, y_pred_proba, 
                                                  multi_class='ovo', average='macro'),
                'roc_auc_ovo_weighted': roc_auc_score(y_true, y_pred_proba, 
                                                     multi_class='ovo', average='weighted')
            }
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC: {str(e)}")
            roc_metrics = {
                'roc_auc_ovr_macro': 0.0,
                'roc_auc_ovr_weighted': 0.0,
                'roc_auc_ovo_macro': 0.0,
                'roc_auc_ovo_weighted': 0.0
            }
        
        return roc_metrics
    
    def calculate_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   class_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate per-class metrics"""
        class_report = classification_report(y_true, y_pred, 
                                           target_names=class_names, 
                                           output_dict=True, 
                                           zero_division=0)
        
        per_class_metrics = {}
        for class_name in class_names:
            if class_name in class_report:
                per_class_metrics[class_name] = class_report[class_name]
        
        return per_class_metrics
    
    def calculate_confusion_matrix_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                         class_names: List[str]) -> Dict[str, Any]:
        """Calculate confusion matrix and related metrics"""
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Calculate per-class accuracy (diagonal elements of normalized CM)
        per_class_accuracy = np.diag(cm_normalized)
        
        # Calculate misclassification rates
        misclassification_matrix = 1 - cm_normalized
        
        cm_metrics = {
            'confusion_matrix': cm,
            'confusion_matrix_normalized': cm_normalized,
            'per_class_accuracy': {class_names[i]: acc for i, acc in enumerate(per_class_accuracy)},
            'misclassification_matrix': misclassification_matrix,
            'total_misclassifications': int(np.sum(cm) - np.trace(cm))
        }
        
        return cm_metrics
    
    def calculate_prediction_confidence_metrics(self, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate metrics related to prediction confidence"""
        # Maximum probability for each prediction (confidence)
        max_probs = np.max(y_pred_proba, axis=1)
        
        # Entropy (uncertainty measure)
        entropy = -np.sum(y_pred_proba * np.log(y_pred_proba + 1e-15), axis=1)
        
        # Probability margin (difference between top 2 probabilities)
        sorted_probs = np.sort(y_pred_proba, axis=1)
        prob_margin = sorted_probs[:, -1] - sorted_probs[:, -2]
        
        confidence_metrics = {
            'mean_confidence': float(np.mean(max_probs)),
            'std_confidence': float(np.std(max_probs)),
            'min_confidence': float(np.min(max_probs)),
            'max_confidence': float(np.max(max_probs)),
            'mean_entropy': float(np.mean(entropy)),
            'std_entropy': float(np.std(entropy)),
            'mean_prob_margin': float(np.mean(prob_margin)),
            'std_prob_margin': float(np.std(prob_margin))
        }
        
        return confidence_metrics
    
    def calculate_risk_based_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   class_names: List[str]) -> Dict[str, Any]:
        """Calculate metrics based on risk levels"""
        # Assume class ordering: Low (0), Medium (1), High (2)
        risk_mapping = {name: i for i, name in enumerate(class_names)}
        
        # Calculate risk-weighted accuracy (higher penalty for larger misclassifications)
        risk_weights = np.abs(y_true - y_pred)
        weighted_accuracy = 1 - np.mean(risk_weights) / (len(class_names) - 1)
        
        # Calculate over/under estimation rates
        over_estimation = np.mean(y_pred > y_true)  # Predicting higher risk than actual
        under_estimation = np.mean(y_pred < y_true)  # Predicting lower risk than actual
        
        # High-risk precision and recall (critical for medical applications)
        high_risk_class = len(class_names) - 1  # Assuming last class is highest risk
        high_risk_precision = precision_score(y_true == high_risk_class, 
                                            y_pred == high_risk_class, zero_division=0)
        high_risk_recall = recall_score(y_true == high_risk_class, 
                                      y_pred == high_risk_class, zero_division=0)
        
        risk_metrics = {
            'risk_weighted_accuracy': float(weighted_accuracy),
            'over_estimation_rate': float(over_estimation),
            'under_estimation_rate': float(under_estimation),
            'high_risk_precision': float(high_risk_precision),
            'high_risk_recall': float(high_risk_recall),
            'exact_risk_match_rate': float(np.mean(y_true == y_pred))
        }
        
        return risk_metrics
    
    def generate_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 y_pred_proba: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        logger.info("Generating comprehensive evaluation report...")
        
        evaluation_report = {}
        
        # Basic metrics
        evaluation_report['basic_metrics'] = self.calculate_basic_metrics(y_true, y_pred)
        
        # ROC AUC metrics
        evaluation_report['roc_auc_metrics'] = self.calculate_multiclass_roc_auc(y_true, y_pred_proba)
        
        # Per-class metrics
        evaluation_report['per_class_metrics'] = self.calculate_per_class_metrics(
            y_true, y_pred, class_names)
        
        # Confusion matrix metrics
        evaluation_report['confusion_matrix_metrics'] = self.calculate_confusion_matrix_metrics(
            y_true, y_pred, class_names)
        
        # Confidence metrics
        evaluation_report['confidence_metrics'] = self.calculate_prediction_confidence_metrics(
            y_pred_proba)
        
        # Risk-based metrics
        evaluation_report['risk_based_metrics'] = self.calculate_risk_based_metrics(
            y_true, y_pred, class_names)
        
        # Summary statistics
        evaluation_report['summary'] = {
            'total_samples': len(y_true),
            'n_classes': len(class_names),
            'class_names': class_names,
            'class_distribution': {
                class_names[i]: int(np.sum(y_true == i)) 
                for i in range(len(class_names))
            }
        }
        
        logger.info("Evaluation report generated successfully!")
        return evaluation_report
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: List[str], normalize: bool = True,
                             save_path: str = None, title: str = None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            cmap = 'Blues'
        else:
            fmt = 'd'
            cmap = 'Blues'
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Proportion' if normalize else 'Count'})
        
        if title:
            plt.title(title)
        else:
            plt.title('Normalized Confusion Matrix' if normalize else 'Confusion Matrix')
        
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                       class_names: List[str], save_path: str = None):
        """Plot ROC curves for each class"""
        n_classes = len(class_names)
        
        # Binarize the output for multiclass ROC
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        plt.figure(figsize=(10, 8))
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            auc_score = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
            
            plt.plot(fpr, tpr, linewidth=2, 
                    label=f'{class_names[i]} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Each Class')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                    class_names: List[str], save_path: str = None):
        """Plot precision-recall curves for each class"""
        n_classes = len(class_names)
        
        # Binarize the output for multiclass PR curves
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        plt.figure(figsize=(10, 8))
        
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
            avg_precision = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
            
            plt.plot(recall, precision, linewidth=2,
                    label=f'{class_names[i]} (AP = {avg_precision:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves for Each Class')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
            logger.info(f"Precision-recall curves saved to {save_path}")
        
        plt.show()
    
    def plot_prediction_confidence_distribution(self, y_pred_proba: np.ndarray,
                                              save_path: str = None):
        """Plot distribution of prediction confidence scores"""
        max_probs = np.max(y_pred_proba, axis=1)
        
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(max_probs, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Maximum Probability (Confidence)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Confidence')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.boxplot(max_probs, vert=True)
        plt.ylabel('Maximum Probability (Confidence)')
        plt.title('Confidence Score Box Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
            logger.info(f"Confidence distribution plot saved to {save_path}")
        
        plt.show()
    
    def plot_calibration_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                               class_names: List[str], save_path: str = None):
        """Plot calibration curves to assess probability calibration"""
        n_classes = len(class_names)
        
        # Binarize for calibration curves
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        plt.figure(figsize=(12, 8))
        
        for i in range(n_classes):
            plt.subplot(2, 2, i+1)
            
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true_bin[:, i], y_pred_proba[:, i], n_bins=10)
            
            plt.plot(mean_predicted_value, fraction_of_positives, "s-",
                    linewidth=2, label=f'{class_names[i]}')
            plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Fraction of Positives')
            plt.title(f'Calibration Curve - {class_names[i]}')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
            logger.info(f"Calibration curves saved to {save_path}")
        
        plt.show()
    
    def generate_evaluation_summary(self, evaluation_report: Dict[str, Any]) -> str:
        """Generate a text summary of evaluation results"""
        summary_lines = []
        
        summary_lines.append("="*60)
        summary_lines.append("MODEL EVALUATION SUMMARY")
        summary_lines.append("="*60)
        
        # Basic metrics
        basic_metrics = evaluation_report['basic_metrics']
        summary_lines.append("\nBASIC METRICS:")
        summary_lines.append("-"*30)
        summary_lines.append(f"Accuracy:           {basic_metrics['accuracy']:.4f}")
        summary_lines.append(f"F1 Score (Weighted): {basic_metrics['f1_weighted']:.4f}")
        summary_lines.append(f"Precision (Weighted): {basic_metrics['precision_weighted']:.4f}")
        summary_lines.append(f"Recall (Weighted):   {basic_metrics['recall_weighted']:.4f}")
        
        # ROC AUC metrics
        roc_metrics = evaluation_report['roc_auc_metrics']
        summary_lines.append(f"\nROC AUC (OvR):      {roc_metrics['roc_auc_ovr_weighted']:.4f}")
        summary_lines.append(f"ROC AUC (OvO):      {roc_metrics['roc_auc_ovo_weighted']:.4f}")
        
        # Risk-based metrics
        risk_metrics = evaluation_report['risk_based_metrics']
        summary_lines.append("\nRISK-BASED METRICS:")
        summary_lines.append("-"*30)
        summary_lines.append(f"Risk-Weighted Accuracy: {risk_metrics['risk_weighted_accuracy']:.4f}")
        summary_lines.append(f"High Risk Precision:    {risk_metrics['high_risk_precision']:.4f}")
        summary_lines.append(f"High Risk Recall:       {risk_metrics['high_risk_recall']:.4f}")
        summary_lines.append(f"Over-estimation Rate:   {risk_metrics['over_estimation_rate']:.4f}")
        summary_lines.append(f"Under-estimation Rate:  {risk_metrics['under_estimation_rate']:.4f}")
        
        # Confidence metrics
        conf_metrics = evaluation_report['confidence_metrics']
        summary_lines.append("\nCONFIDENCE METRICS:")
        summary_lines.append("-"*30)
        summary_lines.append(f"Mean Confidence:    {conf_metrics['mean_confidence']:.4f}")
        summary_lines.append(f"Std Confidence:     {conf_metrics['std_confidence']:.4f}")
        summary_lines.append(f"Mean Entropy:       {conf_metrics['mean_entropy']:.4f}")
        
        # Per-class performance
        per_class = evaluation_report['per_class_metrics']
        summary_lines.append("\nPER-CLASS PERFORMANCE:")
        summary_lines.append("-"*30)
        for class_name, metrics in per_class.items():
            summary_lines.append(f"{class_name:10s}: F1={metrics['f1-score']:.3f}, "
                                f"Precision={metrics['precision']:.3f}, "
                                f"Recall={metrics['recall']:.3f}")
        
        summary_lines.append("="*60)
        
        return "\n".join(summary_lines)
    
    def save_evaluation_report(self, evaluation_report: Dict[str, Any], 
                              filepath: str):
        """Save evaluation report to JSON file"""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        json_report = self._prepare_report_for_json(evaluation_report)
        
        with open(filepath, 'w') as f:
            json.dump(json_report, f, indent=2)
        
        logger.info(f"Evaluation report saved to {filepath}")
    
    def _prepare_report_for_json(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare evaluation report for JSON serialization"""
        json_report = {}
        
        for key, value in report.items():
            if isinstance(value, dict):
                json_report[key] = self._prepare_report_for_json(value)
            elif isinstance(value, np.ndarray):
                json_report[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                json_report[key] = float(value)
            else:
                json_report[key] = value
        
        return json_report
