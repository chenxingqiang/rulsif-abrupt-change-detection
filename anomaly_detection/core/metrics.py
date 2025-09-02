"""
Evaluation metrics for anomaly detection algorithms.

This module provides various metrics for evaluating the performance of
anomaly detection algorithms, including standard metrics and time series
specific metrics.
"""

import numpy as np
from typing import Union, List, Tuple, Optional
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, average_precision_score, confusion_matrix
)


class AnomalyMetrics:
    """
    Comprehensive evaluation metrics for anomaly detection.
    
    This class provides various metrics for evaluating anomaly detection
    algorithms, including standard classification metrics and specialized
    metrics for anomaly detection tasks.
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        pass
    
    @staticmethod
    def compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                            y_scores: Optional[np.ndarray] = None) -> dict:
        """
        Compute basic classification metrics.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels (0 for normal, 1 for anomaly).
            
        y_pred : np.ndarray
            Predicted labels (0 for normal, 1 for anomaly).
            
        y_scores : np.ndarray, optional
            Anomaly scores (higher = more anomalous).
            
        Returns
        -------
        dict
            Dictionary containing computed metrics.
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Additional metrics from confusion matrix
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        metrics['true_positives'] = tp
        
        # Specificity and sensitivity
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # FPR and FNR
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Score-based metrics (if scores provided)
        if y_scores is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
                metrics['average_precision'] = average_precision_score(y_true, y_scores)
            except ValueError:
                metrics['roc_auc'] = np.nan
                metrics['average_precision'] = np.nan
        
        return metrics
    
    @staticmethod
    def compute_time_series_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                  y_scores: Optional[np.ndarray] = None,
                                  window_size: int = 1) -> dict:
        """
        Compute time series specific metrics.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels for time series.
            
        y_pred : np.ndarray
            Predicted labels for time series.
            
        y_scores : np.ndarray, optional
            Anomaly scores for time series.
            
        window_size : int, default=1
            Window size for smoothing predictions.
            
        Returns
        -------
        dict
            Dictionary containing time series metrics.
        """
        metrics = {}
        
        # Apply window smoothing if specified
        if window_size > 1:
            y_pred_smooth = AnomalyMetrics._apply_window_smoothing(y_pred, window_size)
            y_true_smooth = AnomalyMetrics._apply_window_smoothing(y_true, window_size)
        else:
            y_pred_smooth = y_pred
            y_true_smooth = y_true
        
        # Compute basic metrics on smoothed data
        basic_metrics = AnomalyMetrics.compute_basic_metrics(y_true_smooth, y_pred_smooth, y_scores)
        metrics.update(basic_metrics)
        
        # Time series specific metrics
        metrics['detection_delay'] = AnomalyMetrics._compute_detection_delay(y_true, y_pred)
        metrics['false_alarm_rate'] = AnomalyMetrics._compute_false_alarm_rate(y_true, y_pred)
        metrics['missed_detection_rate'] = AnomalyMetrics._compute_missed_detection_rate(y_true, y_pred)
        
        return metrics
    
    @staticmethod
    def _apply_window_smoothing(y: np.ndarray, window_size: int) -> np.ndarray:
        """Apply sliding window smoothing to predictions."""
        from scipy.ndimage import uniform_filter1d
        return uniform_filter1d(y.astype(float), size=window_size) > 0.5
    
    @staticmethod
    def _compute_detection_delay(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute average detection delay for anomalies."""
        delays = []
        
        # Find true anomaly segments
        anomaly_segments = AnomalyMetrics._find_anomaly_segments(y_true)
        
        for start, end in anomaly_segments:
            # Find first detection in this segment
            segment_pred = y_pred[start:end+1]
            if np.any(segment_pred):
                first_detection = np.where(segment_pred)[0][0]
                delay = first_detection
                delays.append(delay)
        
        return np.mean(delays) if delays else 0.0
    
    @staticmethod
    def _compute_false_alarm_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute false alarm rate."""
        normal_indices = np.where(y_true == 0)[0]
        if len(normal_indices) == 0:
            return 0.0
        
        false_alarms = np.sum(y_pred[normal_indices])
        return false_alarms / len(normal_indices)
    
    @staticmethod
    def _compute_missed_detection_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute missed detection rate."""
        anomaly_indices = np.where(y_true == 1)[0]
        if len(anomaly_indices) == 0:
            return 0.0
        
        missed_detections = len(anomaly_indices) - np.sum(y_pred[anomaly_indices])
        return missed_detections / len(anomaly_indices)
    
    @staticmethod
    def _find_anomaly_segments(y: np.ndarray) -> List[Tuple[int, int]]:
        """Find continuous segments of anomalies."""
        segments = []
        start = None
        
        for i, label in enumerate(y):
            if label == 1 and start is None:
                start = i
            elif label == 0 and start is not None:
                segments.append((start, i - 1))
                start = None
        
        # Handle case where anomaly extends to end
        if start is not None:
            segments.append((start, len(y) - 1))
        
        return segments
    
    @staticmethod
    def compute_ranking_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> dict:
        """
        Compute ranking-based metrics for anomaly detection.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels.
            
        y_scores : np.ndarray
            Anomaly scores.
            
        Returns
        -------
        dict
            Dictionary containing ranking metrics.
        """
        metrics = {}
        
        try:
            # ROC AUC
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
            
            # Average Precision
            metrics['average_precision'] = average_precision_score(y_true, y_scores)
            
            # Precision at different recall levels
            recall_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            precision_at_recall = {}
            
            for recall_level in recall_levels:
                threshold = np.percentile(y_scores, (1 - recall_level) * 100)
                y_pred_at_threshold = (y_scores >= threshold).astype(int)
                precision_at_recall[f'precision_at_recall_{recall_level}'] = precision_score(
                    y_true, y_pred_at_threshold, zero_division=0
                )
            
            metrics.update(precision_at_recall)
            
        except ValueError as e:
            metrics['error'] = str(e)
        
        return metrics
    
    @staticmethod
    def compare_algorithms(results: dict) -> dict:
        """
        Compare multiple algorithms based on their metrics.
        
        Parameters
        ----------
        results : dict
            Dictionary with algorithm names as keys and metric dictionaries as values.
            
        Returns
        -------
        dict
            Comparison summary.
        """
        comparison = {}
        
        # Extract common metrics
        common_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        for metric in common_metrics:
            comparison[metric] = {}
            for algo_name, algo_metrics in results.items():
                if metric in algo_metrics and not np.isnan(algo_metrics[metric]):
                    comparison[metric][algo_name] = algo_metrics[metric]
        
        # Find best algorithm for each metric
        best_algorithms = {}
        for metric, algo_scores in comparison.items():
            if algo_scores:
                best_algo = max(algo_scores.items(), key=lambda x: x[1])
                best_algorithms[metric] = best_algo
        
        comparison['best_algorithms'] = best_algorithms
        
        return comparison
    
    @staticmethod
    def generate_report(metrics: dict, algorithm_name: str = "Algorithm") -> str:
        """
        Generate a formatted report from metrics.
        
        Parameters
        ----------
        metrics : dict
            Dictionary of computed metrics.
            
        algorithm_name : str
            Name of the algorithm for the report.
            
        Returns
        -------
        str
            Formatted report string.
        """
        report = f"\n{'='*50}\n"
        report += f"ANOMALY DETECTION EVALUATION REPORT\n"
        report += f"Algorithm: {algorithm_name}\n"
        report += f"{'='*50}\n\n"
        
        # Basic metrics
        report += "BASIC METRICS:\n"
        report += f"  Accuracy:     {metrics.get('accuracy', 'N/A'):.4f}\n"
        report += f"  Precision:    {metrics.get('precision', 'N/A'):.4f}\n"
        report += f"  Recall:       {metrics.get('recall', 'N/A'):.4f}\n"
        report += f"  F1-Score:     {metrics.get('f1', 'N/A'):.4f}\n"
        report += f"  Specificity:  {metrics.get('specificity', 'N/A'):.4f}\n"
        report += f"  Sensitivity:  {metrics.get('sensitivity', 'N/A'):.4f}\n\n"
        
        # Score-based metrics
        if 'roc_auc' in metrics:
            report += "SCORE-BASED METRICS:\n"
            report += f"  ROC AUC:      {metrics.get('roc_auc', 'N/A'):.4f}\n"
            report += f"  Avg Precision: {metrics.get('average_precision', 'N/A'):.4f}\n\n"
        
        # Time series metrics (if available)
        time_series_metrics = ['detection_delay', 'false_alarm_rate', 'missed_detection_rate']
        if any(metric in metrics for metric in time_series_metrics):
            report += "TIME SERIES METRICS:\n"
            report += f"  Detection Delay:      {metrics.get('detection_delay', 'N/A'):.4f}\n"
            report += f"  False Alarm Rate:     {metrics.get('false_alarm_rate', 'N/A'):.4f}\n"
            report += f"  Missed Detection Rate: {metrics.get('missed_detection_rate', 'N/A'):.4f}\n\n"
        
        # Confusion matrix
        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            report += "CONFUSION MATRIX:\n"
            report += f"  True Negatives:  {cm[0, 0]}\n"
            report += f"  False Positives: {cm[0, 1]}\n"
            report += f"  False Negatives: {cm[1, 0]}\n"
            report += f"  True Positives:  {cm[1, 1]}\n\n"
        
        report += f"{'='*50}\n"
        
        return report

