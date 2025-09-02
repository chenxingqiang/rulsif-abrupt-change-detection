"""
Visualization utilities for anomaly detection.

This module provides various plotting and visualization functions for
anomaly detection results, including time series plots, score distributions,
and comparison charts.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Union, Dict, Any
import warnings

# Set default style
plt.style.use('default')
sns.set_palette("husl")


class AnomalyVisualizer:
    """
    Comprehensive visualization tools for anomaly detection results.
    
    This class provides various plotting functions to visualize:
    - Time series data with detected anomalies
    - Anomaly score distributions
    - Algorithm comparison charts
    - Confusion matrices
    - ROC curves and precision-recall curves
    """
    
    def __init__(self, style: str = 'default', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Parameters
        ----------
        style : str, default='default'
            Matplotlib style to use.
            
        figsize : tuple, default=(12, 8)
            Default figure size (width, height).
        """
        self.style = style
        self.figsize = figsize
        self.colors = sns.color_palette("husl", 10)
        
        # Set style
        try:
            plt.style.use(style)
        except OSError:
            warnings.warn(f"Style '{style}' not found, using default")
            plt.style.use('default')
    
    def plot_time_series_with_anomalies(self, time_series: np.ndarray, 
                                       anomaly_labels: Optional[np.ndarray] = None,
                                       anomaly_scores: Optional[np.ndarray] = None,
                                       time_index: Optional[np.ndarray] = None,
                                       title: str = "Time Series with Anomalies",
                                       figsize: Optional[Tuple[int, int]] = None,
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot time series data with detected anomalies.
        
        Parameters
        ----------
        time_series : np.ndarray
            Time series data of shape (n_samples, n_features).
            
        anomaly_labels : np.ndarray, optional
            Binary anomaly labels (0 for normal, 1 for anomaly).
            
        anomaly_scores : np.ndarray, optional
            Continuous anomaly scores.
            
        time_index : np.ndarray, optional
            Time index for x-axis. If None, uses sample indices.
            
        title : str
            Plot title.
            
        figsize : tuple, optional
            Figure size. If None, uses default.
            
        save_path : str, optional
            Path to save the plot.
            
        Returns
        -------
        plt.Figure
            The created figure.
        """
        if figsize is None:
            figsize = self.figsize
        
        fig, axes = plt.subplots(2, 1, figsize=figsize, 
                                gridspec_kw={'height_ratios': [3, 1]})
        
        if time_index is None:
            time_index = np.arange(len(time_series))
        
        # Plot time series
        ax1 = axes[0]
        if time_series.ndim == 1:
            ax1.plot(time_index, time_series, 'b-', alpha=0.7, label='Time Series')
        else:
            for i in range(time_series.shape[1]):
                ax1.plot(time_index, time_series[:, i], alpha=0.7, 
                        label=f'Feature {i+1}')
        
        # Highlight anomalies if labels provided
        if anomaly_labels is not None:
            anomaly_indices = np.where(anomaly_labels == 1)[0]
            if len(anomaly_indices) > 0:
                ax1.scatter(time_index[anomaly_indices], 
                           time_series[anomaly_indices] if time_series.ndim == 1 
                           else time_series[anomaly_indices, 0],
                           color='red', s=50, alpha=0.8, label='Detected Anomalies')
        
        ax1.set_title(title)
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot anomaly scores if provided
        if anomaly_scores is not None:
            ax2 = axes[1]
            ax2.plot(time_index, anomaly_scores, 'g-', alpha=0.7, label='Anomaly Score')
            
            # Add threshold line
            threshold = np.percentile(anomaly_scores, 95)
            ax2.axhline(y=threshold, color='red', linestyle='--', alpha=0.8, 
                       label=f'Threshold (95th percentile)')
            
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Anomaly Score')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_anomaly_scores_distribution(self, scores: np.ndarray, 
                                        labels: Optional[np.ndarray] = None,
                                        title: str = "Anomaly Scores Distribution",
                                        figsize: Optional[Tuple[int, int]] = None,
                                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distribution of anomaly scores.
        
        Parameters
        ----------
        scores : np.ndarray
            Anomaly scores.
            
        labels : np.ndarray, optional
            True labels for coloring.
            
        title : str
            Plot title.
            
        figsize : tuple, optional
            Figure size.
            
        save_path : str, optional
            Path to save the plot.
            
        Returns
        -------
        plt.Figure
            The created figure.
        """
        if figsize is None:
            figsize = self.figsize
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        if labels is not None:
            normal_scores = scores[labels == 0]
            anomaly_scores = scores[labels == 1]
            
            ax1.hist(normal_scores, bins=30, alpha=0.7, label='Normal', 
                    color='blue', density=True)
            ax1.hist(anomaly_scores, bins=30, alpha=0.7, label='Anomaly', 
                    color='red', density=True)
            ax1.legend()
        else:
            ax1.hist(scores, bins=30, alpha=0.7, color='gray', density=True)
        
        ax1.set_xlabel('Anomaly Score')
        ax1.set_ylabel('Density')
        ax1.set_title('Score Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        if labels is not None:
            data_to_plot = [scores[labels == 0], scores[labels == 1]]
            labels_plot = ['Normal', 'Anomaly']
            ax2.boxplot(data_to_plot, labels=labels_plot, patch_artist=True)
            ax2.set_ylabel('Anomaly Score')
            ax2.set_title('Score Distribution by Class')
        else:
            ax2.boxplot(scores)
            ax2.set_ylabel('Anomaly Score')
            ax2.set_title('Score Distribution')
        
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray,
                            labels: List[str] = None,
                            title: str = "Confusion Matrix",
                            normalize: bool = False,
                            figsize: Optional[Tuple[int, int]] = None,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Parameters
        ----------
        confusion_matrix : np.ndarray
            2x2 confusion matrix.
            
        labels : list, optional
            Class labels. Defaults to ['Normal', 'Anomaly'].
            
        title : str
            Plot title.
            
        normalize : bool, default=False
            Whether to normalize the matrix.
            
        figsize : tuple, optional
            Figure size.
            
        save_path : str, optional
            Path to save the plot.
            
        Returns
        -------
        plt.Figure
            The created figure.
        """
        if labels is None:
            labels = ['Normal', 'Anomaly']
        
        if figsize is None:
            figsize = (8, 6)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if normalize:
            cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                       xticklabels=labels, yticklabels=labels, ax=ax)
        else:
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=labels, yticklabels=labels, ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray,
                       title: str = "ROC Curve",
                       figsize: Optional[Tuple[int, int]] = None,
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ROC curve.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels.
            
        y_scores : np.ndarray
            Anomaly scores.
            
        title : str
            Plot title.
            
        figsize : tuple, optional
            Figure size.
            
        save_path : str, optional
            Path to save the plot.
            
        Returns
        -------
        plt.Figure
            The created figure.
        """
        from sklearn.metrics import roc_curve, auc
        
        if figsize is None:
            figsize = (8, 6)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_scores: np.ndarray,
                                   title: str = "Precision-Recall Curve",
                                   figsize: Optional[Tuple[int, int]] = None,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot precision-recall curve.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels.
            
        y_scores : np.ndarray
            Anomaly scores.
            
        title : str
            Plot title.
            
        figsize : tuple, optional
            Figure size.
            
        save_path : str, optional
            Path to save the plot.
            
        Returns
        -------
        plt.Figure
            The created figure.
        """
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        if figsize is None:
            figsize = (8, 6)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)
        
        ax.plot(recall, precision, color='darkorange', lw=2,
               label=f'PR curve (AP = {avg_precision:.2f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_algorithm_comparison(self, results: Dict[str, Dict[str, float]],
                                 metrics: List[str] = None,
                                 title: str = "Algorithm Comparison",
                                 figsize: Optional[Tuple[int, int]] = None,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of multiple algorithms.
        
        Parameters
        ----------
        results : dict
            Dictionary with algorithm names as keys and metric dictionaries as values.
            
        metrics : list, optional
            List of metrics to compare. If None, uses common metrics.
            
        title : str
            Plot title.
            
        figsize : tuple, optional
            Figure size.
            
        save_path : str, optional
            Path to save the plot.
            
        Returns
        -------
        plt.Figure
            The created figure.
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        if figsize is None:
            figsize = (12, 8)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        algorithms = list(results.keys())
        x = np.arange(len(metrics))
        width = 0.8 / len(algorithms)
        
        for i, (algo_name, algo_metrics) in enumerate(results.items()):
            values = [algo_metrics.get(metric, 0) for metric in metrics]
            ax.bar(x + i * width, values, width, label=algo_name, alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
        ax.set_xticklabels(metrics, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_feature_importance(self, feature_names: List[str], 
                               importance_scores: np.ndarray,
                               title: str = "Feature Importance",
                               top_k: Optional[int] = None,
                               figsize: Optional[Tuple[int, int]] = None,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot feature importance scores.
        
        Parameters
        ----------
        feature_names : list
            List of feature names.
            
        importance_scores : np.ndarray
            Feature importance scores.
            
        title : str
            Plot title.
            
        top_k : int, optional
            Number of top features to show. If None, shows all.
            
        figsize : tuple, optional
            Figure size.
            
        save_path : str, optional
            Path to save the plot.
            
        Returns
        -------
        plt.Figure
            The created figure.
        """
        if figsize is None:
            figsize = (10, 6)
        
        # Sort features by importance
        sorted_indices = np.argsort(importance_scores)[::-1]
        
        if top_k is not None:
            sorted_indices = sorted_indices[:top_k]
        
        sorted_names = [feature_names[i] for i in sorted_indices]
        sorted_scores = importance_scores[sorted_indices]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.barh(range(len(sorted_names)), sorted_scores, color=self.colors[:len(sorted_names)])
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Importance Score')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
            ax.text(score + 0.01, i, f'{score:.3f}', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_dashboard(self, time_series: np.ndarray,
                        anomaly_labels: np.ndarray,
                        anomaly_scores: np.ndarray,
                        confusion_matrix: np.ndarray,
                        title: str = "Anomaly Detection Dashboard",
                        figsize: Optional[Tuple[int, int]] = None,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive dashboard with multiple plots.
        
        Parameters
        ----------
        time_series : np.ndarray
            Time series data.
            
        anomaly_labels : np.ndarray
            Detected anomaly labels.
            
        anomaly_scores : np.ndarray
            Anomaly scores.
            
        confusion_matrix : np.ndarray
            Confusion matrix.
            
        title : str
            Dashboard title.
            
        figsize : tuple, optional
            Figure size.
            
        save_path : str, optional
            Path to save the dashboard.
            
        Returns
        -------
        plt.Figure
            The created dashboard figure.
        """
        if figsize is None:
            figsize = (16, 12)
        
        fig = plt.figure(figsize=figsize)
        fig.suptitle(title, fontsize=16)
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Time series plot
        ax1 = fig.add_subplot(gs[0, :])
        time_index = np.arange(len(time_series))
        ax1.plot(time_index, time_series, 'b-', alpha=0.7, label='Time Series')
        
        # Highlight anomalies
        anomaly_indices = np.where(anomaly_labels == 1)[0]
        if len(anomaly_indices) > 0:
            ax1.scatter(time_index[anomaly_indices], time_series[anomaly_indices],
                       color='red', s=30, alpha=0.8, label='Detected Anomalies')
        
        ax1.set_title('Time Series with Detected Anomalies')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Anomaly scores
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(time_index, anomaly_scores, 'g-', alpha=0.7, label='Anomaly Score')
        threshold = np.percentile(anomaly_scores, 95)
        ax2.axhline(y=threshold, color='red', linestyle='--', alpha=0.8,
                   label=f'Threshold (95th percentile)')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Anomaly Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Score distribution
        ax3 = fig.add_subplot(gs[2, 0])
        normal_scores = anomaly_scores[anomaly_labels == 0]
        anomaly_scores_subset = anomaly_scores[anomaly_labels == 1]
        ax3.hist(normal_scores, bins=20, alpha=0.7, label='Normal', color='blue', density=True)
        ax3.hist(anomaly_scores_subset, bins=20, alpha=0.7, label='Anomaly', color='red', density=True)
        ax3.set_xlabel('Anomaly Score')
        ax3.set_ylabel('Density')
        ax3.set_title('Score Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Confusion matrix
        ax4 = fig.add_subplot(gs[2, 1])
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'], ax=ax4)
        ax4.set_title('Confusion Matrix')
        
        # Statistics
        ax5 = fig.add_subplot(gs[2, 2])
        ax5.axis('off')
        
        # Calculate statistics
        total_samples = len(anomaly_labels)
        total_anomalies = np.sum(anomaly_labels)
        anomaly_rate = total_anomalies / total_samples
        
        stats_text = f"""
        STATISTICS
        
        Total Samples: {total_samples}
        Detected Anomalies: {total_anomalies}
        Anomaly Rate: {anomaly_rate:.2%}
        
        Score Statistics:
        Mean: {np.mean(anomaly_scores):.3f}
        Std: {np.std(anomaly_scores):.3f}
        Min: {np.min(anomaly_scores):.3f}
        Max: {np.max(anomaly_scores):.3f}
        """
        
        ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_all_plots(self, save_dir: str, prefix: str = "anomaly_detection"):
        """
        Save all created plots to a directory.
        
        Parameters
        ----------
        save_dir : str
            Directory to save plots.
            
        prefix : str
            Prefix for filenames.
        """
        import os
        
        os.makedirs(save_dir, exist_ok=True)
        
        # This method would need to be called after creating plots
        # Implementation depends on how plots are stored
        pass

