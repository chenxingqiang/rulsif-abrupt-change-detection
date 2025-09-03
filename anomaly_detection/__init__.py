"""
Awesome Anomaly Detection

A comprehensive Python library for anomaly detection including:
- Classical methods (Isolation Forest, LOF, One-Class SVM, etc.)
- Deep learning methods (Autoencoders, GANs, LSTM, etc.)
- Time series methods (RULSIF, streaming methods, etc.)
- Application-specific methods (KPI, Log, Driving data, etc.)

This library provides a unified interface for various anomaly detection algorithms,
making it easy to compare and use different approaches for your specific use case.
"""

# Version info
__version__ = "0.1.0"
__author__ = "Xingqiang Chen"
__email__ = "joy6677@qq.com"

# Core imports
try:
    from .core.base import BaseAnomalyDetector
    from .core.metrics import AnomalyMetrics
    from .core.visualization import AnomalyVisualizer
except ImportError:
    # Core modules not yet implemented
    pass

# Classical methods
try:
    # from .classical.isolation_forest import IsolationForest  # TODO: Implement
    # from .classical.lof import LOF  # TODO: Implement
    # from .classical.one_class_svm import OneClassSVM  # TODO: Implement
    # from .classical.pca_based import PCABased  # TODO: Implement
    # from .classical.clustering import ClusteringAnomaly  # TODO: Implement
    # from .classical.correlation import CorrelationAnomaly  # TODO: Implement
    pass
except ImportError:
    # Classical modules not yet implemented
    pass

# Deep learning methods
try:
    # from .deep_learning.autoencoder.vanilla_ae import AutoEncoder  # TODO: Implement
    # from .deep_learning.autoencoder.variational_ae import VariationalAutoEncoder  # TODO: Implement
    # from .deep_learning.autoencoder.robust_ae import RobustAutoEncoder  # TODO: Implement
    # from .deep_learning.gan_based.gan_anomaly import GANAnomaly  # TODO: Implement
    # from .deep_learning.lstm_based.lstm_anomaly import LSTMAnomaly  # TODO: Implement
    # from .deep_learning.hypersphere.hypersphere_learning import HypersphereLearning  # TODO: Implement
    pass
except ImportError:
    # Deep learning modules not yet implemented
    pass

# Time series methods
try:
    from .time_series.rulsif.rulsif import RULSIFDetector
    # from .time_series.streaming import StreamingAnomaly  # TODO: Implement
    # from .time_series.student_t import StudentTAnomaly  # TODO: Implement
except ImportError:
    # Time series modules not yet implemented
    pass

# Application-specific methods
try:
    # from .applications.kpi.kpi_anomaly import KPIAnomaly  # TODO: Implement
    # from .applications.log.log_anomaly import LogAnomaly  # TODO: Implement
    # from .applications.driving.driving_anomaly import DrivingAnomaly  # TODO: Implement
    pass
except ImportError:
    # Application modules not yet implemented
    pass

# Utility functions
try:
    # from .utils.config import AnomalyConfig  # TODO: Implement
    # from .utils.data_loader import DataLoader  # TODO: Implement
    pass
except ImportError:
    # Utility modules not yet implemented
    pass

# Define what gets imported with "from anomaly_detection import *"
__all__ = [
    # Core
    "BaseAnomalyDetector",
    "AnomalyMetrics", 
    "AnomalyVisualizer",
    
    # Classical
    # "IsolationForest",  # TODO: Implement
    # "LOF",  # TODO: Implement
    # "OneClassSVM",  # TODO: Implement
    # "PCABased",  # TODO: Implement
    # "ClusteringAnomaly",  # TODO: Implement
    # "CorrelationAnomaly",  # TODO: Implement
    
    # Deep Learning
    # "AutoEncoder",  # TODO: Implement
    # "VariationalAutoEncoder",  # TODO: Implement
    # "RobustAutoEncoder",  # TODO: Implement
    # "GANAnomaly",  # TODO: Implement
    # "LSTMAnomaly",  # TODO: Implement
    # "HypersphereLearning",  # TODO: Implement
    
    # Time Series
    "RULSIFDetector",
    # "StreamingAnomaly",  # TODO: Implement
    # "StudentTAnomaly",  # TODO: Implement
    
    # Applications
    # "KPIAnomaly",  # TODO: Implement
    # "LogAnomaly",  # TODO: Implement
    # "DrivingAnomaly",  # TODO: Implement
    
    # Utils
    # "AnomalyConfig",  # TODO: Implement
    # "DataLoader",  # TODO: Implement
]

# Package metadata
__package_info__ = {
    "name": "awesome-anomaly-detection",
    "version": __version__,
    "description": "A comprehensive Python library for anomaly detection",
    "author": __author__,
    "email": __email__,
    "url": "https://github.com/chenxingqiang/awesome-anomaly-detection",
    "license": "MIT",
    "python_requires": ">=3.7",
    "keywords": ["anomaly detection", "outlier detection", "machine learning", "deep learning", "time series"],
}

