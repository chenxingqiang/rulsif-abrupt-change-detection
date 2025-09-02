"""
Core module for Awesome Anomaly Detection.

This module contains the base classes and core functionality that all
anomaly detection algorithms inherit from.
"""

from .base import BaseAnomalyDetector
from .metrics import AnomalyMetrics
from .visualization import AnomalyVisualizer

__all__ = [
    "BaseAnomalyDetector",
    "AnomalyMetrics",
    "AnomalyVisualizer",
]

