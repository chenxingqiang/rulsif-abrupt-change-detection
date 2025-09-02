"""
Time series anomaly detection module.

This module contains algorithms specifically designed for detecting anomalies
in time series data, including RULSIF, streaming methods, and other
time-aware approaches.
"""

from .rulsif.rulsif import RULSIFDetector
# from .streaming import StreamingAnomaly  # TODO: Implement this module
# from .student_t import StudentTAnomaly   # TODO: Implement this module

__all__ = [
    "RULSIFDetector",
    # "StreamingAnomaly", 
    # "StudentTAnomaly",
]

