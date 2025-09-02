"""
RULSIF (Relative Unconstrained Least-Squares Importance Fitting) module.

This module implements the RULSIF algorithm for change-point detection
in time series data, originally proposed by Liu et al. (2012).
"""

from .rulsif import RULSIFDetector
from .kernels import GaussianKernel, LinearKernel, PolynomialKernel

__all__ = [
    "RULSIFDetector",
    "GaussianKernel",
    "LinearKernel", 
    "PolynomialKernel",
]

