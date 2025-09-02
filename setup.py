#!/usr/bin/env python3
"""
Setup script for Awesome Anomaly Detection package.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements(filename):
    with open(filename, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="awesome-anomaly-detection",
    version="0.1.0",
    author="Xingqiang Chen",
    author_email="your.email@example.com",
    description="A comprehensive Python library for anomaly detection including classical, deep learning, and time series methods",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/awesome-anomaly-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
        "deep": ["torch", "tensorflow"],
        "full": ["torch", "tensorflow", "scikit-learn", "pandas", "matplotlib", "seaborn"],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="anomaly detection, outlier detection, machine learning, deep learning, time series",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/awesome-anomaly-detection/issues",
        "Source": "https://github.com/yourusername/awesome-anomaly-detection",
        "Documentation": "https://awesome-anomaly-detection.readthedocs.io/",
    },
)
