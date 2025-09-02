# Awesome Anomaly Detection 🚨

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/awesome-anomaly-detection.svg)](https://badge.fury.io/py/awesome-anomaly-detection)
[![Documentation](https://readthedocs.org/projects/awesome-anomaly-detection/badge/?version=latest)](https://awesome-anomaly-detection.readthedocs.io/)

A comprehensive Python library for anomaly detection including classical methods, deep learning approaches, time series methods, and application-specific algorithms. This library provides a unified interface for various anomaly detection algorithms, making it easy to compare and use different approaches for your specific use case.

## 🌟 Features

- **Unified API**: Consistent interface across all algorithms
- **Comprehensive Coverage**: From classical to state-of-the-art methods
- **Easy Comparison**: Built-in benchmarking and comparison tools
- **Extensible**: Easy to add new algorithms
- **Well Documented**: Examples and tutorials for each method
- **Production Ready**: Optimized implementations with proper error handling

## 🚀 Quick Start

### Installation

```bash
# Install the core package
pip install awesome-anomaly-detection

# Install with deep learning support
pip install awesome-anomaly-detection[deep]

# Install with full dependencies
pip install awesome-anomaly-detection[full]

# Install from source
git clone https://github.com/yourusername/awesome-anomaly-detection.git
cd awesome-anomaly-detection
pip install -e .
```

### Basic Usage

```python
from anomaly_detection.time_series import RULSIFDetector
from anomaly_detection.classical import IsolationForest
import numpy as np

# Generate sample data
np.random.seed(42)
data = np.random.randn(1000, 3)

# Use RULSIF for time series change detection
rulsif = RULSIFDetector(alpha=0.5, n_kernels=50)
rulsif.fit(data)
scores = rulsif.score_samples(data)
change_points = rulsif.detect_changes(data)

# Use Isolation Forest for general anomaly detection
iso_forest = IsolationForest(contamination=0.1)
iso_forest.fit(data)
predictions = iso_forest.predict(data)
```

## 📚 Algorithm Gallery

### Classical Methods

#### Isolation Forest - ICDM 2008
- **Paper**: [Isolation Forest](https://ieeexplore.ieee.org/document/4781136)
- **Description**: An efficient anomaly detection algorithm based on the principle that anomalies are easier to isolate than normal points
- **Use Case**: High-dimensional data, fast detection
- **Implementation**: `anomaly_detection.classical.IsolationForest`

#### LOF: Identifying Density-Based Local Outliers - SIGMOD 2000
- **Paper**: [LOF: Identifying Density-Based Local Outliers](https://dl.acm.org/doi/10.1145/335191.335388)
- **Description**: Density-based local outlier detection using k-nearest neighbors
- **Use Case**: Local density variations, clustering-based detection
- **Implementation**: `anomaly_detection.classical.LOF`

#### Extended Isolation Forest
- **Description**: Improved version of Isolation Forest with better handling of axis-parallel splits
- **Use Case**: Better performance on structured data
- **Implementation**: `anomaly_detection.classical.ExtendedIsolationForest`

#### Support Vector Method for Novelty Detection - NIPS 2000
- **Paper**: [Support Vector Method for Novelty Detection](https://papers.nips.cc/paper/1999/hash/8725fb777f25776ffa5dde2722984a28-Abstract.html)
- **Description**: One-class SVM for novelty detection
- **Use Case**: Novelty detection, high-dimensional spaces
- **Implementation**: `anomaly_detection.classical.OneClassSVM`

#### One-Class Classification
- **Paper**: [One-Class SVMs for Document Classification](https://www.jmlr.org/papers/v2/scholkopf01a.html)
- **Description**: One-class classification methods for anomaly detection
- **Use Case**: Document classification, novelty detection
- **Implementation**: `anomaly_detection.classical.OneClassClassification`

#### Support Vector Data Description
- **Description**: Support Vector Data Description for one-class classification
- **Use Case**: One-class learning, boundary detection
- **Implementation**: `anomaly_detection.classical.SVDD`

#### Can I Trust My One-Class Classification?
- **Description**: Reliability assessment for one-class classifiers
- **Use Case**: Quality assessment, confidence estimation
- **Implementation**: `anomaly_detection.classical.OneClassReliability`

#### Efficient Anomaly Detection via Matrix Sketching - NIPS 2018
- **Paper**: [Efficient Anomaly Detection via Matrix Sketching](https://papers.nips.cc/paper/2018/hash/8f468c873a32bb0619eaeb2050ba45d1-Abstract.html)
- **Description**: Matrix sketching techniques for efficient anomaly detection
- **Use Case**: Large-scale data, streaming data
- **Implementation**: `anomaly_detection.classical.MatrixSketching`

#### PCA-based Methods
- **Paper**: [Robust Deep and Inductive Anomaly Detection](https://link.springer.com/chapter/10.1007/978-3-319-71249-9_42)
- **Description**: PCA-based anomaly detection methods
- **Use Case**: Dimensionality reduction, linear anomalies
- **Implementation**: `anomaly_detection.classical.PCABased`

#### A Loss Framework for Calibrated Anomaly Detection - NIPS 2018
- **Paper**: [A Loss Framework for Calibrated Anomaly Detection](https://papers.nips.cc/paper/2018/hash/5e62d03aec0d17facfc5355f90b0a4d1-Abstract.html)
- **Description**: Calibrated anomaly detection using proper loss functions
- **Use Case**: Calibrated scores, uncertainty quantification
- **Implementation**: `anomaly_detection.classical.CalibratedAnomaly`

#### Clustering-based Methods
- **Paper**: [A Practical Algorithm for Distributed Clustering and Outlier Detection](https://papers.nips.cc/paper/2018/hash/8f468c873a32bb0619eaeb2050ba45d1-Abstract.html)
- **Description**: Clustering-based outlier detection algorithms
- **Use Case**: Cluster-based anomalies, group detection
- **Implementation**: `anomaly_detection.classical.ClusteringAnomaly`

#### Correlation-based Methods
- **Paper**: [Detecting Multiple Periods and Periodic Patterns](https://dl.acm.org/doi/10.1145/3132847.3132866)
- **Description**: Correlation-based anomaly detection
- **Use Case**: Time series, periodic patterns
- **Implementation**: `anomaly_detection.classical.CorrelationAnomaly`

#### Ranking-based Methods
- **Paper**: [Ranking Causal Anomalies via Temporal and Dynamical Analysis](https://dl.acm.org/doi/10.1145/2939672.2939854)
- **Description**: Ranking-based causal anomaly detection
- **Use Case**: Causal analysis, temporal dependencies
- **Implementation**: `anomaly_detection.classical.RankingAnomaly`

### Deep Learning Methods

#### Generative Methods

##### Variational Autoencoder based Anomaly Detection
- **Description**: VAE-based anomaly detection using reconstruction probability
- **Use Case**: Complex patterns, generative modeling
- **Implementation**: `anomaly_detection.deep_learning.autoencoder.VariationalAutoEncoder`

##### Auto-encoder Methods
- **Paper**: [Learning Sparse Representation with Variational Auto-encoder](https://arxiv.org/abs/1704.04003)
- **Description**: Sparse autoencoder for anomaly detection
- **Use Case**: Feature learning, sparse representations
- **Implementation**: `anomaly_detection.deep_learning.autoencoder.AutoEncoder`

##### Robust Deep Autoencoders - KDD 2017
- **Paper**: [Anomaly Detection with Robust Deep Autoencoders](https://dl.acm.org/doi/10.1145/3097983.3098067)
- **Description**: Robust autoencoders for anomaly detection
- **Use Case**: Robust detection, adversarial robustness
- **Implementation**: `anomaly_detection.deep_learning.autoencoder.RobustAutoEncoder`

##### Deep Autoencoding Gaussian Mixture Model - ICLR 2018
- **Paper**: [Deep Autoencoding Gaussian Mixture Model](https://openreview.net/forum?id=BJJtn0cjg)
- **Description**: DAGMM for unsupervised anomaly detection
- **Use Case**: Density estimation, mixture modeling
- **Implementation**: `anomaly_detection.deep_learning.autoencoder.DAGMM`

##### Generative Probabilistic Novelty Detection - NIPS 2018
- **Paper**: [Generative Probabilistic Novelty Detection](https://papers.nips.cc/paper/2018/hash/5e62d03aec0d17facfc5355f90b0a4d1-Abstract.html)
- **Description**: GAN-based novelty detection
- **Use Case**: Novelty detection, generative modeling
- **Implementation**: `anomaly_detection.deep_learning.gan_based.GANAnomaly`

#### Variational Auto-encoder Methods

##### Multidimensional Time Series - ACML 2018
- **Paper**: [Multidimensional Time Series Anomaly Detection](https://www.acml-conf.org/2018/conference/accepted-papers/)
- **Description**: GRU-based Gaussian Mixture VAE for time series
- **Use Case**: Time series, sequential data
- **Implementation**: `anomaly_detection.deep_learning.autoencoder.GRUGaussianVAE`

##### Robot-Assisted Feeding - IEEE Robotics 2018
- **Paper**: [A Multimodel Anomaly Detector for Robot-Assisted Feeding](https://ieeexplore.ieee.org/document/8453268)
- **Description**: LSTM-based VAE for robotics applications
- **Use Case**: Robotics, multimodal data
- **Implementation**: `anomaly_detection.deep_learning.lstm_based.LSTMVAE`

#### GAN-based Methods

##### Unsupervised Anomaly Detection - IPMI 2017
- **Paper**: [Unsupervised Anomaly Detection with GANs](https://link.springer.com/chapter/10.1007/978-3-319-59050-9_12)
- **Description**: GAN-based anomaly detection for marker discovery
- **Use Case**: Medical imaging, marker discovery
- **Implementation**: `anomaly_detection.deep_learning.gan_based.GANAnomaly`

##### Efficient-GAN-Based Anomaly Detection - ICLR Workshop 2018
- **Description**: Efficient GAN implementations for anomaly detection
- **Use Case**: Efficiency, large-scale data
- **Implementation**: `anomaly_detection.deep_learning.gan_based.EfficientGAN`

#### Hypersphere Learning

##### Multi-view Time-Series - CIKM 2017
- **Paper**: [Anomaly Detection in Dynamic Networks](https://dl.acm.org/doi/10.1145/3132847.3132866)
- **Description**: Multi-view hypersphere learning for dynamic networks
- **Use Case**: Network analysis, multi-view data
- **Implementation**: `anomaly_detection.deep_learning.hypersphere.MultiViewHypersphere`

##### Deep into Hypersphere - IJCAI 2018
- **Paper**: [Deep into Hypersphere: Robust and Unsupervised Anomaly Discovery](https://www.ijcai.org/proceedings/2018/0470.pdf)
- **Description**: Robust hypersphere learning for anomaly discovery
- **Use Case**: Robust detection, unsupervised learning
- **Implementation**: `anomaly_detection.deep_learning.hypersphere.RobustHypersphere`

#### One-Class Classification

##### High-dimensional Anomaly Detection - Pattern Recognition 2018
- **Paper**: [High-dimensional and Large-scale Anomaly Detection](https://www.sciencedirect.com/science/article/abs/pii/S0031320318300228)
- **Description**: Linear one-class SVM with deep learning
- **Use Case**: High-dimensional data, scalability
- **Implementation**: `anomaly_detection.deep_learning.one_class.DeepOneClassSVM`

##### Optimal Single-Class Classification - NIPS 2007
- **Paper**: [Optimal Single-Class Classification](https://papers.nips.cc/paper/2007/hash/5e62d03aec0d17facfc5355f90b0a4d1-Abstract.html)
- **Description**: Optimal strategies for single-class classification
- **Use Case**: Optimal classification, theoretical foundations
- **Implementation**: `anomaly_detection.deep_learning.one_class.OptimalOneClass`

##### Deep One-Class Classification - ICML 2018
- **Paper**: [Deep One-Class Classification](https://proceedings.mlr.press/v80/ruff18a.html)
- **Description**: Deep learning for one-class classification
- **Use Case**: Deep learning, one-class problems
- **Implementation**: `anomaly_detection.deep_learning.one_class.DeepOneClass`

#### Energy-based Methods

##### Deep Structured Energy Based Models - ICML 2016
- **Paper**: [Deep Structured Energy Based Models](https://proceedings.mlr.press/v48/zhai16.html)
- **Description**: Energy-based models for anomaly detection
- **Use Case**: Energy modeling, structured data
- **Implementation**: `anomaly_detection.deep_learning.energy.EnergyBasedAnomaly`

#### Time Series Methods

##### Generalized Student-t - AAAI 2013
- **Paper**: [A Generalized Student-t Based Approach](https://ojs.aaai.org/index.php/AAAI/article/view/8641)
- **Description**: Student-t distribution for mixed-type anomaly detection
- **Use Case**: Mixed data types, statistical modeling
- **Implementation**: `anomaly_detection.time_series.StudentTAnomaly`

##### Stochastic Online Anomaly Analysis - IJCAI 2017
- **Paper**: [Stochastic Online Anomaly Analysis](https://www.ijcai.org/proceedings/2017/0329.pdf)
- **Description**: Online anomaly analysis for streaming time series
- **Use Case**: Streaming data, online learning
- **Implementation**: `anomaly_detection.time_series.StreamingAnomaly`

##### LSTM for Anomaly Detection
- **Description**: Long short-term memory networks for anomaly detection
- **Use Case**: Sequential data, temporal dependencies
- **Implementation**: `anomaly_detection.deep_learning.lstm_based.LSTMAnomaly`

##### LSTM Encoder-Decoder - ICML 2016 Workshop
- **Paper**: [LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection](https://arxiv.org/abs/1607.00148)
- **Description**: LSTM encoder-decoder for multi-sensor data
- **Use Case**: Multi-sensor data, reconstruction-based
- **Implementation**: `anomaly_detection.deep_learning.lstm_based.LSTMEncoderDecoder`

#### Interpretation Methods

##### Contextual Outlier Interpretation - IJCAI 2018
- **Paper**: [Contextual Outlier Interpretation](https://www.ijcai.org/proceedings/2018/0329.pdf)
- **Description**: Contextual interpretation of outliers
- **Use Case**: Interpretability, explainable AI
- **Implementation**: `anomaly_detection.deep_learning.interpretation.ContextualInterpretation`

#### Evaluation Metrics

##### Precision and Recall for Time Series - NIPS 2018
- **Paper**: [Precision and Recall for Time Series](https://papers.nips.cc/paper/2018/hash/5e62d03aec0d17facfc5355f90b0a4d1-Abstract.html)
- **Description**: Time series-specific evaluation metrics
- **Use Case**: Evaluation, benchmarking
- **Implementation**: `anomaly_detection.core.metrics.TimeSeriesMetrics`

#### Geometric Transformation

##### Deep Anomaly Detection Using Geometric Transformations - NIPS 2018
- **Paper**: [Deep Anomaly Detection Using Geometric Transformations](https://papers.nips.cc/paper/2018/hash/5e62d03aec0d17facfc5355f90b0a4d1-Abstract.html)
- **Description**: Geometric transformations for anomaly detection
- **Use Case**: Geometric invariance, transformation robustness
- **Implementation**: `anomaly_detection.deep_learning.geometric.GeometricAnomaly`

#### Feedback Methods

##### Incorporating Feedback - KDD 2017 Workshop
- **Paper**: [Incorporating Feedback into Tree-based Anomaly Detection](https://dl.acm.org/doi/10.1145/3097983.3098067)
- **Description**: Interactive anomaly detection with feedback
- **Use Case**: Interactive detection, human feedback
- **Implementation**: `anomaly_detection.deep_learning.feedback.FeedbackAnomaly`

##### Feedback-Guided Anomaly Discovery - KDD 2018
- **Paper**: [Feedback-Guided Anomaly Discovery](https://dl.acm.org/doi/10.1145/3219819.3219820)
- **Description**: Online optimization with feedback guidance
- **Use Case**: Online learning, feedback integration
- **Implementation**: `anomaly_detection.deep_learning.feedback.FeedbackGuided`

### Time Series Methods

#### RULSIF (Relative Unconstrained Least-Squares Importance Fitting)
- **Paper**: [Change-Point Detection in Time-Series Data by Relative Density-Ratio Estimation](https://arxiv.org/abs/1208.1963)
- **Description**: RULSIF algorithm for change-point detection in time series
- **Use Case**: Change point detection, distribution shifts
- **Implementation**: `anomaly_detection.time_series.RULSIFDetector`

### Application-Specific Methods

#### KPI Anomaly Detection

##### Unsupervised KPI Anomaly Detection - WWW 2018
- **Paper**: [Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs](https://dl.acm.org/doi/10.1145/3178876.3185996)
- **Description**: VAE-based anomaly detection for seasonal KPIs
- **Use Case**: Web applications, seasonal patterns
- **Implementation**: `anomaly_detection.applications.kpi.KPIAnomaly`

#### Log Anomaly Detection

##### DeepLog - CCS 2017
- **Paper**: [DeepLog: Anomaly Detection and Diagnosis from System Logs](https://dl.acm.org/doi/10.1145/3133956.3134015)
- **Description**: Deep learning for system log anomaly detection
- **Use Case**: System monitoring, log analysis
- **Implementation**: `anomaly_detection.applications.log.DeepLog`

##### Mining Invariants from Logs - USENIX 2010
- **Paper**: [Mining Invariants from Logs for System Problem Detection](https://www.usenix.org/legacy/events/osdi10/tech/full_papers/Xu.pdf)
- **Description**: Invariant mining for system problem detection
- **Use Case**: System invariants, problem detection
- **Implementation**: `anomaly_detection.applications.log.InvariantMining`

#### Driving Data Anomaly Detection
- **Description**: Anomaly detection for driving behavior analysis
- **Use Case**: Automotive, driving safety
- **Implementation**: `anomaly_detection.applications.driving.DrivingAnomaly`

### Survey Papers

- **Anomaly Detection in Dynamic Networks: A Survey**
- **Anomaly Detection: A Survey**
- **A Survey of Recent Trends in One Class Classification**
- **A Survey on Unsupervised Outlier Detection in High-dimensional Numerical Data**

## 📖 Documentation

For detailed documentation, visit [https://awesome-anomaly-detection.readthedocs.io/](https://awesome-anomaly-detection.readthedocs.io/)

## 🧪 Examples

Check out the `examples/` directory for comprehensive usage examples:

- **Classical Methods**: `examples/classical_examples.py`
- **Deep Learning**: `examples/deep_learning_examples.py`
- **Time Series**: `examples/time_series_examples.py`
- **Applications**: `examples/application_examples.py`

## 🚀 Quick Examples

### Classical Anomaly Detection

```python
from anomaly_detection.classical import IsolationForest, LOF, OneClassSVM
import numpy as np

# Generate sample data
np.random.seed(42)
data = np.random.randn(1000, 3)

# Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(data)
iso_scores = iso_forest.score_samples(data)

# Local Outlier Factor
lof = LOF(n_neighbors=20)
lof.fit(data)
lof_scores = lof.score_samples(data)

# One-Class SVM
ocsvm = OneClassSVM(kernel='rbf', nu=0.1)
ocsvm.fit(data)
ocsvm_scores = ocsvm.score_samples(data)
```

### Deep Learning Anomaly Detection

```python
from anomaly_detection.deep_learning.autoencoder import AutoEncoder
import torch

# Autoencoder for anomaly detection
autoencoder = AutoEncoder(
    input_dim=3,
    hidden_dims=[64, 32, 16],
    latent_dim=8
)

autoencoder.fit(data, epochs=100, batch_size=32)
reconstruction_scores = autoencoder.score_samples(data)
```

### Time Series Change Detection

```python
from anomaly_detection.time_series import RULSIFDetector

# RULSIF for change point detection
rulsif = RULSIFDetector(
    alpha=0.5,
    n_kernels=50,
    n_folds=5,
    random_state=42
)

# Split data into reference and test periods
reference_data = data[:500]
test_data = data[500:]

rulsif.fit(reference_data=reference_data, test_data=test_data)
change_scores = rulsif.score_samples(data)
change_points = rulsif.detect_changes(data, threshold=0.1)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Adding New Algorithms

To add a new anomaly detection algorithm:

1. Create a new module in the appropriate directory
2. Inherit from the appropriate base class (`BaseAnomalyDetector`, `UnsupervisedAnomalyDetector`, or `SupervisedAnomalyDetector`)
3. Implement the required methods (`fit`, `predict`, `score_samples`)
4. Add tests in the `tests/` directory
5. Update the main `__init__.py` file
6. Add documentation and examples

## 📊 Benchmarks

We provide comprehensive benchmarks comparing different algorithms on various datasets. See `examples/benchmarks.py` for details.

## 🏗️ Architecture

The library is designed with a modular architecture:

```
anomaly_detection/
├── core/           # Base classes and interfaces
├── classical/      # Classical anomaly detection methods
├── deep_learning/  # Deep learning approaches
├── time_series/    # Time series specific methods
├── applications/   # Application-specific implementations
├── utils/          # Utility functions and helpers
└── examples/       # Usage examples and tutorials
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- All the researchers and authors whose papers are referenced in this library
- The open-source community for inspiration and tools
- Contributors and users of this library

## 📞 Contact

- **Author**: Xingqiang Chen
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **Project**: [https://github.com/yourusername/awesome-anomaly-detection](https://github.com/yourusername/awesome-anomaly-detection)

## 📚 References

For academic use, please cite the relevant papers for the algorithms you use. Each algorithm implementation includes references to the original papers.

---

**Star this repository if you find it useful! ⭐**
