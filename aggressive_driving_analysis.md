# 🚗 Aggressive Driving Analysis Pipeline

## 📋 Overview

`aggressive_driving_analysis.py` is a comprehensive driving behavior analysis pipeline script designed to detect and analyze aggressive driving events. The script integrates three main stages: data preprocessing, anomaly detection, and event classification.

## 🔄 Pipeline Stages

### **Stage 1: Data Preprocessing**
- **Function**: Clean and prepare driving data for analysis
- **Trigger Condition**: First run (`first_time = True`)
- **Output**: Preprocessed and standardized data

### **Stage 2: Anomaly Detection**
- **Function**: Identify abrupt changes in driving patterns
- **Trigger Condition**: Non-evaluation mode (`only_evaluation = False`)
- **Algorithm**: RULSIF-based anomaly detection
- **Output**: Anomaly scores and change point markers

### **Stage 3: Event Classification**
- **Function**: Analyze and classify detected anomalies as aggressive driving events
- **Trigger Condition**: Always executed
- **Output**: Aggressive driving event list and classification results

## 🚀 Usage

### **Basic Execution**
```bash
# Run complete pipeline
python aggressive_driving_analysis.py

# Or use Python module method
python -m aggressive_driving_analysis
```

### **Configuration Options**
Set the following parameters in `src/config.py`:

```python
# First run flag
first_time = True    # True: Execute data preprocessing, False: Skip preprocessing

# Evaluation-only mode
only_evaluation = False    # True: Skip anomaly detection, False: Execute complete detection
```

### **Execution Modes**

#### **Complete Mode (Default)**
```python
first_time = True
only_evaluation = False
```
- ✅ Data Preprocessing
- ✅ Anomaly Detection
- ✅ Event Classification

#### **Skip Preprocessing Mode**
```python
first_time = False
only_evaluation = False
```
- ⏭️ Skip data preprocessing
- ✅ Anomaly Detection
- ✅ Event Classification

#### **Evaluation-Only Mode**
```python
first_time = False
only_evaluation = True
```
- ⏭️ Skip data preprocessing
- ⏭️ Skip anomaly detection
- ✅ Event Classification

## 📁 File Structure

```
project_root/
├── aggressive_driving_analysis.py    # Main pipeline script (renamed from apply_main.py)
├── src/
│   ├── config.py                     # Configuration file
│   ├── driving_data_preprocess.py    # Data preprocessing module
│   ├── find_aggressive_driving_event.py  # Event detection module
│   └── parallel_aggressive_driving_detection.py  # Anomaly detection module
├── examples/                         # Examples and test data
├── tests/                           # Test suite
└── README_aggressive_driving_analysis.md  # This documentation
```

## 🔧 Dependencies

### **Core Dependencies**
```bash
pip install numpy scipy scikit-learn pandas matplotlib seaborn
```

### **Optional Dependencies**
```bash
# Deep learning support
pip install torch tensorflow

# Anomaly detection extensions
pip install pyod

# Development and testing
pip install pytest pytest-cov black flake8
```

## 📊 Output Results

### **Data Preprocessing Output**
- Cleaned driving data files
- Standardized and normalized feature data
- Data quality reports

### **Anomaly Detection Output**
- Anomaly score time series
- Change point markers
- Detection performance metrics

### **Event Classification Output**
- Aggressive driving event list
- Event severity scores
- Classification confidence levels

## 🎯 Use Cases

### **Research Applications**
- Driving behavior analysis research
- Anomaly detection algorithm evaluation
- Traffic safety data analysis

### **Practical Applications**
- Fleet management systems
- Insurance risk assessment
- Driver training evaluation
- Traffic safety monitoring

## 🚨 Important Notes

### **Data Requirements**
- Input data must contain necessary driving metrics
- Data format must comply with preprocessing module requirements
- Recommend using standardized time series data

### **Performance Considerations**
- Large-scale data processing may require significant time
- Recommend setting `first_time = False` after first run
- Anomaly detection stage is computationally intensive, ensure sufficient resources

### **Configuration Recommendations**
- Use default configuration for first run
- Adjust anomaly detection parameters based on data characteristics
- Regularly update configuration to adapt to new analysis requirements

## 🔍 Troubleshooting

### **Common Issues**

#### **Import Errors**
```bash
# Ensure running in correct directory
cd /path/to/project
python aggressive_driving_analysis.py
```

#### **Data Format Errors**
- Check input data format
- Verify data column names and types
- Validate data integrity

#### **Insufficient Memory**
- Reduce batch processing size
- Use data sampling for testing
- Increase system memory

### **Debug Mode**
Add debug information to script:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Related Documentation

- [Project Main README](README.md)
- [RULSIF Anomaly Detection Guide](examples/README_original_format.md)
- [PyOD Integration Guide](examples/README_PyOD_Integration.md)
- [Testing Guide](tests/README.md)

## 🤝 Contributing

Welcome to submit issues and improvement suggestions:
1. Fork the project
2. Create feature branch
3. Commit changes
4. Create Pull Request

## 📞 Contact

- **Author**: Xingqiang Chen
- **Project**: [GitHub Repository](https://github.com/chenxingqiang/rulsif-abrupt-change-detection)
- **Issue Feedback**: [GitHub Issues](https://github.com/chenxingqiang/rulsif-abrupt-change-detection/issues)

---

**🚗 Making data-driven driving behavior analysis safer and smarter!** ✨
