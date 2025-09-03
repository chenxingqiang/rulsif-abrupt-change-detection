# 🚗 激进驾驶分析流水线 (Aggressive Driving Analysis Pipeline)

## 📋 概述

`aggressive_driving_analysis.py` 是一个完整的驾驶行为分析流水线脚本，用于检测和分析激进驾驶事件。该脚本集成了数据预处理、异常检测和事件分类三个主要阶段。

## 🔄 流水线阶段

### **阶段 1: 数据预处理 (Data Preprocessing)**
- **功能**: 清理和准备驾驶数据进行分析
- **触发条件**: 首次运行 (`first_time = True`)
- **输出**: 预处理后的标准化数据

### **阶段 2: 异常检测 (Anomaly Detection)**
- **功能**: 识别驾驶模式中的异常变化
- **触发条件**: 非仅评估模式 (`only_evaluation = False`)
- **算法**: 基于RULSIF的异常检测
- **输出**: 异常分数和变化点标记

### **阶段 3: 事件分类 (Event Classification)**
- **功能**: 分析和分类检测到的异常为激进驾驶事件
- **触发条件**: 始终执行
- **输出**: 激进驾驶事件列表和分类结果

## 🚀 使用方法

### **基本运行**
```bash
# 运行完整流水线
python aggressive_driving_analysis.py

# 或者使用Python模块方式
python -m aggressive_driving_analysis
```

### **配置选项**
在 `src/config.py` 中设置以下参数：

```python
# 首次运行标志
first_time = True    # True: 执行数据预处理, False: 跳过预处理

# 仅评估模式
only_evaluation = False    # True: 跳过异常检测, False: 执行完整检测
```

### **运行模式**

#### **完整模式 (默认)**
```python
first_time = True
only_evaluation = False
```
- ✅ 数据预处理
- ✅ 异常检测
- ✅ 事件分类

#### **跳过预处理模式**
```python
first_time = False
only_evaluation = False
```
- ⏭️ 跳过数据预处理
- ✅ 异常检测
- ✅ 事件分类

#### **仅评估模式**
```python
first_time = False
only_evaluation = True
```
- ⏭️ 跳过数据预处理
- ⏭️ 跳过异常检测
- ✅ 事件分类

## 📁 文件结构

```
project_root/
├── aggressive_driving_analysis.py    # 主流水线脚本 (重命名自 apply_main.py)
├── src/
│   ├── config.py                     # 配置文件
│   ├── driving_data_preprocess.py    # 数据预处理模块
│   ├── find_aggressive_driving_event.py  # 事件查找模块
│   └── parallel_aggressive_driving_detection.py  # 异常检测模块
├── examples/                         # 示例和测试数据
├── tests/                           # 测试套件
└── README_aggressive_driving_analysis.md  # 本说明文档
```

## 🔧 依赖要求

### **核心依赖**
```bash
pip install numpy scipy scikit-learn pandas matplotlib seaborn
```

### **可选依赖**
```bash
# 深度学习支持
pip install torch tensorflow

# 异常检测扩展
pip install pyod

# 开发和测试
pip install pytest pytest-cov black flake8
```

## 📊 输出结果

### **数据预处理输出**
- 清理后的驾驶数据文件
- 标准化和归一化的特征数据
- 数据质量报告

### **异常检测输出**
- 异常分数时间序列
- 变化点标记
- 检测性能指标

### **事件分类输出**
- 激进驾驶事件列表
- 事件严重程度评分
- 分类置信度

## 🎯 使用场景

### **研究用途**
- 驾驶行为分析研究
- 异常检测算法评估
- 交通安全数据分析

### **实际应用**
- 车队管理系统
- 保险风险评估
- 驾驶员培训评估
- 交通安全监控

## 🚨 注意事项

### **数据要求**
- 输入数据必须包含必要的驾驶指标
- 数据格式必须符合预处理模块要求
- 建议使用标准化的时间序列数据

### **性能考虑**
- 大规模数据处理可能需要较长时间
- 建议在首次运行后设置 `first_time = False`
- 异常检测阶段计算密集，请确保足够的计算资源

### **配置建议**
- 首次运行使用默认配置
- 根据数据特点调整异常检测参数
- 定期更新配置以适应新的分析需求

## 🔍 故障排除

### **常见问题**

#### **导入错误**
```bash
# 确保在正确的目录中运行
cd /path/to/project
python aggressive_driving_analysis.py
```

#### **数据格式错误**
- 检查输入数据格式
- 确认数据列名和类型
- 验证数据完整性

#### **内存不足**
- 减少批处理大小
- 使用数据采样进行测试
- 增加系统内存

### **调试模式**
在脚本中添加调试信息：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 相关文档

- [项目主README](README.md)
- [RULSIF异常检测说明](examples/README_original_format.md)
- [PyOD集成指南](examples/README_PyOD_Integration.md)
- [测试说明](tests/README.md)

## 🤝 贡献指南

欢迎提交问题和改进建议：
1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 📞 联系方式

- **作者**: Xingqiang Chen
- **项目**: [GitHub Repository](https://github.com/chenxingqiang/rulsif-abrupt-change-detection)
- **问题反馈**: [GitHub Issues](https://github.com/chenxingqiang/rulsif-abrupt-change-detection/issues)

---

**🚗 让数据驱动的驾驶行为分析更安全、更智能！** ✨
