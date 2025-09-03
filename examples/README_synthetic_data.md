# 合成驾驶数据使用说明

## 概述

这个目录包含了为异常检测研究生成的合成驾驶数据。数据模拟了真实驾驶行为，包括正常驾驶模式和激进驾驶事件。

## 数据文件

### 主要数据文件
- **`10.csv`** 到 **`89.csv`**: 每个文件代表一个驾驶员的驾驶数据
- **总计**: 41个驾驶员，每个驾驶员5000个数据点

### 数据列说明

| 列名 | 描述 | 单位 | 范围 |
|------|------|------|------|
| `timestamp` | 时间戳 | - | 2024-01-01 00:00:00 开始 |
| `speed` | 车速 | km/h | 0-130 |
| `acceleration` | 加速度 | m/s² | -20 到 +20 |
| `rpm` | 发动机转速 | RPM | 800-6000 |
| `fuel_consumption` | 燃油消耗 | L/100km | 0-15 |
| `engine_temperature` | 发动机温度 | °C | 70-110 |
| `brake_pressure` | 制动压力 | % | 0-100 |
| `steering_angle` | 转向角度 | 度 | -25 到 +25 |
| `lateral_acceleration` | 横向加速度 | m/s² | -3 到 +3 |
| `throttle_position` | 油门位置 | % | 0-100 |
| `gear` | 档位 | - | 1-6 |
| `driver_id` | 驾驶员ID | - | 10, 11, 12, ... |
| `aggressive_event` | 激进驾驶事件标记 | 布尔值 | True/False |

## 数据特征

### 正常驾驶模式
- **速度变化**: 基于正弦波的自然变化
- **相关性**: 各指标之间存在合理的相关性
- **噪声**: 添加了适度的随机噪声模拟真实情况

### 激进驾驶事件
- **突然加速**: 速度突然增加25-40 km/h
- **急刹车**: 速度突然减少30-45 km/h，制动压力达到100%
- **急转弯**: 转向角度突然增加20-35度
- **事件频率**: 约2%的数据点标记为激进事件

### 驾驶员特征
每个驾驶员都有独特的驾驶风格：
- **保守型**: 基础速度较低，激进事件较少
- **正常型**: 中等速度和事件频率
- **激进型**: 较高速度，较多激进事件

## 使用方法

### 1. 加载数据
```python
import pandas as pd

# 加载单个驾驶员数据
driver_10 = pd.read_csv('synthetic_driving_data/10.csv')

# 加载多个驾驶员数据
import glob
csv_files = glob.glob('synthetic_driving_data/*.csv')
all_data = pd.concat([pd.read_csv(f) for f in csv_files])
```

### 2. 数据预处理
```python
# 转换时间戳
driver_10['timestamp'] = pd.to_datetime(driver_10['timestamp'])

# 提取特征
features = ['speed', 'acceleration', 'rpm', 'fuel_consumption', 
           'engine_temperature', 'brake_pressure', 'steering_angle', 
           'lateral_acceleration', 'throttle_position']

X = driver_10[features].values
```

### 3. 异常检测
```python
from anomaly_detection.time_series import RULSIFDetector

# 初始化检测器
detector = RULSIFDetector(alpha=0.5, n_kernels=50)

# 分割数据为参考期和测试期
split_point = len(X) // 2
reference_data = X[:split_point]
test_data = X[split_point:]

# 训练检测器
detector.fit(X, reference_data=reference_data, test_data=test_data)

# 检测异常
anomaly_scores = detector.score_samples(X)
detected_anomalies = detector.detect_changes(X)
```

### 4. 评估结果
```python
# 计算检测准确率
from sklearn.metrics import accuracy_score, classification_report

# 真实标签（激进事件标记）
y_true = driver_10['aggressive_event'].values

# 预测标签（检测到的异常）
y_pred = detected_anomalies

# 评估指标
accuracy = accuracy_score(y_true, y_pred)
print(f"检测准确率: {accuracy:.3f}")

# 详细报告
print(classification_report(y_true, y_pred))
```

## 数据生成脚本

### 主要脚本
- **`simple_data_generator.py`**: 快速生成合成数据
- **`generate_synthetic_driving_data.py`**: 完整功能的数据生成器

### 重新生成数据
```bash
# 生成所有驾驶员数据
python simple_data_generator.py

# 或者使用完整版本
python generate_synthetic_driving_data.py
```

### 自定义参数
```python
# 修改数据生成参数
df = generate_driving_data(
    driver_id='test_driver',
    n_samples=10000,  # 增加样本数量
    # 其他参数...
)
```

## 数据质量

### 真实性
- 基于真实驾驶行为的物理模型
- 合理的数值范围和相关性
- 适当的噪声水平

### 一致性
- 所有驾驶员数据格式一致
- 时间序列连续性良好
- 标记数据可用于监督学习

### 多样性
- 不同驾驶风格
- 各种异常事件类型
- 适合算法比较研究

## 注意事项

1. **数据规模**: 每个文件约900KB，总数据量约37MB
2. **内存使用**: 加载所有数据需要足够的内存
3. **时间戳**: 数据按秒采样，适合时间序列分析
4. **标签质量**: `aggressive_event`列提供了真实标签用于评估

## 扩展建议

### 添加新特征
- GPS坐标和路线信息
- 天气条件
- 交通状况
- 车辆类型和配置

### 增加事件类型
- 疲劳驾驶
- 分心驾驶
- 恶劣天气驾驶
- 交通违规

### 数据增强
- 更多驾驶员
- 不同时间段
- 各种驾驶环境
- 传感器故障模拟

## 联系信息

如有问题或建议，请联系项目维护者或提交Issue。
