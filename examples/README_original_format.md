# 原始格式合成驾驶数据使用说明

## 概述

这个目录包含了完全符合原始 `Aggressive_Drive_detection.ipynb` notebook 格式要求的合成驾驶数据。数据格式、列名、数据结构都与原始notebook完全一致，可以直接替换原始数据文件使用。

## 数据格式

### 列名（完全匹配原始notebook）
```python
['Car_ID', 'Time', 'Car_Orientation', 'Pitch_Rate', 'Roll_Rate', 
 'Acceleration', 'Velocity', 'Steering_Wheel_Angle', 'Yaw_Rate']
```

### 数据说明

| 列名 | 描述 | 单位 | 范围 | 说明 |
|------|------|------|------|------|
| `Car_ID` | 车辆ID | - | 10-89 | 驾驶员编号，整数 |
| `Time` | 时间索引 | - | 0-8000 | 从0开始递增，有重置点 |
| `Car_Orientation` | 车辆朝向 | 度 | 0-360 | 车辆前进方向角度 |
| `Pitch_Rate` | 俯仰角速度 | 度/秒 | -20到+20 | 车辆前后倾斜角速度 |
| `Roll_Rate` | 翻滚角速度 | 度/秒 | -15到+15 | 车辆左右倾斜角速度 |
| `Acceleration` | 加速度 | m/s² | -15到+15 | 车辆前后加速度 |
| `Velocity` | 速度 | km/h | 0-130 | 车辆行驶速度 |
| `Steering_Wheel_Angle` | 方向盘角度 | 度 | -45到+45 | 方向盘转向角度 |
| `Yaw_Rate` | 偏航角速度 | 度/秒 | -25到+25 | 车辆左右转向角速度 |

## 数据特征

### 时间重置特性
- **Time列重置**: 模拟多个驾驶会话，Time会在特定点重置为0
- **分段处理**: 原始notebook会查找Time==0的点来分割数据
- **段长度**: 每个段都大于151个样本（符合notebook要求）

### 驾驶行为模拟
- **正常驾驶**: 平稳的速度变化，合理的转向角度
- **激进事件**: 突然加速、急刹车、急转弯等异常行为
- **物理一致性**: 各指标之间存在合理的物理关系

### 数据规模
- **驾驶员数量**: 41个（与原始notebook完全一致）
- **样本数量**: 每个驾驶员8000个样本
- **总数据量**: 328,000个样本
- **分段文件**: 133个分段文件

## 使用方法

### 1. 直接替换原始数据
```python
# 原始notebook中的路径
data_path = './original_format_data/'  # 替换为这个目录

# 数据文件名完全一致
data_name = ['10.csv','56.csv','64.csv','72.csv','84.csv',
              '11.csv','56_0.csv','64_0.csv','73.csv','85.csv',
              # ... 其他文件名
              '55.csv','63_2.csv','71.csv','83.csv']
```

### 2. 运行原始notebook
```python
# 修改notebook中的路径
data_path = './original_format_data/'  # 指向合成数据目录
data_save_path = './output_check/'     # 输出目录

# 其他代码保持不变
for name in data_name:
    data = pd.read_csv(data_path+name, index_col=None, low_memory=False)
    print("Driving Car ID Set:", set(data.ID))
    # ... 后续处理代码
```

### 3. 数据加载验证
```python
import pandas as pd

# 加载数据
data = pd.read_csv('./original_format_data/10.csv', index_col=None, low_memory=False)

# 检查列名
print("Columns:", list(data.columns))

# 检查车辆ID
print("Car IDs:", set(data['Car_ID']))

# 检查时间重置点
time_zero_points = data[data['Time'] == 0].index.tolist()
print("Time reset points:", time_zero_points)

# 检查数据形状
print("Data shape:", data.shape)
```

## 文件结构

```
original_format_data/
├── 10.csv                    # 驾驶员10的主数据文件
├── 11.csv                    # 驾驶员11的主数据文件
├── ...                       # 其他驾驶员文件
├── 83.csv                    # 驾驶员83的主数据文件
├── OrderRight_10_0001.csv    # 驾驶员10的第1段数据
├── OrderRight_10_0002.csv    # 驾驶员10的第2段数据
├── OrderRight_10_0003.csv    # 驾驶员10的第3段数据
├── ...                       # 其他分段文件
└── OrderRight_83_0004.csv    # 驾驶员83的第4段数据
```

## 与原始notebook的兼容性

### ✅ 完全兼容的特性
- **列名**: 完全一致
- **数据类型**: 整数和浮点数类型匹配
- **时间重置**: Time列有多个重置点（Time==0）
- **分段处理**: 自动生成符合要求的OrderRight_*.csv文件
- **数据规模**: 每个文件8000个样本，符合原始要求

### ✅ 模拟的原始行为
- **ID检测**: `set(data.ID)` 会正确返回驾驶员ID
- **时间分割**: `data[data.Time==0]` 会找到分割点
- **段长度检查**: 所有段都大于151个样本
- **文件保存**: 自动生成OrderRight_和LessLength_文件

## 数据生成脚本

### 主要脚本
- **`generate_original_format_data.py`**: 生成符合原始格式的数据
- **`test_original_format.py`**: 验证数据格式的测试脚本

### 重新生成数据
```bash
# 生成所有驾驶员数据
python generate_original_format_data.py

# 验证数据格式
python test_original_format.py
```

### 自定义参数
```python
# 修改数据生成参数
df, reset_points = generate_original_format_data(
    driver_id='test_driver',
    n_samples=10000,  # 增加样本数量
    # 其他参数...
)
```

## 测试验证

### 运行测试
```bash
python test_original_format.py
```

### 测试项目
1. **列名验证**: 确保列名完全匹配
2. **时间重置**: 验证Time列的重置行为
3. **数据类型**: 检查各列的数据类型
4. **数值范围**: 验证数值在合理范围内
5. **分段文件**: 检查分段文件的生成
6. **notebook兼容性**: 模拟原始notebook的处理流程

## 使用建议

### 1. 直接替换
- 将 `data_path` 指向 `./original_format_data/`
- 其他代码完全不需要修改
- 直接运行原始notebook

### 2. 数据验证
- 运行测试脚本确保数据质量
- 检查分段文件的数量和大小
- 验证时间重置点的正确性

### 3. 性能优化
- 合成数据加载速度更快
- 内存使用更高效
- 适合开发和测试

## 注意事项

1. **数据真实性**: 这是合成数据，用于算法开发和测试
2. **物理一致性**: 数据基于物理模型生成，保持合理性
3. **随机性**: 每次生成的数据略有不同，但格式一致
4. **文件大小**: 每个主文件约1.1MB，总数据量约37MB

## 扩展功能

### 添加更多驾驶员
```python
# 在driver_ids列表中添加新的ID
driver_ids = ['10', '56', '64', '72', '84', 'new_driver_100', ...]
```

### 调整数据特征
```python
# 修改激进驾驶事件的频率
sudden_accel_events = np.random.random(n_samples) < 0.01  # 1% -> 2%
```

### 增加新的传感器数据
```python
# 在DataFrame中添加新列
df['new_sensor'] = new_sensor_data
```

## 联系信息

如有问题或建议，请联系项目维护者或提交Issue。

---

**重要提示**: 这个数据格式完全符合原始 `Aggressive_Drive_detection.ipynb` notebook 的要求，可以直接替换原始数据文件使用，无需修改任何代码！
