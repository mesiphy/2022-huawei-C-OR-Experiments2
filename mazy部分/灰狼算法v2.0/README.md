# 汽车生产线车辆排序优化系统

基于**灰狼优化算法 (Grey Wolf Optimizer, GWO)** 的汽车生产线 PBS 缓冲区车辆排序优化求解器，结合离散事件仿真技术，实现多目标车辆调度优化。

## 📋 项目概述

在汽车制造的涂装-总装之间，PBS (Painted Body Storage) 缓冲区负责重新排序车辆，以满足总装线的生产约束。本项目通过元启发式算法优化车辆出库顺序，最小化以下目标：

- **Z1**: 混动车间隔违规次数
- **Z2**: 四驱车聚类不平衡惩罚
- **Z3**: 返回道使用次数
- **Z4**: 总生产耗时

## 🗂️ 项目结构

```
p1v2/
├── main.py                  # 程序入口
├── data_loader.py           # 数据加载与预处理
├── optimizer/
│   └── gwo.py               # 灰狼优化算法核心实现
├── simulation/
│   ├── simulator.py         # 生产线离散事件仿真器
│   └── trajectory_recorder.py  # 车辆轨迹记录器
└── evaluation/
    └── objective.py         # 多目标评估函数
```

### 模块说明

| 模块 | 文件 | 功能 |
|------|------|------|
| **优化器** | `optimizer/gwo.py` | 灰狼优化算法，负责决策向量的迭代优化 |
| **仿真器** | `simulation/simulator.py` | 模拟车辆在 6 条车道和返回道中的流动过程 |
| **记录器** | `simulation/trajectory_recorder.py` | 记录每辆车在每个时间步的位置 |
| **评估器** | `evaluation/objective.py` | 计算排序方案的多目标加权得分 |

## 🚀 快速开始

### 环境要求

- Python 3.11+
- NumPy
- Pandas
- Matplotlib

### 安装依赖

```bash
pip install numpy pandas matplotlib openpyxl
```

### 运行程序

```bash
cd p1v2
python3 main.py
```

### 数据格式

输入数据为 Excel 文件，需包含以下列：

| 列索引 | 内容 | 说明 |
|--------|------|------|
| 0 | 车辆 ID | 唯一标识 |
| 1 | 其他属性 | - |
| 2 | 燃油类型 | `燃油` / `混动` |
| 3 | 驱动类型 | `两驱` / `四驱` |

## ⚙️ 参数配置

在 `main.py` 中可调整以下参数：

```python
model = GreyWolfOptimizer(
    dimension=6000,          # 决策空间维度
    max_iterations=400,      # 最大迭代次数
    population_size=40,      # 种群大小
    lower_bound=0,           # 位置下界
    upper_bound=0.999999999  # 位置上界
)
```

## 📊 输出结果

程序运行后将：

1. 输出收敛曲线图（Fitness vs Iteration）
2. 返回最优解向量和对应得分
3. 可选导出详细的车辆轨迹矩阵到 Excel

## 🔧 算法原理

### 灰狼优化 (GWO)

模拟灰狼社群的狩猎行为：
- **α 狼**：当前最优解
- **β 狼**：次优解
- **δ 狼**：第三优解
- **ω 狼**：其余个体

每次迭代中，所有个体根据 α、β、δ 的位置更新自身位置，逐步收敛到最优解。

### 仿真模型

模拟 PBS 缓冲区的物理结构：
- 6 条平行车道（每条 28 个车位）
- 1 条返回道（28 个车位）
- 输入/输出横移机

## 📝 License

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！
