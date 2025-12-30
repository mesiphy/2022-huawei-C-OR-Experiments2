# 汽车生产线车辆排序优化系统 (P2 版本)

基于**灰狼优化算法 (Grey Wolf Optimizer, GWO)** 的汽车生产线 PBS 缓冲区车辆排序优化求解器，结合离散事件仿真技术，实现多目标车辆调度优化。

**P2 版本**在 P1 基础上扩展了决策空间，引入更精细的横移机控制策略。

---

## 📋 项目概述

在汽车制造的涂装-总装之间，PBS (Painted Body Storage) 缓冲区负责重新排序车辆，以满足总装线的生产约束。本项目通过元启发式算法优化车辆出库顺序，最小化以下目标：

- **Z1**: 混动车间隔违规次数
- **Z2**: 四驱车聚类不平衡惩罚
- **Z3**: 返回道使用次数
- **Z4**: 总生产耗时

---

## 🆚 P1 与 P2 版本对比

### 核心差异总览

| 特征 | P1 版本 | P2 版本 |
|------|---------|---------|
| **决策变量数** | 每步 **2** 个 | 每步 **4** 个 |
| **决策空间维度** | 6000 (3000步 × 2) | 14000 (3500步 × 4) |
| **输入横移机控制** | 被动响应 | 主动控制（是否工作） |
| **出车策略** | FIFO 固定 | 可变策略 |
| **热启动支持** | 无 | 支持加载 P1 解 |
| **目标车道记忆** | 无 | 有 `target_lane` 变量 |

---

### 决策变量详解

#### P1 版本：2 列决策变量

```
decision_vector.reshape(-1, 2)
├── Col 0: 车道选择 (Lane Selection)
│   └── 决定车辆进入哪条车道 [0-5]
└── Col 1: 返回决策 (Return Decision)
    └── 0.5 为阈值，决定是送出还是送回返回道
```

#### P2 版本：4 列决策变量

```
decision_vector.reshape(-1, 4)
├── Col 0: 输入横移机启用 (Input Hoist Activation) 【P2 新增】
│   └── > 0.5: 从 PBS 取车
│   └── < 0.5: 优先接返回道的车
├── Col 1: 输出横移机决策 (Output Decision)
│   └── ≤ 0.5: 送出
│   └── > 0.5: 送回返回道
├── Col 2: 车道选择 (Lane Selection)
│   └── 决定车辆进入哪条车道 [0-5]
└── Col 3: 出车策略 (Car Selection Strategy) 【P2 新增】
    └── 决定从出车队列中选择哪辆车
```

---

## 🔄 仿真运行规则对比

### 输入横移机逻辑

#### P1 规则

```
如果 PBS 有车 且 返回道 10 车位为空:
    从 PBS 取车，送入选定车道
如果 返回道 10 车位有车:
    优先接返回道的车
```

#### P2 规则（增强版）

```
如果 PBS 有车:
    如果 decision[Col 0] > 0.5 且 返回道 10 车位为空:
        从 PBS 取车，送入选定车道
        记录 target_lane = 选定车道  【P2 新增】
    如果 decision[Col 0] < 0.5 或 返回道 10 车位有车:
        优先接返回道的车
        记录 target_lane = 选定车道  【P2 新增】
```

**关键区别**：
- P1 的输入横移机是**被动响应**，只根据返回道状态决定行为
- P2 的输入横移机是**主动控制**，可以通过 `Col 0` 主动决定是否工作

---

### 输出横移机逻辑

#### P1 规则

```
当 1 车位有车时:
    如果 返回道末端阻塞:
        强制送出
    否则:
        根据 decision[Col 1] 决定送出或送回
```

#### P2 规则（增强版）

```
当 1 车位有车时:
    使用 decision[Col 3] 从出车队列中选择车辆  【P2 新增】
    如果 返回道末端阻塞:
        强制送出
    否则:
        根据 decision[Col 1] 决定送出或送回
```

**关键区别**：
- P1 采用 **FIFO**（先进先出）策略，总是取队列第一辆车
- P2 可以通过 `Col 3` **选择**队列中的特定车辆

---

### 目标车道记忆机制 (P2 新增)

P2 引入 `target_lane` 变量，用于记忆横移机当前的工作目标车道：

```python
target_lane = 0  # 初始化

# 在输入横移机选择车道后
target_lane = selected_lane_index  # 记录目标

# 在后续操作中使用 target_lane 而非 selected_lane_index
# 确保跨时间步的一致性
```

**作用**：避免在多时间步操作中丢失车道信息，提高仿真稳定性。

---

## 🗂️ 项目结构

```
p2v1.5/
├── main.py                      # 程序入口（含热启动）
├── data_loader.py               # 数据加载与预处理
├── optimizer/
│   └── gwo.py                   # 灰狼优化算法核心实现
├── simulation/
│   ├── simulator.py             # 生产线离散事件仿真器
│   └── trajectory_recorder.py   # 车辆轨迹记录器
└── evaluation/
    └── objective.py             # 多目标评估函数
```

### 模块说明

| 模块 | 文件 | 功能 |
|------|------|------|
| **优化器** | `optimizer/gwo.py` | 灰狼优化算法，负责决策向量的迭代优化 |
| **仿真器** | `simulation/simulator.py` | 模拟车辆在 6 条车道和返回道中的流动过程 |
| **记录器** | `simulation/trajectory_recorder.py` | 记录每辆车在每个时间步的位置 |
| **评估器** | `evaluation/objective.py` | 计算排序方案的多目标加权得分 |

---

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
cd p2v1.5
python3 main.py
```

---

## 🔥 热启动机制 (P2 特有)

P2 版本支持从 P1 的最优解热启动，加速收敛：

```python
# 加载 P1 解
p1_best_x = np.load('best_x_p1.npy')

# P1 → P2 映射
# P1 Col 0 (车道) → P2 Col 2
# P1 Col 1 (返回) → P2 Col 1
p2_init_matrix[:valid_steps, 2] = p1_matrix[:valid_steps, 0]
p2_init_matrix[:valid_steps, 1] = p1_matrix[:valid_steps, 1]
```

### 种群扩散策略

```python
# 第 0 号狼保留纯净热启动数据
# 第 1-39 号狼带变异继承
for k in range(1, 40):
    noise = np.random.normal(0, 0.01, size=len(base_wolf))
    new_wolf = base_wolf + noise
    # 强制修正 Col 0 (Input Hoist Activation)
    temp_matrix[mask_work, 0] = 0.8 + random()
```

---

## ⚙️ 参数配置

在 `main.py` 中可调整以下参数：

```python
# 颗粒度参数
TARGET_STEPS = 3500          # 目标仿真步数
VARS_PER_STEP_P2 = 4         # 每步变量数
DIM_P2 = 14000               # 总维度

# 优化器参数
model = GreyWolfOptimizer(
    dimension=14000,         # 决策空间维度
    max_iterations=400,      # 最大迭代次数
    population_size=40,      # 种群大小
    lower_bound=0,           # 位置下界
    upper_bound=0.999999999  # 位置上界
)
```

---

## 📊 输出结果

程序运行后将：

1. 输出收敛曲线图（Fitness vs Iteration）
2. 返回最优解向量和对应得分
3. 可选导出详细的车辆轨迹矩阵到 Excel

---

## 🔧 物理模型参数

### PBS 缓冲区结构

- 6 条平行车道（每条 28 个车位）
- 1 条返回道（28 个车位）
- 输入/输出横移机

### 时间参数

```python
shuttle_in_out_times = [6, 4, 2, 0, 4, 6]      # 横移机到各车道时间
return_lane_transit_times = [8, 6, 4, 2, 4, 6]  # 返回道往返时间
time_return_to_lane = [4, 3, 2, 1, 1, 2]        # 返回道到车道时间
```

---


## 🤝 贡献

欢迎提交 Issue 和 Pull Request！
