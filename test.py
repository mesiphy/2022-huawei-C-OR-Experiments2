import gurobipy as gp
from gurobipy import GRB
import numpy as np
import os
os.environ["GRB_LICENSE_FILE"] = "/Library/gurobi1203/gurobi.lic"
# ==========================================
# 1. 数据准备 (微型数据集: 6辆车)
# ==========================================
# ID: 0-5
# H (Hybrid): 1=混动, 0=燃油
# D (Drive): 1=四驱(4WD), 0=两驱(2WD) (注意：模型中需将4/2转化为0/1)
# Arrival: 到达顺序
# cars_data = [
#     {'id': 0, 'H': 1, 'D': 0},  # 混动, 两驱
#     {'id': 1, 'H': 1, 'D': 0},  # 混动, 两驱
#     {'id': 2, 'H': 0, 'D': 1},  # 燃油, 四驱
#     {'id': 3, 'H': 0, 'D': 1},  # 燃油, 四驱
#     {'id': 4, 'H': 1, 'D': 1},  # 混动, 四驱
#     {'id': 5, 'H': 0, 'D': 0},  # 燃油, 两驱
# ]
from data_preparation import load_data
file_path = 'data/附件1.xlsx'
# 读取全部数据
print("正在从Excel加载数据...")
full_cars_data = load_data(file_path)

if not full_cars_data:
    print("错误：未读取到数据，请检查路径！")
    exit()

# 【严重警告】为了测试模型逻辑，请务必先只取前 20-30 辆车！
# 现在的 MILP 模型在处理超过 50 辆车时会变得极慢。
# 跑通这 20 辆车后，我们再讨论如何处理全量数据。
N_LIMIT = 50
cars_data = full_cars_data[:N_LIMIT]

print(f"\n当前测试规模: N = {len(cars_data)} 辆车")
print("-" * 30)


N = len(cars_data)
K = 6  # 进车道数量
P = 10  # 车道深度
P_range = range(1, P + 1)  # 1..10

# 权重
W1, W2, W3, W4 = 0.4, 0.3, 0.2, 0.1

# ==========================================
# 2. 模型初始化
# ==========================================
model = gp.Model("PBS_Scheduling_MILP")
model.setParam('TimeLimit', 300)  # 设置最长求解时间300秒

# ==========================================
# 3. 定义决策变量
# ==========================================

# y[i, s]: 车i是否排在出车序列的第s位
y = model.addVars(N, N, vtype=GRB.BINARY, name="y")

# x[i, k, p]: 车i是否存放在车道k的车位p (p=1靠近出口)
x = model.addVars(N, K, P, vtype=GRB.BINARY, name="x")

# r[i]: 车i是否使用了返回道 (简化：如果出车序位 < 物理FIFO序位，则必须用返回道)
r = model.addVars(N, vtype=GRB.BINARY, name="r")

# 辅助变量：出车序列属性
u = model.addVars(N, vtype=GRB.BINARY, name="u_seq_hybrid")  # 位置s是混动吗
f = model.addVars(N, vtype=GRB.BINARY, name="f_seq_4wd")  # 位置s是四驱吗

# ==========================================
# 4. 核心约束
# ==========================================

# 4.1 唯一性约束
# 每辆车只能在一个出车位置
model.addConstrs((y.sum(i, '*') == 1 for i in range(N)), name="Assign_i")
# 每个出车位置只能有一辆车
model.addConstrs((y.sum('*', s) == 1 for s in range(N)), name="Assign_s")
# 每辆车只能存放在一个车道的一个车位
model.addConstrs((x.sum(i, '*', '*') == 1 for i in range(N)), name="Storage_i")
# 车道物理约束：每个格子最多一辆车
model.addConstrs((x.sum('*', k, p) <= 1 for k in range(K) for p in range(P)), name="Cap_kp")

# 【新增】: 强制均匀分布 (Round-Robin 策略)
# 这会极大地帮助求解器，因为它不再需要猜测车该放哪了
# -------------------------------------------------------
print("应用启发式策略: 强制车辆均匀分布到各车道...")
for i in range(N):
    # 简单的轮询策略: 0->Lane0, 1->Lane1, ..., 6->Lane0
    target_lane = i % K

    # 目标车位: 随着车道填满，往后排
    # 0-5号车 -> Pos 0 (出口)
    # 6-11号车 -> Pos 1
    target_pos = i // K

    if target_pos < P:  # 确保不超过车道深度
        # 强制 x[i, target_lane, target_pos] = 1
        model.addConstr(x[i, target_lane, target_pos] == 1, name=f"Force_Lane_{i}")

# 4.2 链接属性变量 (Linking Attributes to Sequence)
# u[s] = sum(H_i * y[i,s])
for s in range(N):
    model.addConstr(u[s] == gp.quicksum(cars_data[i]['H'] * y[i, s] for i in range(N)))
    model.addConstr(f[s] == gp.quicksum(cars_data[i]['D'] * y[i, s] for i in range(N)))

# 4.3 物理FIFO约束 (最关键约束)
# 如果车i比车j先到达(i<j)，且它们在同一车道k，则i的位置p必须比j小 (p=1是出口，先进的离出口近)
# 遍历所有车对 (i, j) 其中 i < j
for i in range(N):
    for j in range(i + 1, N):
        for k in range(K):
            # 逻辑：若 x[i,k,p1]=1 且 x[j,k,p2]=1 => p1 < p2
            # 利用Gurobi的Indicator约束或Big-M
            # 这里为了演示清晰，使用简单逻辑：如果同车道，Sum(p*x_i) < Sum(p*x_j)

            # 定义二进制变量 same_lane_k: 两车是否都在车道k
            same_lane = model.addVar(vtype=GRB.BINARY)
            model.addConstr(same_lane >= x.sum(i, k, '*') + x.sum(j, k, '*') - 1)

            # 获取位置编号 (0-9)
            pos_i = gp.quicksum(p * x[i, k, p] for p in range(P))
            pos_j = gp.quicksum(p * x[j, k, p] for p in range(P))

            # 如果同车道，强制 p_i < p_j (即 p_i + 1 <= p_j)
            model.addConstr((same_lane == 1) >> (pos_i + 1 <= pos_j), name=f"FIFO_{i}_{j}_{k}")

# 4.4 返回道逻辑 (简化版)
# 如果车i想“插队”到车j前面出库(s_i < s_j)，但车i实际上是后进PBS的(i > j)，
# 且它们不在同一车道(FIFO已处理同车道)，那么通常意味着需要复杂的调度或返回道。
# 此处直接引用你的模型逻辑：Minimizing returns.
# 为了代码可运行，我们假设 r[i] 为惩罚项，由求解器根据目标函数决定是否尽量避免乱序。

# ==========================================
# 5. 目标函数实现
# ==========================================

# --- Z1: 混动间隔 (Hybrid Spacing) ---
# 目标：每两辆混动之间，最好间隔2辆非混动
# 实现：遍历窗口。如果 u[s]=1 (混动)，看 u[s+1], u[s+2], u[s+3]
# 这是一个软约束，用惩罚项实现
# --- Z1: 混动间隔 (Hybrid Spacing) ---
z1_penalties = []
for s in range(N - 3):
    is_hybrid = u[s]  # u[s] 是变量

    # 【修复 1】: 检查 s 和 s+1 是否同时为混动
    # 逻辑转换: pen1 >= u[s] + u[s+1] - 1
    # 原理:
    #   如果都是1: 1 + 1 - 1 = 1 <= pen1 (强制 pen1 为 1)
    #   如果有一个0: 1 + 0 - 1 = 0 <= pen1 (允许 pen1 为 0)
    pen1 = model.addVar(vtype=GRB.BINARY)
    model.addConstr(pen1 >= is_hybrid + u[s + 1] - 1, name=f"Z1_adj_{s}")

    # 【修复 2】: 检查 s 和 s+2 是否同时为混动
    # 逻辑转换: pen2 >= u[s] + u[s+2] - 1
    pen2 = model.addVar(vtype=GRB.BINARY)
    model.addConstr(pen2 >= is_hybrid + u[s + 2] - 1, name=f"Z1_gap_{s}")

    z1_penalties.append(pen1)
    z1_penalties.append(pen2)

Z1_score = 100 - gp.quicksum(z1_penalties) * 5

# --- Z2: 驱动比例 (Drive Ratio) - 动态分块近似 ---
# 为了演示，使用固定窗口 (Window Size=4) 来检查局部比例是否为 1:1
# 你的模型中提到了方案B（滑动窗口），这在代码中更容易实现且鲁棒
z2_penalties = []
window_size = 4
if N >= window_size:
    for s in range(N - window_size + 1):
        # 计算窗口内的四驱数量
        sum_4wd = gp.quicksum(f[s + j] for j in range(window_size))

        # 理想数量是 window_size / 2
        diff = model.addVar(lb=-window_size, ub=window_size)
        model.addConstr(diff == sum_4wd - (window_size / 2))

        # 绝对值惩罚
        abs_diff = model.addVar()
        model.addGenConstrAbs(abs_diff, diff)
        z2_penalties.append(abs_diff)

Z2_score = 100 - gp.quicksum(z2_penalties) * 5

# --- Z3: 返回道使用 ---
Z3_score = 100 - r.sum() * 5  # 每次使用扣分

# --- Z4: 调度时间 (Time) ---
# 简化：序列越接近原始到达顺序，通常时间越短 (因为不需要横移机反复倒腾)
# 使用序列混乱度作为时间的Proxy
order_diff = []
for i in range(N):
    # s_i 是车i的出车位置
    s_i = gp.quicksum(s * y[i, s] for s in range(N))
    # 它是第i个来的
    diff_i = model.addVar(lb=-N, ub=N)
    model.addConstr(diff_i == s_i - i)
    abs_diff_i = model.addVar()
    model.addGenConstrAbs(abs_diff_i, diff_i)
    order_diff.append(abs_diff_i)

Z4_score = 100 - gp.quicksum(order_diff) * 0.5

# 总目标
Total_Objective = W1 * Z1_score + W2 * Z2_score + W3 * Z3_score + W4 * Z4_score

model.setObjective(Total_Objective, GRB.MAXIMIZE)

# ==========================================
# 6. 求解与输出
# ==========================================
print("开始求解...")
model.optimize()

if model.status == GRB.OPTIMAL:
    print("\n=== 最优调度方案 ===")

    # 构建输出序列
    final_seq = [-1] * N
    for i in range(N):
        for s in range(N):
            if y[i, s].x > 0.5:
                final_seq[s] = i

    print(f"出车序列 (Car IDs): {final_seq}")

    print("\n--- 属性序列 ---")
    seq_h = [cars_data[i]['H'] for i in final_seq]
    seq_d = [cars_data[i]['D'] for i in final_seq]
    print(f"动力 (H): {seq_h}")
    print(f"驱动 (D): {seq_d}")

    print("\n--- 车道存放情况 ---")
    for k in range(K):
        lane_cars = []
        for p in range(P):  # p=0..9
            for i in range(N):
                if x[i, k, p].x > 0.5:
                    lane_cars.append(f"Car{i}")
        if lane_cars:
            print(f"车道 {k + 1}: {lane_cars} (左侧靠近出口)")

    print(f"\n预估总分: {model.ObjVal:.2f}")

else:
    print("无解或求解超时")