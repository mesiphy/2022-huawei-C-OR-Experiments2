import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from data_preparation import load_data  # 你的数据读取函数


def generate_pbs_animation(result_file="result11.xlsx", data_file="data/附件1.xlsx"):
    print("1. 正在加载数据...")
    # 读取调度结果矩阵
    df_matrix = pd.read_excel(result_file, index_col=0)
    # 读取原始车辆属性 (为了画颜色: 混动=红, 燃油=蓝)
    cars_data = load_data(data_file)
    # 创建 ID -> 属性映射
    car_props = {c['id']: c for c in cars_data}

    # 获取最大时间步
    max_time = df_matrix.columns[-1]
    # 如果时间太长，限制一下动画长度用于测试 (比如前 200秒)
    # max_time = 200

    print(f"2. 初始化画布 (总时长: {max_time}秒)...")

    # 设置画布
    fig, ax = plt.subplots(figsize=(12, 6))

    # 颜色定义
    COLOR_HYBRID = '#FF6B6B'  # 红色系 (混动)
    COLOR_GAS = '#4ECDC4'  # 蓝色系 (燃油)

    def draw_background():
        """绘制PBS的静态背景 (6条车道, 每道10个格子)"""
        ax.clear()
        ax.set_xlim(-2, 12)  # X轴范围: 左侧缓冲区(-2) -> 10...1 -> 出口(11)
        ax.set_ylim(0, 8)  # Y轴范围: 6条车道

        # 绘制车道线和格子
        for lane in range(1, 7):
            # 车道轴线 y
            y_center = 7 - lane

            # 绘制车道标签
            ax.text(-1.5, y_center, f"Lane {lane}", va='center', ha='right', fontsize=12, fontweight='bold')

            # 绘制10个车位 (从右向左: 1号位在 x=10, 10号位在 x=1)
            # 注意：题目中 P1 靠近出口。为了视觉习惯，我们设定：
            # X=10 是出口 (Pos 1), X=1 是入口 (Pos 10)
            for pos in range(1, 11):  # Pos 1 to 10
                # 视觉坐标 x
                vis_x = 11 - pos  # Pos 1 -> x=10, Pos 10 -> x=1

                # 绘制方格
                rect = mpatches.Rectangle((vis_x - 0.4, y_center - 0.4), 0.8, 0.8,
                                          fill=False, edgecolor='#DDDDDD', linestyle='--')
                ax.add_patch(rect)

                # 标号
                if lane == 6:  # 只在最下面标一次位置号
                    ax.text(vis_x, 0.5, f"P{pos}", ha='center', fontsize=8, color='gray')

        # 标注区域
        ax.text(11, 7.5, "To Assembly ->", ha='center', fontsize=10, color='green')
        ax.text(0, 7.5, "<- From Paint", ha='center', fontsize=10, color='gray')
        ax.set_title(f"PBS Simulation - Time: 0s", fontsize=16)
        ax.axis('off')  # 隐藏坐标轴

    def update(t):
        """每一帧的更新函数"""
        # 清除上一帧的车辆，保留背景太慢，所以直接全清再重画背景
        draw_background()
        ax.set_title(f"PBS Simulation - Time: {t}s", fontsize=16)

        # 获取当前时间 t 的所有车辆状态
        if t not in df_matrix.columns:
            return

        current_states = df_matrix[t]  # 获取这一列

        # ------------------------------------------------
        # 核心逻辑: 推算每辆车的位置
        # ------------------------------------------------
        # 我们按照车道将车分组
        lane_cars = {k: [] for k in range(1, 7)}

        moving_cars = []  # 正在进车道或换道的
        exiting_cars = []  # 正在离开的

        for car_id_str, status in current_states.items():
            if pd.isna(status) or status == '': continue

            # car_id_str 可能是 "Car0", 提取数字 ID
            try:
                cid = int(car_id_str.replace("Car", ""))
            except:
                continue

            status = str(status)

            if "PBS_Out" in status:
                # 刚离开的车画在右边
                exiting_cars.append(cid)
            elif "L" in status:
                # 解析车道 L1, L2...
                # 假设状态格式是 "L1_Move" 或 "L1_Pos1"
                try:
                    lane_num = int(status.split('_')[0].replace('L', ''))
                    lane_cars[lane_num].append(cid)
                except:
                    pass

        # ------------------------------------------------
        # 绘制逻辑
        # ------------------------------------------------

        # 1. 绘制车道内的车 (自动推算位置 FIFO)
        for lane_num, car_list in lane_cars.items():
            # 这里的 car_list 顺序是任意的，我们需要按 "Original_ID" 或 "进入时间" 排序
            # 简便方法：ID小的通常先来 (如果不适用，需要额外读取进入时间)
            # 假设 ID 小的在前面 (靠近出口 Pos 1)
            car_list.sort()

            for i, cid in enumerate(car_list):
                # i=0 -> 最早的车 -> Pos 1 (X=10)
                # i=1 -> Pos 2 (X=9)
                pos_idx = i + 1
                if pos_idx > 10: pos_idx = 10  # 溢出显示在入口

                vis_x = 11 - pos_idx  # 转换坐标
                vis_y = 7 - lane_num

                # 确定颜色
                ctype = car_props[cid]['H']  # 1=混动, 0=燃油
                color = COLOR_HYBRID if ctype == 1 else COLOR_GAS
                shape_marker = 's' if car_props[cid]['D'] == 1 else 'o'  # 四驱方形，两驱圆形

                # 画车
                circle = mpatches.Circle((vis_x, vis_y), 0.35, color=color, ec='black')
                ax.add_patch(circle)
                # 写ID
                ax.text(vis_x, vis_y, str(cid), ha='center', va='center', color='white', fontsize=7, fontweight='bold')

        # 2. 绘制刚出库的车 (动画效果)
        for i, cid in enumerate(exiting_cars):
            # 简单画在右侧
            ax.text(11.5, 4, f"Out:{cid}", fontsize=8, color='green')

    print("3. 开始生成动画 (这可能需要几分钟)...")
    # 创建动画对象
    # frames: 帧数 (时间范围)
    # interval: 每帧间隔 (毫秒), 50ms = 20fps
    ani = animation.FuncAnimation(fig, update, frames=range(0, 500, 50), interval=100)  # 每2秒采一帧加快速度

    # 保存
    save_path = "pbs_simulation.gif"
    print(f"4. 正在保存到 {save_path} ...")

    # 使用 Pillow Writer (不需要安装 ffmpeg)
    ani.save(save_path, writer='pillow', fps=10)
    print("完成！请打开 pbs_simulation.gif 查看。")


if __name__ == "__main__":
    generate_pbs_animation()