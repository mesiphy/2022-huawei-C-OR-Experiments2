import os
from data_preparation import load_data
from pbs_simulator import PBSSimulator

# TODO: 导入你的优化算法类
# 假设你的文件名是 optimize_sa.py，类名是 PBS_SimulatedAnnealing
from optimize_sa import PBS_SimulatedAnnealing


def solve_for_file(input_file, output_file, area_code_file):
    print(f"\n{'=' * 40}")
    print(f"正在处理: {input_file}")
    print(f"{'=' * 40}")

    # 1. 读取数据 (Data Layer)
    if not os.path.exists(input_file):
        print(f"错误: 找不到文件 {input_file}")
        return

    cars_data = load_data(input_file)
    if not cars_data:
        print("数据读取失败，终止。")
        return

    # 2. 运行优化算法 (The Brain)
    print(">>> 启动模拟退火优化器...")
    # 初始化优化器
    optimizer = PBS_SimulatedAnnealing(cars_data)

    # 执行优化
    # TODO: 检查你的 optimize_sa.py，调用正确的主函数
    # 假设 run() 返回最优的【出车ID列表】
    # 如果你的 run() 返回的是对象，请提取出序列部分
    best_sequence_ids = optimizer.run()

    print(f"优化完成。生成序列长度: {len(best_sequence_ids)}")

    # 3. 运行仿真验证 (The Validator)
    print(">>> 启动离散事件仿真器 (验证约束6 & 7)...")
    simulator = PBSSimulator(cars_data, best_sequence_ids, area_code_file=area_code_file)

    # 运行仿真
    success, z3, z4 = simulator.run()

    # 4. 输出结果
    if success:
        print(f"\n>>> 验证成功！")
        print(f"    Z3 (返回道使用次数): {z3}")
        print(f"    Z4 (总调度时间): {z4}")

        # 导出 Result Excel
        simulator.export_result(output_file)
        print(f"    结果文件已生成: {output_file}")
    else:
        print(f"\n>>> 验证失败！发生死锁或超时。")
        print("    建议：调整优化算法中 Z3 的惩罚权重，减少对返回道的依赖。")


if __name__ == "__main__":
    # 配置路径
    data_dir = 'data'  # 假设你的Excel都在data文件夹下

    # 任务 1: 处理附件1
    input1 = os.path.join(data_dir, '附件1.xlsx')
    output1 = 'result11.xlsx'
    area_code_path = os.path.join(data_dir, '附件3.xlsx')

    solve_for_file(input1, output1, area_code_path)

    # 任务 2: 处理附件2 (如果需要)
    # input2 = os.path.join(data_dir, '附件2.xlsx')
    # output2 = 'result12.xlsx'
    # solve_for_file(input2, output2, area_code_path)