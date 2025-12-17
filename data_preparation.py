import pandas as pd

def load_data(file_path):
    print(f"正在读取 Excel 文件: {file_path} ...")

    # 1. 读取 Excel 文件 (.xlsx)
    # engine='openpyxl' 是读取 xlsx 的标准引擎
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
    except FileNotFoundError:
        print("错误：找不到文件，请检查路径。")
        return []
    except Exception as e:
        print(f"读取错误: {e}")
        return []

    print("成功读取。列名:", df.columns.tolist())

    # 2. 数据清洗与映射 (逻辑与之前一致)
    cars_data = []

    for index, row in df.iterrows():
        # 辅助函数：安全获取单元格内容并转为字符串
        def get_val(col_name):
            val = row.get(col_name)
            return str(val).strip() if pd.notna(val) else ""

        # --- 动力属性映射 (H) ---
        power = get_val('动力')
        # 只要包含"混动"二字即为1，否则为0
        h_val = 1 if '混动' in power else 0

        # --- 驱动属性映射 (D) ---
        drive = get_val('驱动')
        # 兼容 "四驱" 中文或数字 4
        d_val = 1 if '四驱' in drive or drive == '4' else 0

        # --- 构建数据字典 ---
        car_info = {
            'id': index,  # 内部模型ID (0, 1, 2...)
            'original_id': row.get('进车顺序', index + 1),
            'H': h_val,
            'D': d_val
        }
        cars_data.append(car_info)

    print(f"成功处理 {len(cars_data)} 条车辆数据。")
    return cars_data


# --- 测试调用 ---
if __name__ == "__main__":
    # 请确保这里的文件名后缀是 .xlsx
    file_path = 'data/附件1.xlsx'

    data = load_data(file_path)

    # 验证前5条
    if data:
        print("\n--- 数据预览 (前5条) ---")
        for car in data[:5]:
            print(car)