import os
# 如果采用了方法二，记得保留这行；如果用了方法一，这行可以删掉
os.environ["GRB_LICENSE_FILE"] = "/Library/gurobi1203/gurobi.lic"

import gurobipy as gp

try:
    # 尝试启动环境
    env = gp.Env()
    print("\n✅ 成功！Gurobi 已识别许可，当前为学术版/完整版。")
except gp.GurobiError as e:
    print(f"\n❌ 失败。Gurobi 仍然报错: {e}")