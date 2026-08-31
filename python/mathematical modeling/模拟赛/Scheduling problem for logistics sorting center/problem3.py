import math
from pathlib import Path
import pandas as pd
import pulp


# 1. 基本参数
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "output" / "问题二_最优排班结果.xlsx"
OUTPUT_FILE = BASE_DIR / "output" / "问题三_人员分配结果.xlsx"

WORK_DAYS = 23
MAX_CONSECUTIVE = 7
# 给求解器预留的候选工人数上界
MAX_WORKERS = 650


# 2. 计算一个工人的最大连续工作天数
def get_max_consecutive(schedule):
    current = 0
    best = 0
    for value in schedule:
        if value == 1:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


# 3. 主程序
def main():
    # 3.1 检查问题二结果文件
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"没有找到问题二结果：{INPUT_FILE}")
    print(f"正在读取：{INPUT_FILE}")

    # 3.2 读取问题二每日汇总
    summary_df = pd.read_excel(INPUT_FILE, sheet_name="每日汇总")

    # 3.3 检查需要的列
    required_columns = {"天", "最优总工人数"}
    if not required_columns.issubset(summary_df.columns):
        raise ValueError("问题二每日汇总表缺少‘天’或‘最优总工人数’列")

    # 3.4 转换数据类型
    summary_df["天"] = summary_df["天"].astype(int)
    summary_df["最优总工人数"] = summary_df["最优总工人数"].astype(int)

    # 3.5 建立每天最低人数需求字典
    demand = {int(row["天"]): int(row["最优总工人数"]) for _, row in summary_df.iterrows()}

    # 30天
    D = sorted(demand.keys())
    if D != list(range(1, 31)):
        raise ValueError("每日汇总必须包含第1天到第30天")

    # 4. 计算理论下界
    total_required_days = sum(demand[d] for d in D)
    max_daily_demand = max(demand[d] for d in D)
    workday_lower_bound = math.ceil(total_required_days / WORK_DAYS)
    theoretical_lower_bound = max(max_daily_demand, workday_lower_bound)

    print()
    print("=" * 60)
    print("第三问理论下界分析")
    print("=" * 60)
    print(f"30天最低需求总工日：{total_required_days}")
    print(f"单日最大需求人数：{max_daily_demand}")
    print(f"按每人工作23天计算的下界：{workday_lower_bound}")
    print(f"综合理论下界：{theoretical_lower_bound}")

    # 5. 候选员工集合
    if MAX_WORKERS < theoretical_lower_bound:
        raise ValueError("MAX_WORKERS设置过小")
    I = list(range(1, MAX_WORKERS + 1))

    # 6. 建立MILP模型
    model = pulp.LpProblem(name="Monthly_Workforce_Scheduling", sense=pulp.LpMinimize)

    # 7. 是否招聘员工i
    hire = pulp.LpVariable.dicts("hire", I, cat=pulp.LpBinary)

    # 8. 员工i第d天是否工作
    work = pulp.LpVariable.dicts("work", (I, D), cat=pulp.LpBinary)

    # 9. 目标函数：招聘人数最少
    model += (pulp.lpSum(hire[i] for i in I), "Minimize_Total_Workers")

    # 10. 每天实际出勤人数必须满足问题二需求
    for d in D:
        model += (pulp.lpSum(work[i][d] for i in I) >= demand[d], f"Daily_Demand_{d}")

    # 11. 每名被招聘工人必须工作23天
    for i in I:
        model += (pulp.lpSum(work[i][d] for d in D) == WORK_DAYS * hire[i], f"Work_23_Days_{i}")

    # 12. 未招聘员工不能上班
    for i in I:
        for d in D:
            model += (work[i][d] <= hire[i], f"Work_Only_If_Hired_{i}_{d}")

    # 13. 连续工作不能超过7天
    for i in I:
        for start in range(1, 24):
            model += (
                pulp.lpSum(work[i][d] for d in range(start, start + 8)) <= MAX_CONSECUTIVE,
                f"Max_7_Consecutive_{i}_{start}",
            )

    # 14. 对称性破除
    for i in range(1, MAX_WORKERS):
        model += (hire[i] >= hire[i + 1], f"Hire_Order_{i}")

    # 15. 创建HiGHS求解器
    solver = pulp.HiGHS(msg=False)

    # 16. 求解
    print()
    print("开始求解第三问……")
    model.solve(solver)
    status = pulp.LpStatus[model.status]
    print(f"求解状态：{status}")
    if status != "Optimal":
        raise RuntimeError(f"第三问求解失败：{status}")

    # 17. 提取最优招聘人数
    hired_workers = [i for i in I if pulp.value(hire[i]) > 0.5]
    optimal_workers = len(hired_workers)

    print()
    print("=" * 60)
    print("第三问求解结果")
    print("=" * 60)
    print(f"理论下界：{theoretical_lower_bound}")
    print(f"最优招聘人数：{optimal_workers}")

    # 18. 提取每名员工30天安排
    employee_result = []
    for i in hired_workers:
        schedule = [int(round(pulp.value(work[i][d]))) for d in D]
        row = {"工人编号": f"W{i:03d}"}
        for index, d in enumerate(D):
            row[f"第{d}天"] = "上班" if schedule[index] == 1 else "休息"
        row["工作天数"] = sum(schedule)
        row["最大连续工作天数"] = get_max_consecutive(schedule)
        employee_result.append(row)

    # 19. 提取每天实际出勤情况
    daily_result = []
    for d in D:
        actual_workers = sum(int(round(pulp.value(work[i][d]))) for i in hired_workers)
        daily_result.append({
            "天": d,
            "问题二最低需求人数": demand[d],
            "第三问实际出勤人数": actual_workers,
            "富余人数": actual_workers - demand[d],
        })

    # 20. 一致性检查
    for row in employee_result:
        if row["工作天数"] != WORK_DAYS:
            raise RuntimeError(f"{row['工人编号']}工作天数不是23天")
        if row["最大连续工作天数"] > MAX_CONSECUTIVE:
            raise RuntimeError(f"{row['工人编号']}连续工作超过7天")
    for row in daily_result:
        if row["第三问实际出勤人数"] < row["问题二最低需求人数"]:
            raise RuntimeError(f"第{row['天']}天人数不足")

    # 21. 汇总结果
    total_actual_days = optimal_workers * WORK_DAYS
    total_surplus_days = total_actual_days - total_required_days
    result_summary = pd.DataFrame([{
        "理论下界": theoretical_lower_bound,
        "最优招聘人数": optimal_workers,
        "30天最低需求总工日": total_required_days,
        "实际总工日": total_actual_days,
        "富余总工日": total_surplus_days,
        "每人工作天数": WORK_DAYS,
        "最大连续工作天数": MAX_CONSECUTIVE,
    }])

    employee_df = pd.DataFrame(employee_result)
    daily_df = pd.DataFrame(daily_result)

    # 22. 保存Excel
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        result_summary.to_excel(writer, sheet_name="月度汇总", index=False)
        daily_df.to_excel(writer, sheet_name="每日出勤", index=False)
        employee_df.to_excel(writer, sheet_name="员工工作日安排", index=False)

    print()
    print("=" * 60)
    print("第三问第一阶段求解完成")
    print("=" * 60)
    print(f"结果文件：{OUTPUT_FILE}")


# 23. Python程序入口
if __name__ == "__main__":
    main()
