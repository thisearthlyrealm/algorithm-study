import math
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import pulp

# 中文字体设置，防止图内中文乱码
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 基本参数
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "附件1.xlsx"
OUTPUT_DIR = BASE_DIR / "output"
PROCESS_RATE = 25
LOW_PROCESS_RATE = 10
SHIFT_LENGTH = 8
SHIFT_NUMBER = 5
DEADLINE_HOUR = 16
EARLY_END_HOUR = 12
SOLVE_ONLY_DAY = None



IMG_PREFIX = "问题二"
WORKER_SHIFT_CAPACITY = ((SHIFT_LENGTH - 1)*PROCESS_RATE+LOW_PROCESS_RATE)

# 2. 时间格式转换函数
def time_string(hour):
    # 将整数小时转换成时间字符串。
    return  f"{hour:02d}:00"

# 3. 数据检查函数
def check_data(df):
    """
    检查附件1的数据格式是否满足模型要求。
    """
    required_columns = {"天", "小时", "进货量"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Excel必须包含以下三列：{required_columns}")

    df["天"] = df["天"].astype(int)
    df["小时"] = df["小时"].astype(int)
    df["进货量"] = df["进货量"].astype(int)

    if (df["进货量"] < 0).any():
        raise ValueError("进货量不能为负数")

    if df.duplicated(subset=["天", "小时"]).any():
        raise ValueError("数据中存在重复的‘天-小时’记录")

    days=sorted(df["天"].unique())
    if len(days) != 30:
        print(f"题目要求应为30天。")

    for day in days:
        day_data = df[df["天"] == day]
        hours = sorted(day_data["小时"].tolist())
        if hours != list(range(24)):
            raise ValueError(f"第 {day} 天小时数据不完整")

    print("数据检查通过：天、小时和进货量均符合要求。")
    return df

# 4. 求解单独一天
def solve_one_day(day,day_data):
    """
    对某一天建立混合整数线性规划模型，
    并进行三层字典序优化。
    """

    # 4.1 整理当天24小时进货量
    day_data = day_data.sort_values("小时").reset_index(drop=True)
    Q = {int(row["小时"]): int(row["进货量"]) for _, row in day_data.iterrows()}
    T = list(range(24))
    S = list(range(17))
    EARLY_T = list(range(DEADLINE_HOUR))
    total_goods = sum(Q.values())
    early_goods = sum(Q[t] for t in range(EARLY_END_HOUR))

    # 4.2 设置大M
    M = math.ceil(total_goods / PROCESS_RATE)

    # 4.3 建立优化模型
    model = pulp.LpProblem(
        name=f"Day_{day}_Scheduling",
        sense=pulp.LpMinimize
    )

    # 4.4 决策变量 x[s]
    x = pulp.LpVariable.dicts("s",S,lowBound=0,cat=pulp.LpInteger)

    # 4.5 决策变量 y[s]
    y = pulp.LpVariable.dicts("y",S,cat=pulp.LpBinary)

    # 4.6 每小时在岗人数 H[t]
    H = pulp.LpVariable.dicts("H",T,lowBound=0,cat=pulp.LpInteger)

    # 4.7 每小时实际处理量 P[t]
    P = pulp.LpVariable.dicts("P",T,lowBound=0,cat=pulp.LpInteger)

    # 4.8 每小时积压量 B[t]
    B = pulp.LpVariable.dicts("B",T,lowBound=0,cat=pulp.LpInteger)

    # 4.9 峰值人数 H_max
    H_max = pulp.LpVariable("H_max",lowBound=0,cat=pulp.LpInteger)

    # 4.10 属于班次 s 的工人中，在小时 t 只有10件/小时处理能力的人数 r
    r = pulp.LpVariable.dicts("r",(S,T),lowBound=0,cat=pulp.LpInteger)

    # 4.11 第 t 个小时，所有班次加起来一共有多少人处于10件/小时状态。
    R = pulp.LpVariable.dicts("R",T,lowBound=0,cat=pulp.LpInteger)

    # 4.12 第 t 小时处理掉的“0—12点到达货物 P_early
    P_early = pulp.LpVariable.dicts("P_early",EARLY_T,lowBound=0,cat=pulp.LpInteger)

    # 4.13 第 t 小时结束以后，还剩多少0—12点到达的货没有处理
    B_early = pulp.LpVariable.dicts("B_early",EARLY_T,lowBound=0,cat=pulp.LpInteger)

    # 5. 约束条件
    # 5.1 每天恰好设置5个班次
    model +=(pulp.lpSum(y[s] for s in S) == SHIFT_NUMBER,"Exactly_Five_Shifts")

    # 5.2 x和y之间的关联
    for s in S:
        model += (x[s] <= M*y[s],f"Shift_Upper_{s}")
        model += (x[s] >= y[s],f"Shift_Lower_{s}")

    # 5.3 计算每小时在岗人数
    for t in T:
        model += (H[t] == pulp.lpSum(x[s] for s in S if s <= t <= s+SHIFT_LENGTH-1),f"Workers_At_Hour_{t}")

    # 5.4.1 每小时处理能力约束

    for t in T:
        model += (R[t] == pulp.lpSum(r[s][t] for s in S),f"Low_Rate_Workers_At_Hour_{t}")

    # 5.4.2
    for t in T:
        model += (P[t] <= PROCESS_RATE*H[t] - (PROCESS_RATE - LOW_PROCESS_RATE) * R[t],f"Processing_Capacity_{t}")

    # 5.5 第0小时的货物平衡
    model += (B[0] == Q[0] - P[0],"Backlog_0")

    # 5.6 第1~23小时货物动态平衡
    for t in range(1,24):
        model += (B[t] == B[t-1] + Q[t] - P[t],f"Backlog_{t}")

    # 5.7 当天货物当天全部处理完
    model += (B[23] == 0,"End_Of_Day_Clear")

    # 5.8 峰值人数约束
    for t in T:
        model += (H[t] <= H_max,f"Peak_Workers_{t}")

    # 5.9 低效率人时约束
    for s in S:
        model+= (pulp.lpSum(r[s][t] for t in T) == x[s],f"One_Low_Rate_Hour_Per_Worker_Shift_{s}")
        for t in T:
            if s <= t <= s+SHIFT_LENGTH-1:
                model += (r[s][t] <= x[s],f"Low_Rate_Inside_Shift_{s}_{t}")
            else:
                model += (r[s][t] == 0,f"Low_Rate_Outside_Shift_{s}_{t}")
    # 5.10 0—12点货物16点前完成约束
    for t in EARLY_T:
        model += (P_early[t] <= P[t],f"Early_Processed_Not_Exceed_Total_{t}")
        if t == 0:
            model += (B_early[t] == Q[t] -P_early[t],f"Early_Backlog_{t}")
        elif t<EARLY_END_HOUR:
            model += (B_early[t] == B_early[t-1] + Q[t] -P_early[t],f"Early_Backlog_{t}")
        else:
            model += (B_early[t] == B_early[t-1] - P_early[t],f"Early_Backlog_{t}")
    model += (B_early[DEADLINE_HOUR-1] == 0,"Early_Goods_Clear_Before_16")

    # 6. 创建HiGHS求解器
    solver = pulp.HiGHS(msg=False)

    # 7. 第一层优化：总工人数最少
    Z1 = pulp.lpSum(x[s] for s in S)
    model.setObjective(Z1)
    model.solve(solver)
    if pulp.LpStatus[model.status] != "Optimal":
        raise RuntimeError(f"第{day}天第一层优化失败：",f"{pulp.LpStatus[model.status]}")
    Z1_star = int(round(pulp.value(Z1)))
    model += (Z1 == Z1_star,"Fix_First_Objective")

    # 8. 第二层优化：峰值在岗人数最少
    model.setObjective(H_max)
    model.solve(solver)
    if pulp.LpStatus[model.status] != "Optimal":
        raise RuntimeError(f"第{day}天第二层优化失败："f"{pulp.LpStatus[model.status]}")
    Z2_star = int(round(pulp.value(H_max)))
    model += (H_max == Z2_star,"Fix_Second_Objective")

    # 9. 第三层优化：累计积压量最少
    Z3 = pulp.lpSum(B[t] for t in T)
    model.setObjective(Z3)
    model.solve(solver)
    if pulp.LpStatus[model.status] != "Optimal":
        raise RuntimeError(f"第{day}天第三层优化失败："f"{pulp.LpStatus[model.status]}")
    Z3_star = int(round(pulp.value(Z3)))

    # 10. 提取5个班次结果
    shift_result = []
    selected_shifts = [s for s in S if pulp.value(y[s])>0.5]
    selected_shifts.sort()
    for number,s in enumerate(selected_shifts,start=1):
        workers = int(round(pulp.value(x[s])))
        shift_result.append({
            "天": day,
            "班次": number,
            "开始小时": s,
            "结束小时": s + SHIFT_LENGTH,
            "开始时间": time_string(s),
            "结束时间": time_string(
                s + SHIFT_LENGTH
            ),
            "工人数": workers
        })
    # 11. 提取逐小时运行情况
    hourly_result = []
    cumulative_in = 0
    cumulative_out = 0
    for t in T:
        incoming = Q[t]
        workers = int(round(pulp.value(H[t])))
        low_workers = int(round(pulp.value(R[t])))
        processed = int(round(pulp.value(P[t])))
        backlog = int(round(pulp.value(B[t])))
        capacity = (
                PROCESS_RATE * workers
                - (PROCESS_RATE - LOW_PROCESS_RATE) * low_workers
        )
        if t < DEADLINE_HOUR:
            early_processed = int(round(pulp.value(P_early[t])))
            early_backlog = int(round(pulp.value(B_early[t])))
        else:
            early_processed = 0
            early_backlog = 0
        cumulative_in += incoming
        cumulative_out += processed
        hourly_result.append({
            "天": day,
            "小时": t,
            "时间": time_string(t),
            "进货量": incoming,
            "在岗人数": workers,
            "10件效率人数": low_workers,
            "最大处理能力": capacity,
            "实际处理量": processed,
            "小时末积压量": backlog,
            "0-12点货实际处理量": early_processed,
            "0-12点货小时末积压量": early_backlog,
            "累计进货量": cumulative_in,
            "累计处理量": cumulative_out
        })
    # 12. 理论总量下界
    theoretical_min = math.ceil(total_goods/WORKER_SHIFT_CAPACITY)

    # 13. 每日汇总
    summary = {
        "天": day,
        "总进货量": total_goods,
        "理论总量下界": theoretical_min,
        "最优总工人数": Z1_star,
        "峰值在岗人数": Z2_star,
        "累计积压量": Z3_star,
        "是否达到理论下界":Z1_star == theoretical_min
    }

    # 14. 控制台输出
    print()
    print("=" * 60)
    print(f"第 {day} 天求解完成")
    print(f"总进货量：{total_goods}")
    print(f"0—12点进货量：{early_goods}")
    print(f"理论总量下界：{theoretical_min}")
    print(f"第一层最优总人数：{Z1_star}")
    print(f"第二层最优峰值人数：{Z2_star}")
    print(f"第三层最小累计积压：{Z3_star}")
    print(
        f"16点时0—12点货积压："
        f"{int(round(pulp.value(B_early[15])))}"
    )
    print("\n班次安排：")
    for row in shift_result:
        print(
            f"班次{row['班次']}："
            f"{row['开始时间']}"
            f" - "
            f"{row['结束时间']}，"
            f"{row['工人数']}人"
        )
    print(
        f"\n当天结束积压量："
        f"{hourly_result[-1]['小时末积压量']}"
    )
    return (
        summary,
        shift_result,
        hourly_result
    )
    # 15. 绘制30天最优工人数变化图
def plot_daily_workers(summary_df):
    plt.figure(figsize=(10, 5))
    plt.plot(summary_df["天"],summary_df["最优总工人数"],marker="o")
    plt.xlabel("天数")
    plt.ylabel("最优工人数")
    plt.title("30天每日最优工人数")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/ f"{IMG_PREFIX}_30天最优工人数.png",dpi=300)
    plt.close()

    # 16. 绘制某一天进货量与处理量
def plot_incoming_processed(hourly_df, day):
    data = hourly_df[hourly_df["天"] == day]
    plt.figure(figsize=(10, 5))
    plt.plot(data["小时"],data["进货量"],marker="o",label="进货量")
    plt.plot(data["小时"],data["实际处理量"],marker="s",label="处理量")
    plt.xlabel("小时")
    plt.ylabel("件数")
    plt.title(f"第{day}天 进货量与处理量")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/ f"{IMG_PREFIX}_第{day}天进货与处理.png",dpi=300)
    plt.close()

    # 17. 绘制某一天积压变化图
def plot_backlog(hourly_df, day):
    data = hourly_df[hourly_df["天"] == day]
    plt.figure(figsize=(10, 5))
    plt.plot(data["小时"],data["小时末积压量"],marker="o")
    plt.xlabel("小时")
    plt.ylabel("积压量(件)")
    plt.title(f"第{day}天 积压量变化")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/ f"{IMG_PREFIX}_第{day}天积压量.png",dpi=300)
    plt.close()

    # 18. 绘制某一天0-12点货物积压变化图
def plot_early_backlog(hourly_df, day):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    data = hourly_df[hourly_df["天"] == day]
    plt.figure(figsize=(10, 5))
    plt.plot(data["小时"], data["0-12点货小时末积压量"], marker="o",
             color="orange", label="0-12点货物积压")
    plt.axvline(x=16, color="red", linestyle="--", linewidth=1.5,
                label="16:00 截止")
    plt.xlabel("小时")
    plt.ylabel("积压量(件)")
    plt.title(f"第{day}天 0-12点货物积压变化")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{IMG_PREFIX}_第{day}天0-12点货积压.png", dpi=300)
    plt.close()

    # 19. 绘制某一天在岗人数与10件效率人数图
def plot_workers_vs_lowrate(hourly_df, day):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    data = hourly_df[hourly_df["天"] == day]
    plt.figure(figsize=(10, 5))
    plt.plot(data["小时"], data["在岗人数"], marker="o", label="在岗人数")
    plt.plot(data["小时"], data["10件效率人数"], marker="s", label="10件效率人数")
    plt.xlabel("小时")
    plt.ylabel("人数")
    plt.title(f"第{day}天 每小时在岗人数与10件效率人数")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{IMG_PREFIX}_第{day}天在岗与低效人数.png", dpi=300)
    plt.close()

    # 20. 主程序
def main():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"没有找到文件：{DATA_FILE}")
    print(f"正在读取：{DATA_FILE}")
    df = pd.read_excel(DATA_FILE)
    df = check_data(df)
    if SOLVE_ONLY_DAY is None:
        days_to_solve = sorted(df["天"].unique())
    else:
        days_to_solve = [SOLVE_ONLY_DAY]
    print(f"准备求解日期：{days_to_solve}")
    all_summary = []
    all_shifts = []
    all_hourly = []
    for day in days_to_solve:
        day_data = df[df["天"] == day].copy()
        summary, shifts, hourly = solve_one_day(day,day_data)
        all_summary.append(summary)
        all_shifts.extend(shifts)
        all_hourly.extend(hourly)
    summary_df = pd.DataFrame(all_summary)
    shift_df = pd.DataFrame(all_shifts)
    hourly_df = pd.DataFrame(all_hourly)
    output_excel = (OUTPUT_DIR/ "问题二_最优排班结果.xlsx")
    with pd.ExcelWriter(output_excel,engine="openpyxl") as writer:
        summary_df.to_excel(writer,sheet_name="每日汇总",index=False)
        shift_df.to_excel(writer,sheet_name="班次方案",index=False)
        hourly_df.to_excel(writer,sheet_name="逐小时运行",index=False)
    if SOLVE_ONLY_DAY is None:plot_daily_workers(summary_df)
    first_day = days_to_solve[0]
    plot_incoming_processed(hourly_df,first_day)
    plot_backlog(hourly_df,first_day)
    plot_early_backlog(hourly_df,first_day)
    plot_workers_vs_lowrate(hourly_df,first_day)
    print()
    print("=" * 60)
    print("问题二求解完成")
    print("=" * 60)
    print(f"Excel结果：{output_excel}")
    print(f"图片结果：{OUTPUT_DIR}")
    # 19. Python程序入口
if __name__ == "__main__":
    main()