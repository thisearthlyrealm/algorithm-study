import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import differential_evolution,NonlinearConstraint,minimize_scalar

# 中文字体设置
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]
plt.rcParams["axes.unicode_minus"]=False

# 1. 基本参数
BASE_DIR=Path(__file__).resolve().parent
OUTPUT_DIR=BASE_DIR/"output"

# 1.1 标准D-H参数：题目表1原值直接使用
DH_A=[0,300,1200,300,0,0]
DH_ALPHA=[0,-90,0,-90,-90,-90]
DH_D=[600,0,0,1200,0,0]

# 1.2 零位关节角 / °
THETA_ZERO=np.array([0,-90,0,180,-90,0],dtype=float)

# 1.3 目标点 / mm
TARGET=np.array([1500.0,1200.0,200.0],dtype=float)

# 1.4 六个关节角范围 / °
JOINT_BOUNDS=[(-160,160),(-150,15),(-200,80),(-180,180),(-120,120),(-180,180)]

# 1.5 转动惯量 / kg·m²
INERTIA=np.array([0.5,0.3,0.4,0.6,0.2,0.4],dtype=float)

# 1.6 题目给出的平均角速度 / rad/s
OMEGA=np.array([2.0,1.5,1.0,2.5,3.0,2.0],dtype=float)

# 1.7 机械臂质量与末端载重之和 / kg
# 由于题目未给出各连杆质量和质心位置，这里将5 kg等效集中于末端
MASS=5.0

# 1.8 重力加速度 / m/s²
G=9.81

# 1.9 动力学路径离散点数
PATH_POINTS=121

# 1.10 epsilon约束法扫描范围 / mm
EPSILON_LIST=np.arange(0,201,20,dtype=float)

# 1.11 优化变量：x=[theta1,theta2,theta3,theta4]
# theta5、theta6不影响末端位置，且任何额外转动只会增加非负能耗，因此固定在零位
OPT_BOUNDS=[JOINT_BOUNDS[0],JOINT_BOUNDS[1],JOINT_BOUNDS[2],JOINT_BOUNDS[3]]

# 1.12 第一问正theta4分支的精确几何解
# 该解满足phi=theta1+theta2=0；theta1、theta2的具体分配在问题2中按新能耗模型重新优化
BASELINE_THETA3=4.7815847477
BASELINE_THETA4=84.4207640324

# 2. 从优化变量恢复六个关节角
def x_to_theta(x):
    """x=[theta1,theta2,theta3,theta4]，theta5、theta6固定在零位。"""
    return np.array([float(x[0]),float(x[1]),float(x[2]),float(x[3]),
                     THETA_ZERO[4],THETA_ZERO[5]],dtype=float)

# 3. 标准D-H末端位置解析式
def end_position(x):
    """
    第一问SDH模型的解析式：
    phi=theta1+theta2
    A=1200*cos(theta3)+300*cos(theta3+theta4)+300
    X=A*cos(phi)-1200*sin(phi)
    Y=A*sin(phi)+1200*cos(phi)
    Z=600-1200*sin(theta3)-300*sin(theta3+theta4)
    """
    theta1,theta2,theta3,theta4=x
    phi_rad=np.deg2rad(theta1+theta2)
    theta3_rad=np.deg2rad(theta3)
    theta34_rad=np.deg2rad(theta3+theta4)
    A=1200*np.cos(theta3_rad)+300*np.cos(theta34_rad)+300
    x_end=A*np.cos(phi_rad)-1200*np.sin(phi_rad)
    y_end=A*np.sin(phi_rad)+1200*np.cos(phi_rad)
    z_end=600-1200*np.sin(theta3_rad)-300*np.sin(theta34_rad)
    return np.array([x_end,y_end,z_end],dtype=float)

# 4. 末端位置误差 / mm
def position_error(x):
    return float(np.linalg.norm(end_position(x)-TARGET))

# 5. 三次时间标度下的关节轨迹
def build_motion_profile(theta_final,n_points=PATH_POINTS):
    """
    使用三次时间标度 s(xi)=3*xi^2-2*xi^3，xi=t/T，保证各关节起止角速度均为0。
    总动作时间 T=max_i(|Delta theta_i|/omega_i)，使各关节平均转速不超过题给平均角速度。
    """
    theta0_rad=np.deg2rad(THETA_ZERO)
    thetaf_rad=np.deg2rad(theta_final)
    delta=thetaf_rad-theta0_rad
    duration_each=np.abs(delta)/OMEGA
    T=float(np.max(duration_each))
    if T<1e-12:
        t=np.array([0.0,1.0],dtype=float)
        theta_path=np.vstack([theta0_rad,theta0_rad])
        qdot=np.zeros((2,6),dtype=float)
        qdd=np.zeros((2,6),dtype=float)
        return t,theta_path,qdot,qdd,0.0
    t=np.linspace(0.0,T,n_points)
    xi=t/T
    s=3.0*xi**2-2.0*xi**3
    ds_dt=(6.0*xi-6.0*xi**2)/T
    d2s_dt2=(6.0-12.0*xi)/(T**2)
    theta_path=theta0_rad[None,:]+s[:,None]*delta[None,:]
    qdot=ds_dt[:,None]*delta[None,:]
    qdd=d2s_dt2[:,None]*delta[None,:]
    return t,theta_path,qdot,qdd,T

# 6. 末端等效集中质量产生的重力广义力矩
def gravity_torque(theta_path):
    """
    末端高度（m）：z=0.6-1.2*sin(theta3)-0.3*sin(theta3+theta4)
    由虚功原理 tau_g,i=m*g*partial(z)/partial(theta_i)；
    当前SDH解析式下高度仅与theta3、theta4有关。
    """
    theta3=theta_path[:,2]
    theta4=theta_path[:,3]
    theta34=theta3+theta4
    dz_dtheta3=-1.2*np.cos(theta3)-0.3*np.cos(theta34)
    dz_dtheta4=-0.3*np.cos(theta34)
    tau_g=np.zeros_like(theta_path)
    tau_g[:,2]=MASS*G*dz_dtheta3
    tau_g[:,3]=MASS*G*dz_dtheta4
    return tau_g

# 7. 简化关节动力学等效机械能耗
def mechanical_energy(theta_final,n_points=PATH_POINTS):
    """
    简化动力学：tau_i=I_i*ddot(theta_i)+tau_g,i，P_i=tau_i*dot(theta_i)。
    题目未给出电机效率与再生制动效率，采用绝对机械功 E=int sum_i|P_i(t)|dt，
    驱动与制动过程的机械功均计入。
    """
    t,theta_path,qdot,qdd,T=build_motion_profile(theta_final,n_points=n_points)
    if T==0.0:
        return {"motion_time":0.0,"inertia_work":0.0,"gravity_work":0.0,
                "total_energy":0.0,"positive_work":0.0,"braking_work":0.0}
    tau_inertia=qdd*INERTIA[None,:]
    tau_g=gravity_torque(theta_path)
    tau_total=tau_inertia+tau_g
    power_inertia=tau_inertia*qdot
    power_gravity=tau_g*qdot
    power_total=tau_total*qdot
    total_power_sum=np.sum(power_total,axis=1)
    inertia_work=float(np.trapezoid(np.sum(np.abs(power_inertia),axis=1),t))
    gravity_work=float(np.trapezoid(np.sum(np.abs(power_gravity),axis=1),t))
    total_energy=float(np.trapezoid(np.sum(np.abs(power_total),axis=1),t))
    positive_work=float(np.trapezoid(np.maximum(total_power_sum,0.0),t))
    braking_work=float(np.trapezoid(np.maximum(-total_power_sum,0.0),t))
    return {"motion_time":float(T),"inertia_work":inertia_work,"gravity_work":gravity_work,
            "total_energy":total_energy,"positive_work":positive_work,"braking_work":braking_work}

# 8. 优化目标：等效总机械能耗 / J
def total_energy(x):
    return mechanical_energy(x_to_theta(x))["total_energy"]

# 9. 第一问正theta4精确分支下，重新寻找最节能的theta1、theta2分配
def build_zero_error_baseline():
    """正theta4分支满足phi=0（theta2=-theta1），新能耗模型下重新分配theta1、theta2。"""
    theta1_low=max(JOINT_BOUNDS[0][0],-JOINT_BOUNDS[1][1])
    theta1_high=min(JOINT_BOUNDS[0][1],-JOINT_BOUNDS[1][0])
    def baseline_objective(theta1):
        theta=np.array([theta1,-theta1,BASELINE_THETA3,BASELINE_THETA4,
                        THETA_ZERO[4],THETA_ZERO[5]],dtype=float)
        return mechanical_energy(theta)["total_energy"]
    opt=minimize_scalar(baseline_objective,bounds=(theta1_low,theta1_high),
                        method="bounded",options={"xatol":1e-10,"maxiter":500})
    theta1=float(opt.x)
    x=np.array([theta1,-theta1,BASELINE_THETA3,BASELINE_THETA4],dtype=float)
    theta=x_to_theta(x)
    p=end_position(x)
    error=position_error(x)
    energy=mechanical_energy(theta)
    return {
        "epsilon":0.0,"x":x,"theta":theta,"position":p,"error":float(error),
        "motion_time":energy["motion_time"],"inertia_work":energy["inertia_work"],
        "gravity_work":energy["gravity_work"],"positive_work":energy["positive_work"],
        "braking_work":energy["braking_work"],"total_energy":energy["total_energy"],
        "solver_success":bool(opt.success),"feasible":bool(error<=1e-4),
        "iterations":0,"evaluations":int(opt.nfev),
        "message":"第一问正theta4精确分支；theta1、theta2按新能耗模型重新分配"
    }

# 10. 给定epsilon，使用差分进化求最小能耗
def solve_epsilon(epsilon,x0,seed=2026):
    """求解min E(x)，满足position_error(x)<=epsilon，变量x=[theta1,theta2,theta3,theta4]。"""
    error_constraint=NonlinearConstraint(position_error,0.0,epsilon)
    result=differential_evolution(
        total_energy,bounds=OPT_BOUNDS,constraints=(error_constraint,),
        strategy="best1bin",maxiter=600,popsize=20,mutation=(0.5,1.0),
        recombination=0.7,tol=1e-8,atol=1e-8,
        x0=np.array(x0,dtype=float),polish=False,seed=seed,workers=1,
        updating="immediate",disp=False)
    x_opt=np.array(result.x,dtype=float)
    theta_opt=x_to_theta(x_opt)
    p_opt=end_position(x_opt)
    error=position_error(x_opt)
    energy=mechanical_energy(theta_opt)
    feasible=(error<=epsilon+1e-4)
    return {
        "epsilon":float(epsilon),"x":x_opt,"theta":theta_opt,"position":p_opt,
        "error":float(error),"motion_time":energy["motion_time"],
        "inertia_work":energy["inertia_work"],"gravity_work":energy["gravity_work"],
        "positive_work":energy["positive_work"],"braking_work":energy["braking_work"],
        "total_energy":energy["total_energy"],"solver_success":bool(result.success),
        "feasible":bool(feasible),"iterations":int(result.nit),
        "evaluations":int(result.nfev),"message":str(result.message)
    }

# 11. epsilon扫描
def solve_pareto():
    results=[]
    baseline=build_zero_error_baseline()
    print("\n【零误差基准验证】")
    print("-"*70)
    print(f"关节角 / ° = {baseline['theta']}")
    print(f"末端位置 / mm = {baseline['position']}")
    print(f"末端误差 / mm = {baseline['error']}")
    print(f"动作时间 / s = {baseline['motion_time']}")
    print(f"惯性绝对功 / J = {baseline['inertia_work']}")
    print(f"重力绝对功 / J = {baseline['gravity_work']}")
    print(f"等效总机械能耗 / J = {baseline['total_energy']}")
    results.append(baseline)
    x0=baseline["x"].copy()
    for index,epsilon in enumerate(EPSILON_LIST[1:],start=1):
        print(f"\n正在求解 epsilon={epsilon:.0f} mm ...")
        result=solve_epsilon(epsilon,x0,seed=2026+index)
        print(f"实际误差 = {result['error']:.6f} mm")
        print(f"动作时间 = {result['motion_time']:.6f} s")
        print(f"总机械能耗 = {result['total_energy']:.6f} J")
        print(f"约束可行 = {result['feasible']}")
        print(f"进化代数 = {result['iterations']}")
        if not result["feasible"]:
            print("警告：该结果没有满足epsilon误差约束")
        if (not result["solver_success"] and result["feasible"]):
            print("提示：DE未满足内部收敛判据，但当前结果满足实际误差约束。")
        results.append(result)
        x0=result["x"].copy()
    return results

# 12. 打印Pareto结果表
def print_pareto_results(results):
    base_energy=results[0]["total_energy"]
    print("\n"+"="*178)
    print("                              问题2：误差—能耗 Pareto 扫描结果（简化动力学模型）")
    print("="*178)
    print(f"{'eps/mm':>8}{'实际误差':>13}{'phi':>11}{'theta1':>11}{'theta2':>11}{'theta3':>11}{'theta4':>11}{'T/s':>10}{'E惯性':>12}{'E重力':>12}{'E总/J':>12}{'降幅/%':>11}{'可行':>8}{'代数':>8}")
    print("-"*178)
    for item in results:
        x=item["x"]
        theta=item["theta"]
        phi=theta[0]+theta[1]
        reduction=(base_energy-item["total_energy"])/base_energy*100.0
        print(f"{item['epsilon']:>8.0f}{item['error']:>13.6f}{phi:>11.6f}{theta[0]:>11.6f}{theta[1]:>11.6f}{theta[2]:>11.6f}{theta[3]:>11.6f}{item['motion_time']:>10.6f}{item['inertia_work']:>12.6f}{item['gravity_work']:>12.6f}{item['total_energy']:>12.6f}{reduction:>11.2f}{str(item['feasible']):>8}{item['iterations']:>8}")
    print("-"*178)
    print("说明：E惯性、E重力是分项绝对功诊断量；E总按|总关节机械功率|积分，不要求等于两者简单相加。")

# 13. 保存Pareto数据到xlsx
def save_results_xlsx(results,save_path):
    base_energy=results[0]["total_energy"]
    rows=[]
    for item in results:
        theta=item["theta"]
        p=item["position"]
        reduction=(base_energy-item["total_energy"])/base_energy*100.0
        rows.append({
            "epsilon_mm":item["epsilon"],
            "actual_error_mm":item["error"],
            "phi_deg":theta[0]+theta[1],
            "theta1_deg":theta[0],
            "theta2_deg":theta[1],
            "theta3_deg":theta[2],
            "theta4_deg":theta[3],
            "theta5_deg":theta[4],
            "theta6_deg":theta[5],
            "end_x_mm":p[0],
            "end_y_mm":p[1],
            "end_z_mm":p[2],
            "motion_time_s":item["motion_time"],
            "inertia_abs_work_J":item["inertia_work"],
            "gravity_abs_work_J":item["gravity_work"],
            "positive_actuator_work_J":item["positive_work"],
            "braking_work_J":item["braking_work"],
            "total_energy_J":item["total_energy"],
            "energy_reduction_percent":reduction,
            "solver_success":item["solver_success"],
            "feasible":item["feasible"],
            "iterations":item["iterations"],
            "evaluations":item["evaluations"]
        })
    pd.DataFrame(rows).to_excel(save_path,index=False)

# 14. 绘制误差—能耗Pareto曲线
def plot_pareto_curve(results,save_path):
    errors=[item["error"] for item in results]
    energies=[item["total_energy"] for item in results]
    fig=plt.figure(figsize=(9,6))
    ax=fig.add_subplot(111)
    ax.plot(errors,energies,marker="o")
    ax.set_xlabel("末端位置误差 / mm")
    ax.set_ylabel("等效机械能耗 / J")
    ax.set_title("末端误差—等效机械能耗 Pareto 曲线")
    ax.grid(True,alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path,dpi=300,bbox_inches="tight")
    plt.close(fig)

# 15. 绘制最优关节角随epsilon变化
def plot_joint_angles(results,save_path):
    eps=[item["epsilon"] for item in results]
    theta1=[item["theta"][0] for item in results]
    theta2=[item["theta"][1] for item in results]
    theta3=[item["theta"][2] for item in results]
    theta4=[item["theta"][3] for item in results]
    fig=plt.figure(figsize=(10,6))
    ax=fig.add_subplot(111)
    ax.plot(eps,theta1,marker="o",label="theta1")
    ax.plot(eps,theta2,marker="o",label="theta2")
    ax.plot(eps,theta3,marker="o",label="theta3")
    ax.plot(eps,theta4,marker="o",label="theta4")
    ax.set_xlabel("允许末端误差 epsilon / mm")
    ax.set_ylabel("最优关节角 / °")
    ax.set_title("不同误差约束下的最优关节角")
    ax.grid(True,alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path,dpi=300,bbox_inches="tight")
    plt.close(fig)

# 16. 获取三次时间标度下的末端高度路径
def get_height_path(theta_final,n_points=PATH_POINTS):
    t,theta_path,_,_,T=build_motion_profile(theta_final,n_points=n_points)
    theta3=theta_path[:,2]
    theta4=theta_path[:,3]
    theta34=theta3+theta4
    heights_mm=600-1200*np.sin(theta3)-300*np.sin(theta34)
    progress=t/T if T>0.0 else np.array([0.0,1.0],dtype=float)
    return progress,np.array(heights_mm,dtype=float)

# 17. 绘制零误差与200 mm方案的高度路径
def plot_height_comparison(results,save_path):
    zero_solution=results[0]
    max_error_solution=results[-1]
    s_zero,h_zero=get_height_path(zero_solution["theta"])
    s_200,h_200=get_height_path(max_error_solution["theta"])
    fig=plt.figure(figsize=(9,6))
    ax=fig.add_subplot(111)
    ax.plot(s_zero,h_zero,label="零误差方案")
    ax.plot(s_200,h_200,label="200 mm误差方案")
    ax.set_xlabel("归一化运动时间 t/T")
    ax.set_ylabel("末端高度 Z / mm")
    ax.set_title("零误差与200 mm方案的末端高度路径")
    ax.grid(True,alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path,dpi=300,bbox_inches="tight")
    plt.close(fig)

# 18. 绘制零误差与200 mm方案的关节总机械功率
def get_total_power_path(theta_final,n_points=PATH_POINTS):
    t,theta_path,qdot,qdd,T=build_motion_profile(theta_final,n_points=n_points)
    if T==0.0:
        return np.array([0.0,1.0]),np.zeros(2,dtype=float)
    tau_inertia=qdd*INERTIA[None,:]
    tau_g=gravity_torque(theta_path)
    tau_total=tau_inertia+tau_g
    power_total=np.sum(tau_total*qdot,axis=1)
    return t,np.array(power_total,dtype=float)

def plot_power_comparison(results,save_path):
    zero_solution=results[0]
    max_error_solution=results[-1]
    t_zero,p_zero=get_total_power_path(zero_solution["theta"])
    t_200,p_200=get_total_power_path(max_error_solution["theta"])
    fig=plt.figure(figsize=(9,6))
    ax=fig.add_subplot(111)
    ax.plot(t_zero,p_zero,label="零误差方案")
    ax.plot(t_200,p_200,label="200 mm误差方案")
    ax.axhline(0.0,linewidth=1.0)
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("关节总机械功率 / W")
    ax.set_title("零误差与200 mm方案的机械功率对比")
    ax.grid(True,alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path,dpi=300,bbox_inches="tight")
    plt.close(fig)

# 19. 主程序
def main():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

    # 19.1 题目参数检查
    print("\n【1】题目给定动力学参数")
    print("-"*70)
    for i in range(6):
        print(f"关节{i+1}: I={INERTIA[i]:.3f} kg·m², omega={OMEGA[i]:.3f} rad/s")
    print(f"等效末端集中质量 = {MASS:.3f} kg")

    # 19.2 第一问几何解验证
    print("\n【2】第一问正theta4精确分支几何验证")
    print("-"*70)
    x_test=np.array([0.0,0.0,BASELINE_THETA3,BASELINE_THETA4],dtype=float)
    p_test=end_position(x_test)
    error_test=position_error(x_test)
    print("末端位置 / mm =",p_test)
    print("末端误差 / mm =",error_test)

    # 19.3 epsilon约束法 + DE
    print("\n【3】开始epsilon约束法 + 差分进化求解")
    print("-"*70)
    results=solve_pareto()

    # 19.4 打印结果
    print_pareto_results(results)

    # 19.5 保存xlsx
    xlsx_path=OUTPUT_DIR/"第二问_误差能耗Pareto数据.xlsx"
    save_results_xlsx(results,xlsx_path)

    # 19.6 Pareto曲线
    pareto_path=OUTPUT_DIR/"第二问_误差能耗Pareto曲线.png"
    plot_pareto_curve(results,pareto_path)

    # 19.7 最优关节角变化
    angle_path=OUTPUT_DIR/"第二问_最优关节角变化.png"
    plot_joint_angles(results,angle_path)

    # 19.8 高度路径
    height_path=OUTPUT_DIR/"第二问_末端高度路径对比.png"
    plot_height_comparison(results,height_path)

    # 19.9 机械功率对比
    power_path=OUTPUT_DIR/"第二问_机械功率对比.png"
    plot_power_comparison(results,power_path)

    # 19.10 输出文件位置
    print("\n"+"="*78)
    print("                              结果文件")
    print("="*78)
    print(f"Pareto数据：output/{xlsx_path.name}")
    print(f"Pareto曲线：output/{pareto_path.name}")
    print(f"最优关节角变化：output/{angle_path.name}")
    print(f"末端高度路径对比：output/{height_path.name}")
    print(f"机械功率对比：output/{power_path.name}")
    print("="*78)

if __name__=="__main__":
    main()
