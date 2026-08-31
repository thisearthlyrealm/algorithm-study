import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import differential_evolution

# 中文字体设置，防止图内中文乱码
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]
plt.rcParams["axes.unicode_minus"]=False


# ============================================================
# 1. 基本参数
# ============================================================

BASE_DIR=Path(__file__).resolve().parent
OUTPUT_DIR=BASE_DIR/"output"

# 解释B：题目表1参数本身直接按标准D-H（SDH）参数使用
DH_A=[0,300,1200,300,0,0]          # a_i / mm
DH_ALPHA=[0,-90,0,-90,-90,-90]     # alpha_i / °
DH_D=[600,0,0,1200,0,0]            # d_i / mm

# 零位关节角 / °
THETA_ZERO=[0,-90,0,180,-90,0]

# 问题1目标点 / mm
TARGET=np.array([1500.0,1200.0,200.0])

# 题目给定关节角范围 / °
JOINT_BOUNDS=[(-160,160),(-150,15),(-200,80),(-180,180),(-120,120),(-180,180)]

# 标准D-H直接解释下，末端位置中theta1和theta2主要通过phi=theta1+theta2
# 共同起作用，因此将4维搜索降为3维：x=[phi,theta3,theta4]
PHI_BOUNDS=(JOINT_BOUNDS[0][0]+JOINT_BOUNDS[1][0],JOINT_BOUNDS[0][1]+JOINT_BOUNDS[1][1])

POSITION_BOUNDS=[PHI_BOUNDS,JOINT_BOUNDS[2],JOINT_BOUNDS[3]]


# ============================================================
# 2. 标准D-H齐次变换矩阵
# ============================================================

def sdh_matrix(a,alpha_deg,d,theta_deg):
    """根据标准D-H（SDH）参数构造4×4齐次变换矩阵。"""
    alpha=np.deg2rad(alpha_deg)
    theta=np.deg2rad(theta_deg)
    ca=np.cos(alpha); sa=np.sin(alpha)
    ct=np.cos(theta); st=np.sin(theta)
    T=np.array([[ct,-st*ca,st*sa,a*ct],[st,ct*ca,-ct*sa,a*st],[0,sa,ca,d],[0,0,0,1]],dtype=float)
    return T


# ============================================================
# 3. 六自由度机械臂正运动学
# ============================================================

def forward_kinematics(theta):
    """输入6个关节角（度），返回J0~J6位置(7,3)与累计变换矩阵列表。"""
    if len(theta)!=6:
        raise ValueError("theta必须包含6个关节角")
    T=np.eye(4)
    positions=[np.array([0.0,0.0,0.0])]
    transforms=[T.copy()]
    for i in range(6):
        A=sdh_matrix(DH_A[i],DH_ALPHA[i],DH_D[i],theta[i])
        T=T@A
        transforms.append(T.copy())
        positions.append(T[:3,3].copy())
    positions=np.array(positions)
    # 清除极小浮点误差
    positions[np.abs(positions)<1e-10]=0.0
    return positions,transforms


# ============================================================
# 4. phi拆分为theta1、theta2
# ============================================================

def split_phi(phi):
    """在满足关节角范围的连续等价解中，选择相对零位平方调整量最小的一组代表解。"""
    theta1_low=max(JOINT_BOUNDS[0][0],phi-JOINT_BOUNDS[1][1])
    theta1_high=min(JOINT_BOUNDS[0][1],phi-JOINT_BOUNDS[1][0])
    if theta1_low>theta1_high:
        raise ValueError("当前phi不存在可行的theta1、theta2")
    # 最小化(theta1-θ1_zero)^2+(theta2-θ2_zero)^2，且theta2=phi-theta1
    theta1_zero=THETA_ZERO[0]
    theta2_zero=THETA_ZERO[1]
    theta1_best=(theta1_zero+phi-theta2_zero)/2
    theta1=float(np.clip(theta1_best,theta1_low,theta1_high))
    theta2=float(phi-theta1)
    return theta1,theta2


# ============================================================
# 5. 问题1末端位置误差
# ============================================================

def position_error(x):
    """优化变量x=[phi,theta3,theta4]，其中phi=theta1+theta2。"""
    phi,theta3,theta4=x[0],x[1],x[2]
    theta1,theta2=split_phi(phi)
    theta=[theta1,theta2,theta3,theta4,THETA_ZERO[4],THETA_ZERO[5]]
    positions,_=forward_kinematics(theta)
    return float(np.linalg.norm(positions[-1]-TARGET))


# ============================================================
# 6. 差分进化求解问题1
# ============================================================

def solve_problem1(seed=2026,theta4_bounds=None):
    convergence=[]
    def callback(xk,convergence_val):
        convergence.append(position_error(xk))
    # 如果没有指定theta4范围，就搜索完整范围
    if theta4_bounds is None:
        theta4_bounds=JOINT_BOUNDS[3]
    bounds=[PHI_BOUNDS,JOINT_BOUNDS[2],theta4_bounds]
    result=differential_evolution(
        position_error,bounds=bounds,strategy="best1bin",
        maxiter=1500,popsize=30,mutation=(0.5,1.0),recombination=0.8,
        tol=1e-10,atol=1e-9,callback=callback,polish=True,
        rng=np.random.default_rng(seed))
    phi_opt=float(result.x[0])
    theta3_opt=float(result.x[1])
    theta4_opt=float(result.x[2])
    theta1_opt,theta2_opt=split_phi(phi_opt)
    theta_opt=[theta1_opt,theta2_opt,theta3_opt,theta4_opt,THETA_ZERO[4],THETA_ZERO[5]]
    positions_opt,transforms_opt=forward_kinematics(theta_opt)
    final_error=float(np.linalg.norm(positions_opt[-1]-TARGET))
    return (theta_opt,positions_opt,transforms_opt,final_error,result,convergence,phi_opt)


# ============================================================
# 6.1 分别搜索两支逆运动学解
# ============================================================

def solve_two_branches():
    """分别在theta4>0和theta4<0区域搜索，稳定获得两个主要逆运动学分支。"""
    branch_positive=solve_problem1(seed=2026,theta4_bounds=(0,180))
    branch_negative=solve_problem1(seed=2027,theta4_bounds=(-180,0))
    return (branch_positive,branch_negative)


# ============================================================
# 6.2 从两支解中选取展示用代表解
# ============================================================

def representative_score(theta):
    """相对零位状态的关节角平方调整量，仅用于展示选解，不作问题1主目标。"""
    theta=np.array(theta[:4],dtype=float)
    theta_zero=np.array(THETA_ZERO[:4],dtype=float)
    return float(np.sum((theta-theta_zero)**2))


def choose_representative_solution(branch_positive,branch_negative):
    theta_pos,theta_neg=branch_positive[0],branch_negative[0]
    error_pos,error_neg=branch_positive[3],branch_negative[3]
    score_pos,score_neg=representative_score(theta_pos),representative_score(theta_neg)
    print("\n"+"="*72)
    print("                  两支逆运动学解比较")
    print("="*72)
    print("\n正theta4分支：")
    print(f"theta = ({theta_pos[0]:.6f},{theta_pos[1]:.6f},{theta_pos[2]:.6f},{theta_pos[3]:.6f},{theta_pos[4]:.6f},{theta_pos[5]:.6f})")
    print(f"末端误差 = {error_pos:.6e} mm")
    print(f"代表性调整指标 = {score_pos:.6f}")
    print("\n负theta4分支：")
    print(f"theta = ({theta_neg[0]:.6f},{theta_neg[1]:.6f},{theta_neg[2]:.6f},{theta_neg[3]:.6f},{theta_neg[4]:.6f},{theta_neg[5]:.6f})")
    print(f"末端误差 = {error_neg:.6e} mm")
    print(f"代表性调整指标 = {score_neg:.6f}")
    print("-"*72)
    if score_pos<=score_neg:
        print("第一问推荐展示：正theta4分支")
        return branch_positive
    else:
        print("第一问推荐展示：负theta4分支")
        return branch_negative


# ============================================================
# 7. 多随机种子实验
# ============================================================

def test_multiple_seeds():
    seeds=[2026,2027,2028,1107,42]
    results=[]
    print("\n"+"="*104)
    print("                         差分进化多随机种子实验")
    print("="*104)
    print(f"{'Seed':<8}{'phi':>13}{'theta1':>13}{'theta2':>13}{'theta3':>13}{'theta4':>13}{'误差/mm':>18}{'代数':>10}")
    print("-"*104)
    for seed in seeds:
        theta_opt,positions_opt,transforms_opt,final_error,result,convergence,phi_opt=solve_problem1(seed)
        results.append([seed,phi_opt,theta_opt[0],theta_opt[1],theta_opt[2],theta_opt[3],final_error,result.nit])
        print(f"{seed:<8}{phi_opt:>13.6f}{theta_opt[0]:>13.6f}{theta_opt[1]:>13.6f}{theta_opt[2]:>13.6f}{theta_opt[3]:>13.6f}{final_error:>18.6e}{result.nit:>10}")
    print("-"*104)
    return results


# ============================================================
# 8. 输出问题1优化结果
# ============================================================

def print_problem1_result(theta_opt,positions_opt,final_error,result,phi_opt):
    end_position=positions_opt[-1]
    print(f"phi = theta1+theta2 = {phi_opt:.6f}°")
    print(f"theta = ({theta_opt[0]:.6f},{theta_opt[1]:.6f},{theta_opt[2]:.6f},{theta_opt[3]:.6f},{theta_opt[4]:.6f},{theta_opt[5]:.6f})")
    print(f"P6 = ({end_position[0]:.6f},{end_position[1]:.6f},{end_position[2]:.6f})")
    print(f"E = {final_error:.6e} mm")
    print(f"收敛: {result.success} | 进化代数: {result.nit} | 目标函数计算: {result.nfev}")


# ============================================================
# 9. 输出连续等价解范围
# ============================================================

def print_solution_family(phi):
    """在给定phi下输出theta1、theta2的连续可行范围：theta2=phi-theta1"""
    theta1_low=max(JOINT_BOUNDS[0][0],phi-JOINT_BOUNDS[1][1])
    theta1_high=min(JOINT_BOUNDS[0][1],phi-JOINT_BOUNDS[1][0])
    theta2_low=phi-theta1_high
    theta2_high=phi-theta1_low
    print("\n"+"="*64)
    print("                    连续等价解范围")
    print("="*64)
    print(f"phi = theta1 + theta2 = {phi:.10f}°")
    print(f"theta1 ∈ [{theta1_low:.10f},{theta1_high:.10f}]°")
    print(f"theta2 ∈ [{theta2_low:.10f},{theta2_high:.10f}]°")
    print("并满足：theta2 = phi - theta1")
    print("="*64)


# ============================================================
# 10. 输出零位状态
# ============================================================

def solve_zero_state():
    """计算零位状态下各关节坐标与零位末端误差。"""
    positions,transforms=forward_kinematics(THETA_ZERO)
    print("\n===== 六自由度机械臂零位状态 =====")
    print("\n【1】各关节空间坐标 / mm")
    print(f"{'关节':<8}{'X':>16}{'Y':>16}{'Z':>16}")
    for i,p in enumerate(positions):
        print(f"{'J'+str(i):<8}{p[0]:>16.2f}{p[1]:>16.2f}{p[2]:>16.2f}")
    end=positions[-1]
    print("\n【2】机械臂末端位置")
    print(f"P6 = ({end[0]:.2f},{end[1]:.2f},{end[2]:.2f}) mm")
    print(f"E0 = {np.linalg.norm(end-TARGET):.6f} mm")
    return positions,transforms


# ============================================================
# 11. 绘制零位机械臂简图
# ============================================================

def plot_arm_schematic(positions,save_path):
    fig=plt.figure(figsize=(8,7))
    ax=fig.add_subplot(111,projection="3d")
    x,y,z=positions[:,0],positions[:,1],positions[:,2]
    ax.plot(x,y,z,marker="o",linewidth=2)
    # J0~J3
    for i in range(4):
        ax.text(x[i],y[i],z[i]+40,f"J{i}")
    # 在当前SDH模型中J4、J5、J6原点重合
    ax.text(x[4],y[4],z[4]+40,"J4/J5/J6")
    ax.set_xlabel("X / mm"); ax.set_ylabel("Y / mm"); ax.set_zlabel("Z / mm")
    ax.set_title("六自由度机械臂零位状态简图")
    # 扩大范围，避免SDH零位状态被截断
    ax.set_xlim(-1800,2300); ax.set_ylim(-1800,2300); ax.set_zlim(-500,2500)
    ax.view_init(elev=22,azim=-55)
    ax.set_box_aspect((4100,4100,3000))
    fig.savefig(save_path,dpi=300,bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 12. 绘制问题1优化前后机械臂对比图
# ============================================================

def plot_optimization_comparison(positions_zero,positions_opt,save_path):
    fig=plt.figure(figsize=(9,8))
    ax=fig.add_subplot(111,projection="3d")
    # 零位机械臂
    ax.plot(positions_zero[:,0],positions_zero[:,1],positions_zero[:,2],marker="o",linewidth=2,linestyle="--",label="零位状态")
    # 优化后机械臂
    ax.plot(positions_opt[:,0],positions_opt[:,1],positions_opt[:,2],marker="o",linewidth=2.5,label="优化后状态")
    # 目标点
    ax.scatter(TARGET[0],TARGET[1],TARGET[2],marker="*",s=180,label="目标点")
    # 标注优化后的J0~J3
    for i in range(4):
        ax.text(positions_opt[i,0],positions_opt[i,1],positions_opt[i,2]+40,f"J{i}")
    # J4/J5/J6
    ax.text(positions_opt[4,0],positions_opt[4,1],positions_opt[4,2]+40,"J4/J5/J6")
    ax.set_xlabel("X / mm"); ax.set_ylabel("Y / mm"); ax.set_zlabel("Z / mm")
    ax.set_title("机械臂关节角优化前后位姿对比")
    ax.set_xlim(-1800,2300); ax.set_ylim(-1800,2300); ax.set_zlim(-500,2500)
    ax.view_init(elev=25,azim=-55)
    ax.set_box_aspect((4100,4100,3000))
    ax.legend()
    fig.savefig(save_path,dpi=300,bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 13. 绘制差分进化收敛曲线
# ============================================================

def plot_convergence(convergence,save_path):
    fig=plt.figure(figsize=(9,6))
    ax=fig.add_subplot(111)
    generations=np.arange(1,len(convergence)+1)
    # 对数坐标不能出现0，因此给极小值设置下限
    convergence_safe=np.maximum(np.asarray(convergence,dtype=float),1e-15)
    ax.plot(generations,convergence_safe,marker="o",linewidth=1.5,markersize=3,label="当代最优末端误差")
    ax.set_yscale("log")
    ax.set_xlabel("进化代数"); ax.set_ylabel("最优末端误差 / mm")
    ax.set_title("差分进化收敛曲线")
    ax.grid(True,alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path,dpi=300,bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 14. 主程序
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

    # 14.1 零位状态
    positions_zero,transforms_zero=solve_zero_state()
    zero_path=OUTPUT_DIR/"第一问_SDH_零位机械臂简图.png"
    plot_arm_schematic(positions_zero,zero_path)
    print("\n【零位图文件】")
    print(f"output/{zero_path.name}")

    # 14.2 多随机种子实验
    seed_results=test_multiple_seeds()

    # 14.3 分别搜索两支精确逆运动学解
    branch_positive,branch_negative=solve_two_branches()

    # 14.4 输出正theta4分支
    print("\n\n【正 theta4 分支】")
    print_problem1_result(branch_positive[0],branch_positive[1],branch_positive[3],branch_positive[4],branch_positive[6])

    # 14.5 输出负theta4分支
    print("\n\n【负 theta4 分支】")
    print_problem1_result(branch_negative[0],branch_negative[1],branch_negative[3],branch_negative[4],branch_negative[6])

    # 14.6 从两支中选一组用于第一问画图展示
    representative=choose_representative_solution(branch_positive,branch_negative)
    (theta_opt,positions_opt,transforms_opt,final_error,result,convergence,phi_opt)=representative

    # 14.7 输出连续等价解范围
    print_solution_family(phi_opt)

    # 14.8 优化前后对比图
    comp_path=OUTPUT_DIR/"第一问_SDH_机械臂优化前后对比.png"
    plot_optimization_comparison(positions_zero,positions_opt,comp_path)

    # 14.9 收敛曲线
    conv_path=OUTPUT_DIR/"第一问_SDH_差分进化收敛曲线.png"
    plot_convergence(convergence,conv_path)

    print("\n"+"="*64)
    print("                    结果文件")
    print("="*64)
    print(f"零位机械臂简图：output/{zero_path.name}")
    print(f"优化前后对比图：output/{comp_path.name}")
    print(f"差分进化收敛曲线：output/{conv_path.name}")
    print("="*64)


if __name__=="__main__":
    main()
