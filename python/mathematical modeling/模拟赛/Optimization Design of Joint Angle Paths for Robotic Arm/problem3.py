# ============================================================
# A题问题3 合并最终版
# 来源：problem3.1.py + problem3.2.py
#
# RUN_MODE:
#   "explore" -> 只运行全局快速探索
#   "verify"  -> 只运行最终高精度核验（已有探索结果时可直接使用warm-start）
#   "all"     -> 先全局探索，再高精度核验
# ============================================================

RUN_MODE="all"

import math
import heapq
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import differential_evolution


# ============================================================
# 0. 绘图设置
# ============================================================

plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]
plt.rcParams["axes.unicode_minus"]=False


# ============================================================
# 1. 基本参数
# ============================================================

BASE_DIR=Path(__file__).resolve().parent
OUTPUT_DIR=BASE_DIR/"output"

# Sheet1：20×20栅格，0=可行区域，1=障碍物
GRID=np.array([
    [0,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,0,1,0],
    [0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,0,0,1,1],
    [0,1,0,1,0,0,1,0,0,0,0,0,1,0,0,1,1,0,0,0],
    [1,0,0,0,1,0,1,1,0,0,0,0,0,0,0,1,1,1,0,1],
    [1,0,1,1,0,1,0,1,1,0,0,1,1,0,0,0,1,1,1,1],
    [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0],
    [0,0,0,1,0,0,0,0,0,1,0,1,0,1,0,1,1,1,1,1],
    [0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0],
    [0,1,1,0,0,0,0,0,0,0,1,1,0,0,1,0,0,1,0,0],
    [0,0,0,1,1,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0],
    [0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,1,1,0,1],
    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,1],
    [0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,0,1,1,1,0,1,0,0,0,1,0,0,0,1,0,0,0],
    [0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,1,0,0,0]
],dtype=int)

ROWS,COLS=GRID.shape
CELL_SIZE=200.0

# 0-based栅格坐标
START=(0,0)
END=(19,19)
TARGET_Z=200.0

# 题目零位 / °
THETA_ZERO=np.array([0.0,-90.0,0.0,180.0,-90.0,0.0],dtype=float)

# 题目关节范围 / °
JOINT_BOUNDS=[(-160.0,160.0),(-150.0,15.0),(-200.0,80.0),
              (-180.0,180.0),(-120.0,120.0),(-180.0,180.0)]

# 沿用第二问4维优化变量：x=[theta1,theta2,theta3,theta4]
OPT_BOUNDS=[JOINT_BOUNDS[0],JOINT_BOUNDS[1],JOINT_BOUNDS[2],JOINT_BOUNDS[3]]

# 第二问动力学参数
INERTIA=np.array([0.5,0.3,0.4,0.6,0.2,0.4],dtype=float)
OMEGA=np.array([2.0,1.5,1.0,2.5,3.0,2.0],dtype=float)
MASS=5.0
G=9.81
OPT_PATH_POINTS=41
FINAL_PATH_POINTS=121

# 第三问误差上限
MAX_ERROR=200.0

# 地图物理边界 / mm
MAP_X_MIN=-CELL_SIZE/2.0
MAP_X_MAX=(COLS-1)*CELL_SIZE+CELL_SIZE/2.0
MAP_Y_MIN=-CELL_SIZE/2.0
MAP_Y_MAX=(ROWS-1)*CELL_SIZE+CELL_SIZE/2.0

# ---------- 多随机种子 ----------
FEASIBILITY_SEEDS=[2026,2027]
ENERGY_SEEDS=[2026]

# ---------- DE参数 ----------
FEASIBILITY_MAXITER=90
FEASIBILITY_POPSIZE=9
ENERGY_MAXITER=80
ENERGY_POPSIZE=8

# ---------- 自适应扫描 ----------
COARSE_STEP=10.0
FINE_STEP=1.0
ENERGY_DROP_TRIGGER=0.10

# ---------- Pareto工程容差 ----------
PARETO_ERROR_TOL=0.5      # mm
PARETO_ENERGY_TOL=1e-3    # J

# ---------- 罚函数（碰撞/越界是硬约束，罚很重） ----------
COLLISION_PENALTY=1e6
ERROR_PENALTY=1e4

# 可行性数值容差
ERROR_FEAS_TOL=1e-3
COLLISION_FEAS_TOL=1e-12

OBSTACLES=[(r,c) for r in range(ROWS) for c in range(COLS) if GRID[r,c]==1]

OBSTACLE_RECTS=[]
for r,c in OBSTACLES:
    cx=c*CELL_SIZE
    cy=r*CELL_SIZE
    half=CELL_SIZE/2.0
    OBSTACLE_RECTS.append((cx-half,cx+half,cy-half,cy+half))


# ============================================================
# 2. 坐标转换与A*
# ============================================================

def grid_to_world(cell):
    r,c=cell
    return c*CELL_SIZE,r*CELL_SIZE


def local_target(base_cell):
    bx,by=grid_to_world(base_cell)
    tx,ty=grid_to_world(END)
    return np.array([tx-bx,ty-by,TARGET_Z],dtype=float)


def manhattan(a,b):
    return abs(a[0]-b[0])+abs(a[1]-b[1])


def astar_path(start,goal):
    queue=[]
    heapq.heappush(queue,(manhattan(start,goal),0,start))
    came_from={}
    g_score={start:0}
    closed=set()
    while queue:
        _,current_g,current=heapq.heappop(queue)
        if current in closed:
            continue
        closed.add(current)
        if current==goal:
            path=[current]
            while current in came_from:
                current=came_from[current]
                path.append(current)
            path.reverse()
            return path
        r,c=current
        for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr,nc=r+dr,c+dc
            if not (0<=nr<ROWS and 0<=nc<COLS):
                continue
            neighbor=(nr,nc)
            if GRID[nr,nc]==1:
                continue
            # 货物格不允许底座占据或穿过
            if neighbor==END:
                continue
            new_g=current_g+1
            if new_g<g_score.get(neighbor,float("inf")):
                g_score[neighbor]=new_g
                came_from[neighbor]=current
                f=new_g+manhattan(neighbor,goal)
                heapq.heappush(queue,(f,new_g,neighbor))
    return None


# ============================================================
# 3. 标准D-H末端位置解析式（与第二问一致）
# ============================================================

def x_to_theta(x):
    """x=[theta1,theta2,theta3,theta4]，theta5、theta6固定零位。"""
    return np.array([float(x[0]),float(x[1]),float(x[2]),float(x[3]),
                     THETA_ZERO[4],THETA_ZERO[5]],dtype=float)


def end_position(x):
    theta1,theta2,theta3,theta4=x
    phi_rad=np.deg2rad(theta1+theta2)
    theta3_rad=np.deg2rad(theta3)
    theta34_rad=np.deg2rad(theta3+theta4)
    A=1200*np.cos(theta3_rad)+300*np.cos(theta34_rad)+300
    x_end=A*np.cos(phi_rad)-1200*np.sin(phi_rad)
    y_end=A*np.sin(phi_rad)+1200*np.cos(phi_rad)
    z_end=600-1200*np.sin(theta3_rad)-300*np.sin(theta34_rad)
    return np.array([x_end,y_end,z_end],dtype=float)


def position_error(x,target):
    return float(np.linalg.norm(end_position(x)-target))


# ============================================================
# 4. 第二问三次时间标度动力学能耗模型
# ============================================================

def build_motion_profile(theta_final,n_points=FINAL_PATH_POINTS):
    """s(xi)=3xi^2-2xi^3，T=max_i(|Delta theta_i|/omega_i)。"""
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


def gravity_torque(theta_path):
    """末端质量等效集中于末端，高度仅与theta3、theta4有关。"""
    theta3=theta_path[:,2]
    theta4=theta_path[:,3]
    theta34=theta3+theta4
    dz_dtheta3=-1.2*np.cos(theta3)-0.3*np.cos(theta34)
    dz_dtheta4=-0.3*np.cos(theta34)
    tau_g=np.zeros_like(theta_path)
    tau_g[:,2]=MASS*G*dz_dtheta3
    tau_g[:,3]=MASS*G*dz_dtheta4
    return tau_g


def mechanical_energy(theta_final,n_points=FINAL_PATH_POINTS):
    """tau=I*qdd+tau_g，P=tau*qdot，E=int sum_i|P_i(t)|dt。"""
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
    return {"motion_time":float(T),"inertia_work":inertia_work,
            "gravity_work":gravity_work,"total_energy":total_energy,
            "positive_work":positive_work,"braking_work":braking_work}


def total_energy(x):
    # DE搜索阶段用41点加速；最终可行解仍用121点重新核算
    return mechanical_energy(x_to_theta(x),n_points=OPT_PATH_POINTS)["total_energy"]


# ============================================================
# 5. 主要关节点世界坐标
# ============================================================

def joint_positions(x,base_cell):
    theta1,theta2,theta3,theta4=x
    phi=theta1+theta2
    phi_rad=math.radians(phi)
    theta3_rad=math.radians(theta3)
    theta34_rad=math.radians(theta3+theta4)
    cp,sp=math.cos(phi_rad),math.sin(phi_rad)
    c3,s3=math.cos(theta3_rad),math.sin(theta3_rad)
    c34,s34=math.cos(theta34_rad),math.sin(theta34_rad)
    bx,by=grid_to_world(base_cell)
    J0=(bx,by,0.0)
    J1=(bx,by,600.0)
    J2=(bx+300.0*cp,by+300.0*sp,600.0)
    J3=(bx+300.0*(4.0*c3+1.0)*cp,by+300.0*(4.0*c3+1.0)*sp,600.0-1200.0*s3)
    B=1200.0*c3+300.0*c34+300.0
    J4=(bx-1200.0*sp+B*cp,by+B*sp+1200.0*cp,600.0-1200.0*s3-300.0*s34)
    return [J0,J1,J2,J3,J4]


# ============================================================
# 6. 地图边界与最终抓取构型碰撞
# ============================================================

def segment_rectangle_intersection(x1,y1,x2,y2,xmin,xmax,ymin,ymax):
    """Liang-Barsky，与障碍边界接触也视为碰撞。"""
    dx=x2-x1
    dy=y2-y1
    p=[-dx,dx,-dy,dy]
    q=[x1-xmin,xmax-x1,y1-ymin,ymax-y1]
    u1=0.0
    u2=1.0
    for pi,qi in zip(p,q):
        if abs(pi)<1e-12:
            if qi<0.0:
                return False
        else:
            t=qi/pi
            if pi<0.0:
                if t>u2:
                    return False
                u1=max(u1,t)
            else:
                if t<u1:
                    return False
                u2=min(u2,t)
    return True


def collision_violation(x,base_cell):
    """
    =0 表示J0~J4均在地图边界内且连杆XY投影不穿障碍格；>0 表示越界或碰撞。
    """
    joints=joint_positions(x,base_cell)
    violation=0.0
    # 地图边界
    for px,py,pz in joints:
        if px<MAP_X_MIN:
            violation+=(MAP_X_MIN-px)/CELL_SIZE
        elif px>MAP_X_MAX:
            violation+=(px-MAP_X_MAX)/CELL_SIZE
        if py<MAP_Y_MIN:
            violation+=(MAP_Y_MIN-py)/CELL_SIZE
        elif py>MAP_Y_MAX:
            violation+=(py-MAP_Y_MAX)/CELL_SIZE
    # 连杆-障碍
    for i in range(len(joints)-1):
        x1,y1,_=joints[i]
        x2,y2,_=joints[i+1]
        seg_xmin,seg_xmax=min(x1,x2),max(x1,x2)
        seg_ymin,seg_ymax=min(y1,y2),max(y1,y2)
        for xmin,xmax,ymin,ymax in OBSTACLE_RECTS:
            # 快速包围盒剔除
            if seg_xmax<xmin or seg_xmin>xmax or seg_ymax<ymin or seg_ymin>ymax:
                continue
            if segment_rectangle_intersection(x1,y1,x2,y2,xmin,xmax,ymin,ymax):
                violation+=1.0
    return float(violation)


def collision_free(x,base_cell):
    return collision_violation(x,base_cell)<=COLLISION_FEAS_TOL


# ============================================================
# 7. A*可达底座的必要工作空间预筛
# ============================================================

def generate_base_candidates():
    """水平工作半径约[1200,sqrt(1800^2+1200^2)]，允许200mm误差，向内外扩200mm。"""
    gx,gy=grid_to_world(END)
    rho_low=max(0.0,1200.0-MAX_ERROR)
    rho_high=math.sqrt(1800.0**2+1200.0**2)+MAX_ERROR
    candidates=[]
    for r in range(ROWS):
        for c in range(COLS):
            cell=(r,c)
            if GRID[r,c]!=0 or cell==END:
                continue
            path=astar_path(START,cell)
            if path is None:
                continue
            bx,by=grid_to_world(cell)
            rho=math.hypot(gx-bx,gy-by)
            if not (rho_low<=rho<=rho_high):
                continue
            candidates.append({
                "base_cell":cell,
                "base_world":(bx,by),
                "target":local_target(cell),
                "path":path,
                "steps_one_way":len(path)-1
            })
    return candidates


# ============================================================
# 8. 第一阶段：每个底座多seed求严格避障下最小误差
# ============================================================

def solve_min_error_one_seed(candidate,seed):
    base_cell=candidate["base_cell"]
    target=candidate["target"]
    def objective(x):
        violation=collision_violation(x,base_cell)
        if violation>COLLISION_FEAS_TOL:
            return COLLISION_PENALTY*(1.0+violation)
        return position_error(x,target)
    result=differential_evolution(
        objective,bounds=OPT_BOUNDS,strategy="best1bin",
        maxiter=FEASIBILITY_MAXITER,popsize=FEASIBILITY_POPSIZE,
        mutation=(0.5,1.0),recombination=0.8,tol=1e-6,atol=1e-6,
        polish=False,seed=seed,workers=1,updating="immediate",disp=False)
    x=np.array(result.x,dtype=float)
    return {
        "x":x,
        "error":float(position_error(x,target)),
        "violation":float(collision_violation(x,base_cell)),
        "seed":int(seed),
        "solver_success":bool(result.success),
        "iterations":int(result.nit),
        "evaluations":int(result.nfev)
    }


def solve_min_error_multiseed(candidate):
    trials=[solve_min_error_one_seed(candidate,seed) for seed in FEASIBILITY_SEEDS]
    valid=[item for item in trials if item["violation"]<=COLLISION_FEAS_TOL]
    if valid:
        best=min(valid,key=lambda item:item["error"])
    else:
        # 无严格可行结果时，保留违约最小者用于诊断
        best=min(trials,key=lambda item:(item["violation"],item["error"]))
    return {
        **candidate,
        "x_min_error":best["x"],
        "min_error":best["error"],
        "collision_violation":best["violation"],
        "best_seed":best["seed"],
        "within_200":bool(best["violation"]<=COLLISION_FEAS_TOL
                          and best["error"]<=MAX_ERROR+ERROR_FEAS_TOL)
    }


# ============================================================
# 9. 每个底座构造10mm粗扫描epsilon列表
# ============================================================

def build_coarse_epsilons(min_error):
    """例如min_error=162.573 -> 163,170,180,190,200，不再直接ceil到20mm档。"""
    start=float(math.ceil(min_error))
    if start>MAX_ERROR:
        return []
    values=[start]
    first_grid=math.ceil(start/COARSE_STEP)*COARSE_STEP
    v=first_grid
    while v<=MAX_ERROR+1e-9:
        values.append(float(v))
        v+=COARSE_STEP
    values.append(float(MAX_ERROR))
    values=sorted({round(v,6) for v in values if v<=MAX_ERROR+1e-9})
    return values


# ============================================================
# 10. 给定底座+epsilon，多seed求最小动力学能耗
# ============================================================

def solve_energy_one_seed(base_item,epsilon,x0,seed):
    base_cell=base_item["base_cell"]
    target=base_item["target"]
    def objective(x):
        violation=collision_violation(x,base_cell)
        if violation>COLLISION_FEAS_TOL:
            return COLLISION_PENALTY*(1.0+violation)
        error=position_error(x,target)
        if error>epsilon:
            return COLLISION_PENALTY+ERROR_PENALTY*(error-epsilon)
        # 只有真正进入可行域后才计算动力学能耗，节省大量时间
        return total_energy(x)
    result=differential_evolution(
        objective,bounds=OPT_BOUNDS,strategy="best1bin",
        maxiter=ENERGY_MAXITER,popsize=ENERGY_POPSIZE,
        mutation=(0.5,1.0),recombination=0.8,tol=1e-6,atol=1e-6,
        x0=np.array(x0,dtype=float),polish=False,seed=seed,workers=1,
        updating="immediate",disp=False)
    x=np.array(result.x,dtype=float)
    violation=collision_violation(x,base_cell)
    error=position_error(x,target)
    feasible=(violation<=COLLISION_FEAS_TOL and error<=epsilon+ERROR_FEAS_TOL)
    if not feasible:
        return None
    theta=x_to_theta(x)
    energy=mechanical_energy(theta)
    p_local=end_position(x)
    bx,by=base_item["base_world"]
    p_world=np.array([bx+p_local[0],by+p_local[1],p_local[2]],dtype=float)
    return {
        "base_cell":base_cell,
        "base_world":base_item["base_world"],
        "target_local":target,
        "path":base_item["path"],
        "steps_one_way":base_item["steps_one_way"],
        "epsilon":float(epsilon),
        "actual_error":float(error),
        "x":x,
        "theta":theta,
        "motion_time":energy["motion_time"],
        "inertia_work":energy["inertia_work"],
        "gravity_work":energy["gravity_work"],
        "positive_work":energy["positive_work"],
        "braking_work":energy["braking_work"],
        "total_energy":energy["total_energy"],
        "end_local":p_local,
        "end_world":p_world,
        "seed":int(seed),
        "solver_success":bool(result.success),
        "iterations":int(result.nit),
        "evaluations":int(result.nfev),
        "scan_type":"coarse"
    }


def solve_energy_multiseed(base_item,epsilon,x0_list,scan_type):
    """同一(base,epsilon)跑3个随机种子，取最低能耗可行解；x0_list不足3个时自动重复最后一个。"""
    if len(x0_list)==0:
        x0_list=[base_item["x_min_error"]]
    while len(x0_list)<len(ENERGY_SEEDS):
        x0_list.append(np.array(x0_list[-1],dtype=float))
    trials=[]
    for idx,seed in enumerate(ENERGY_SEEDS):
        result=solve_energy_one_seed(base_item,epsilon,x0=x0_list[idx],seed=seed)
        if result is not None:
            result["scan_type"]=scan_type
            trials.append(result)
    if not trials:
        return None
    return min(trials,key=lambda item:item["total_energy"])


# ============================================================
# 11. 缓存与粗扫描
# ============================================================

def solution_key(base_cell,epsilon):
    return (int(base_cell[0]),int(base_cell[1]),round(float(epsilon),6))


def find_previous_x0(cache,base_cell,epsilon,default_x):
    """找同一底座epsilon以下最近的已算解，作为warm-start。"""
    candidates=[]
    for key,item in cache.items():
        r,c,eps=key
        if (r,c)==base_cell and eps<epsilon-1e-9:
            candidates.append((eps,item["x"]))
    if not candidates:
        return np.array(default_x,dtype=float)
    candidates.sort(key=lambda pair:pair[0])
    return np.array(candidates[-1][1],dtype=float)


def coarse_energy_scan(feasible_bases):
    cache={}
    print("\n【阶段3】10 mm自适应粗扫描 + 单随机种子能耗优化（快速探索）")
    for b_idx,base_item in enumerate(feasible_bases,start=1):
        eps_list=build_coarse_epsilons(base_item["min_error"])
        print(f"\n底座({base_item['base_cell'][0]+1},{base_item['base_cell'][1]+1}) "
              f"min_error={base_item['min_error']:.3f} mm")
        for epsilon in eps_list:
            key=solution_key(base_item["base_cell"],epsilon)
            if key in cache:
                continue
            x_prev=find_previous_x0(cache,base_item["base_cell"],epsilon,base_item["x_min_error"])
            x0_list=[x_prev.copy(),x_prev.copy(),x_prev.copy()]
            solution=solve_energy_multiseed(base_item,epsilon,x0_list=x0_list,scan_type="coarse")
            if solution is None:
                print(f"  eps={epsilon:>6.1f} mm -> 未获得可行能耗解")
                continue
            cache[key]=solution
            print(f"  eps={epsilon:>6.1f} mm error={solution['actual_error']:>9.3f} "
                  f"E={solution['total_energy']:>10.6f} J seed={solution['seed']}")
    return cache


# ============================================================
# 12. 在给定允许误差下，寻找当前全局最低能耗方案
# ============================================================

def global_best_at_allowance(cache,allowance):
    feasible=[item for item in cache.values()
              if item["actual_error"]<=allowance+ERROR_FEAS_TOL]
    if not feasible:
        return None
    return min(feasible,key=lambda item:(item["total_energy"],item["actual_error"],item["steps_one_way"]))


def build_global_best_sequence(cache):
    """以粗扫描出现过的epsilon为横坐标，形成全局最优序列。"""
    allowances=sorted({key[2] for key in cache.keys()})
    sequence=[]
    for allowance in allowances:
        best=global_best_at_allowance(cache,allowance)
        if best is None:
            continue
        sequence.append({"allowance":float(allowance),"best_base":best["base_cell"],
                         "best_energy":best["total_energy"],"best_error":best["actual_error"]})
    return sequence


# ============================================================
# 13. 自动识别“底座切换/能耗突变”区间
# ============================================================

def detect_refine_intervals(sequence):
    """连续点底座变化或能耗相对下降>=10%时，加入1mm精扫区间。"""
    intervals=[]
    for i in range(len(sequence)-1):
        a,b=sequence[i],sequence[i+1]
        base_changed=a["best_base"]!=b["best_base"]
        energy_drop=0.0
        if a["best_energy"]>1e-12:
            energy_drop=(a["best_energy"]-b["best_energy"])/a["best_energy"]
        if base_changed or energy_drop>=ENERGY_DROP_TRIGGER:
            left=int(math.floor(a["allowance"]))
            right=int(math.ceil(b["allowance"]))
            if right>left:
                intervals.append((left,right))
    if not intervals:
        return []
    intervals.sort()
    merged=[list(intervals[0])]
    for left,right in intervals[1:]:
        if left<=merged[-1][1]+1:
            merged[-1][1]=max(merged[-1][1],right)
        else:
            merged.append([left,right])
    return [(left,right) for left,right in merged]


# ============================================================
# 14. 关键区间1 mm精扫
# ============================================================

def fine_scan_intervals(cache,feasible_bases,intervals):
    if not intervals:
        print("\n未检测到需要1 mm精扫的区间。")
        return cache
    print("\n【阶段4】关键区间1 mm精扫")
    base_map={item["base_cell"]:item for item in feasible_bases}
    for left,right in intervals:
        print(f"\n精扫区间：{left}~{right} mm")
        for epsilon in np.arange(max(0,left),min(MAX_ERROR,right)+0.1,FINE_STEP):
            epsilon=float(round(epsilon,6))
            for base_cell,base_item in base_map.items():
                if base_item["min_error"]>epsilon+ERROR_FEAS_TOL:
                    continue
                key=solution_key(base_cell,epsilon)
                if key in cache:
                    continue
                x_prev=find_previous_x0(cache,base_cell,epsilon,base_item["x_min_error"])
                solution=solve_energy_multiseed(base_item,epsilon,
                                                x0_list=[x_prev.copy(),x_prev.copy(),x_prev.copy()],
                                                scan_type="fine")
                if solution is None:
                    continue
                cache[key]=solution
                print(f"  base=({base_cell[0]+1},{base_cell[1]+1}) eps={epsilon:>6.1f} "
                      f"error={solution['actual_error']:>9.3f} E={solution['total_energy']:>10.6f}")
    return cache


# ============================================================
# 15. 工程容差Pareto前沿
# ============================================================

def pareto_front_tolerant(solutions):
    """误差0.5mm内视为同一精度等级，能耗1e-3J内视为同一能耗等级。"""
    items=list(solutions)
    front=[]
    for i,item in enumerate(items):
        dominated=False
        for j,other in enumerate(items):
            if i==j:
                continue
            energy_better=other["total_energy"]<item["total_energy"]-PARETO_ENERGY_TOL
            error_not_worse=other["actual_error"]<=item["actual_error"]+PARETO_ERROR_TOL
            error_better=other["actual_error"]<item["actual_error"]-PARETO_ERROR_TOL
            energy_not_worse=other["total_energy"]<=item["total_energy"]+PARETO_ENERGY_TOL
            if (energy_better and error_not_worse) or (error_better and energy_not_worse):
                dominated=True
                break
        if not dominated:
            front.append(item)
    front=sorted(front,key=lambda item:(item["actual_error"],item["total_energy"]))
    cleaned=[]
    for item in front:
        if not cleaned:
            cleaned.append(item)
            continue
        last=cleaned[-1]
        if abs(item["actual_error"]-last["actual_error"])<=PARETO_ERROR_TOL:
            if item["total_energy"]<last["total_energy"]:
                cleaned[-1]=item
        else:
            cleaned.append(item)
    return cleaned


# ============================================================
# 16. 保存结果
# ============================================================

def feasibility_dataframe(items):
    rows=[]
    for item in items:
        rows.append({
            "base_row":item["base_cell"][0]+1,
            "base_col":item["base_cell"][1]+1,
            "base_x_mm":item["base_world"][0],
            "base_y_mm":item["base_world"][1],
            "min_error_mm":item["min_error"],
            "collision_violation":item["collision_violation"],
            "best_seed":item["best_seed"],
            "within_200":item["within_200"],
            "one_way_steps":item["steps_one_way"],
            "round_trip_steps":2*item["steps_one_way"],
            "one_way_distance_mm":item["steps_one_way"]*CELL_SIZE,
            "round_trip_distance_mm":2*item["steps_one_way"]*CELL_SIZE
        })
    return pd.DataFrame(rows)


def solutions_dataframe(items):
    rows=[]
    for item in items:
        theta=item["theta"]
        x=item["x"]
        rows.append({
            "base_row":item["base_cell"][0]+1,
            "base_col":item["base_cell"][1]+1,
            "scan_type":item["scan_type"],
            "epsilon_mm":item["epsilon"],
            "actual_error_mm":item["actual_error"],
            "phi_deg":theta[0]+theta[1],
            "theta1_deg":theta[0],
            "theta2_deg":theta[1],
            "theta3_deg":theta[2],
            "theta4_deg":theta[3],
            "theta5_deg":theta[4],
            "theta6_deg":theta[5],
            "motion_time_s":item["motion_time"],
            "inertia_abs_work_J":item["inertia_work"],
            "gravity_abs_work_J":item["gravity_work"],
            "positive_actuator_work_J":item["positive_work"],
            "braking_work_J":item["braking_work"],
            "total_energy_J":item["total_energy"],
            "end_world_x_mm":item["end_world"][0],
            "end_world_y_mm":item["end_world"][1],
            "end_world_z_mm":item["end_world"][2],
            "seed":item["seed"],
            "solver_success":item["solver_success"],
            "iterations":item["iterations"],
            "evaluations":item["evaluations"],
            "one_way_steps":item["steps_one_way"],
            "round_trip_steps":2*item["steps_one_way"],
            "round_trip_distance_mm":2*item["steps_one_way"]*CELL_SIZE
        })
    return pd.DataFrame(rows)


def switch_dataframe(sequence):
    rows=[]
    previous_base=None
    for item in sequence:
        if previous_base is None or item["best_base"]!=previous_base:
            rows.append({
                "allowance_mm":item["allowance"],
                "best_base_row":item["best_base"][0]+1,
                "best_base_col":item["best_base"][1]+1,
                "best_actual_error_mm":item["best_error"],
                "best_energy_J":item["best_energy"]
            })
            previous_base=item["best_base"]
    return pd.DataFrame(rows)


# ============================================================
# 17. 绘图底图
# ============================================================

def draw_grid_background(ax):
    for r,c in OBSTACLES:
        ax.add_patch(Rectangle((c-0.5,r-0.5),1,1))
    for k in range(COLS+1):
        ax.plot([k-0.5,k-0.5],[-0.5,ROWS-0.5],linewidth=0.4)
    for k in range(ROWS+1):
        ax.plot([-0.5,COLS-0.5],[k-0.5,k-0.5],linewidth=0.4)
    ax.set_xlim(-0.5,COLS-0.5)
    ax.set_ylim(ROWS-0.5,-0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.set_xticklabels(range(1,COLS+1))
    ax.set_yticklabels(range(1,ROWS+1))
    ax.set_xlabel("列")
    ax.set_ylabel("行")


# ============================================================
# 18. 图1：底座最小可行误差分布
# ============================================================

def plot_min_error_map(feasibility_items,save_path):
    fig,ax=plt.subplots(figsize=(10,9))
    draw_grid_background(ax)
    feasible=[item for item in feasibility_items if item["within_200"]]
    if feasible:
        xs=[item["base_cell"][1] for item in feasible]
        ys=[item["base_cell"][0] for item in feasible]
        vals=[item["min_error"] for item in feasible]
        sc=ax.scatter(xs,ys,c=vals,s=90,marker="o")
        fig.colorbar(sc,ax=ax,label="严格避障下最小末端误差 / mm")
        for item in feasible:
            r,c=item["base_cell"]
            ax.text(c+0.12,r-0.12,f"{item['min_error']:.1f}",fontsize=8)
    ax.scatter([START[1]],[START[0]],marker="*",s=180,label="Start")
    ax.scatter([END[1]],[END[0]],marker="X",s=150,label="货物")
    ax.set_title("第三问：严格避障下可行底座及最小误差")
    ax.legend(loc="upper left",bbox_to_anchor=(1.02,1.0))
    fig.tight_layout()
    fig.savefig(save_path,dpi=300,bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 19. 图2：最终全局Pareto
# ============================================================

def plot_pareto(all_solutions,front,save_path):
    fig,ax=plt.subplots(figsize=(9,6))
    if all_solutions:
        ax.scatter([item["actual_error"] for item in all_solutions],
                   [item["total_energy"] for item in all_solutions],
                   alpha=0.45,label="全部可行方案")
    if front:
        ax.plot([item["actual_error"] for item in front],
                [item["total_energy"] for item in front],
                marker="o",linewidth=2,label="工程容差Pareto前沿")
        for item in front:
            ax.annotate(f"({item['base_cell'][0]+1},{item['base_cell'][1]+1})",
                        (item["actual_error"],item["total_energy"]),
                        xytext=(5,5),textcoords="offset points",fontsize=8)
    ax.set_xlabel("末端位置误差 / mm")
    ax.set_ylabel("等效机械能耗 / J")
    ax.set_title("第三问：精细扫描误差—能耗Pareto前沿")
    ax.grid(True,alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path,dpi=300,bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 20. 图3：最优底座随允许误差变化
# ============================================================

def plot_best_base_switch(sequence,save_path):
    if not sequence:
        return
    allowances=[item["allowance"] for item in sequence]
    bases=[]
    for item in sequence:
        if item["best_base"] not in bases:
            bases.append(item["best_base"])
    base_to_index={base:i for i,base in enumerate(bases)}
    values=[base_to_index[item["best_base"]] for item in sequence]
    fig,ax=plt.subplots(figsize=(9,5))
    ax.step(allowances,values,where="post",linewidth=2)
    ax.scatter(allowances,values,s=25)
    ax.set_yticks(range(len(bases)))
    ax.set_yticklabels([f"({r+1},{c+1})" for r,c in bases])
    ax.set_xlabel("允许末端误差 / mm")
    ax.set_ylabel("全局最低能耗底座")
    ax.set_title("第三问：最优底座随误差容限变化")
    ax.grid(True,alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path,dpi=300,bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 21. 图4：最低能耗代表方案A*路径
# ============================================================

def plot_base_path(solution,save_path):
    fig,ax=plt.subplots(figsize=(9,9))
    draw_grid_background(ax)
    path=solution["path"]
    xs=[c for r,c in path]
    ys=[r for r,c in path]
    ax.plot(xs,ys,marker="o",linewidth=2,label="去程路径")
    ax.plot(list(reversed(xs)),list(reversed(ys)),linestyle="--",linewidth=2,label="返回路径")
    ax.scatter([START[1]],[START[0]],marker="*",s=180,label="Start")
    ax.scatter([END[1]],[END[0]],marker="X",s=150,label="货物")
    br,bc=solution["base_cell"]
    ax.scatter([bc],[br],marker="s",s=130,label="选定底座点")
    ax.set_title("第三问：最低能耗代表方案底座路径")
    ax.legend(loc="upper left",bbox_to_anchor=(1.02,1.0))
    fig.tight_layout()
    fig.savefig(save_path,dpi=300,bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 22. 图5：最低能耗代表抓取构型
# ============================================================

def plot_grasp_pose(solution,save_path):
    fig,ax=plt.subplots(figsize=(9,9))
    draw_grid_background(ax)
    joints=joint_positions(solution["x"],solution["base_cell"])
    cols=[p[0]/CELL_SIZE for p in joints]
    rows=[p[1]/CELL_SIZE for p in joints]
    ax.plot(cols,rows,marker="o",linewidth=2,label="机械臂主要连杆")
    for i,(cx,cy) in enumerate(zip(cols,rows)):
        text="J4/J5/J6" if i==4 else f"J{i}"
        ax.text(cx+0.08,cy-0.08,text,fontsize=9)
    ax.scatter([END[1]],[END[0]],marker="X",s=150,label="货物目标位置")
    p_world=solution["end_world"]
    ax.scatter([p_world[0]/CELL_SIZE],[p_world[1]/CELL_SIZE],marker="P",s=120,label="实际末端位置")
    br,bc=solution["base_cell"]
    ax.scatter([bc],[br],marker="s",s=130,label="底座位置")
    ax.set_title("第三问：最低能耗代表抓取构型")
    ax.legend(loc="upper left",bbox_to_anchor=(1.02,1.0))
    fig.tight_layout()
    fig.savefig(save_path,dpi=300,bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 23. 主程序（explore阶段）
# ============================================================

def run_explore():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    print("\n第三问：A* + 4D-DE + 第二问动力学能耗 + 快速探索 + 局部1mm精扫")

    # 阶段1：底座预筛
    candidates=generate_base_candidates()
    print(f"\n【阶段1】A*+工作空间预筛候选底座数量：{len(candidates)}")

    # 阶段2：每底座多seed最小误差
    feasibility=[]
    print("\n【阶段2】3随机种子搜索严格避障下最小末端误差")
    for i,candidate in enumerate(candidates,start=1):
        result=solve_min_error_multiseed(candidate)
        feasibility.append(result)
        r,c=result["base_cell"]
        print(f"[{i:>2}/{len(candidates)}] 底座=({r+1},{c+1}) "
              f"min_error={result['min_error']:.3f} mm seed={result['best_seed']} "
              f"200mm内可行={result['within_200']}")

    feasible_bases=[item for item in feasibility if item["within_200"]]
    print(f"\n严格避障且最小误差<=200mm的底座格：{len(feasible_bases)}")
    if not feasible_bases:
        print("没有可行底座，程序结束。")
        feasibility_dataframe(feasibility).to_excel(
            OUTPUT_DIR/"第三问_底座最小可行误差.xlsx",index=False)
        return

    # 阶段3：10mm粗扫描
    cache=coarse_energy_scan(feasible_bases)
    coarse_sequence=build_global_best_sequence(cache)
    intervals=detect_refine_intervals(coarse_sequence)
    print(f"\n自动检测到的1 mm精扫区间：{intervals}")

    # 阶段4：1mm精扫
    cache=fine_scan_intervals(cache,feasible_bases,intervals)

    # 阶段5：最终结果整理
    all_solutions=list(cache.values())
    front=pareto_front_tolerant(all_solutions)
    final_sequence=build_global_best_sequence(cache)
    representative=min(all_solutions,key=lambda item:(item["total_energy"],item["actual_error"],item["steps_one_way"]))

    print("\n"+"="*88)
    print("                         第三问最终结果")
    print("="*88)
    print(f"全部可行误差-能耗方案数：{len(all_solutions)}")
    print(f"工程容差Pareto点数：{len(front)}")
    print("\n【最低能耗代表方案】")
    print(f"底座格（1-based） = ({representative['base_cell'][0]+1},{representative['base_cell'][1]+1})")
    print(f"允许误差epsilon = {representative['epsilon']:.3f} mm")
    print(f"实际误差 = {representative['actual_error']:.6f} mm")
    print(f"动作时间 = {representative['motion_time']:.6f} s")
    print(f"惯性绝对功 = {representative['inertia_work']:.6f} J")
    print(f"重力绝对功 = {representative['gravity_work']:.6f} J")
    print(f"等效总机械能耗 = {representative['total_energy']:.6f} J")
    print(f"关节角 / ° = {representative['theta']}")
    print(f"底座往返距离 = {2*representative['steps_one_way']*CELL_SIZE:.0f} mm")
    print("="*88)

    # 保存xlsx
    feasibility_path=OUTPUT_DIR/"第三问_底座最小可行误差.xlsx"
    all_path=OUTPUT_DIR/"第三问_全部误差能耗方案.xlsx"
    pareto_path=OUTPUT_DIR/"第三问_全局Pareto前沿.xlsx"
    switch_path=OUTPUT_DIR/"第三问_最优底座切换点.xlsx"
    feasibility_dataframe(feasibility).to_excel(feasibility_path,index=False)
    solutions_dataframe(all_solutions).to_excel(all_path,index=False)
    solutions_dataframe(front).to_excel(pareto_path,index=False)
    switch_dataframe(final_sequence).to_excel(switch_path,index=False)

    # 绘图
    fig1=OUTPUT_DIR/"第三问_底座最小误差分布.png"
    fig2=OUTPUT_DIR/"第三问_全局Pareto前沿.png"
    fig3=OUTPUT_DIR/"第三问_最优底座切换.png"
    fig4=OUTPUT_DIR/"第三问_最低能耗底座路径.png"
    fig5=OUTPUT_DIR/"第三问_最低能耗抓取构型.png"
    plot_min_error_map(feasibility,fig1)
    plot_pareto(all_solutions,front,fig2)
    plot_best_base_switch(final_sequence,fig3)
    plot_base_path(representative,fig4)
    plot_grasp_pose(representative,fig5)

    print("\n结果已保存到 output 文件夹：")
    for p in [feasibility_path,all_path,pareto_path,switch_path,fig1,fig2,fig3,fig4,fig5]:
        print(f"  {p.name}")


# ============================================================
# 最终高精度核验模块参数（problem3.2 / problem3.3）
# ============================================================

COLLISION_TOL=COLLISION_FEAS_TOL
ERROR_TOL=ERROR_FEAS_TOL

VERIFY_BASES_1BASED=[(12,16),(13,15),(13,16),(15,10),(15,11),(15,12),(16,10)]
COMPETITIVE_BASES_1BASED=[(13,16),(15,12),(15,10),(15,11)]
SWITCH_INTERVALS=[
    ((13,16),(15,12),75,90),
    ((15,12),(15,10),108,120),
    ((15,10),(15,11),124,140)
]

MIN_ERROR_SEEDS=[2026,2027,2028,1107]
MIN_ERROR_MAXITER=220
MIN_ERROR_POPSIZE=14

SWITCH_SCAN_SEED=2026
SWITCH_SCAN_MAXITER=75
SWITCH_SCAN_POPSIZE=8

VERIFY_ENERGY_SEEDS=[2026,2027,2028]
VERIFY_ENERGY_MAXITER=150
VERIFY_ENERGY_POPSIZE=10

ENDPOINT_SEEDS=[2026,2027,2028]
ENDPOINT_MAXITER=160
ENDPOINT_POPSIZE=10

PREVIOUS_ALL_PATH=OUTPUT_DIR/"第三问_全部误差能耗方案.xlsx"


# ============================================================
# 最终高精度核验模块
# ============================================================

def to_zero_based(cell_1based):
    return (cell_1based[0]-1,cell_1based[1]-1)


def make_base_item(cell_1based):
    base_cell=to_zero_based(cell_1based)
    path=astar_path(START,base_cell)
    if path is None:
        raise RuntimeError(f"底座{cell_1based}无法由Start通过A*到达")
    return {
        "base_cell":base_cell,
        "base_1based":cell_1based,
        "base_world":grid_to_world(base_cell),
        "target":local_target(base_cell),
        "path":path,
        "steps_one_way":len(path)-1
    }


def load_previous_solutions():
    if not PREVIOUS_ALL_PATH.exists():
        print(f"\n未找到上一版结果文件：{PREVIOUS_ALL_PATH}")
        print("程序仍可运行，但warm-start效果会变差。")
        return pd.DataFrame()
    df=pd.read_excel(PREVIOUS_ALL_PATH)
    print(f"\n已读取上一版warm-start数据：{len(df)}条")
    return df


def previous_x_for_base(previous_df,base_1based,prefer_min_error=True,epsilon=None):
    if previous_df.empty:
        return None
    r,c=base_1based
    sub=previous_df[(previous_df["base_row"]==r)&(previous_df["base_col"]==c)].copy()
    if sub.empty:
        return None
    if epsilon is not None:
        sub["eps_distance"]=np.abs(sub["epsilon_mm"]-epsilon)
        row=sub.sort_values(["eps_distance","total_energy_J"]).iloc[0]
    elif prefer_min_error:
        row=sub.sort_values("actual_error_mm").iloc[0]
    else:
        row=sub.sort_values("total_energy_J").iloc[0]
    return np.array([row["theta1_deg"],row["theta2_deg"],row["theta3_deg"],row["theta4_deg"]],dtype=float)


def solve_min_error_seed(base_item,seed,x0):
    base_cell=base_item["base_cell"]
    target=base_item["target"]
    def objective(x):
        violation=collision_violation(x,base_cell)
        if violation>COLLISION_TOL:
            return COLLISION_PENALTY*(1.0+violation)
        return position_error(x,target)
    kwargs={}
    if x0 is not None:
        kwargs["x0"]=np.array(x0,dtype=float)
    result=differential_evolution(
        objective,bounds=OPT_BOUNDS,strategy="best1bin",
        maxiter=MIN_ERROR_MAXITER,popsize=MIN_ERROR_POPSIZE,
        mutation=(0.5,1.0),recombination=0.8,tol=1e-8,atol=1e-8,
        polish=False,seed=seed,workers=1,updating="immediate",disp=False,**kwargs)
    x=np.array(result.x,dtype=float)
    return {
        "x":x,
        "error":position_error(x,target),
        "violation":collision_violation(x,base_cell),
        "seed":seed,
        "nit":result.nit,
        "nfev":result.nfev
    }


def verify_min_error(base_item,previous_df):
    warm=previous_x_for_base(previous_df,base_item["base_1based"],prefer_min_error=True)
    trials=[solve_min_error_seed(base_item,seed,warm) for seed in MIN_ERROR_SEEDS]
    feasible=[item for item in trials if item["violation"]<=COLLISION_TOL]
    if not feasible:
        best=min(trials,key=lambda item:(item["violation"],item["error"]))
    else:
        best=min(feasible,key=lambda item:item["error"])
    return best


def solve_energy_seed(base_item,epsilon,seed,x0,maxiter,popsize):
    base_cell=base_item["base_cell"]
    target=base_item["target"]
    def objective(x):
        violation=collision_violation(x,base_cell)
        if violation>COLLISION_TOL:
            return COLLISION_PENALTY*(1.0+violation)
        error=position_error(x,target)
        if error>epsilon:
            return COLLISION_PENALTY+ERROR_PENALTY*(error-epsilon)
        return mechanical_energy(x_to_theta(x),n_points=FINAL_PATH_POINTS)["total_energy"]
    kwargs={}
    if x0 is not None:
        kwargs["x0"]=np.array(x0,dtype=float)
    result=differential_evolution(
        objective,bounds=OPT_BOUNDS,strategy="best1bin",
        maxiter=maxiter,popsize=popsize,
        mutation=(0.5,1.0),recombination=0.8,tol=1e-8,atol=1e-8,
        polish=False,seed=seed,workers=1,updating="immediate",disp=False,**kwargs)
    x=np.array(result.x,dtype=float)
    violation=collision_violation(x,base_cell)
    error=position_error(x,target)
    if violation>COLLISION_TOL or error>epsilon+ERROR_TOL:
        return None
    theta=x_to_theta(x)
    energy=mechanical_energy(theta)
    p_local=end_position(x)
    bx,by=base_item["base_world"]
    p_world=np.array([bx+p_local[0],by+p_local[1],p_local[2]])
    return {
        "base_1based":base_item["base_1based"],
        "base_cell":base_cell,
        "epsilon":float(epsilon),
        "actual_error":float(error),
        "x":x,
        "theta":theta,
        "motion_time":energy["motion_time"],
        "inertia_work":energy["inertia_work"],
        "gravity_work":energy["gravity_work"],
        "positive_work":energy["positive_work"],
        "braking_work":energy["braking_work"],
        "total_energy":energy["total_energy"],
        "end_world":p_world,
        "seed":seed,
        "nit":result.nit,
        "nfev":result.nfev,
        "steps_one_way":base_item["steps_one_way"],
        "path":base_item["path"]
    }


def verify_solve_energy_multiseed(base_item,epsilon,seeds,maxiter,popsize,previous_df,x0_override=None):
    if x0_override is not None:
        warm=np.array(x0_override,dtype=float)
    else:
        warm=previous_x_for_base(previous_df,base_item["base_1based"],prefer_min_error=False,epsilon=epsilon)
    trials=[]
    for seed in seeds:
        item=solve_energy_seed(base_item,epsilon,seed,warm,maxiter,popsize)
        if item is not None:
            trials.append(item)
    if not trials:
        return None
    return min(trials,key=lambda item:item["total_energy"])


def scan_pair_interval(base_a,base_b,left,right,verified_min_error,previous_df):
    rows=[]
    warm_a=previous_x_for_base(previous_df,base_a["base_1based"],epsilon=left)
    warm_b=previous_x_for_base(previous_df,base_b["base_1based"],epsilon=left)
    for epsilon in range(left,right+1):
        result_a=None
        result_b=None
        min_a=verified_min_error[base_a["base_1based"]]["error"]
        min_b=verified_min_error[base_b["base_1based"]]["error"]
        if min_a<=epsilon+ERROR_TOL:
            result_a=solve_energy_seed(base_a,epsilon,SWITCH_SCAN_SEED,warm_a,
                                       SWITCH_SCAN_MAXITER,SWITCH_SCAN_POPSIZE)
            if result_a is not None:
                warm_a=result_a["x"]
        if min_b<=epsilon+ERROR_TOL:
            result_b=solve_energy_seed(base_b,epsilon,SWITCH_SCAN_SEED,warm_b,
                                       SWITCH_SCAN_MAXITER,SWITCH_SCAN_POPSIZE)
            if result_b is not None:
                warm_b=result_b["x"]
        energy_a=np.nan if result_a is None else result_a["total_energy"]
        energy_b=np.nan if result_b is None else result_b["total_energy"]
        if result_a is None and result_b is None:
            winner=None
        elif result_b is None:
            winner=base_a["base_1based"]
        elif result_a is None:
            winner=base_b["base_1based"]
        elif energy_a<=energy_b:
            winner=base_a["base_1based"]
        else:
            winner=base_b["base_1based"]
        rows.append({
            "epsilon_mm":epsilon,
            "baseA":str(base_a["base_1based"]),
            "energy_A_J":energy_a,
            "error_A_mm":np.nan if result_a is None else result_a["actual_error"],
            "baseB":str(base_b["base_1based"]),
            "energy_B_J":energy_b,
            "error_B_mm":np.nan if result_b is None else result_b["actual_error"],
            "winner":str(winner)
        })
    return pd.DataFrame(rows)


def detect_first_switch(scan_df,base_b_1based):
    target=str(base_b_1based)
    for _,row in scan_df.iterrows():
        if row["winner"]==target:
            return int(row["epsilon_mm"])
    return None


def verify_switch_neighborhood(base_a,base_b,switch_epsilon,previous_df):
    if switch_epsilon is None:
        return []
    results=[]
    for epsilon in [switch_epsilon-1,switch_epsilon,switch_epsilon+1]:
        if epsilon<0 or epsilon>MAX_ERROR:
            continue
        for base_item in [base_a,base_b]:
            result=verify_solve_energy_multiseed(base_item,epsilon,VERIFY_ENERGY_SEEDS,
                                                 VERIFY_ENERGY_MAXITER,VERIFY_ENERGY_POPSIZE,previous_df)
            if result is not None:
                results.append(result)
    return results


def verify_200_endpoint(base_items,verified_min_error,previous_df):
    results=[]
    print("\n【阶段4】epsilon=200 mm：7个底座统一高精度能耗复核")
    for base_item in base_items:
        base=base_item["base_1based"]
        if verified_min_error[base]["error"]>MAX_ERROR+ERROR_TOL:
            continue
        result=verify_solve_energy_multiseed(base_item,200.0,ENDPOINT_SEEDS,
                                             ENDPOINT_MAXITER,ENDPOINT_POPSIZE,previous_df)
        if result is None:
            print(f"底座{base}: 未得到200mm可行能耗解")
            continue
        results.append(result)
        print(f"底座{base}: error={result['actual_error']:.3f} mm, "
              f"E={result['total_energy']:.6f} J, seed={result['seed']}")
    return results


def energy_results_dataframe(items):
    rows=[]
    for item in items:
        theta=item["theta"]
        rows.append({
            "base_row":item["base_1based"][0],
            "base_col":item["base_1based"][1],
            "epsilon_mm":item["epsilon"],
            "actual_error_mm":item["actual_error"],
            "theta1_deg":theta[0],
            "theta2_deg":theta[1],
            "theta3_deg":theta[2],
            "theta4_deg":theta[3],
            "theta5_deg":theta[4],
            "theta6_deg":theta[5],
            "motion_time_s":item["motion_time"],
            "inertia_abs_work_J":item["inertia_work"],
            "gravity_abs_work_J":item["gravity_work"],
            "positive_work_J":item["positive_work"],
            "braking_work_J":item["braking_work"],
            "total_energy_J":item["total_energy"],
            "end_world_x_mm":item["end_world"][0],
            "end_world_y_mm":item["end_world"][1],
            "end_world_z_mm":item["end_world"][2],
            "seed":item["seed"],
            "iterations":item["nit"],
            "evaluations":item["nfev"],
            "one_way_steps":item["steps_one_way"],
            "round_trip_distance_mm":2*item["steps_one_way"]*CELL_SIZE
        })
    return pd.DataFrame(rows)




def run_verify():
    print("\n"+"="*92)
    print("第三问核验版：7底座高精度最小误差 + 3组切换区间逐毫米核验 + 200mm端点复核")
    print("="*92)

    previous_df=load_previous_solutions()
    base_items={cell:make_base_item(cell) for cell in VERIFY_BASES_1BASED}

    # 阶段1：7个底座最小误差复核
    print("\n【阶段1】7个底座高精度最小误差复核")
    verified_min_error={}
    min_rows=[]
    for cell in VERIFY_BASES_1BASED:
        base_item=base_items[cell]
        best=verify_min_error(base_item,previous_df)
        verified_min_error[cell]=best
        min_rows.append({
            "base_row":cell[0],
            "base_col":cell[1],
            "verified_min_error_mm":best["error"],
            "collision_violation":best["violation"],
            "best_seed":best["seed"],
            "within_200":bool(best["violation"]<=COLLISION_TOL and best["error"]<=MAX_ERROR+ERROR_TOL),
            "one_way_steps":base_item["steps_one_way"],
            "round_trip_distance_mm":2*base_item["steps_one_way"]*CELL_SIZE
        })
        print(f"底座{cell}: verified_min_error={best['error']:.6f} mm, "
              f"seed={best['seed']}, violation={best['violation']:.3e}")
    min_df=pd.DataFrame(min_rows)
    min_path=OUTPUT_DIR/"第三问_底座最小误差.xlsx"
    min_df.to_excel(min_path,index=False)

    # 阶段2：3个切换区间逐毫米单seed扫描
    print("\n【阶段2】3个候选切换区间逐毫米扫描")
    pair_scan_frames=[]
    detected_switches=[]
    for pair_index,(cell_a,cell_b,left,right) in enumerate(SWITCH_INTERVALS,start=1):
        print(f"\n切换区间{pair_index}: {cell_a} -> {cell_b}, {left}~{right} mm")
        df=scan_pair_interval(base_items[cell_a],base_items[cell_b],left,right,
                              verified_min_error,previous_df)
        df.insert(0,"pair_index",pair_index)
        pair_scan_frames.append(df)
        switch_epsilon=detect_first_switch(df,cell_b)
        detected_switches.append({"pair_index":pair_index,"base_from":str(cell_a),
                                 "base_to":str(cell_b),
                                 "first_integer_switch_epsilon_mm":switch_epsilon})
        print(f"检测到的整数mm切换点：{switch_epsilon}")
    pair_scan_df=pd.concat(pair_scan_frames,ignore_index=True)
    pair_scan_path=OUTPUT_DIR/"第三问_切换区间逐毫米.xlsx"
    pair_scan_df.to_excel(pair_scan_path,index=False)
    switch_summary_df=pd.DataFrame(detected_switches)
    switch_summary_path=OUTPUT_DIR/"第三问_底座切换点.xlsx"
    switch_summary_df.to_excel(switch_summary_path,index=False)

    # 阶段3：切换点附近3seed正式复核
    print("\n【阶段3】切换点±1 mm，3随机种子正式复核")
    switch_verified_results=[]
    for info,interval in zip(detected_switches,SWITCH_INTERVALS):
        cell_a,cell_b,_,_=interval
        switch_epsilon=info["first_integer_switch_epsilon_mm"]
        local_results=verify_switch_neighborhood(base_items[cell_a],base_items[cell_b],
                                                 switch_epsilon,previous_df)
        switch_verified_results.extend(local_results)
        print(f"{cell_a}->{cell_b}, 切换点附近完成 {len(local_results)} 个正式核验解")
    switch_verified_df=energy_results_dataframe(switch_verified_results)
    switch_verified_path=OUTPUT_DIR/"第三问_切换点高精度方案.xlsx"
    switch_verified_df.to_excel(switch_verified_path,index=False)

    # 阶段4：epsilon=200，7底座统一复核
    endpoint_results=verify_200_endpoint(list(base_items.values()),verified_min_error,previous_df)
    endpoint_df=energy_results_dataframe(endpoint_results)
    endpoint_path=OUTPUT_DIR/"第三问_200mm七底座能耗.xlsx"
    endpoint_df.to_excel(endpoint_path,index=False)
    if not endpoint_results:
        print("200mm端点没有得到可行解，程序结束。")
        return
    final_representative=min(endpoint_results,key=lambda item:(item["total_energy"],item["actual_error"],item["steps_one_way"]))

    # 输出代表方案
    print("\n"+"="*92)
    print("第三问：epsilon=200 mm最低能耗方案")
    print("="*92)
    print(f"底座 = {final_representative['base_1based']}")
    print(f"实际误差 = {final_representative['actual_error']:.6f} mm")
    print(f"总机械能耗 = {final_representative['total_energy']:.6f} J")
    print(f"动作时间 = {final_representative['motion_time']:.6f} s")
    print(f"惯性绝对功 = {final_representative['inertia_work']:.6f} J")
    print(f"重力绝对功 = {final_representative['gravity_work']:.6f} J")
    print(f"关节角 / ° = {final_representative['theta']}")
    print(f"往返距离 = {2*final_representative['steps_one_way']*CELL_SIZE:.0f} mm")
    print("="*92)

    # 图：最低能耗底座路径/抓取构型已在explore阶段生成，此处不重复输出
    print("\n结果文件：")
    for p in [min_path,pair_scan_path,switch_summary_path,switch_verified_path,endpoint_path]:
        print(f"  {p.name}")


# ============================================================
# 统一入口
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    mode=RUN_MODE.lower().strip()
    print("="*88)
    print("A题问题3 合并最终版")
    print(f"RUN_MODE = {RUN_MODE}")
    print("="*88)
    if mode=="explore":
        run_explore()
    elif mode=="verify":
        run_verify()
    elif mode=="all":
        run_explore()
        run_verify()
    else:
        raise ValueError(f"未知RUN_MODE={RUN_MODE!r}，只能是 'explore'、'verify' 或 'all'")


if __name__=="__main__":
    main()
