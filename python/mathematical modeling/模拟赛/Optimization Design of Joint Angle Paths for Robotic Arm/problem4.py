# ============================================================
# A题问题4 整体整合最终版
# 多货物：A* + SDH + 4D-DE + 动力学能耗 + 组合枚举 + Pareto
#
# 继承问题3：
# 1. 标准D-H末端位置解析式
# 2. 三次时间标度动力学能耗
# 3. 主要连杆-障碍碰撞 + 关节地图边界约束
# 4. A* 四邻域底座路径
#
# 问题4新增：
# 1. Sheet2五个货物
# 2. 每个货物独立筛选候选抓取底座
# 3. 每个候选底座生成误差—能耗候选抓取状态
# 4. 枚举五个货物的抓取状态组合
# 5. 每组状态精确枚举120种访问顺序，找最短底座路径
# 6. 全局Pareto + 归一化理想点折中代表方案
#
# 建模口径：
# - 能耗仅计每次抓取动作的机械臂能耗，不计底座移动能耗
# - 每个货物末端误差<=200 mm
# - RETURN_TO_START 控制是否返回Start（默认True，与问题3一致）
# ============================================================

RUN_MODE="all"
FAST_MODE=True
RETURN_TO_START=True

import math
import heapq
import itertools
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import differential_evolution

BASE_DIR=Path(__file__).resolve().parent
OUTPUT_DIR=BASE_DIR/"output"

plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]
plt.rcParams["axes.unicode_minus"]=False


# ============================================================
# 1. Sheet2栅格（0=可行，1=障碍；Start和target1~5转换为0）
# ============================================================

GRID=np.array([
    [0,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,0,1,0],
    [0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,0,0,1,1],
    [0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0],
    [1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,1,1,0,1],
    [1,0,1,1,0,1,0,1,1,0,0,1,1,0,0,0,1,1,1,1],
    [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0],
    [0,0,0,1,0,0,0,0,0,1,0,1,0,1,0,1,1,1,1,1],
    [0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0],
    [0,1,1,0,0,0,0,0,0,0,1,1,0,0,1,0,0,1,0,0],
    [0,0,0,1,1,0,1,0,0,0,1,0,1,1,0,0,0,0,0,0],
    [0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,1,1,0,1],
    [0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1],
    [0,1,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,1,0,0],
    [0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,1,0,0,0,0],
    [0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,0,1,1,1,0,1,0,0,0,1,0,0,0,1,1,0,0],
    [0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,1,0,0,0]
],dtype=int)

ROWS,COLS=GRID.shape
CELL_SIZE=200.0
START=(0,0)

TARGETS={
    "target1":(0,14),
    "target2":(7,10),
    "target3":(15,8),
    "target4":(17,1),
    "target5":(19,19)
}
TARGET_NAMES=list(TARGETS.keys())
TARGET_Z=200.0
TARGET_CELLS=set(TARGETS.values())
MAX_ERROR=200.0


# ============================================================
# 2. 机械臂参数（与问题2、3一致）
# ============================================================

THETA_ZERO=np.array([0.0,-90.0,0.0,180.0,-90.0,0.0],dtype=float)
JOINT_BOUNDS=[(-160.0,160.0),(-150.0,15.0),(-200.0,80.0),
              (-180.0,180.0),(-120.0,120.0),(-180.0,180.0)]
OPT_BOUNDS=[JOINT_BOUNDS[0],JOINT_BOUNDS[1],JOINT_BOUNDS[2],JOINT_BOUNDS[3]]

# 可行性搜索使用几何变量 y=[phi,theta3,theta4]
GEOM_BOUNDS=[(-310.0,175.0),JOINT_BOUNDS[2],JOINT_BOUNDS[3]]

INERTIA=np.array([0.5,0.3,0.4,0.6,0.2,0.4],dtype=float)
OMEGA=np.array([2.0,1.5,1.0,2.5,3.0,2.0],dtype=float)
MASS=5.0
G=9.81
OPT_PATH_POINTS=41
FINAL_PATH_POINTS=121


# ============================================================
# 3. 搜索规模
# ============================================================

if FAST_MODE:
    GEOM_CANDIDATE_LIMIT=24
    FEASIBILITY_SEEDS=[2026]
    FEASIBILITY_MAXITER=55
    FEASIBILITY_POPSIZE=6
    KEEP_BY_ERROR=3
    KEEP_BY_PROXY_ENERGY=3
    KEEP_BY_ROUTE=2
    MAX_RETAINED_BASES=5
    EPSILON_LEVELS=[120.0,160.0,200.0]
    ENERGY_SEEDS=[2026]
    ENERGY_MAXITER=55
    ENERGY_POPSIZE=6
    MAX_STATES_PER_TARGET=4
else:
    GEOM_CANDIDATE_LIMIT=40
    FEASIBILITY_SEEDS=[2026,2027]
    FEASIBILITY_MAXITER=100
    FEASIBILITY_POPSIZE=10
    KEEP_BY_ERROR=5
    KEEP_BY_PROXY_ENERGY=5
    KEEP_BY_ROUTE=3
    MAX_RETAINED_BASES=8
    EPSILON_LEVELS=[80.0,120.0,160.0,180.0,200.0]
    ENERGY_SEEDS=[2026,2027]
    ENERGY_MAXITER=120
    ENERGY_POPSIZE=10
    MAX_STATES_PER_TARGET=6


# ============================================================
# 4. 数值与Pareto容差
# ============================================================

COLLISION_PENALTY=1e6
ERROR_PENALTY=1e4
COLLISION_TOL=1e-12
ERROR_TOL=1e-3
STATE_ERROR_TOL=0.5
STATE_ENERGY_TOL=1e-3
GLOBAL_ERROR_TOL=1.0
GLOBAL_ENERGY_TOL=1e-2


# ============================================================
# 5. 障碍矩形与地图边界
# ============================================================

OBSTACLES=[(r,c) for r in range(ROWS) for c in range(COLS) if GRID[r,c]==1]

OBSTACLE_RECTS=[]
for r,c in OBSTACLES:
    cx,cy=c*CELL_SIZE,r*CELL_SIZE
    half=CELL_SIZE/2.0
    OBSTACLE_RECTS.append((cx-half,cx+half,cy-half,cy+half))

MAP_X_MIN=-CELL_SIZE/2.0
MAP_X_MAX=(COLS-1)*CELL_SIZE+CELL_SIZE/2.0
MAP_Y_MIN=-CELL_SIZE/2.0
MAP_Y_MAX=(ROWS-1)*CELL_SIZE+CELL_SIZE/2.0


# ============================================================
# 6. 栅格工具
# ============================================================

def grid_to_world(cell):
    r,c=cell
    return c*CELL_SIZE,r*CELL_SIZE


def local_target(base_cell,target_cell):
    bx,by=grid_to_world(base_cell)
    tx,ty=grid_to_world(target_cell)
    return np.array([tx-bx,ty-by,TARGET_Z],dtype=float)


def is_base_cell_allowed(cell):
    r,c=cell
    if not (0<=r<ROWS and 0<=c<COLS):
        return False
    if GRID[r,c]!=0:
        return False
    # 底座不能占据货物格
    if cell in TARGET_CELLS:
        return False
    return True


def is_transit_cell_allowed(cell):
    r,c=cell
    if not (0<=r<ROWS and 0<=c<COLS):
        return False
    if GRID[r,c]!=0:
        return False
    # 货物格不允许底座穿过
    if cell in TARGET_CELLS:
        return False
    return True


def manhattan(a,b):
    return abs(a[0]-b[0])+abs(a[1]-b[1])


def astar_path(start,goal):
    if start==goal:
        return [start]
    if start!=START and not is_transit_cell_allowed(start):
        return None
    if not is_base_cell_allowed(goal):
        return None
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
            neighbor=(r+dr,c+dc)
            if neighbor!=goal:
                if not is_transit_cell_allowed(neighbor):
                    continue
            else:
                if not is_base_cell_allowed(neighbor):
                    continue
            new_g=current_g+1
            if new_g<g_score.get(neighbor,float("inf")):
                g_score[neighbor]=new_g
                came_from[neighbor]=current
                heapq.heappush(queue,(new_g+manhattan(neighbor,goal),new_g,neighbor))
    return None


def bfs_distance_map(start):
    """单源最短步数，判断所有底座是否与Start连通。"""
    dist={start:0}
    q=deque([start])
    while q:
        current=q.popleft()
        r,c=current
        for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nxt=(r+dr,c+dc)
            if nxt in dist:
                continue
            if not is_transit_cell_allowed(nxt):
                continue
            dist[nxt]=dist[current]+1
            q.append(nxt)
    return dist


# ============================================================
# 7. 标准D-H末端位置
# ============================================================

def x_to_theta(x):
    return np.array([float(x[0]),float(x[1]),float(x[2]),float(x[3]),
                     THETA_ZERO[4],THETA_ZERO[5]],dtype=float)


def end_position(x):
    theta1,theta2,theta3,theta4=x
    phi=np.deg2rad(theta1+theta2)
    t3=np.deg2rad(theta3)
    t34=np.deg2rad(theta3+theta4)
    A=1200.0*np.cos(t3)+300.0*np.cos(t34)+300.0
    px=A*np.cos(phi)-1200.0*np.sin(phi)
    py=A*np.sin(phi)+1200.0*np.cos(phi)
    pz=600.0-1200.0*np.sin(t3)-300.0*np.sin(t34)
    return np.array([px,py,pz],dtype=float)


def position_error(x,target_local):
    return float(np.linalg.norm(end_position(x)-target_local))


# ============================================================
# 8. 问题2动力学能耗
# ============================================================

def build_motion_profile(theta_final,n_points=FINAL_PATH_POINTS):
    """s(xi)=3xi^2-2xi^3，T=max_i(|Delta theta_i|/omega_i)。"""
    theta0_rad=np.deg2rad(THETA_ZERO)
    thetaf_rad=np.deg2rad(theta_final)
    delta=thetaf_rad-theta0_rad
    T=float(np.max(np.abs(delta)/OMEGA))
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
                "positive_work":0.0,"braking_work":0.0,"total_energy":0.0}
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
            "gravity_work":gravity_work,"positive_work":positive_work,
            "braking_work":braking_work,"total_energy":total_energy}


def quick_total_energy(x):
    return mechanical_energy(x_to_theta(x),n_points=OPT_PATH_POINTS)["total_energy"]


# ============================================================
# 9. 主要关节点与碰撞
# ============================================================

def joint_positions(x,base_cell):
    theta1,theta2,theta3,theta4=x
    phi=math.radians(theta1+theta2)
    t3=math.radians(theta3)
    t34=math.radians(theta3+theta4)
    cp,sp=math.cos(phi),math.sin(phi)
    c3,s3=math.cos(t3),math.sin(t3)
    c34,s34=math.cos(t34),math.sin(t34)
    bx,by=grid_to_world(base_cell)
    J0=(bx,by,0.0)
    J1=(bx,by,600.0)
    J2=(bx+300.0*cp,by+300.0*sp,600.0)
    J3=(bx+300.0*(4.0*c3+1.0)*cp,by+300.0*(4.0*c3+1.0)*sp,600.0-1200.0*s3)
    B=1200.0*c3+300.0*c34+300.0
    J4=(bx-1200.0*sp+B*cp,by+B*sp+1200.0*cp,600.0-1200.0*s3-300.0*s34)
    return [J0,J1,J2,J3,J4]


def segment_rectangle_intersection(x1,y1,x2,y2,xmin,xmax,ymin,ymax):
    """Liang-Barsky，与障碍边界接触也视为碰撞。"""
    dx,dy=x2-x1,y2-y1
    p=[-dx,dx,-dy,dy]
    q=[x1-xmin,xmax-x1,y1-ymin,ymax-y1]
    u1,u2=0.0,1.0
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
    """J0~J4地图边界 + 主要连杆XY投影不穿障碍格；>0表示越界或碰撞。"""
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
            if seg_xmax<xmin or seg_xmin>xmax or seg_ymax<ymin or seg_ymin>ymax:
                continue
            if segment_rectangle_intersection(x1,y1,x2,y2,xmin,xmax,ymin,ymax):
                violation+=1.0
    return float(violation)


# ============================================================
# 10. 候选底座几何预筛 + 解析IK容差球预筛
# ============================================================

def generate_geometric_base_candidates(target_cell,reachable_dist):
    """工作空间必要条件 + Start连通预筛。"""
    tx,ty=grid_to_world(target_cell)
    rho_low=max(0.0,1200.0-MAX_ERROR)
    rho_high=math.sqrt(1800.0**2+1200.0**2)+MAX_ERROR
    raw=[]
    for cell,steps in reachable_dist.items():
        if not is_base_cell_allowed(cell):
            continue
        bx,by=grid_to_world(cell)
        rho=math.hypot(tx-bx,ty-by)
        if not (rho_low<=rho<=rho_high):
            continue
        raw.append({"base_cell":cell,"rho":rho,"start_steps":steps})
    raw=sorted(raw,key=lambda item:(item["start_steps"],item["rho"],item["base_cell"][0],item["base_cell"][1]))
    return raw


def make_error_offsets(step=50.0):
    """半径200mm误差球内规则采样，用于寻找可行抓取种子。"""
    values=np.arange(-MAX_ERROR,MAX_ERROR+1e-9,step,dtype=float)
    offsets=[]
    for dx in values:
        for dy in values:
            for dz in values:
                if dx*dx+dy*dy+dz*dz<=MAX_ERROR**2+1e-9:
                    offsets.append(np.array([dx,dy,dz],dtype=float))
    offsets.sort(key=lambda v:float(np.dot(v,v)))
    return offsets


COARSE_ERROR_OFFSETS=make_error_offsets(step=50.0)


def make_rescue_offsets(n_points=6000,seed=20260825):
    """固定随机种子的容差球加密采样，规则采样失败时启用。"""
    rng=np.random.default_rng(seed)
    offsets=[np.zeros(3,dtype=float)]
    for _ in range(n_points):
        direction=rng.normal(size=3)
        norm=float(np.linalg.norm(direction))
        if norm<1e-12:
            continue
        direction=direction/norm
        radius=MAX_ERROR*rng.random()**(1.0/3.0)
        offsets.append(direction*radius)
    offsets.sort(key=lambda v:float(np.dot(v,v)))
    return offsets


RESCUE_ERROR_OFFSETS=None


def phi_to_x(phi,theta3,theta4):
    """给定phi=theta1+theta2，恢复一组满足关节角范围的theta1、theta2。"""
    low=max(JOINT_BOUNDS[0][0],phi-JOINT_BOUNDS[1][1])
    high=min(JOINT_BOUNDS[0][1],phi-JOINT_BOUNDS[1][0])
    if low>high:
        return None
    theta1_star=(phi+90.0)/2.0
    theta1=float(np.clip(theta1_star,low,high))
    theta2=float(phi-theta1)
    return np.array([theta1,theta2,float(theta3),float(theta4)],dtype=float)


def analytic_ik_solutions(local_point):
    """
    由SDH解析式恢复(phi,theta3,theta4)：
    x=A cos(phi)-1200 sin(phi)，y=A sin(phi)+1200 cos(phi)
    A=1200 cos(theta3)+300 cos(theta3+theta4)+300
    z=600-1200 sin(theta3)-300 sin(theta3+theta4)
    """
    px,py,pz=map(float,local_point)
    rho=math.hypot(px,py)
    if rho<1200.0-1e-9:
        return []
    A_abs=math.sqrt(max(0.0,rho*rho-1200.0**2))
    beta=math.atan2(py,px)
    A_values=[0.0] if A_abs<1e-12 else [A_abs,-A_abs]
    solutions=[]
    for A in A_values:
        gamma=math.atan2(1200.0,A)
        phi0=beta-gamma
        for phi_rad in [phi0-2.0*math.pi,phi0,phi0+2.0*math.pi]:
            phi_deg=math.degrees(phi_rad)
            if not (GEOM_BOUNDS[0][0]-1e-9<=phi_deg<=GEOM_BOUNDS[0][1]+1e-9):
                continue
            u,v=A-300.0,600.0-pz
            radius=math.hypot(u,v)
            cos_theta4=(radius*radius-1200.0**2-300.0**2)/(2.0*1200.0*300.0)
            if cos_theta4<-1.0-1e-9 or cos_theta4>1.0+1e-9:
                continue
            cos_theta4=float(np.clip(cos_theta4,-1.0,1.0))
            alpha=math.atan2(v,u)
            angle=math.acos(cos_theta4)
            for theta4_rad in [angle,-angle]:
                theta3_rad=alpha-math.atan2(300.0*math.sin(theta4_rad),1200.0+300.0*math.cos(theta4_rad))
                theta3_deg0=math.degrees(theta3_rad)
                theta4_deg=math.degrees(theta4_rad)
                for theta3_deg in [theta3_deg0-360.0,theta3_deg0,theta3_deg0+360.0]:
                    if not (JOINT_BOUNDS[2][0]-1e-9<=theta3_deg<=JOINT_BOUNDS[2][1]+1e-9):
                        continue
                    if not (JOINT_BOUNDS[3][0]-1e-9<=theta4_deg<=JOINT_BOUNDS[3][1]+1e-9):
                        continue
                    x=phi_to_x(phi_deg,theta3_deg,theta4_deg)
                    if x is not None:
                        solutions.append(x)
    unique=[]
    for item in solutions:
        if not any(np.linalg.norm(item-old)<1e-7 for old in unique):
            unique.append(item)
    return unique


def analytic_feasible_seed(base_cell,target_cell,offsets):
    """目标点200mm误差球内采样+解析IK+碰撞检查，返回该底座最小误差可行种子。"""
    target_local=local_target(base_cell,target_cell)
    best=None
    for offset in offsets:
        error=float(np.linalg.norm(offset))
        if best is not None and error>=best["error"]-1e-12:
            continue
        endpoint=target_local+offset
        for x in analytic_ik_solutions(endpoint):
            violation=collision_violation(x,base_cell)
            if violation>COLLISION_TOL:
                continue
            actual_error=position_error(x,target_local)
            if actual_error<=MAX_ERROR+ERROR_TOL:
                if best is None or actual_error<best["error"]:
                    best={"x":np.array(x,dtype=float),"error":float(actual_error),
                          "violation":float(violation),"seed":"analytic"}
    return best


# ============================================================
# 11. 最小误差DE
# ============================================================

def solve_min_error_seed(base_cell,target_cell,seed,x0=None):
    """3维几何DE细化解析IK找到的可行种子。"""
    target_local=local_target(base_cell,target_cell)
    def objective(y):
        phi,theta3,theta4=y
        x=phi_to_x(phi,theta3,theta4)
        if x is None:
            return COLLISION_PENALTY
        violation=collision_violation(x,base_cell)
        if violation>COLLISION_TOL:
            return COLLISION_PENALTY*(1.0+violation)
        return position_error(x,target_local)
    kwargs={}
    if x0 is not None:
        x0=np.array(x0,dtype=float)
        kwargs["x0"]=np.array([x0[0]+x0[1],x0[2],x0[3]],dtype=float)
    result=differential_evolution(
        objective,bounds=GEOM_BOUNDS,strategy="best1bin",
        maxiter=FEASIBILITY_MAXITER,popsize=FEASIBILITY_POPSIZE,
        mutation=(0.5,1.0),recombination=0.8,tol=1e-6,atol=1e-6,
        polish=False,seed=seed,workers=1,updating="immediate",disp=False,**kwargs)
    phi,theta3,theta4=np.array(result.x,dtype=float)
    x=phi_to_x(phi,theta3,theta4)
    if x is None:
        return None
    return {"x":x,"error":position_error(x,target_local),
            "violation":collision_violation(x,base_cell),"seed":seed}


def solve_min_error_multiseed(base_cell,target_cell,analytic_seed=None):
    """解析IK种子本身作为候选；DE在其附近进一步降低误差。"""
    trials=[]
    if analytic_seed is not None:
        trials.append({"x":np.array(analytic_seed["x"],dtype=float),
                       "error":float(analytic_seed["error"]),
                       "violation":float(analytic_seed["violation"]),"seed":"analytic"})
    warm=None if analytic_seed is None else analytic_seed["x"]
    for seed in FEASIBILITY_SEEDS:
        result=solve_min_error_seed(base_cell,target_cell,seed,x0=warm)
        if result is not None:
            trials.append(result)
    if not trials:
        return {"x":np.zeros(4,dtype=float),"error":float("inf"),
                "violation":float("inf"),"seed":None}
    valid=[item for item in trials if item["violation"]<=COLLISION_TOL]
    if not valid:
        return min(trials,key=lambda item:(item["violation"],item["error"]))
    return min(valid,key=lambda item:item["error"])


# ============================================================
# 12. 每个货物的候选底座筛选
# ============================================================

def screen_target_bases(target_name,target_cell,reachable_dist):
    print(f"\n【{target_name}】工作空间 + 解析IK容差球预筛")
    geom_candidates=generate_geometric_base_candidates(target_cell,reachable_dist)
    print(f"工作空间候选底座数：{len(geom_candidates)}")
    # 第一步：规则50mm容差球 + 解析IK + 障碍约束
    analytic_candidates=[]
    for item in geom_candidates:
        base_cell=item["base_cell"]
        seed=analytic_feasible_seed(base_cell,target_cell,COARSE_ERROR_OFFSETS)
        if seed is not None:
            analytic_candidates.append((item,seed))
            print(f"  coarse可行 base=({base_cell[0]+1},{base_cell[1]+1}) seed_error={seed['error']:.3f} mm")
    # 规则采样失败时启用固定随机种子加密救援
    if not analytic_candidates:
        print(f"{target_name}：规则容差球未发现可行底座，启动6000点加密解析IK扫描...")
        global RESCUE_ERROR_OFFSETS
        if RESCUE_ERROR_OFFSETS is None:
            RESCUE_ERROR_OFFSETS=make_rescue_offsets()
        for item in geom_candidates:
            base_cell=item["base_cell"]
            seed=analytic_feasible_seed(base_cell,target_cell,RESCUE_ERROR_OFFSETS)
            if seed is not None:
                analytic_candidates.append((item,seed))
                print(f"  rescue可行 base=({base_cell[0]+1},{base_cell[1]+1}) seed_error={seed['error']:.3f} mm")
    if not analytic_candidates:
        raise RuntimeError(f"{target_name} 在全部工作空间候选底座中，仍未找到误差<=200 mm且满足障碍约束的构型。")
    # 第二步：对解析预筛可行底座做3维DE细化
    screened=[]
    print(f"{target_name}：解析预筛发现{len(analytic_candidates)}个可行底座，开始3维DE细化最小误差。")
    for item,analytic_seed in analytic_candidates:
        base_cell=item["base_cell"]
        result=solve_min_error_multiseed(base_cell,target_cell,analytic_seed=analytic_seed)
        if result["violation"]<=COLLISION_TOL:
            proxy_energy=mechanical_energy(x_to_theta(result["x"]),n_points=OPT_PATH_POINTS)["total_energy"]
        else:
            proxy_energy=float("inf")
        screened.append({"target_name":target_name,"target_cell":target_cell,"base_cell":base_cell,
                         "rho":item["rho"],"start_steps":item["start_steps"],
                         "x_min_error":result["x"],"min_error":result["error"],
                         "violation":result["violation"],"proxy_energy":proxy_energy})
        print(f"  base=({base_cell[0]+1},{base_cell[1]+1}) e_min={result['error']:.3f} collision={result['violation']:.3e}")
    feasible=[item for item in screened
              if item["violation"]<=COLLISION_TOL and item["min_error"]<=MAX_ERROR+ERROR_TOL]
    if not feasible:
        raise RuntimeError(f"{target_name} 解析预筛已找到可行种子，但DE细化后列表异常为空。")
    # 误差、代理能耗、Start路径三类排名做并集
    selected=[]
    def add_ranked(items,n):
        selected_cells={x["base_cell"] for x in selected}
        for item in items[:n]:
            if item["base_cell"] not in selected_cells:
                selected.append(item)
                selected_cells.add(item["base_cell"])
    add_ranked(sorted(feasible,key=lambda x:x["min_error"]),KEEP_BY_ERROR)
    add_ranked(sorted(feasible,key=lambda x:x["proxy_energy"]),KEEP_BY_PROXY_ENERGY)
    add_ranked(sorted(feasible,key=lambda x:x["start_steps"]),KEEP_BY_ROUTE)
    if len(selected)>MAX_RETAINED_BASES:
        def norm(arr):
            amin,amax=float(np.min(arr)),float(np.max(arr))
            if abs(amax-amin)<1e-12:
                return np.zeros_like(arr,dtype=float)
            return (arr-amin)/(amax-amin)
        e_values=np.array([x["min_error"] for x in selected],dtype=float)
        E_values=np.array([x["proxy_energy"] for x in selected],dtype=float)
        L_values=np.array([x["start_steps"] for x in selected],dtype=float)
        score=norm(e_values)+norm(E_values)+0.3*norm(L_values)
        selected=[selected[i] for i in np.argsort(score)[:MAX_RETAINED_BASES]]
    print(f"{target_name}：误差<=200的可行底座{len(feasible)}个，保留{len(selected)}个进入能耗优化。")
    return screened,selected


# ============================================================
# 13. 给定epsilon的能耗优化
# ============================================================

def solve_energy_seed(target_name,target_cell,base_item,epsilon,x0,seed):
    base_cell=base_item["base_cell"]
    target_local=local_target(base_cell,target_cell)
    def objective(x):
        violation=collision_violation(x,base_cell)
        if violation>COLLISION_TOL:
            return COLLISION_PENALTY*(1.0+violation)
        error=position_error(x,target_local)
        if error>epsilon:
            return COLLISION_PENALTY+ERROR_PENALTY*(error-epsilon)
        return quick_total_energy(x)
    result=differential_evolution(
        objective,bounds=OPT_BOUNDS,strategy="best1bin",
        maxiter=ENERGY_MAXITER,popsize=ENERGY_POPSIZE,
        mutation=(0.5,1.0),recombination=0.8,tol=1e-6,atol=1e-6,
        x0=np.array(x0,dtype=float),polish=False,seed=seed,workers=1,
        updating="immediate",disp=False)
    x=np.array(result.x,dtype=float)
    violation=collision_violation(x,base_cell)
    error=position_error(x,target_local)
    if violation>COLLISION_TOL or error>epsilon+ERROR_TOL:
        return None
    theta=x_to_theta(x)
    energy=mechanical_energy(theta,n_points=FINAL_PATH_POINTS)
    p_local=end_position(x)
    bx,by=grid_to_world(base_cell)
    end_world=np.array([bx+p_local[0],by+p_local[1],p_local[2]],dtype=float)
    return {"target_name":target_name,"target_cell":target_cell,"base_cell":base_cell,
            "epsilon":float(epsilon),"actual_error":float(error),"x":x,"theta":theta,
            "motion_time":energy["motion_time"],"inertia_work":energy["inertia_work"],
            "gravity_work":energy["gravity_work"],"positive_work":energy["positive_work"],
            "braking_work":energy["braking_work"],"total_energy":energy["total_energy"],
            "end_world":end_world,"source":"epsilon_DE"}


def solve_energy_multiseed(target_name,target_cell,base_item,epsilon,x0):
    trials=[]
    for seed in ENERGY_SEEDS:
        result=solve_energy_seed(target_name,target_cell,base_item,epsilon,x0,seed)
        if result is not None:
            trials.append(result)
    if not trials:
        return None
    return min(trials,key=lambda x:x["total_energy"])


# ============================================================
# 14. 单货物候选状态Pareto
# ============================================================

def target_state_pareto(states):
    front=[]
    for i,item in enumerate(states):
        dominated=False
        for j,other in enumerate(states):
            if i==j:
                continue
            if (other["actual_error"]<=item["actual_error"]+STATE_ERROR_TOL
                    and other["total_energy"]<=item["total_energy"]+STATE_ENERGY_TOL
                    and (other["actual_error"]<item["actual_error"]-STATE_ERROR_TOL
                         or other["total_energy"]<item["total_energy"]-STATE_ENERGY_TOL)):
                dominated=True
                break
        if not dominated:
            front.append(item)
    front=sorted(front,key=lambda x:(x["actual_error"],x["total_energy"]))
    return front


def limit_states(states,max_states):
    if len(states)<=max_states:
        return states
    states=sorted(states,key=lambda x:x["actual_error"])
    selected_indices={0,int(np.argmin([x["total_energy"] for x in states])),len(states)-1}
    if max_states>3:
        extra=np.linspace(0,len(states)-1,max_states,dtype=int)
        selected_indices.update(extra.tolist())
    selected=[states[i] for i in sorted(selected_indices)]
    if len(selected)>max_states:
        selected=selected[:max_states]
    return selected


# ============================================================
# 15. 为一个货物生成候选抓取状态
# ============================================================

def build_target_states(target_name,target_cell,retained_bases):
    all_states=[]
    for base_item in retained_bases:
        # 1. 最小误差状态本身也作为一个候选
        x_min=np.array(base_item["x_min_error"],dtype=float)
        theta_min=x_to_theta(x_min)
        energy_min=mechanical_energy(theta_min,n_points=FINAL_PATH_POINTS)
        p_local=end_position(x_min)
        bx,by=grid_to_world(base_item["base_cell"])
        all_states.append({"target_name":target_name,"target_cell":target_cell,
                           "base_cell":base_item["base_cell"],"epsilon":float(base_item["min_error"]),
                           "actual_error":float(base_item["min_error"]),"x":x_min,"theta":theta_min,
                           "motion_time":energy_min["motion_time"],"inertia_work":energy_min["inertia_work"],
                           "gravity_work":energy_min["gravity_work"],"positive_work":energy_min["positive_work"],
                           "braking_work":energy_min["braking_work"],"total_energy":energy_min["total_energy"],
                           "end_world":np.array([bx+p_local[0],by+p_local[1],p_local[2]],dtype=float),
                           "source":"min_error"})
        # 2. epsilon能耗优化
        warm=x_min.copy()
        eps_values=[eps for eps in EPSILON_LEVELS if eps>=base_item["min_error"]-ERROR_TOL]
        for epsilon in eps_values:
            result=solve_energy_multiseed(target_name,target_cell,base_item,epsilon,warm)
            if result is None:
                continue
            all_states.append(result)
            warm=result["x"].copy()
    front=target_state_pareto(all_states)
    selected=limit_states(front,MAX_STATES_PER_TARGET)
    for idx,state in enumerate(selected,start=1):
        state["state_id"]=f"{target_name}_S{idx}"
    print(f"{target_name}：生成{len(all_states)}个状态，单货物Pareto后{len(front)}个，最终保留{len(selected)}个。")
    return all_states,selected


# ============================================================
# 16. A*距离缓存
# ============================================================

PATH_CACHE={}

def cached_path(a,b):
    key=(a,b)
    if key in PATH_CACHE:
        return PATH_CACHE[key]
    path=astar_path(a,b)
    PATH_CACHE[(a,b)]=path
    if path is not None:
        PATH_CACHE[(b,a)]=list(reversed(path))
    return path


def cached_steps(a,b):
    path=cached_path(a,b)
    if path is None:
        return float("inf")
    return len(path)-1


# ============================================================
# 17. 给定五个抓取状态，枚举120种访问顺序求最短路线
# ============================================================

def best_order_for_state_combo(state_by_target):
    best=None
    for order in itertools.permutations(TARGET_NAMES):
        total_steps=0.0
        current=START
        feasible=True
        for target_name in order:
            base_cell=state_by_target[target_name]["base_cell"]
            steps=cached_steps(current,base_cell)
            if not np.isfinite(steps):
                feasible=False
                break
            total_steps+=steps
            current=base_cell
        if not feasible:
            continue
        if RETURN_TO_START:
            steps=cached_steps(current,START)
            if not np.isfinite(steps):
                continue
            total_steps+=steps
        if best is None or total_steps<best["steps"]:
            best={"order":order,"steps":int(total_steps)}
    return best


# ============================================================
# 18. 全局状态组合
# ============================================================

def enumerate_global_combinations(selected_states_by_target):
    state_lists=[selected_states_by_target[name] for name in TARGET_NAMES]
    total_combinations=1
    for states in state_lists:
        total_combinations*=len(states)
    print(f"\n五个货物状态组合数：{total_combinations}")
    rows=[]
    for combo_index,combo in enumerate(itertools.product(*state_lists),start=1):
        state_by_target={name:state for name,state in zip(TARGET_NAMES,combo)}
        route=best_order_for_state_combo(state_by_target)
        if route is None:
            continue
        total_error=sum(state["actual_error"] for state in combo)
        total_energy=sum(state["total_energy"] for state in combo)
        rows.append({"combo_id":combo_index,"total_error_mm":float(total_error),
                     "total_energy_J":float(total_energy),"route_steps":route["steps"],
                     "route_distance_mm":route["steps"]*CELL_SIZE,
                     "order":route["order"],"state_by_target":state_by_target})
    print(f"可行全局组合数：{len(rows)}")
    return rows


# ============================================================
# 19. 全局误差—能耗Pareto
# ============================================================

def global_pareto(solutions):
    front=[]
    for i,item in enumerate(solutions):
        dominated=False
        for j,other in enumerate(solutions):
            if i==j:
                continue
            if (other["total_error_mm"]<=item["total_error_mm"]+GLOBAL_ERROR_TOL
                    and other["total_energy_J"]<=item["total_energy_J"]+GLOBAL_ENERGY_TOL
                    and (other["total_error_mm"]<item["total_error_mm"]-GLOBAL_ERROR_TOL
                         or other["total_energy_J"]<item["total_energy_J"]-GLOBAL_ENERGY_TOL)):
                dominated=True
                break
        if not dominated:
            front.append(item)
    front=sorted(front,key=lambda x:(x["total_error_mm"],x["total_energy_J"],x["route_steps"]))
    cleaned=[]
    for item in front:
        merged=False
        for k,old in enumerate(cleaned):
            if (abs(item["total_error_mm"]-old["total_error_mm"])<=GLOBAL_ERROR_TOL
                    and abs(item["total_energy_J"]-old["total_energy_J"])<=GLOBAL_ENERGY_TOL):
                if item["route_steps"]<old["route_steps"]:
                    cleaned[k]=item
                merged=True
                break
        if not merged:
            cleaned.append(item)
    return cleaned


# ============================================================
# 20. 理想点折中代表方案
# ============================================================

def choose_representatives(front):
    if not front:
        raise RuntimeError("全局Pareto前沿为空。")
    accuracy=min(front,key=lambda x:(x["total_error_mm"],x["total_energy_J"],x["route_steps"]))
    energy=min(front,key=lambda x:(x["total_energy_J"],x["total_error_mm"],x["route_steps"]))
    errors=np.array([x["total_error_mm"] for x in front],dtype=float)
    energies=np.array([x["total_energy_J"] for x in front],dtype=float)
    def norm(arr):
        amin,amax=float(np.min(arr)),float(np.max(arr))
        if abs(amax-amin)<1e-12:
            return np.zeros_like(arr)
        return (arr-amin)/(amax-amin)
    ne,nE=norm(errors),norm(energies)
    scores=np.sqrt(ne**2+nE**2)
    min_score=float(np.min(scores))
    candidate_indices=np.where(np.abs(scores-min_score)<1e-12)[0]
    best_index=min(candidate_indices,key=lambda i:front[i]["route_steps"])
    balanced=front[int(best_index)].copy()
    balanced["ideal_point_score"]=min_score
    return {"accuracy":accuracy,"energy":energy,"balanced":balanced}


# ============================================================
# 21. DataFrame
# ============================================================

def screened_bases_dataframe(all_screened):
    rows=[]
    for item in all_screened:
        r,c=item["base_cell"]
        rows.append({"target":item["target_name"],"base_row":r+1,"base_col":c+1,
                     "rho_mm":item["rho"],"start_steps":item["start_steps"],
                     "min_error_mm":item["min_error"],"collision_violation":item["violation"],
                     "proxy_energy_J":item["proxy_energy"]})
    return pd.DataFrame(rows)


def states_dataframe(states):
    rows=[]
    for state in states:
        theta=state["theta"]
        br,bc=state["base_cell"]
        tr,tc=state["target_cell"]
        p=state["end_world"]
        rows.append({"state_id":state.get("state_id",""),"target":state["target_name"],
                     "target_row":tr+1,"target_col":tc+1,"base_row":br+1,"base_col":bc+1,
                     "source":state["source"],"epsilon_mm":state["epsilon"],
                     "actual_error_mm":state["actual_error"],
                     "theta1_deg":theta[0],"theta2_deg":theta[1],"theta3_deg":theta[2],
                     "theta4_deg":theta[3],"theta5_deg":theta[4],"theta6_deg":theta[5],
                     "motion_time_s":state["motion_time"],"inertia_abs_work_J":state["inertia_work"],
                     "gravity_abs_work_J":state["gravity_work"],"total_energy_J":state["total_energy"],
                     "end_world_x_mm":p[0],"end_world_y_mm":p[1],"end_world_z_mm":p[2]})
    return pd.DataFrame(rows)


def global_solutions_dataframe(solutions):
    rows=[]
    for item in solutions:
        state_map=item["state_by_target"]
        row={"combo_id":item["combo_id"],"total_error_mm":item["total_error_mm"],
             "total_energy_J":item["total_energy_J"],"route_steps":item["route_steps"],
             "route_distance_mm":item["route_distance_mm"],"order":" -> ".join(item["order"])}
        for name in TARGET_NAMES:
            state=state_map[name]
            br,bc=state["base_cell"]
            row[f"{name}_state"]=state["state_id"]
            row[f"{name}_base"]=f"({br+1},{bc+1})"
            row[f"{name}_error_mm"]=state["actual_error"]
            row[f"{name}_energy_J"]=state["total_energy"]
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# 22. 最终路线细节
# ============================================================

def build_route_segments(solution):
    order=solution["order"]
    state_map=solution["state_by_target"]
    segments=[]
    current=START
    for seq,target_name in enumerate(order,start=1):
        destination=state_map[target_name]["base_cell"]
        path=cached_path(current,destination)
        if path is None:
            raise RuntimeError("最终方案存在不可达路径。")
        segments.append({"sequence":seq,"from":"Start" if current==START else str(current),
                         "to_target":target_name,"to_base_cell":destination,
                         "steps":len(path)-1,"distance_mm":(len(path)-1)*CELL_SIZE,"path":path})
        current=destination
    if RETURN_TO_START:
        path=cached_path(current,START)
        if path is None:
            raise RuntimeError("最终方案无法返回Start。")
        segments.append({"sequence":len(order)+1,"from":str(current),"to_target":"Return_Start",
                         "to_base_cell":START,"steps":len(path)-1,"distance_mm":(len(path)-1)*CELL_SIZE,
                         "path":path})
    return segments


def final_task_dataframe(solution):
    rows=[]
    for seq,target_name in enumerate(solution["order"],start=1):
        state=solution["state_by_target"][target_name]
        theta=state["theta"]
        tr,tc=state["target_cell"]
        br,bc=state["base_cell"]
        rows.append({"sequence":seq,"target":target_name,"target_row":tr+1,"target_col":tc+1,
                     "base_row":br+1,"base_col":bc+1,"actual_error_mm":state["actual_error"],
                     "total_energy_J":state["total_energy"],
                     "theta1_deg":theta[0],"theta2_deg":theta[1],"theta3_deg":theta[2],
                     "theta4_deg":theta[3],"theta5_deg":theta[4],"theta6_deg":theta[5],
                     "motion_time_s":state["motion_time"]})
    return pd.DataFrame(rows)


# ============================================================
# 23. 绘图
# ============================================================

def draw_grid_background(ax):
    for r,c in OBSTACLES:
        ax.add_patch(Rectangle((c-0.5,r-0.5),1,1))
    for k in range(COLS+1):
        ax.plot([k-0.5,k-0.5],[-0.5,ROWS-0.5],linewidth=0.35)
    for k in range(ROWS+1):
        ax.plot([-0.5,COLS-0.5],[k-0.5,k-0.5],linewidth=0.35)
    ax.set_xlim(-0.5,COLS-0.5)
    ax.set_ylim(ROWS-0.5,-0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.set_xticklabels(range(1,COLS+1))
    ax.set_yticklabels(range(1,ROWS+1))
    ax.set_xlabel("列")
    ax.set_ylabel("行")


def plot_sheet2_map(save_path):
    fig,ax=plt.subplots(figsize=(10,9))
    draw_grid_background(ax)
    ax.scatter([START[1]],[START[0]],marker="*",s=180,label="Start")
    for name,cell in TARGETS.items():
        r,c=cell
        ax.scatter([c],[r],marker="X",s=120)
        ax.text(c+0.15,r-0.15,name,fontsize=9)
    ax.set_title("第四问：Sheet2障碍物与五个货物位置")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path,dpi=300,bbox_inches="tight")
    plt.close(fig)


def plot_global_pareto(front,representatives,save_path):
    fig,ax=plt.subplots(figsize=(9,6))
    ax.plot([x["total_error_mm"] for x in front],
            [x["total_energy_J"] for x in front],
            marker="o",linewidth=1.8,label="全局Pareto前沿")
    balanced=representatives["balanced"]
    ax.scatter([balanced["total_error_mm"]],[balanced["total_energy_J"]],
               marker="*",s=180,label="理想点折中方案")
    ax.set_xlabel("五次抓取总末端误差 / mm")
    ax.set_ylabel("五次抓取总机械能耗 / J")
    ax.set_title("第四问：总末端误差—总机械能耗 Pareto 前沿")
    ax.grid(True,alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path,dpi=300,bbox_inches="tight")
    plt.close(fig)


def plot_final_route(solution,save_path):
    fig,ax=plt.subplots(figsize=(10,9))
    draw_grid_background(ax)
    segments=build_route_segments(solution)
    for segment in segments:
        path=segment["path"]
        ax.plot([c for r,c in path],[r for r,c in path],
                marker="o",linewidth=1.8,label=f"段{segment['sequence']}")
    ax.scatter([START[1]],[START[0]],marker="*",s=180,label="Start")
    for name,cell in TARGETS.items():
        r,c=cell
        ax.scatter([c],[r],marker="X",s=110)
        ax.text(c+0.12,r-0.12,name,fontsize=8)
    for seq,target_name in enumerate(solution["order"],start=1):
        state=solution["state_by_target"][target_name]
        br,bc=state["base_cell"]
        ax.scatter([bc],[br],marker="s",s=90)
        ax.text(bc+0.1,br+0.25,f"B{seq}",fontsize=8)
    ax.set_title("第四问：折中方案底座最优移动路径")
    ax.legend(loc="upper left",bbox_to_anchor=(1.02,1.0))
    fig.tight_layout()
    fig.savefig(save_path,dpi=300,bbox_inches="tight")
    plt.close(fig)


def plot_joint_angle_path(solution,save_path):
    order=solution["order"]
    x_axis=np.arange(1,len(order)+1)
    theta_matrix=np.array([solution["state_by_target"][name]["theta"][:4] for name in order],dtype=float)
    fig,ax=plt.subplots(figsize=(10,6))
    for j in range(4):
        ax.plot(x_axis,theta_matrix[:,j],marker="o",label=f"theta{j+1}")
    ax.set_xticks(x_axis)
    ax.set_xticklabels([f"{i}:{name}" for i,name in enumerate(order,start=1)])
    ax.set_xlabel("抓取顺序")
    ax.set_ylabel("目标关节角 / °")
    ax.set_title("第四问：五次抓取最优关节角路径")
    ax.grid(True,alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path,dpi=300,bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 24. 主程序（explore阶段）
# ============================================================

def run_explore():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    print("="*96)
    print("A题问题4：五货物 + A* + 4D-DE + 动力学能耗 + 组合枚举 + 全局Pareto")
    print(f"FAST_MODE = {FAST_MODE}")
    print(f"RETURN_TO_START = {RETURN_TO_START}")
    print("="*96)
    print("\nSheet2目标：")
    for name,cell in TARGETS.items():
        wx,wy=grid_to_world(cell)
        print(f"{name}: grid=({cell[0]+1},{cell[1]+1}), world=({wx:.0f},{wy:.0f},{TARGET_Z:.0f}) mm")
    print(f"\n障碍物数量：{len(OBSTACLES)}")
    # 阶段1：Start连通域
    reachable_dist=bfs_distance_map(START)
    print(f"Start连通的可通行栅格数：{len(reachable_dist)}")
    # 阶段2：五个货物分别筛候选底座
    all_screened=[]
    retained_bases_by_target={}
    for target_name,target_cell in TARGETS.items():
        screened,retained=screen_target_bases(target_name,target_cell,reachable_dist)
        all_screened.extend(screened)
        retained_bases_by_target[target_name]=retained
    # 阶段3：候选抓取状态
    all_states=[]
    selected_states_by_target={}
    print("\n"+"="*96)
    print("开始生成每个货物的误差—能耗候选状态")
    print("="*96)
    for target_name,target_cell in TARGETS.items():
        target_all_states,target_selected=build_target_states(target_name,target_cell,retained_bases_by_target[target_name])
        all_states.extend(target_all_states)
        selected_states_by_target[target_name]=target_selected
        if not target_selected:
            raise RuntimeError(f"{target_name} 没有保留下来的抓取状态。")
    # 阶段4：全局组合 + 120顺序精确枚举
    global_solutions=enumerate_global_combinations(selected_states_by_target)
    if not global_solutions:
        raise RuntimeError("没有得到可行的五货物联合方案。")
    front=global_pareto(global_solutions)
    representatives=choose_representatives(front)
    final_solution=representatives["balanced"]
    # 输出核心结果
    print("\n"+"="*96)
    print("第四问：全局Pareto及代表方案")
    print("="*96)
    print(f"全局组合方案数：{len(global_solutions)}")
    print(f"全局Pareto点数：{len(front)}")
    print("\n【精度优先方案】")
    print(f"总误差 = {representatives['accuracy']['total_error_mm']:.6f} mm")
    print(f"总能耗 = {representatives['accuracy']['total_energy_J']:.6f} J")
    print(f"底座总路程 = {representatives['accuracy']['route_distance_mm']:.0f} mm")
    print("\n【能耗优先方案】")
    print(f"总误差 = {representatives['energy']['total_error_mm']:.6f} mm")
    print(f"总能耗 = {representatives['energy']['total_energy_J']:.6f} J")
    print(f"底座总路程 = {representatives['energy']['route_distance_mm']:.0f} mm")
    print("\n【理想点折中代表方案】")
    print(f"货物访问顺序 = {' -> '.join(final_solution['order'])}")
    print(f"总末端误差 = {final_solution['total_error_mm']:.6f} mm")
    print(f"总机械能耗 = {final_solution['total_energy_J']:.6f} J")
    print(f"底座总移动距离 = {final_solution['route_distance_mm']:.0f} mm")
    print(f"理想点归一化距离 = {final_solution['ideal_point_score']:.6f}")
    print("\n五次抓取详细方案：")
    for seq,target_name in enumerate(final_solution["order"],start=1):
        state=final_solution["state_by_target"][target_name]
        br,bc=state["base_cell"]
        print(f"{seq}. {target_name}: base=({br+1},{bc+1}), error={state['actual_error']:.3f} mm, "
              f"E={state['total_energy']:.6f} J, theta={state['theta']}")
    # 保存Excel
    screened_path=OUTPUT_DIR/"第四问_候选底座筛选.xlsx"
    states_path=OUTPUT_DIR/"第四问_候选抓取状态.xlsx"
    combinations_path=OUTPUT_DIR/"第四问_全局组合方案.xlsx"
    pareto_path=OUTPUT_DIR/"第四问_全局Pareto前沿.xlsx"
    final_path=OUTPUT_DIR/"第四问_最终折中方案.xlsx"
    screened_bases_dataframe(all_screened).to_excel(screened_path,index=False)
    selected_flat=[state for name in TARGET_NAMES for state in selected_states_by_target[name]]
    states_dataframe(selected_flat).to_excel(states_path,index=False)
    global_solutions_dataframe(global_solutions).to_excel(combinations_path,index=False)
    global_solutions_dataframe(front).to_excel(pareto_path,index=False)
    route_segments=build_route_segments(final_solution)
    route_rows=[]
    for item in route_segments:
        br,bc=item["to_base_cell"]
        route_rows.append({"sequence":item["sequence"],"from":item["from"],
                           "to_target":item["to_target"],"to_base_row":br+1,"to_base_col":bc+1,
                           "steps":item["steps"],"distance_mm":item["distance_mm"],
                           "path":str([(r+1,c+1) for r,c in item["path"]])})
    summary_df=pd.DataFrame([{"scheme":"balanced","visit_order":" -> ".join(final_solution["order"]),
                              "total_error_mm":final_solution["total_error_mm"],
                              "total_energy_J":final_solution["total_energy_J"],
                              "route_steps":final_solution["route_steps"],
                              "route_distance_mm":final_solution["route_distance_mm"],
                              "return_to_start":RETURN_TO_START,
                              "ideal_point_score":final_solution["ideal_point_score"]}])
    with pd.ExcelWriter(final_path) as writer:
        summary_df.to_excel(writer,sheet_name="summary",index=False)
        final_task_dataframe(final_solution).to_excel(writer,sheet_name="task_detail",index=False)
        pd.DataFrame(route_rows).to_excel(writer,sheet_name="route",index=False)
    # 绘图
    map_fig=OUTPUT_DIR/"第四问_Sheet2地图与目标.png"
    pareto_fig=OUTPUT_DIR/"第四问_总误差总能耗Pareto.png"
    route_fig=OUTPUT_DIR/"第四问_最终底座移动路径.png"
    joint_fig=OUTPUT_DIR/"第四问_最终关节角路径.png"
    plot_sheet2_map(map_fig)
    plot_global_pareto(front,representatives,pareto_fig)
    plot_final_route(final_solution,route_fig)
    plot_joint_angle_path(final_solution,joint_fig)
    print("\n"+"="*96)
    print("结果文件")
    print("="*96)
    for path in [screened_path,states_path,combinations_path,pareto_path,final_path,
                 map_fig,pareto_fig,route_fig,joint_fig]:
        print(f"  {path.name}")


# ============================================================
# 25. 高精度核验模块
# ============================================================

EXPLORE_STATES_PATH=OUTPUT_DIR/"第四问_候选抓取状态.xlsx"
VERIFY_ENERGY_SEEDS=[2026,2027,2028]
VERIFY_ENERGY_MAXITER=150
VERIFY_ENERGY_POPSIZE=10
VERIFY_PATH_POINTS=FINAL_PATH_POINTS


def load_explore_selected_states():
    """读取探索阶段保留的单货物Pareto候选状态。"""
    if not EXPLORE_STATES_PATH.exists():
        raise FileNotFoundError(f"未找到探索阶段候选状态文件：{EXPLORE_STATES_PATH}\n"
                                f"请先运行RUN_MODE='explore'，或直接使用RUN_MODE='all'。")
    df=pd.read_excel(EXPLORE_STATES_PATH)
    required=["state_id","target","base_row","base_col","epsilon_mm","actual_error_mm",
              "theta1_deg","theta2_deg","theta3_deg","theta4_deg","theta5_deg","theta6_deg"]
    missing=[col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"探索结果缺少字段：{missing}")
    states_by_target={name:[] for name in TARGET_NAMES}
    for _,row in df.iterrows():
        target_name=str(row["target"])
        if target_name not in TARGETS:
            continue
        target_cell=TARGETS[target_name]
        base_cell=(int(row["base_row"])-1,int(row["base_col"])-1)
        x=np.array([row["theta1_deg"],row["theta2_deg"],row["theta3_deg"],row["theta4_deg"]],dtype=float)
        theta=x_to_theta(x)
        target_local=local_target(base_cell,target_cell)
        actual_error=position_error(x,target_local)
        energy=mechanical_energy(theta,n_points=VERIFY_PATH_POINTS)
        p_local=end_position(x)
        bx,by=grid_to_world(base_cell)
        end_world=np.array([bx+p_local[0],by+p_local[1],p_local[2]],dtype=float)
        states_by_target[target_name].append({"state_id":str(row["state_id"]),
            "target_name":target_name,"target_cell":target_cell,"base_cell":base_cell,
            "epsilon":float(row["epsilon_mm"]),"actual_error":float(actual_error),"x":x,"theta":theta,
            "motion_time":energy["motion_time"],"inertia_work":energy["inertia_work"],
            "gravity_work":energy["gravity_work"],"positive_work":energy["positive_work"],
            "braking_work":energy["braking_work"],"total_energy":energy["total_energy"],
            "end_world":end_world,"source":"explore_warm"})
    for name in TARGET_NAMES:
        if not states_by_target[name]:
            raise RuntimeError(f"{name} 没有可用于高精度核验的探索状态。")
    return states_by_target


def make_verified_state(target_name,target_cell,base_cell,epsilon,x,state_id,source,seed=None):
    """对一个候选x做正式121点核算，验证误差/障碍约束。"""
    x=np.array(x,dtype=float)
    violation=collision_violation(x,base_cell)
    error=position_error(x,local_target(base_cell,target_cell))
    if violation>COLLISION_TOL or error>epsilon+ERROR_TOL:
        return None
    theta=x_to_theta(x)
    energy=mechanical_energy(theta,n_points=VERIFY_PATH_POINTS)
    p_local=end_position(x)
    bx,by=grid_to_world(base_cell)
    end_world=np.array([bx+p_local[0],by+p_local[1],p_local[2]],dtype=float)
    return {"state_id":state_id,"target_name":target_name,"target_cell":target_cell,
            "base_cell":base_cell,"epsilon":float(epsilon),"actual_error":float(error),"x":x,"theta":theta,
            "motion_time":energy["motion_time"],"inertia_work":energy["inertia_work"],
            "gravity_work":energy["gravity_work"],"positive_work":energy["positive_work"],
            "braking_work":energy["braking_work"],"total_energy":energy["total_energy"],
            "end_world":end_world,"source":source,"seed":seed}


def verify_energy_seed(warm_state,seed):
    """对探索阶段候选状态做高精度DE核验（121点动力学积分）。"""
    target_name=warm_state["target_name"]
    target_cell=warm_state["target_cell"]
    base_cell=warm_state["base_cell"]
    epsilon=float(warm_state["epsilon"])
    target_local=local_target(base_cell,target_cell)
    def objective(x):
        violation=collision_violation(x,base_cell)
        if violation>COLLISION_TOL:
            return COLLISION_PENALTY*(1.0+violation)
        error=position_error(x,target_local)
        if error>epsilon:
            return COLLISION_PENALTY+ERROR_PENALTY*(error-epsilon)
        return mechanical_energy(x_to_theta(x),n_points=VERIFY_PATH_POINTS)["total_energy"]
    result=differential_evolution(
        objective,bounds=OPT_BOUNDS,strategy="best1bin",
        maxiter=VERIFY_ENERGY_MAXITER,popsize=VERIFY_ENERGY_POPSIZE,
        mutation=(0.5,1.0),recombination=0.8,tol=1e-8,atol=1e-8,
        x0=np.array(warm_state["x"],dtype=float),polish=False,seed=seed,workers=1,
        updating="immediate",disp=False)
    x=np.array(result.x,dtype=float)
    return make_verified_state(target_name=target_name,target_cell=target_cell,base_cell=base_cell,
                               epsilon=epsilon,x=x,state_id=warm_state["state_id"],
                               source="verify_DE",seed=seed)


def verify_one_state(warm_state):
    """原探索解 + 三个正式DE随机种子竞争，避免高精度DE偶然退化。"""
    candidates=[]
    original=make_verified_state(target_name=warm_state["target_name"],target_cell=warm_state["target_cell"],
                                 base_cell=warm_state["base_cell"],epsilon=warm_state["epsilon"],
                                 x=warm_state["x"],state_id=warm_state["state_id"],
                                 source="explore_recheck",seed=None)
    if original is not None:
        candidates.append(original)
    for seed in VERIFY_ENERGY_SEEDS:
        item=verify_energy_seed(warm_state,seed)
        if item is not None:
            candidates.append(item)
    if not candidates:
        return None
    return min(candidates,key=lambda item:(item["total_energy"],item["actual_error"]))


def verify_selected_states(states_by_target):
    """对探索阶段候选抓取状态统一做高精度复核。"""
    verified_by_target={name:[] for name in TARGET_NAMES}
    print("\n"+"="*96)
    print("第四问核验：候选抓取状态多随机种子 + 121点动力学积分")
    print("="*96)
    for target_name in TARGET_NAMES:
        warm_states=states_by_target[target_name]
        print(f"\n【{target_name}】待核验状态数：{len(warm_states)}")
        verified=[]
        for idx,warm in enumerate(warm_states,start=1):
            br,bc=warm["base_cell"]
            item=verify_one_state(warm)
            if item is None:
                print(f"[{idx}/{len(warm_states)}] base=({br+1},{bc+1}) eps={warm['epsilon']:.3f} -> 核验未获得可行解")
                continue
            verified.append(item)
            print(f"[{idx}/{len(warm_states)}] base=({br+1},{bc+1}) eps={item['epsilon']:.3f} "
                  f"error={item['actual_error']:.3f} mm E={item['total_energy']:.6f} J "
                  f"source={item['source']} seed={item.get('seed')}")
        if not verified:
            raise RuntimeError(f"{target_name} 高精度核验后没有可行状态。")
        front=target_state_pareto(verified)
        for idx,state in enumerate(front,start=1):
            state["state_id"]=f"{target_name}_V{idx}"
        verified_by_target[target_name]=front
        print(f"{target_name}：高精度核验后保留{len(front)}个单货物Pareto状态。")
    return verified_by_target


def save_high_precision_result(verified_by_target,global_solutions,front,representatives):
    """保存高精度联合优化结果（仅xlsx，图已在explore阶段生成）。"""
    final_solution=representatives["balanced"]
    verified_states_flat=[state for name in TARGET_NAMES for state in verified_by_target[name]]
    state_path=OUTPUT_DIR/"第四问_高精度候选抓取状态.xlsx"
    combination_path=OUTPUT_DIR/"第四问_高精度全局组合方案.xlsx"
    pareto_path=OUTPUT_DIR/"第四问_高精度全局Pareto前沿.xlsx"
    final_path=OUTPUT_DIR/"第四问_最终高精度方案.xlsx"
    states_dataframe(verified_states_flat).to_excel(state_path,index=False)
    global_solutions_dataframe(global_solutions).to_excel(combination_path,index=False)
    global_solutions_dataframe(front).to_excel(pareto_path,index=False)
    route_segments=build_route_segments(final_solution)
    route_rows=[]
    for item in route_segments:
        br,bc=item["to_base_cell"]
        route_rows.append({"sequence":item["sequence"],"from":item["from"],
                           "to_target":item["to_target"],"to_base_row":br+1,"to_base_col":bc+1,
                           "steps":item["steps"],"distance_mm":item["distance_mm"],
                           "path":str([(r+1,c+1) for r,c in item["path"]])})
    summary_df=pd.DataFrame([{"scheme":"high_precision_balanced",
                              "visit_order":" -> ".join(final_solution["order"]),
                              "total_error_mm":final_solution["total_error_mm"],
                              "total_energy_J":final_solution["total_energy_J"],
                              "route_steps":final_solution["route_steps"],
                              "route_distance_mm":final_solution["route_distance_mm"],
                              "return_to_start":RETURN_TO_START,
                              "ideal_point_score":final_solution["ideal_point_score"],
                              "verify_seeds":str(VERIFY_ENERGY_SEEDS),
                              "verify_maxiter":VERIFY_ENERGY_MAXITER,
                              "verify_popsize":VERIFY_ENERGY_POPSIZE,
                              "energy_path_points":VERIFY_PATH_POINTS}])
    with pd.ExcelWriter(final_path) as writer:
        summary_df.to_excel(writer,sheet_name="summary",index=False)
        final_task_dataframe(final_solution).to_excel(writer,sheet_name="task_detail",index=False)
        pd.DataFrame(route_rows).to_excel(writer,sheet_name="route",index=False)
    return [state_path,combination_path,pareto_path,final_path]


def run_verify():
    """读取探索状态 -> 高精度核验 -> 重新枚举组合 -> 输出高精度折中方案。"""
    states_by_target=load_explore_selected_states()
    verified_by_target=verify_selected_states(states_by_target)
    global_solutions=enumerate_global_combinations(verified_by_target)
    if not global_solutions:
        raise RuntimeError("高精度核验后没有可行联合方案。")
    front=global_pareto(global_solutions)
    representatives=choose_representatives(front)
    final_solution=representatives["balanced"]
    print("\n"+"="*96)
    print("第四问：最终高精度联合优化结果")
    print("="*96)
    print(f"高精度全局组合方案数：{len(global_solutions)}")
    print(f"高精度全局Pareto点数：{len(front)}")
    print("\n【高精度精度优先方案】")
    print(f"总误差 = {representatives['accuracy']['total_error_mm']:.6f} mm")
    print(f"总能耗 = {representatives['accuracy']['total_energy_J']:.6f} J")
    print(f"底座总路程 = {representatives['accuracy']['route_distance_mm']:.0f} mm")
    print("\n【高精度能耗优先方案】")
    print(f"总误差 = {representatives['energy']['total_error_mm']:.6f} mm")
    print(f"总能耗 = {representatives['energy']['total_energy_J']:.6f} J")
    print(f"底座总路程 = {representatives['energy']['route_distance_mm']:.0f} mm")
    print("\n【最终高精度理想点折中方案】")
    print(f"货物访问顺序 = {' -> '.join(final_solution['order'])}")
    print(f"总末端误差 = {final_solution['total_error_mm']:.6f} mm")
    print(f"总机械能耗 = {final_solution['total_energy_J']:.6f} J")
    print(f"底座总移动距离 = {final_solution['route_distance_mm']:.0f} mm")
    print(f"理想点归一化距离 = {final_solution['ideal_point_score']:.6f}")
    print("\n五次抓取高精度详细方案：")
    for seq,target_name in enumerate(final_solution["order"],start=1):
        state=final_solution["state_by_target"][target_name]
        br,bc=state["base_cell"]
        print(f"{seq}. {target_name}: base=({br+1},{bc+1}), error={state['actual_error']:.3f} mm, "
              f"E={state['total_energy']:.6f} J, theta={state['theta']}")
    paths=save_high_precision_result(verified_by_target,global_solutions,front,representatives)
    print("\n高精度结果文件：")
    for path in paths:
        print(f"  {path.name}")


# ============================================================
# 26. 统一入口
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    mode=RUN_MODE.lower().strip()
    print("="*96)
    print("A题问题4 整体整合最终版")
    print(f"RUN_MODE = {RUN_MODE}")
    print(f"FAST_MODE = {FAST_MODE}")
    print(f"RETURN_TO_START = {RETURN_TO_START}")
    print("="*96)
    if mode=="explore":
        run_explore()
    elif mode=="verify":
        run_verify()
    elif mode=="all":
        run_explore()
        run_verify()
    else:
        raise ValueError(f"未知RUN_MODE={RUN_MODE!r}，只能是'explore'、'verify'或'all'。")


if __name__=="__main__":
    main()
