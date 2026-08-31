# ============================================================
# A题问题3 单文件正式版
# Phase 1：目标边长递减搜索 + S*-1 临界核验（求最小死区比例）
# Phase 2：固定临界边长 S* 后重新优化 HPWL
# 说明：Phase 1/2 均有缓存，中断后重跑自动续。
# ============================================================

import re, os, csv, json, math, pickle, random, shutil, statistics
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================
# 1. 参数
# ============================================================

CASES = ["n100", "n200", "n300"]
SEEDS = [2026, 2027, 2028, 42, 1107]
BASE_SEEDS = {"n100": 2028, "n200": 2028, "n300": 2028}
VERIFY_SEEDS = [2026, 2027, 2028, 42, 1107]
CPU_COUNT = os.cpu_count() or 4
MAX_WORKERS = min(3, max(1, CPU_COUNT // 4))
STAGE1_VERSION = "Q3_stage1B_target_descent_v1"
STAGE2_VERSION = "Q3_stage2_hpwl_v1"


# ============================================================
# 2. 数据结构
# ============================================================

@dataclass
class Block:
    name: str
    width: int
    height: int
    rotated: bool = False
    x: int = 0
    y: int = 0

    @property
    def w(self): return self.height if self.rotated else self.width
    @property
    def h(self): return self.width if self.rotated else self.height
    @property
    def area(self): return self.width * self.height


@dataclass
class TreeNode:
    block_idx: int
    parent: Optional[int] = None
    left: Optional[int] = None
    right: Optional[int] = None


def clone_blocks(blocks):
    return [Block(b.name, b.width, b.height, b.rotated, b.x, b.y) for b in blocks]


def clone_nodes(nodes):
    return [TreeNode(n.block_idx, n.parent, n.left, n.right) for n in nodes]


# ============================================================
# 3. B*-Tree + Skyline 解码
# ============================================================

def pack_bstar_tree(blocks, nodes, root):
    for b in blocks:
        b.x = b.y = 0
    skyline = [0] * (sum(max(b.width, b.height) for b in blocks) + 2)
    stack, placed = [root], 0
    while stack:
        u = stack.pop()
        node, b = nodes[u], blocks[nodes[u].block_idx]
        if node.parent is None:
            x = 0
        else:
            pnode, pb = nodes[node.parent], blocks[nodes[node.parent].block_idx]
            if pnode.left == u:
                x = pb.x + pb.w
            elif pnode.right == u:
                x = pb.x
            else:
                raise RuntimeError("B*-Tree 父子关系异常")
        y = max(skyline[x:x+b.w])
        b.x, b.y = x, y
        for xx in range(x, x+b.w):
            skyline[xx] = y + b.h
        placed += 1
        if node.right is not None:
            stack.append(node.right)
        if node.left is not None:
            stack.append(node.left)
    if placed != len(blocks):
        raise RuntimeError(f"仅摆放 {placed}/{len(blocks)} 个模块")
    return max(b.x+b.w for b in blocks), max(b.y+b.h for b in blocks)


# ============================================================
# 4. 数据读取（Phase 2 需要 .nets / .pl）
# ============================================================

def _clean_line(line):
    return line.split("//", 1)[0].strip()


def read_nets(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [_clean_line(line) for line in f]
    nets, i = [], 0
    while i < len(lines):
        m = re.search(r"NetDegree\s*[:：]\s*(\d+)", lines[i], flags=re.I)
        if not m:
            i += 1
            continue
        degree = int(m.group(1))
        pins, i = [], i + 1
        while i < len(lines) and len(pins) < degree:
            if lines[i]:
                pins.append(lines[i].split()[0])
            i += 1
        if len(pins) != degree:
            raise ValueError(f"NetDegree={degree}，实际只读取到 {len(pins)} 个引脚")
        nets.append(pins)
    return nets


def read_pl(file_path):
    terminals = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = _clean_line(raw)
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                x, y = float(parts[1]), float(parts[2])
            except ValueError:
                continue
            terminals[parts[0]] = (x, y)
    return terminals


def find_data_file(project_dir, case, suffix):
    candidates = [project_dir/"data"/f"{case}.{suffix}", project_dir/f"{case}.{suffix}"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"未找到 {case}.{suffix}，已检查：" + "；".join(str(p) for p in candidates))


def compile_nets(blocks, nets, terminal_pos):
    block_index = {b.name: i for i, b in enumerate(blocks)}
    compiled = []
    for net in nets:
        block_ids, fixed_points = [], []
        for name in net:
            if name in block_index:
                block_ids.append(block_index[name])
            elif name in terminal_pos:
                fixed_points.append(terminal_pos[name])
            else:
                raise KeyError(f"未知线网节点：{name}")
        compiled.append((block_ids, fixed_points))
    return compiled


# ============================================================
# 5. 邻域算子（两个 Phase 共用）
# ============================================================

def rotate_random_block(blocks):
    i = random.randrange(len(blocks))
    blocks[i].rotated = not blocks[i].rotated


def swap_random_blocks(nodes):
    i, j = random.sample(range(len(nodes)), 2)
    nodes[i].block_idx, nodes[j].block_idx = nodes[j].block_idx, nodes[i].block_idx


def get_subtree_nodes(nodes, subtree_root):
    result, stack = set(), [subtree_root]
    while stack:
        u = stack.pop()
        if u in result:
            continue
        result.add(u)
        if nodes[u].left is not None:
            stack.append(nodes[u].left)
        if nodes[u].right is not None:
            stack.append(nodes[u].right)
    return result


def _detach(nodes, moved):
    old_parent = nodes[moved].parent
    if old_parent is None:
        return None, None
    if nodes[old_parent].left == moved:
        old_side = "left"; nodes[old_parent].left = None
    elif nodes[old_parent].right == moved:
        old_side = "right"; nodes[old_parent].right = None
    else:
        raise RuntimeError("父子关系异常")
    nodes[moved].parent = None
    return old_parent, old_side


def _reattach(nodes, moved, old_parent, old_side, exclude_subtree=None):
    exclude_subtree = exclude_subtree or set()
    positions = []
    for target in range(len(nodes)):
        if target == moved or target in exclude_subtree:
            continue
        if nodes[target].left is None and not (target == old_parent and old_side == "left"):
            positions.append((target, "left"))
        if nodes[target].right is None and not (target == old_parent and old_side == "right"):
            positions.append((target, "right"))
    if not positions:
        if old_side == "left":
            nodes[old_parent].left = moved
        else:
            nodes[old_parent].right = moved
        nodes[moved].parent = old_parent
        return False
    target, side = random.choice(positions)
    if side == "left":
        nodes[target].left = moved
    else:
        nodes[target].right = moved
    nodes[moved].parent = target
    return True


def move_random_leaf(nodes, root):
    leaves = [i for i in range(len(nodes)) if i != root and nodes[i].left is None and nodes[i].right is None]
    if not leaves:
        return False
    moved = random.choice(leaves)
    old_parent, old_side = _detach(nodes, moved)
    if old_parent is None:
        return False
    return _reattach(nodes, moved, old_parent, old_side)


def move_random_subtree(nodes, root):
    movable = [i for i in range(len(nodes)) if i != root]
    if not movable:
        return False
    moved = random.choice(movable)
    subtree = get_subtree_nodes(nodes, moved)
    old_parent, old_side = _detach(nodes, moved)
    if old_parent is None:
        return False
    return _reattach(nodes, moved, old_parent, old_side, exclude_subtree=subtree)


def move_specific_subtree(nodes, root, moved):
    if moved == root:
        return False
    subtree = get_subtree_nodes(nodes, moved)
    old_parent, old_side = _detach(nodes, moved)
    if old_parent is None:
        return False
    return _reattach(nodes, moved, old_parent, old_side, exclude_subtree=subtree)


def swap_random_children(nodes):
    candidates = [i for i, node in enumerate(nodes) if node.left is not None or node.right is not None]
    if not candidates:
        return False
    u = random.choice(candidates)
    nodes[u].left, nodes[u].right = nodes[u].right, nodes[u].left
    return True


def swap_random_subtrees(nodes, root):
    candidates = [i for i in range(len(nodes)) if i != root]
    if len(candidates) < 2:
        return False
    for _ in range(20):
        u, v = random.sample(candidates, 2)
        if v in get_subtree_nodes(nodes, u) or u in get_subtree_nodes(nodes, v):
            continue
        pu, pv = nodes[u].parent, nodes[v].parent
        if pu is None or pv is None:
            continue
        side_u = "left" if nodes[pu].left == u else "right"
        side_v = "left" if nodes[pv].left == v else "right"
        if pu == pv:
            if side_u == side_v:
                continue
            nodes[pu].left, nodes[pu].right = nodes[pu].right, nodes[pu].left
            return True
        if side_u == "left":
            nodes[pu].left = v
        else:
            nodes[pu].right = v
        if side_v == "left":
            nodes[pv].left = u
        else:
            nodes[pv].right = u
        nodes[u].parent, nodes[v].parent = pv, pu
        return True
    return False


# ============================================================
# 6. Phase 1：可行边界搜索
# ============================================================

def metrics(blocks, nodes, root, L):
    W, H = pack_bstar_tree(blocks, nodes, root)
    ox, oy = max(0.0, W-L), max(0.0, H-L)
    return {"W": W, "H": H, "S": max(W, H), "area": W*H, "ox": ox, "oy": oy,
            "overflow": ox+oy, "feasible": W <= L and H <= L}


def boundary_energy(m):
    return m["ox"]**2 + m["oy"]**2 + 0.04*abs(m["W"]-m["H"])


def global_perturb(blocks, nodes, root):
    r = random.random()
    if r < 0.16:
        rotate_random_block(blocks)
    elif r < 0.36:
        swap_random_blocks(nodes)
    elif r < 0.56:
        if not move_random_leaf(nodes, root):
            swap_random_blocks(nodes)
    elif r < 0.72:
        if not move_random_subtree(nodes, root):
            swap_random_blocks(nodes)
    elif r < 0.86:
        if not swap_random_children(nodes):
            swap_random_blocks(nodes)
    else:
        if not swap_random_subtrees(nodes, root):
            swap_random_blocks(nodes)


def targeted_perturb(blocks, nodes, root, L):
    W, H = max(b.x+b.w for b in blocks), max(b.y+b.h for b in blocks)
    ox, oy = max(0.0, W-L), max(0.0, H-L)
    if random.random() < 0.40:  # 保留40%全局拓扑扰动
        global_perturb(blocks, nodes, root)
        return
    dim = "x" if ox >= oy else "y"
    edge = W if dim == "x" else H
    if dim == "x":
        critical = [i for i, b in enumerate(blocks) if b.x+b.w >= edge-3]
    else:
        critical = [i for i, b in enumerate(blocks) if b.y+b.h >= edge-3]
    if not critical:
        global_perturb(blocks, nodes, root)
        return
    bi = random.choice(critical)
    ni = next(i for i, node in enumerate(nodes) if node.block_idx == bi)
    r = random.random()
    b = blocks[bi]
    helpful_rotate = ((dim == "x" and b.w > b.h) or (dim == "y" and b.h > b.w))
    if r < 0.28 and helpful_rotate:
        b.rotated = not b.rotated
        return
    if r < 0.60:
        nj = random.randrange(len(nodes))
        if nj == ni:
            nj = (nj+1) % len(nodes)
        nodes[ni].block_idx, nodes[nj].block_idx = nodes[nj].block_idx, nodes[ni].block_idx
        return
    if r < 0.80 and ni != root:
        if move_specific_subtree(nodes, root, ni):
            return
    target = ni
    if nodes[target].left is None and nodes[target].right is None:
        target = nodes[target].parent
    if target is not None:
        nodes[target].left, nodes[target].right = nodes[target].right, nodes[target].left
        return
    global_perturb(blocks, nodes, root)


def search_target(base_blocks, base_nodes, root, L, seed, rounds=4, moves_per_temp=450):
    random.seed(seed)
    best_blocks, best_nodes = clone_blocks(base_blocks), clone_nodes(base_nodes)
    best = metrics(best_blocks, best_nodes, root, L)
    if best["feasible"]:
        return True, best_blocks, best_nodes, best
    for round_id in range(1, rounds+1):
        current_blocks, current_nodes = clone_blocks(best_blocks), clone_nodes(best_nodes)
        current = metrics(current_blocks, current_nodes, root, L)
        current_e, best_e = boundary_energy(current), boundary_energy(best)
        T = min(100.0, max(12.0, 0.75*best_e))
        level = 0
        while T > 0.30:
            level += 1
            for _ in range(moves_per_temp):
                cand_blocks, cand_nodes = clone_blocks(current_blocks), clone_nodes(current_nodes)
                targeted_perturb(cand_blocks, cand_nodes, root, L)
                cand = metrics(cand_blocks, cand_nodes, root, L)
                cand_e = boundary_energy(cand)
                delta = cand_e - current_e
                if delta <= 0 or random.random() < math.exp(-delta/T):
                    current_blocks, current_nodes, current, current_e = cand_blocks, cand_nodes, cand, cand_e
                cand_score = (cand["overflow"], cand["S"], abs(cand["W"]-cand["H"]), cand["area"])
                best_score = (best["overflow"], best["S"], abs(best["W"]-best["H"]), best["area"])
                if cand_score < best_score:
                    best_blocks, best_nodes, best = clone_blocks(cand_blocks), clone_nodes(cand_nodes), cand.copy()
                    best_e = cand_e
                    if best["feasible"]:
                        return True, best_blocks, best_nodes, best
            T *= 0.95
    return False, best_blocks, best_nodes, best


def try_target(base_blocks, base_nodes, root, target, seeds, case):
    best_failed = best_failed_blocks = best_failed_nodes = None
    for k, seed in enumerate(seeds, start=1):
        print(f"[{case}] target={target} | attempt={k}/{len(seeds)} | seed={seed}", flush=True)
        ok, blocks, nodes, m = search_target(base_blocks, base_nodes, root, target, seed,
                                             rounds=4, moves_per_temp=420 + len(base_blocks)//2)
        if ok:
            print(f"[{case}] target={target} 成功：{m['W']}×{m['H']}", flush=True)
            return True, blocks, nodes, m
        if (best_failed is None
                or (m["overflow"], m["S"], abs(m["W"]-m["H"]))
                < (best_failed["overflow"], best_failed["S"], abs(best_failed["W"]-best_failed["H"]))):
            best_failed, best_failed_blocks, best_failed_nodes = m, blocks, nodes
        print(f"[{case}] target={target} 暂未成功 | best={m['W']}×{m['H']} | overflow={m['overflow']:.3f}", flush=True)
    return False, best_failed_blocks, best_failed_nodes, best_failed


def run_case(case, project_dir):
    project_dir = Path(project_dir)
    q2_cache = project_dir/"output"/"问题2_三数据集并行"/"cache"/f"{case}_feasible.pkl"
    out_dir = project_dir/"output"/"问题3_Stage1B临界边长"/case
    out_dir.mkdir(parents=True, exist_ok=True)
    result_json, state_pkl = out_dir/f"{case}_Stage1B结果.json", out_dir/f"{case}_Stage1B最优状态.pkl"
    if result_json.exists() and state_pkl.exists():
        try:
            with open(result_json, "r", encoding="utf-8") as f:
                result = json.load(f)
            if result.get("version") == STAGE1_VERSION:
                print(f"[{case}] 读取 Stage1B 缓存：S*={result['S_star']}", flush=True)
                return result
        except Exception:
            pass
    if not q2_cache.exists():
        raise FileNotFoundError(f"未找到问题2缓存：{q2_cache}")
    with open(q2_cache, "rb") as f:
        cache = pickle.load(f)
    blocks, nodes, root = clone_blocks(cache["blocks"]), clone_nodes(cache["nodes"]), cache["root"]
    W0, H0 = pack_bstar_tree(blocks, nodes, root)
    current_S = max(W0, H0)
    total_area = sum(b.area for b in blocks)
    area_lb = math.ceil(math.sqrt(total_area))
    print("=" * 96, flush=True)
    print(f"[{case}] Stage1B 开始 | 起点={W0}×{H0} | S0={current_S} | 面积下界={area_lb}", flush=True)
    print("=" * 96, flush=True)
    current_blocks, current_nodes = clone_blocks(blocks), clone_nodes(nodes)
    history = []
    for step, attempts in [(5, 1), (2, 2), (1, 3)]:
        while current_S - step >= area_lb:
            target = current_S - step
            seeds = [BASE_SEEDS[case] + step*10000 + i*97 + target for i in range(attempts)]
            ok, new_blocks, new_nodes, m = try_target(current_blocks, current_nodes, root, target, seeds, case)
            history.append({"step": step, "target": target, "success": ok,
                            "best_W": m["W"], "best_H": m["H"], "overflow": m["overflow"]})
            if ok:
                current_blocks, current_nodes = clone_blocks(new_blocks), clone_nodes(new_nodes)
                current_S = max(m["W"], m["H"])
            else:
                break
    # S*-1 临界核验：5 seeds 全失败才停止
    while current_S - 1 >= area_lb:
        target = current_S - 1
        print(f"\n[{case}] 临界核验：target=S*-1={target}，执行 5 个独立 Seed", flush=True)
        ok, new_blocks, new_nodes, m = try_target(current_blocks, current_nodes, root, target, VERIFY_SEEDS, case)
        history.append({"step": "verify", "target": target, "success": ok,
                        "best_W": m["W"], "best_H": m["H"], "overflow": m["overflow"]})
        if ok:
            print(f"[{case}] S*-1 核验成功，说明还能继续压缩。", flush=True)
            current_blocks, current_nodes = clone_blocks(new_blocks), clone_nodes(new_nodes)
            current_S = max(m["W"], m["H"])
        else:
            print(f"[{case}] S*-1={target} 五次均未成功，将 S*={current_S} 作为临界边长。", flush=True)
            break
    final_W, final_H = pack_bstar_tree(current_blocks, current_nodes, root)
    d_min = current_S**2 / total_area - 1.0
    result = {"version": STAGE1_VERSION, "case": case, "block_area": total_area,
              "area_lower_bound_side": area_lb, "W": final_W, "H": final_H,
              "S_star": current_S, "dead_space_ratio": d_min, "gap_to_area_lb": current_S-area_lb,
              "history": history}
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(state_pkl, "wb") as f:
        pickle.dump({"version": STAGE1_VERSION, "case": case, "blocks": current_blocks,
                     "nodes": current_nodes, "root": root, "S_star": current_S,
                     "dead_space_ratio": d_min, "block_area": total_area}, f)
    save_layout_csv(current_blocks, out_dir/f"{case}_Stage1B最优坐标.csv")
    plot_layout(current_blocks, current_S, out_dir/f"{case}_Stage1B最优布局.png",
                f"Q3 {case} Critical Square - S={current_S}")
    print(f"\n[{case}] Stage1B 完成 | S*={current_S} | W×H={final_W}×{final_H} | d_min={d_min:.6%}", flush=True)
    return result


def main_phase1():
    project_dir = Path(__file__).resolve().parent
    root_out = project_dir/"output"/"问题3_Stage1B临界边长"
    root_out.mkdir(parents=True, exist_ok=True)
    print("=" * 108)
    print("A题问题3 Stage 1B：目标边长递减搜索 + S*-1 临界核验")
    print(f"CPU逻辑核心数：{CPU_COUNT} | 数据集并行进程数：{MAX_WORKERS}")
    print("策略：5 -> 2 -> 1 递减；最终 S*-1 采用 5 Seed 独立核验")
    print("=" * 108)
    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=mp.get_context("spawn")) as executor:
        futures = {executor.submit(run_case, case, str(project_dir)): case for case in CASES}
        for future in as_completed(futures):
            case = futures[future]
            result = future.result()
            results.append(result)
            print(f"[主进程] {case} 完成 | S*={result['S_star']} | d_min={result['dead_space_ratio']:.6%}", flush=True)
    results.sort(key=lambda r: CASES.index(r["case"]))
    print("\n" + "=" * 108)
    print("【问题3 Stage 1B 最终汇总】")
    print("=" * 108)
    for r in results:
        print(f"{r['case']:>5} | A={r['block_area']:>7} | S*={r['S_star']:>4} | W×H={r['W']}×{r['H']} | "
              f"d_min={r['dead_space_ratio']:.6%} | LB={r['area_lower_bound_side']} | gap={r['gap_to_area_lb']}")
    fields = ["case", "block_area", "area_lower_bound_side", "W", "H", "S_star",
              "dead_space_ratio", "gap_to_area_lb"]
    with open(root_out/"问题3_Stage1B最终汇总.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fields})
    print(f"\nStage 1B 完成。结果目录：{root_out}")


# ============================================================
# 7. Phase 2：固定 S* 后优化 HPWL
# ============================================================

def shape_metrics(blocks, nodes, root, L):
    W, H = pack_bstar_tree(blocks, nodes, root)
    return {"W": W, "H": H, "area": W*H, "feasible": W <= L and H <= L}


def calculate_hpwl(blocks, compiled_nets):
    total = 0.0
    for block_ids, fixed_points in compiled_nets:
        xmin = ymin = float("inf")
        xmax = ymax = float("-inf")
        for idx in block_ids:
            b = blocks[idx]
            x, y = b.x+b.w/2, b.y+b.h/2
            xmin = min(xmin, x); xmax = max(xmax, x)
            ymin = min(ymin, y); ymax = max(ymax, y)
        for x, y in fixed_points:
            xmin = min(xmin, x); xmax = max(xmax, x)
            ymin = min(ymin, y); ymax = max(ymax, y)
        total += (xmax-xmin) + (ymax-ymin)
    return total


def is_overlap(a, b):
    return not (a.x+a.w <= b.x or b.x+b.w <= a.x or a.y+a.h <= b.y or b.y+b.h <= a.y)


def validate_layout(blocks, L):
    overlap = [(blocks[i].name, blocks[j].name)
               for i in range(len(blocks)) for j in range(i+1, len(blocks))
               if is_overlap(blocks[i], blocks[j])]
    out = [b.name for b in blocks if b.x < 0 or b.y < 0 or b.x+b.w > L or b.y+b.h > L]
    return overlap, out


def get_adaptive_probabilities(temperature_ratio):
    if temperature_ratio > 0.60:
        return 0.15, 0.25, 0.35, 0.25
    elif temperature_ratio > 0.20:
        return 0.20, 0.30, 0.35, 0.15
    return 0.30, 0.35, 0.30, 0.05


def adaptive_perturb(blocks, nodes, root, temperature_ratio):
    p1, p2, p3, _ = get_adaptive_probabilities(temperature_ratio)
    r = random.random()
    if r < p1:
        rotate_random_block(blocks); return "rotate"
    elif r < p1+p2:
        swap_random_blocks(nodes); return "swap"
    elif r < p1+p2+p3:
        if move_random_leaf(nodes, root):
            return "leaf_move"
        swap_random_blocks(nodes); return "swap"
    else:
        if move_random_subtree(nodes, root):
            return "subtree_move"
        if move_random_leaf(nodes, root):
            return "leaf_move"
        swap_random_blocks(nodes); return "swap"


def local_perturb(blocks, nodes, root, temperature_ratio=None):
    r = random.random()
    if r < 0.25:
        rotate_random_block(blocks); return "rotate"
    elif r < 0.60:
        swap_random_blocks(nodes); return "swap"
    else:
        if move_random_leaf(nodes, root):
            return "leaf_move"
        swap_random_blocks(nodes); return "swap"


def hpwl_sa(blocks, nodes, root, L, compiled_nets, seed, perturb_fun,
            initial_temp_ratio, cooling_rate, min_temp, moves_per_temp, feasible_retries, prefix):
    random.seed(seed)
    current_blocks, current_nodes = clone_blocks(blocks), clone_nodes(nodes)
    current_shape = shape_metrics(current_blocks, current_nodes, root, L)
    if not current_shape["feasible"]:
        raise ValueError(f"HPWL-SA 起点不可行：{current_shape['W']}×{current_shape['H']} > {L}")
    current_hpwl = calculate_hpwl(current_blocks, compiled_nets)
    best_blocks, best_nodes = clone_blocks(current_blocks), clone_nodes(current_nodes)
    best_shape, best_hpwl = current_shape.copy(), current_hpwl
    T0 = max(1.0, initial_temp_ratio*current_hpwl)
    T = T0
    level, history = 0, [(0, best_hpwl)]
    while T > min_temp:
        level += 1
        feasible_count = accepted = proposal_count = 0
        for _ in range(moves_per_temp):
            found_feasible = False
            for _try in range(feasible_retries):
                proposal_count += 1
                cand_blocks, cand_nodes = clone_blocks(current_blocks), clone_nodes(current_nodes)
                perturb_fun(cand_blocks, cand_nodes, root, T/T0)
                cand_shape = shape_metrics(cand_blocks, cand_nodes, root, L)
                if not cand_shape["feasible"]:
                    continue
                found_feasible = True
                feasible_count += 1
                break
            if not found_feasible:
                continue
            cand_hpwl = calculate_hpwl(cand_blocks, compiled_nets)
            delta = cand_hpwl - current_hpwl
            if delta <= 0 or random.random() < math.exp(-delta/T):
                current_blocks, current_nodes = cand_blocks, cand_nodes
                current_shape, current_hpwl = cand_shape, cand_hpwl
                accepted += 1
            if cand_hpwl < best_hpwl:
                best_blocks, best_nodes = clone_blocks(cand_blocks), clone_nodes(cand_nodes)
                best_shape, best_hpwl = cand_shape.copy(), cand_hpwl
        history.append((level, best_hpwl))
        if level == 1 or level % 20 == 0:
            print(f"{prefix} level={level:3d} | bestHPWL={best_hpwl:12.1f} | "
                  f"feasibleMove={feasible_count/moves_per_temp:.3f} | "
                  f"proposalOK={feasible_count/proposal_count if proposal_count else 0.0:.3f} | "
                  f"accepted={accepted/moves_per_temp:.3f}", flush=True)
        T *= cooling_rate
    return best_blocks, best_nodes, best_shape, best_hpwl, history


def hpwl_worker(case, seed, project_dir):
    project_dir = Path(project_dir)
    q3_dir = project_dir/"output"/"问题3_Stage1B临界边长"/case
    state_path, result_path = q3_dir/f"{case}_Stage1B最优状态.pkl", q3_dir/f"{case}_Stage1B结果.json"
    if not state_path.exists() or not result_path.exists():
        raise FileNotFoundError(f"未找到第三问 Stage1B 状态/结果：{q3_dir}")
    out_dir = project_dir/"output"/"问题3_Stage2最小死区HPWL"/case/"seed_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_cache, state_cache = out_dir/f"{case}_seed{seed}_result.json", out_dir/f"{case}_seed{seed}_state.pkl"
    layout_path, coord_path, conv_path = (out_dir/f"{case}_seed{seed}_最终布局.png",
                                          out_dir/f"{case}_seed{seed}_最终坐标.csv",
                                          out_dir/f"{case}_seed{seed}_收敛曲线.png")
    if result_cache.exists() and state_cache.exists() and layout_path.exists() and coord_path.exists():
        try:
            with open(result_cache, "r", encoding="utf-8") as f:
                result = json.load(f)
            if result.get("version") == STAGE2_VERSION:
                print(f"[{case}|seed={seed}] 读取缓存：HPWL={result['final_hpwl']:.1f}", flush=True)
                return result
        except Exception:
            pass
    with open(state_path, "rb") as f:
        state = pickle.load(f)
    with open(result_path, "r", encoding="utf-8") as f:
        stage1_result = json.load(f)
    blocks, nodes, root = clone_blocks(state["blocks"]), clone_nodes(state["nodes"]), state["root"]
    L = float(stage1_result["S_star"])
    dead_space_ratio = float(stage1_result["dead_space_ratio"])
    nets = read_nets(find_data_file(project_dir, case, "nets"))
    terminal_pos = read_pl(find_data_file(project_dir, case, "pl"))
    compiled_nets = compile_nets(blocks, nets, terminal_pos)
    initial_shape = shape_metrics(blocks, nodes, root, L)
    if not initial_shape["feasible"]:
        raise RuntimeError(f"[{case}] Stage1B 状态异常：{initial_shape['W']}×{initial_shape['H']} > {L}")
    overlap, out = validate_layout(blocks, L)
    if overlap or out:
        raise RuntimeError(f"[{case}] Stage1B 合法性失败：overlap={len(overlap)} out={len(out)}")
    initial_hpwl = calculate_hpwl(blocks, compiled_nets)
    print(f"\n[{case}|seed={seed}] L={L:.0f} | d_min={dead_space_ratio:.6%} | "
          f"起点={initial_shape['W']}×{initial_shape['H']} | Initial HPWL={initial_hpwl:.1f}", flush=True)
    g_blocks, g_nodes, g_shape, g_hpwl, g_history = hpwl_sa(
        blocks, nodes, root, L, compiled_nets, seed=seed, perturb_fun=adaptive_perturb,
        initial_temp_ratio=0.02, cooling_rate=0.93, min_temp=1.0, moves_per_temp=300,
        feasible_retries=3, prefix=f"[{case}|seed={seed}|Global]")
    f_blocks, f_nodes, f_shape, f_hpwl, p_history = hpwl_sa(
        g_blocks, g_nodes, root, L, compiled_nets, seed=seed+100000, perturb_fun=local_perturb,
        initial_temp_ratio=0.003, cooling_rate=0.95, min_temp=1.0, moves_per_temp=400,
        feasible_retries=4, prefix=f"[{case}|seed={seed}|Polish]")
    overlap, out = validate_layout(f_blocks, L)
    if overlap or out:
        raise RuntimeError(f"[{case}|seed={seed}] 最终合法性失败：overlap={len(overlap)} out={len(out)}")
    save_layout_csv(f_blocks, coord_path)
    plot_layout(f_blocks, L, layout_path, f"Q3 {case} Minimum Dead Space HPWL - Seed {seed}")
    merged_history = [(x, y) for x, y in g_history]
    offset = g_history[-1][0] if g_history else 0
    merged_history += [(offset+x, y) for x, y in p_history[1:]]
    plot_convergence(merged_history, conv_path, f"Q3 {case} HPWL Convergence - Seed {seed}")
    result = {"version": STAGE2_VERSION, "case": case, "seed": seed, "L": L,
              "dead_space_ratio": dead_space_ratio, "initial_hpwl": initial_hpwl,
              "global_hpwl": g_hpwl, "final_hpwl": f_hpwl,
              "final_width": f_shape["W"], "final_height": f_shape["H"],
              "global_improvement": (initial_hpwl-g_hpwl)/initial_hpwl,
              "polish_improvement": (g_hpwl-f_hpwl)/g_hpwl,
              "total_improvement": (initial_hpwl-f_hpwl)/initial_hpwl,
              "overlap_count": len(overlap), "out_count": len(out)}
    with open(result_cache, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(state_cache, "wb") as f:
        pickle.dump({"version": STAGE2_VERSION, "case": case, "seed": seed, "blocks": f_blocks,
                     "nodes": f_nodes, "root": root, "L": L,
                     "dead_space_ratio": dead_space_ratio, "final_hpwl": f_hpwl}, f)
    print(f"[{case}|seed={seed}] 完成 | HPWL={f_hpwl:.1f} | W×H={f_shape['W']}×{f_shape['H']} | overlap=0 out=0", flush=True)
    return result


# ============================================================
# 8. 绘图与保存（两 Phase 共用）
# ============================================================

def save_layout_csv(blocks, save_path):
    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "x", "y", "width", "height", "rotated"])
        for b in blocks:
            writer.writerow([b.name, b.x, b.y, b.w, b.h, int(b.rotated)])


def plot_layout(blocks, L, save_path, title):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.add_patch(Rectangle((0, 0), L, L, fill=False, linewidth=2.0, linestyle="--"))
    for b in blocks:
        ax.add_patch(Rectangle((b.x, b.y), b.w, b.h, fill=False, linewidth=0.65))
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(save_path, dpi=220); plt.close()


def plot_convergence(history, save_path, title):
    plt.figure(figsize=(10, 6))
    plt.plot([x for x, _ in history], [y for _, y in history], linewidth=1.2)
    plt.xlabel("Temperature Level"); plt.ylabel("Best HPWL"); plt.title(title)
    plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(save_path, dpi=220); plt.close()


def save_results_csv(results, save_path):
    fields = ["case", "seed", "L", "dead_space_ratio", "initial_hpwl", "global_hpwl", "final_hpwl",
              "final_width", "final_height", "global_improvement",
              "polish_improvement", "total_improvement", "overlap_count", "out_count"]
    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fields})


def main_phase2():
    project_dir = Path(__file__).resolve().parent
    root_out = project_dir/"output"/"问题3_Stage2最小死区HPWL"
    root_out.mkdir(parents=True, exist_ok=True)
    print("=" * 112)
    print("A题问题3 Stage 2：固定临界最小 dead space ratio 后重新优化 HPWL")
    print(f"CPU逻辑核心数：{CPU_COUNT} | 最大并行进程数：{MAX_WORKERS}")
    print("策略：三组临界布局各执行 5 Seed Global SA + Local Polish")
    print("=" * 112)
    tasks = [(case, seed) for case in CASES for seed in SEEDS]
    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=mp.get_context("spawn")) as executor:
        futures = {executor.submit(hpwl_worker, case, seed, str(project_dir)): (case, seed)
                   for case, seed in tasks}
        done = 0
        for future in as_completed(futures):
            case, seed = futures[future]
            result = future.result()
            results.append(result)
            done += 1
            print(f"[主进程] 完成 {done}/{len(tasks)} | {case} Seed={seed} | Final HPWL={result['final_hpwl']:.1f}", flush=True)
    results.sort(key=lambda r: (CASES.index(r["case"]), SEEDS.index(r["seed"])))
    save_results_csv(results, root_out/"问题3_Stage2全部15次结果.csv")
    final_summary = []
    print("\n" + "=" * 112)
    print("【问题3 Stage 2 最终汇总】")
    print("=" * 112)
    for case in CASES:
        case_results = [r for r in results if r["case"] == case]
        if len(case_results) != len(SEEDS):
            raise RuntimeError(f"{case} 结果数量异常：{len(case_results)}/{len(SEEDS)}")
        best = min(case_results, key=lambda r: r["final_hpwl"])
        finals = [r["final_hpwl"] for r in case_results]
        mean_hpwl, std_hpwl = statistics.mean(finals), statistics.pstdev(finals)
        cv = std_hpwl/mean_hpwl if mean_hpwl else 0.0
        print(f"\n[{case}]")
        print(f"{'Seed':>8} {'Initial':>12} {'Global':>12} {'Final':>12} {'W':>6} {'H':>6} {'Improve':>10}")
        for r in case_results:
            print(f"{r['seed']:>8} {r['initial_hpwl']:>12.1f} {r['global_hpwl']:>12.1f} "
                  f"{r['final_hpwl']:>12.1f} {r['final_width']:>6} {r['final_height']:>6} "
                  f"{r['total_improvement']:>9.4%}")
        print(f"稳定性：mean={mean_hpwl:.2f} | std={std_hpwl:.2f} | CV={cv:.4%}")
        print(f"最优：Seed={best['seed']} | HPWL={best['final_hpwl']:.3f} | W×H={best['final_width']}×{best['final_height']}")
        final_summary.append({"case": case, "best_seed": best["seed"], "L": best["L"],
                              "dead_space_ratio": best["dead_space_ratio"], "best_hpwl": best["final_hpwl"],
                              "W": best["final_width"], "H": best["final_height"],
                              "mean_hpwl": mean_hpwl, "std_hpwl": std_hpwl, "cv": cv,
                              "overlap_count": best["overlap_count"], "out_count": best["out_count"]})
    with open(root_out/"问题3_Stage2最终汇总.csv", "w", newline="", encoding="utf-8-sig") as f:
        fields = ["case", "best_seed", "L", "dead_space_ratio", "best_hpwl", "W", "H",
                  "mean_hpwl", "std_hpwl", "cv", "overlap_count", "out_count"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(final_summary)
    print(f"\nStage 2 完成。结果目录：{root_out}")


if __name__ == "__main__":
    mp.freeze_support()
    main_phase1()   # Phase 1：求最小死区比例 S*
    main_phase2()   # Phase 2：固定 S* 优化 HPWL
