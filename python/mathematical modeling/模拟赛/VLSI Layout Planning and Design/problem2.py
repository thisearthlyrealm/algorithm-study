# ============================================================
# A题问题2：n100 / n200 / n300 三数据集多核并行
# 方法：每组构造一次固定轮廓可行初解 + 5随机种子并行 HPWL Global SA + Local Polish
# 特点：Windows多进程 / 自动缓存 / 中断续跑 / 最终合法性核验
# ============================================================

import re, csv, math, os, json, pickle, random, shutil, statistics
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
# 1. 全局参数
# ============================================================

CASES = ["n100", "n200", "n300"]
SEEDS = [2026, 2027, 2028, 42, 1107]
CACHE_VERSION = "Q2_parallel_v1"
PREP_SEEDS = {
    "n100": [2028, 42, 2027, 1107, 2026],
    "n200": [2028, 1107, 42, 2026, 2027],
    "n300": [2028, 2027, 42, 1107, 2026],
}
CPU_COUNT = os.cpu_count() or 4
MAX_WORKERS = max(1, min(6, max(1, CPU_COUNT // 2)))


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
    def w(self) -> int: return self.height if self.rotated else self.width
    @property
    def h(self) -> int: return self.width if self.rotated else self.height
    @property
    def area(self) -> int: return self.width * self.height


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
# 3. 读取 .blocks / .nets / .pl
# ============================================================

def _clean_line(line: str) -> str:
    return line.split("//", 1)[0].strip()


def read_blocks(file_path: str):
    blocks, terminals = [], set()
    with open(file_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = _clean_line(raw)
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            if parts[1] == "block":
                coords = re.findall(r"\((-?\d+)\s*,\s*(-?\d+)\)", line)
                if len(coords) != 4:
                    raise ValueError(f"无法解析 HardBlock：{line}")
                xs = [int(x) for x, _ in coords]
                ys = [int(y) for _, y in coords]
                blocks.append(Block(parts[0], max(xs)-min(xs), max(ys)-min(ys)))
            elif parts[1] == "terminal":
                terminals.add(parts[0])
    return blocks, terminals


def read_nets(file_path: str) -> list[list[str]]:
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


def read_pl(file_path: str) -> dict[str, tuple[float, float]]:
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
# 4. B*-Tree 构造与 Skyline 解码
# ============================================================

def build_random_tree(blocks, rng):
    order = list(range(len(blocks)))
    rng.shuffle(order)
    nodes = [TreeNode(block_idx=i) for i in order]
    root = 0
    slots = [(root, "left"), (root, "right")]
    for idx in range(1, len(nodes)):
        parent, side = slots.pop(rng.randrange(len(slots)))
        if side == "left":
            nodes[parent].left = idx
        else:
            nodes[parent].right = idx
        nodes[idx].parent = parent
        slots.append((idx, "left"))
        slots.append((idx, "right"))
    return nodes, root


def pack_bstar_tree(blocks, nodes, root):
    for b in blocks:
        b.x = b.y = 0
    skyline = [0] * (sum(max(b.width, b.height) for b in blocks) + 2)
    stack, placed = [root], 0
    while stack:
        node_idx = stack.pop()
        node = nodes[node_idx]
        block = blocks[node.block_idx]
        if node.parent is None:
            x = 0
        else:
            pnode, pblock = nodes[node.parent], blocks[nodes[node.parent].block_idx]
            if pnode.left == node_idx:
                x = pblock.x + pblock.w
            elif pnode.right == node_idx:
                x = pblock.x
            else:
                raise RuntimeError("B*-Tree 父子关系异常")
        y = max(skyline[x:x+block.w])
        block.x, block.y = x, y
        for xx in range(x, x+block.w):
            skyline[xx] = y + block.h
        placed += 1
        if node.right is not None:
            stack.append(node.right)
        if node.left is not None:
            stack.append(node.left)
    if placed != len(blocks):
        raise RuntimeError(f"只摆放了 {placed}/{len(blocks)} 个模块")
    return max(b.x+b.w for b in blocks), max(b.y+b.h for b in blocks)


# ============================================================
# 5. 固定轮廓、HPWL 与合法性
# ============================================================

def chip_side(blocks, dead_space_ratio=0.15):
    return math.sqrt(sum(b.area for b in blocks) * (1.0 + dead_space_ratio))


def shape_metrics(blocks, nodes, root, L):
    W, H = pack_bstar_tree(blocks, nodes, root)
    return {"W": W, "H": H, "area": W*H, "ratio": max(W, H)/min(W, H),
            "max_dim": max(W, H), "overflow": max(0.0, W-L)+max(0.0, H-L),
            "feasible": W <= L and H <= L}


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
        total += (xmax-xmin)+(ymax-ymin)
    return total


def is_overlap(a, b):
    return not (a.x+a.w <= b.x or b.x+b.w <= a.x or a.y+a.h <= b.y or b.y+b.h <= a.y)


def validate_layout(blocks, L):
    overlap = [(blocks[i].name, blocks[j].name)
               for i in range(len(blocks)) for j in range(i+1, len(blocks))
               if is_overlap(blocks[i], blocks[j])]
    out = [b.name for b in blocks if b.x < 0 or b.y < 0 or b.x+b.w > L or b.y+b.h > L]
    return overlap, out


# ============================================================
# 6. 四类邻域
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
    if exclude_subtree is None:
        exclude_subtree = set()
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
    leaves = [i for i in range(len(nodes))
              if i != root and nodes[i].left is None and nodes[i].right is None]
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


def swap_random_children(nodes):
    """交换随机节点的左右孩子，安全改变 B*-Tree 几何关系。"""
    candidates = [i for i, node in enumerate(nodes) if node.left is not None or node.right is not None]
    if not candidates:
        return False
    u = random.choice(candidates)
    nodes[u].left, nodes[u].right = nodes[u].right, nodes[u].left
    return True


def swap_random_subtrees(nodes, root):
    """交换两棵互不包含的非根子树。"""
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


def boundary_perturb(blocks, nodes, root):
    """固定正方形可行性专用六邻域，强化拓扑重排能力。"""
    r = random.random()
    if r < 0.15:
        rotate_random_block(blocks); return "rotate"
    elif r < 0.35:
        swap_random_blocks(nodes); return "swap"
    elif r < 0.50:
        if move_random_leaf(nodes, root):
            return "leaf_move"
    elif r < 0.65:
        if move_random_subtree(nodes, root):
            return "subtree_move"
    elif r < 0.82:
        if swap_random_children(nodes):
            return "child_mirror"
    else:
        if swap_random_subtrees(nodes, root):
            return "subtree_swap"
    swap_random_blocks(nodes)
    return "swap"


# ============================================================
# 7. 邻域选择
# ============================================================

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


# ============================================================
# 8. V3 边界可行搜索辅助
# ============================================================

def move_specific_subtree(nodes, root, moved):
    if moved == root:
        return False
    subtree = get_subtree_nodes(nodes, moved)
    old_parent, old_side = _detach(nodes, moved)
    if old_parent is None:
        return False
    return _reattach(nodes, moved, old_parent, old_side, exclude_subtree=subtree)


def boundary_overflow_parts(W, H, L):
    return max(0.0, W-L), max(0.0, H-L)


def boundary_energy(metrics, L):
    """固定轮廓阶段只关心超界量：E=ox^2+oy^2+极小破平局项。"""
    ox, oy = boundary_overflow_parts(metrics["W"], metrics["H"], L)
    return ox*ox + oy*oy + 1e-4*metrics["max_dim"] + 1e-8*metrics["area"]


def targeted_boundary_perturb(blocks, nodes, root, L):
    """优先扰动造成 W/H 超界的边界模块，剩余概率执行全局六邻域。"""
    W = max(b.x+b.w for b in blocks)
    H = max(b.y+b.h for b in blocks)
    ox, oy = boundary_overflow_parts(W, H, L)
    if ox <= 0 and oy <= 0:
        return boundary_perturb(blocks, nodes, root)
    if random.random() < 0.45:
        return boundary_perturb(blocks, nodes, root)
    dim = "x" if ox >= oy else "y"
    edge = W if dim == "x" else H
    tol = 3
    if dim == "x":
        critical_blocks = [i for i, b in enumerate(blocks) if b.x+b.w >= edge-tol]
    else:
        critical_blocks = [i for i, b in enumerate(blocks) if b.y+b.h >= edge-tol]
    if not critical_blocks:
        return boundary_perturb(blocks, nodes, root)
    block_idx = random.choice(critical_blocks)
    node_idx = next(i for i, node in enumerate(nodes) if node.block_idx == block_idx)
    r = random.random()
    b = blocks[block_idx]
    helpful_rotate = ((dim == "x" and b.w > b.h) or (dim == "y" and b.h > b.w))
    if r < 0.30 and helpful_rotate:
        b.rotated = not b.rotated
        return "critical_rotate"
    if r < 0.62:
        other = random.randrange(len(nodes))
        if other == node_idx:
            other = (other+1) % len(nodes)
        nodes[node_idx].block_idx, nodes[other].block_idx = nodes[other].block_idx, nodes[node_idx].block_idx
        return "critical_swap"
    if r < 0.82 and node_idx != root:
        if move_specific_subtree(nodes, root, node_idx):
            return "critical_subtree_move"
    target = node_idx
    if nodes[target].left is None and nodes[target].right is None:
        target = nodes[target].parent if nodes[target].parent is not None else root
    if target is not None:
        nodes[target].left, nodes[target].right = nodes[target].right, nodes[target].left
        return "critical_mirror"
    return boundary_perturb(blocks, nodes, root)


def greedy_boundary_polish(blocks, nodes, root, L, trials=40):
    """对当前最好解做一次便宜的临界边界贪心精修。"""
    base_blocks, base_nodes = clone_blocks(blocks), clone_nodes(nodes)
    base = shape_metrics(base_blocks, base_nodes, root, L)
    best_blocks, best_nodes, best = base_blocks, base_nodes, base
    W, H = base["W"], base["H"]
    ox, oy = boundary_overflow_parts(W, H, L)
    dim = "x" if ox >= oy else "y"
    edge = W if dim == "x" else H
    critical = [i for i, b in enumerate(base_blocks)
                if ((b.x+b.w >= edge-2) if dim == "x" else (b.y+b.h >= edge-2))]
    # 先枚举临界模块旋转
    for bi in critical:
        cand_blocks, cand_nodes = clone_blocks(base_blocks), clone_nodes(base_nodes)
        cand_blocks[bi].rotated = not cand_blocks[bi].rotated
        cand = shape_metrics(cand_blocks, cand_nodes, root, L)
        if (cand["overflow"], cand["max_dim"], cand["area"]) < (best["overflow"], best["max_dim"], best["area"]):
            best_blocks, best_nodes, best = cand_blocks, cand_nodes, cand
    # 再随机枚举若干"临界模块-普通模块"交换
    if critical:
        for _ in range(trials):
            bi = random.choice(critical)
            ni = next(i for i, node in enumerate(base_nodes) if node.block_idx == bi)
            nj = random.randrange(len(base_nodes))
            if ni == nj:
                continue
            cand_blocks, cand_nodes = clone_blocks(base_blocks), clone_nodes(base_nodes)
            cand_nodes[ni].block_idx, cand_nodes[nj].block_idx = cand_nodes[nj].block_idx, cand_nodes[ni].block_idx
            cand = shape_metrics(cand_blocks, cand_nodes, root, L)
            if (cand["overflow"], cand["max_dim"], cand["area"]) < (best["overflow"], best["max_dim"], best["area"]):
                best_blocks, best_nodes, best = cand_blocks, cand_nodes, cand
    return best_blocks, best_nodes, best


# ============================================================
# 9. 紧凑化 SA：只用于构造一次可行初解
# ============================================================

def compact_energy(area, ratio):
    return area + 0.5*(ratio-1.0)/(ratio+1.0)


def compact_metrics(blocks, nodes, root, L):
    m = shape_metrics(blocks, nodes, root, L)
    m["energy"] = compact_energy(m["area"], m["ratio"])
    return m


def select_initial_tree(blocks, seed, L, trials=20):
    rng = random.Random(seed)
    work = clone_blocks(blocks)
    best_nodes = best_root = best = None
    for _ in range(trials):
        nodes, root = build_random_tree(work, rng)
        m = compact_metrics(work, nodes, root, L)
        score = (m["area"], m["ratio"])
        if best is None or score < (best["area"], best["ratio"]):
            best_nodes, best_root, best = clone_nodes(nodes), root, m.copy()
    final = compact_metrics(blocks, best_nodes, best_root, L)
    return best_nodes, best_root, final


def compact_sa(blocks, nodes, root, L, seed, T0, min_temp,
               cooling_rate, moves_per_temp, perturb_fun, prefix):
    random.seed(seed)
    current_blocks, current_nodes = clone_blocks(blocks), clone_nodes(nodes)
    current = compact_metrics(current_blocks, current_nodes, root, L)
    best_blocks, best_nodes, best = clone_blocks(current_blocks), clone_nodes(current_nodes), current.copy()
    T, level = T0, 0
    while T > min_temp:
        level += 1
        for _ in range(moves_per_temp):
            cand_blocks, cand_nodes = clone_blocks(current_blocks), clone_nodes(current_nodes)
            perturb_fun(cand_blocks, cand_nodes, root, T/T0)
            cand = compact_metrics(cand_blocks, cand_nodes, root, L)
            delta = cand["energy"]-current["energy"]
            if delta <= 0 or random.random() < math.exp(-delta/T):
                current_blocks, current_nodes, current = cand_blocks, cand_nodes, cand
            if (cand["area"], cand["ratio"]) < (best["area"], best["ratio"]):
                best_blocks, best_nodes, best = clone_blocks(cand_blocks), clone_nodes(cand_nodes), cand.copy()
        if level == 1 or level % 20 == 0:
            print(f"{prefix} Compact level={level:3d} | best={best['W']}×{best['H']} | "
                  f"A={best['area']} | feasible={best['feasible']}", flush=True)
        T *= cooling_rate
    return best_blocks, best_nodes, best


def boundary_polish(blocks, nodes, root, L, seed, prefix):
    """
    V3 固定轮廓搜索：目标改为平方超界量 ox^2+oy^2；
    55% 邻域针对超界临界模块；每10层贪心精修；4次Reheat。
    """
    best_blocks, best_nodes = clone_blocks(blocks), clone_nodes(nodes)
    best = shape_metrics(best_blocks, best_nodes, root, L)
    if best["feasible"]:
        return best_blocks, best_nodes, best
    rounds, moves_per_temp = 4, 450 + len(blocks)
    for round_id in range(1, rounds+1):
        random.seed(seed + round_id*10000)
        current_blocks, current_nodes = clone_blocks(best_blocks), clone_nodes(best_nodes)
        current = shape_metrics(current_blocks, current_nodes, root, L)
        shake_count = 10 + 5*round_id
        for _ in range(shake_count):
            boundary_perturb(current_blocks, current_nodes, root)
        current = shape_metrics(current_blocks, current_nodes, root, L)
        best_e, current_e = boundary_energy(best, L), boundary_energy(current, L)
        T0 = max(8.0, 0.60*max(best_e, current_e))
        T, level = T0, 0
        print(f"{prefix} Boundary-V3 {round_id}/{rounds} | best={best['W']}×{best['H']} | "
              f"overflow={best['overflow']:.3f} | T0={T0:.2f}", flush=True)
        while T > 0.12:
            level += 1
            for _ in range(moves_per_temp):
                cand_blocks, cand_nodes = clone_blocks(current_blocks), clone_nodes(current_nodes)
                targeted_boundary_perturb(cand_blocks, cand_nodes, root, L)
                cand = shape_metrics(cand_blocks, cand_nodes, root, L)
                cand_e = boundary_energy(cand, L)
                delta = cand_e-current_e
                if delta <= 0 or random.random() < math.exp(-delta/T):
                    current_blocks, current_nodes, current, current_e = cand_blocks, cand_nodes, cand, cand_e
                if (cand["overflow"], cand["max_dim"], cand["area"]) < (best["overflow"], best["max_dim"], best["area"]):
                    best_blocks, best_nodes, best = clone_blocks(cand_blocks), clone_nodes(cand_nodes), cand.copy()
                    best_e = cand_e
                    if best["feasible"]:
                        print(f"{prefix} Boundary-V3 找到可行布局：{best['W']}×{best['H']} | L={L:.3f}", flush=True)
                        return best_blocks, best_nodes, best
            if level % 10 == 0:
                g_blocks, g_nodes, g = greedy_boundary_polish(best_blocks, best_nodes, root, L, trials=30)
                if (g["overflow"], g["max_dim"], g["area"]) < (best["overflow"], best["max_dim"], best["area"]):
                    best_blocks, best_nodes, best = g_blocks, g_nodes, g
                    best_e = boundary_energy(best, L)
                    current_blocks, current_nodes = clone_blocks(best_blocks), clone_nodes(best_nodes)
                    current, current_e = best.copy(), best_e
                    if best["feasible"]:
                        print(f"{prefix} Greedy 找到可行布局：{best['W']}×{best['H']} | L={L:.3f}", flush=True)
                        return best_blocks, best_nodes, best
            if level == 1 or level % 20 == 0:
                ox, oy = boundary_overflow_parts(best["W"], best["H"], L)
                print(f"{prefix} Boundary-V3-R{round_id} level={level:3d} | best={best['W']}×{best['H']} | "
                      f"ox={ox:.3f} oy={oy:.3f} | E={best_e:.3f}", flush=True)
            T *= 0.95
        print(f"{prefix} Boundary-V3 Reheat {round_id} 结束 | 全局最好={best['W']}×{best['H']} | "
              f"overflow={best['overflow']:.3f}", flush=True)
    if best["feasible"]:
        return best_blocks, best_nodes, best
    raise RuntimeError(f"V3 Boundary 搜索仍未找到可行布局：W={best['W']} H={best['H']} L={L:.3f} overflow={best['overflow']:.3f}")


def prepare_from_seed(blocks, L, seed, prefix):
    work = clone_blocks(blocks)
    nodes, root, start = select_initial_tree(work, seed, L, trials=20)
    T0 = max(1.0, 0.02*start["area"])
    g_blocks, g_nodes, g = compact_sa(work, nodes, root, L, seed, T0, 3.0, 0.92, 300, adaptive_perturb, prefix)
    p_blocks, p_nodes, p = compact_sa(g_blocks, g_nodes, root, L, seed+100000,
                                      max(1.0, 0.002*g["area"]), 1.0, 0.95, 500, local_perturb, prefix)
    if p["feasible"]:
        return p_blocks, p_nodes, root, p
    b_blocks, b_nodes, b = boundary_polish(p_blocks, p_nodes, root, L, seed+200000, prefix)
    return b_blocks, b_nodes, root, b


# ============================================================
# 9. 可行初解缓存
# ============================================================

def input_signature(data_dir, case):
    sig = {}
    for ext in ["blocks", "nets", "pl"]:
        st = (data_dir / f"{case}.{ext}").stat()
        sig[ext] = (st.st_size, st.st_mtime_ns)
    return sig


def prepare_case_worker(case, project_dir):
    project_dir = Path(project_dir)
    data_dir = project_dir / "data"
    root_out = project_dir / "output" / "问题2_三数据集并行"
    cache_dir = root_out / "cache"
    case_dir = root_out / case
    cache_dir.mkdir(parents=True, exist_ok=True)
    case_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{case}_feasible.pkl"
    signature = input_signature(data_dir, case)
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            if cache.get("version") == CACHE_VERSION and cache.get("signature") == signature:
                m = shape_metrics(cache["blocks"], cache["nodes"], cache["root"], cache["L"])
                if m["feasible"]:
                    print(f"[{case}] 已读取缓存可行初解：{m['W']}×{m['H']}", flush=True)
                    return {"case": case, "cached": True, "seed": cache["prep_seed"],
                            "W": m["W"], "H": m["H"], "L": cache["L"], "hpwl": cache["feasible_hpwl"]}
        except Exception:
            pass
    blocks_file, nets_file, pl_file = data_dir/f"{case}.blocks", data_dir/f"{case}.nets", data_dir/f"{case}.pl"
    for path in [blocks_file, nets_file, pl_file]:
        if not path.exists():
            raise FileNotFoundError(f"[{case}] 没有找到文件：{path}")
    blocks, terminals = read_blocks(str(blocks_file))
    nets = read_nets(str(nets_file))
    terminal_pos = read_pl(str(pl_file))
    compiled_nets = compile_nets(blocks, nets, terminal_pos)
    L = chip_side(blocks, 0.15)
    terminal_out = [name for name, (x, y) in terminal_pos.items() if x < 0 or y < 0 or x > L or y > L]
    if terminal_out:
        raise ValueError(f"[{case}] 存在越界 Terminal，前10个：{terminal_out[:10]}")
    last_error = None
    for seed in PREP_SEEDS[case]:
        prefix = f"[{case}|prep={seed}]"
        try:
            print(f"{prefix} 开始构造固定轮廓可行初解 | L={L:.3f}", flush=True)
            f_blocks, f_nodes, root, fm = prepare_from_seed(blocks, L, seed, prefix)
            overlap, out = validate_layout(f_blocks, L)
            if overlap or out:
                raise RuntimeError(f"最终合法性核验失败：overlap={len(overlap)} out={len(out)}")
            feasible_hpwl = calculate_hpwl(f_blocks, compiled_nets)
            cache = {"version": CACHE_VERSION, "signature": signature, "case": case, "prep_seed": seed,
                     "L": L, "blocks": f_blocks, "nodes": f_nodes, "root": root,
                     "compiled_nets": compiled_nets, "terminal_pos": terminal_pos,
                     "feasible_hpwl": feasible_hpwl}
            with open(cache_path, "wb") as f:
                pickle.dump(cache, f)
            save_layout_csv(f_blocks, case_dir / f"{case}_统一可行初解坐标.csv")
            plot_layout(f_blocks, L, case_dir / f"{case}_统一可行初解.png",
                        f"{case} Feasible Initial Layout - Prep Seed {seed}")
            print(f"{prefix} 可行初解完成：{fm['W']}×{fm['H']} | HPWL={feasible_hpwl:.1f}", flush=True)
            return {"case": case, "cached": False, "seed": seed, "W": fm["W"], "H": fm["H"],
                    "L": L, "hpwl": feasible_hpwl}
        except Exception as e:
            last_error = e
            print(f"{prefix} 本次失败，自动尝试下一 Seed：{e}", flush=True)
    raise RuntimeError(f"[{case}] 所有可行初解尝试均失败，最后错误：{last_error}")


# ============================================================
# 10. HPWL Global SA + Local Polish
# ============================================================

def hpwl_sa(blocks, nodes, root, L, compiled_nets, seed, perturb_fun,
            initial_temp_ratio, cooling_rate, min_temp, moves_per_temp, prefix):
    random.seed(seed)
    current_blocks, current_nodes = clone_blocks(blocks), clone_nodes(nodes)
    current_shape = shape_metrics(current_blocks, current_nodes, root, L)
    if not current_shape["feasible"]:
        raise ValueError("HPWL-SA 必须从可行布局开始")
    current_hpwl = calculate_hpwl(current_blocks, compiled_nets)
    best_blocks, best_nodes = clone_blocks(current_blocks), clone_nodes(current_nodes)
    best_shape, best_hpwl = current_shape.copy(), current_hpwl
    T0 = max(1.0, initial_temp_ratio*current_hpwl)
    T, level, history = T0, 0, [(0, best_hpwl)]
    while T > min_temp:
        level += 1
        feasible_count = accepted = 0
        for _ in range(moves_per_temp):
            cand_blocks, cand_nodes = clone_blocks(current_blocks), clone_nodes(current_nodes)
            perturb_fun(cand_blocks, cand_nodes, root, T/T0)
            cand_shape = shape_metrics(cand_blocks, cand_nodes, root, L)
            if not cand_shape["feasible"]:
                continue
            feasible_count += 1
            cand_hpwl = calculate_hpwl(cand_blocks, compiled_nets)
            delta = cand_hpwl-current_hpwl
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
                  f"feasible={feasible_count/moves_per_temp:.3f} | accepted={accepted/moves_per_temp:.3f}", flush=True)
        T *= cooling_rate
    return best_blocks, best_nodes, best_shape, best_hpwl, history


def hpwl_worker(case, seed, project_dir):
    project_dir = Path(project_dir)
    root_out = project_dir / "output" / "问题2_三数据集并行"
    cache_path = root_out / "cache" / f"{case}_feasible.pkl"
    case_dir = root_out / case
    result_dir = case_dir / "seed_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_cache = result_dir / f"{case}_seed{seed}_result.json"
    layout_path = result_dir / f"{case}_seed{seed}_最终布局.png"
    coord_path = result_dir / f"{case}_seed{seed}_最终坐标.csv"
    conv_path = result_dir / f"{case}_seed{seed}_收敛曲线.png"
    # 已完成任务直接复用，支持中断后续跑
    if result_cache.exists() and layout_path.exists() and coord_path.exists():
        try:
            with open(result_cache, "r", encoding="utf-8") as f:
                result = json.load(f)
            if result.get("version") == CACHE_VERSION:
                print(f"[{case}|seed={seed}] 已读取历史结果：HPWL={result['final_hpwl']:.1f}", flush=True)
                return result
        except Exception:
            pass
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    blocks = clone_blocks(cache["blocks"])
    nodes = clone_nodes(cache["nodes"])
    root, L, compiled_nets = cache["root"], cache["L"], cache["compiled_nets"]
    initial_hpwl = cache["feasible_hpwl"]
    prefix = f"[{case}|seed={seed}|Global]"
    g_blocks, g_nodes, g_shape, g_hpwl, g_history = hpwl_sa(
        blocks, nodes, root, L, compiled_nets, seed=seed, perturb_fun=adaptive_perturb,
        initial_temp_ratio=0.02, cooling_rate=0.93, min_temp=1.0, moves_per_temp=300, prefix=prefix)
    prefix = f"[{case}|seed={seed}|Polish]"
    f_blocks, f_nodes, f_shape, f_hpwl, p_history = hpwl_sa(
        g_blocks, g_nodes, root, L, compiled_nets, seed=seed+100000, perturb_fun=local_perturb,
        initial_temp_ratio=0.003, cooling_rate=0.95, min_temp=1.0, moves_per_temp=400, prefix=prefix)
    overlap, out = validate_layout(f_blocks, L)
    if overlap or out:
        raise RuntimeError(f"[{case}|seed={seed}] 最终合法性失败：overlap={len(overlap)} out={len(out)}")
    plot_layout(f_blocks, L, layout_path, f"{case} HPWL Optimized - Seed {seed}")
    save_layout_csv(f_blocks, coord_path)
    merged_history = [(x, y, "Global") for x, y in g_history]
    offset = g_history[-1][0] if g_history else 0
    merged_history += [(offset+x, y, "Polish") for x, y in p_history[1:]]
    plot_convergence(merged_history, conv_path, f"{case} HPWL Convergence - Seed {seed}")
    result = {"version": CACHE_VERSION, "case": case, "seed": seed, "L": L,
              "initial_hpwl": initial_hpwl, "global_hpwl": g_hpwl, "final_hpwl": f_hpwl,
              "final_width": f_shape["W"], "final_height": f_shape["H"],
              "global_improvement": (initial_hpwl-g_hpwl)/initial_hpwl,
              "polish_improvement": (g_hpwl-f_hpwl)/g_hpwl,
              "total_improvement": (initial_hpwl-f_hpwl)/initial_hpwl,
              "overlap_count": 0, "out_count": 0,
              "layout_path": str(layout_path), "coord_path": str(coord_path),
              "convergence_path": str(conv_path)}
    with open(result_cache, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[{case}|seed={seed}] 完成 | HPWL={f_hpwl:.1f} | W×H={f_shape['W']}×{f_shape['H']}", flush=True)
    return result


# ============================================================
# 11. 绘图与保存
# ============================================================

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
    plt.plot([x for x, _, _ in history], [y for _, y, _ in history], linewidth=1.2)
    plt.xlabel("Temperature Level"); plt.ylabel("Best Total HPWL"); plt.title(title)
    plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(save_path, dpi=220); plt.close()


def save_layout_csv(blocks, save_path):
    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "x", "y", "width", "height", "rotated"])
        for b in blocks:
            writer.writerow([b.name, b.x, b.y, b.w, b.h, int(b.rotated)])


def save_results_csv(results, save_path):
    fields = ["case", "seed", "L", "initial_hpwl", "global_hpwl", "final_hpwl",
              "final_width", "final_height", "global_improvement",
              "polish_improvement", "total_improvement", "overlap_count", "out_count"]
    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({key: r[key] for key in fields})


# ============================================================
# 12. 主程序：3组初解并行 + 15个 HPWL 任务并行
# ============================================================

def main():
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    root_out = project_dir / "output" / "问题2_三数据集并行"
    root_out.mkdir(parents=True, exist_ok=True)
    print("=" * 110)
    print("A题问题2：n100 / n200 / n300 多核并行正式版")
    print(f"CPU逻辑核心数：{CPU_COUNT} | 最大并行进程数：{MAX_WORKERS}")
    print("策略：每个数据集只构造一次可行初解；随后 5 Seed 仅并行优化 HPWL")
    print("缓存：已完成初解和 Seed 结果会自动复用，中断后重新运行即可续跑")
    print("=" * 110)
    # Phase 1：三个数据集并行构造一次可行初解
    print("\n【Phase 1】并行构造 n100 / n200 / n300 固定轮廓可行初解")
    prep_results = []
    prep_workers = min(3, MAX_WORKERS)
    with ProcessPoolExecutor(max_workers=prep_workers, mp_context=mp.get_context("spawn")) as executor:
        futures = {executor.submit(prepare_case_worker, case, str(project_dir)): case for case in CASES}
        for future in as_completed(futures):
            case = futures[future]
            result = future.result()
            prep_results.append(result)
            print(f"[主进程] {case} 可行初解就绪：W×H={result['W']}×{result['H']} | "
                  f"L={result['L']:.3f} | HPWL={result['hpwl']:.1f}", flush=True)
    # Phase 2：15 个 HPWL 任务并行
    print(f"\n【Phase 2】启动 {len(CASES)*len(SEEDS)} 个 HPWL 优化任务，最多同时运行 {MAX_WORKERS} 个进程")
    results = []
    pending = []
    for case in CASES:
        for seed in SEEDS:
            result_file = root_out / case / "seed_results" / f"{case}_seed{seed}_result.json"
            if result_file.exists():
                try:
                    with open(result_file, "r", encoding="utf-8") as f:
                        r = json.load(f)
                    if r.get("version") == CACHE_VERSION:
                        results.append(r)
                        print(f"[主进程] 复用 {case} Seed={seed} | HPWL={r['final_hpwl']:.1f}")
                        continue
                except Exception:
                    pass
            pending.append((case, seed))
    if pending:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=mp.get_context("spawn")) as executor:
            futures = {executor.submit(hpwl_worker, case, seed, str(project_dir)): (case, seed)
                       for case, seed in pending}
            done = 0
            for future in as_completed(futures):
                case, seed = futures[future]
                result = future.result()
                results.append(result)
                done += 1
                print(f"[主进程] 完成 {done}/{len(pending)} | {case} Seed={seed} | "
                      f"Final HPWL={result['final_hpwl']:.1f}", flush=True)
    results.sort(key=lambda r: (CASES.index(r["case"]), SEEDS.index(r["seed"])))
    save_results_csv(results, root_out / "问题2_全部15次结果.csv")
    # Phase 3：汇总
    final_summary = []
    print("\n" + "=" * 110)
    print("【问题2最终汇总】")
    print("=" * 110)
    for case in CASES:
        case_results = [r for r in results if r["case"] == case]
        if len(case_results) != len(SEEDS):
            raise RuntimeError(f"{case} 结果数量异常：{len(case_results)}/{len(SEEDS)}")
        best = min(case_results, key=lambda r: r["final_hpwl"])
        hpwls = [r["final_hpwl"] for r in case_results]
        mean_hpwl, std_hpwl = statistics.mean(hpwls), statistics.pstdev(hpwls)
        cv = std_hpwl/mean_hpwl if mean_hpwl else 0.0
        print(f"\n[{case}]")
        print(f"{'Seed':>8} {'Initial':>12} {'Global':>12} {'Final':>12} {'W':>6} {'H':>6} {'Improve':>10}")
        for r in case_results:
            print(f"{r['seed']:>8} {r['initial_hpwl']:>12.1f} {r['global_hpwl']:>12.1f} "
                  f"{r['final_hpwl']:>12.1f} {r['final_width']:>6} {r['final_height']:>6} "
                  f"{r['total_improvement']:>9.4%}")
        print(f"稳定性：mean={mean_hpwl:.2f} | std={std_hpwl:.2f} | CV={cv:.4%}")
        print(f"最优：Seed={best['seed']} | HPWL={best['final_hpwl']:.3f} | W×H={best['final_width']}×{best['final_height']}")
        best_layout, best_coord = Path(best["layout_path"]), Path(best["coord_path"])
        if best_layout.exists():
            shutil.copy2(best_layout, root_out / case / f"问题2_{case}_总体最优布局.png")
        if best_coord.exists():
            shutil.copy2(best_coord, root_out / case / f"问题2_{case}_总体最优坐标.csv")
        final_summary.append({"case": case, "best_seed": best["seed"], "L": best["L"],
                              "best_hpwl": best["final_hpwl"], "W": best["final_width"],
                              "H": best["final_height"], "mean_hpwl": mean_hpwl,
                              "std_hpwl": std_hpwl, "cv": cv,
                              "overlap_count": best["overlap_count"], "out_count": best["out_count"]})
    with open(root_out / "问题2_最终汇总.csv", "w", newline="", encoding="utf-8-sig") as f:
        fields = ["case", "best_seed", "L", "best_hpwl", "W", "H",
                  "mean_hpwl", "std_hpwl", "cv", "overlap_count", "out_count"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(final_summary)
    print("\n" + "=" * 110)
    print(f"全部完成。结果目录：{root_out}")
    print("若程序中途被关闭，直接重新运行本文件即可自动复用已完成缓存。")
    print("=" * 110)


if __name__ == "__main__":
    mp.freeze_support()
    main()
