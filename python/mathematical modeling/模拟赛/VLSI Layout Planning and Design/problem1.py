import re
import copy
import csv
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================
# 1. 数据结构
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


# ============================================================
# 2. 读取 .blocks
# ============================================================

def read_blocks(file_path: str) -> list[Block]:
    blocks = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2 or parts[1] != "block":  # 只处理 HardBlock
                continue
            coords = re.findall(r"\((-?\d+)\s*,\s*(-?\d+)\)", line)
            if len(coords) != 4:
                raise ValueError(f"无法解析模块坐标：{line}")
            xs = [int(c[0]) for c in coords]
            ys = [int(c[1]) for c in coords]
            blocks.append(Block(name=parts[0], width=max(xs)-min(xs), height=max(ys)-min(ys)))
    return blocks


# ============================================================
# 3. 初始树：左链基线 / 随机二维树
# ============================================================

def build_initial_tree(blocks: list[Block]):
    if not blocks:
        raise ValueError("没有读取到 HardBlock")
    order = sorted(range(len(blocks)), key=lambda i: blocks[i].height, reverse=True)
    nodes = [TreeNode(block_idx=i) for i in order]
    for i in range(len(nodes)-1):
        nodes[i].left = i + 1
        nodes[i+1].parent = i
    return nodes, 0


def build_random_tree(blocks: list[Block], rng: random.Random):
    """随机二维 B*-Tree：随机顺序 + 随机挂到可用空位。"""
    if not blocks:
        raise ValueError("没有读取到 HardBlock")
    order = list(range(len(blocks)))
    rng.shuffle(order)
    nodes = [TreeNode(block_idx=i) for i in order]
    root = 0
    slots = [(root, "left"), (root, "right")]
    for idx in range(1, len(nodes)):
        parent_idx, side = slots.pop(rng.randrange(len(slots)))
        if side == "left":
            nodes[parent_idx].left = idx
        else:
            nodes[parent_idx].right = idx
        nodes[idx].parent = parent_idx
        slots.append((idx, "left"))
        slots.append((idx, "right"))
    return nodes, root


# ============================================================
# 4. B*-Tree 解码：Contour / Skyline
# ============================================================

def pack_bstar_tree(blocks: list[Block], nodes: list[TreeNode], root: int):
    for block in blocks:
        block.x = block.y = 0
    max_w = sum(max(b.width, b.height) for b in blocks) + 1
    skyline = [0] * (max_w + 1)
    stack, placed = [root], 0
    while stack:
        node_idx = stack.pop()
        node = nodes[node_idx]
        block = blocks[node.block_idx]
        if node.parent is None:
            x = 0
        else:
            pnode = nodes[node.parent]
            pblock = blocks[pnode.block_idx]
            if pnode.left == node_idx:      # 左孩子靠父块右侧
                x = pblock.x + pblock.w
            elif pnode.right == node_idx:   # 右孩子靠父块左下角
                x = pblock.x
            else:
                raise RuntimeError("B*-Tree 父子关系出现错误")
        w, h = block.w, block.h
        y = max(skyline[x:x+w])
        block.x, block.y = x, y
        for xx in range(x, x+w):
            skyline[xx] = y + h
        placed += 1
        if node.right is not None:
            stack.append(node.right)
        if node.left is not None:
            stack.append(node.left)
    if placed != len(blocks):
        raise RuntimeError(f"只摆放了 {placed}/{len(blocks)} 个模块")
    W = max(b.x + b.w for b in blocks)
    H = max(b.y + b.h for b in blocks)
    return W, H, W * H


# ============================================================
# 5. 合法性核验
# ============================================================

def is_overlap(a: Block, b: Block) -> bool:
    return not (a.x + a.w <= b.x or b.x + b.w <= a.x or a.y + a.h <= b.y or b.y + b.h <= a.y)


def validate_layout(blocks: list[Block]):
    negative = [b.name for b in blocks if b.x < 0 or b.y < 0]
    overlap = [(blocks[i].name, blocks[j].name)
               for i in range(len(blocks)) for j in range(i+1, len(blocks))
               if is_overlap(blocks[i], blocks[j])]
    return negative, overlap


# ============================================================
# 6. 指标与目标函数
# ============================================================

def calculate_metrics(blocks: list[Block], W: int, H: int):
    total_block_area = sum(b.area for b in blocks)
    outline_area = W * H
    aspect_ratio = max(W, H) / min(W, H)
    utilization = total_block_area / outline_area
    return {"total_block_area": total_block_area, "outline_width": W, "outline_height": H,
            "outline_area": outline_area, "aspect_ratio": aspect_ratio,
            "utilization": utilization, "dead_space_ratio": 1.0 - utilization}


def objective_energy(area: int, aspect_ratio: float) -> float:
    """面积严格优先，长宽比次优（形状惩罚恒小于0.5）。"""
    shape_penalty = (aspect_ratio - 1.0) / (aspect_ratio + 1.0)
    return area + 0.5 * shape_penalty


def evaluate_layout(blocks: list[Block], nodes: list[TreeNode], root: int):
    W, H, A = pack_bstar_tree(blocks, nodes, root)
    ratio = max(W, H) / min(W, H)
    return {"W": W, "H": H, "area": A, "aspect_ratio": ratio,
            "energy": objective_energy(A, ratio)}


# ============================================================
# 7. 多随机树择优作为 SA 起点
# ============================================================

def select_sa_initial_tree(blocks: list[Block], seed: int, trials: int = 20):
    rng = random.Random(seed)
    work = copy.deepcopy(blocks)
    best_nodes = best_root = None
    best_metrics = None
    for _ in range(trials):
        nodes, root = build_random_tree(work, rng)
        metrics = evaluate_layout(work, nodes, root)
        score = (metrics["area"], metrics["aspect_ratio"])
        if best_metrics is None or score < (best_metrics["area"], best_metrics["aspect_ratio"]):
            best_nodes, best_root, best_metrics = copy.deepcopy(nodes), root, metrics.copy()
    final_metrics = evaluate_layout(blocks, best_nodes, best_root)
    return best_nodes, best_root, final_metrics


# ============================================================
# 8. 四类邻域操作
# ============================================================

def rotate_random_block(blocks: list[Block]):
    idx = random.randrange(len(blocks))
    blocks[idx].rotated = not blocks[idx].rotated


def swap_random_blocks(nodes: list[TreeNode]):
    if len(nodes) < 2:
        return
    i, j = random.sample(range(len(nodes)), 2)
    nodes[i].block_idx, nodes[j].block_idx = nodes[j].block_idx, nodes[i].block_idx


def get_subtree_nodes(nodes: list[TreeNode], subtree_root: int) -> set[int]:
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
    """摘下一个非根节点，返回(old_parent, old_side)。"""
    old_parent = nodes[moved].parent
    if old_parent is None:
        return None, None
    if nodes[old_parent].left == moved:
        side = "left"; nodes[old_parent].left = None
    elif nodes[old_parent].right == moved:
        side = "right"; nodes[old_parent].right = None
    else:
        raise RuntimeError("父子关系异常")
    nodes[moved].parent = None
    return old_parent, side


def _reattach(nodes, moved, old_parent, old_side, exclude_subtree=None):
    """把 moved 挂到随机空位；无可选位置则恢复原状。"""
    if exclude_subtree is None:
        exclude_subtree = set()
    possible_positions = []
    for target in range(len(nodes)):
        if target in exclude_subtree or target == moved:
            continue
        if nodes[target].left is None and not (target == old_parent and old_side == "left"):
            possible_positions.append((target, "left"))
        if nodes[target].right is None and not (target == old_parent and old_side == "right"):
            possible_positions.append((target, "right"))
    if not possible_positions:
        if old_side == "left":
            nodes[old_parent].left = moved
        else:
            nodes[old_parent].right = moved
        nodes[moved].parent = old_parent
        return False
    target, side = random.choice(possible_positions)
    if side == "left":
        nodes[target].left = moved
    else:
        nodes[target].right = moved
    nodes[moved].parent = target
    return True


def move_random_leaf(nodes: list[TreeNode], root: int) -> bool:
    """随机移动一个叶子节点到其他空位。"""
    leaves = [i for i in range(len(nodes)) if i != root and nodes[i].left is None and nodes[i].right is None]
    if not leaves:
        return False
    moved = random.choice(leaves)
    old_parent, old_side = _detach(nodes, moved)
    if old_parent is None:
        return False
    return _reattach(nodes, moved, old_parent, old_side)


def move_random_subtree(nodes: list[TreeNode], root: int) -> bool:
    """随机移动一棵子树到其他空位。"""
    if len(nodes) <= 1:
        return False
    movable = [i for i in range(len(nodes)) if i != root]
    moved = random.choice(movable)
    subtree = get_subtree_nodes(nodes, moved)
    old_parent, old_side = _detach(nodes, moved)
    if old_parent is None:
        return False
    return _reattach(nodes, moved, old_parent, old_side, exclude_subtree=subtree)


# ============================================================
# 9. 邻域选择（全局自适应 / 局部固定）
# ============================================================

def get_adaptive_probabilities(temperature_ratio: float):
    if temperature_ratio > 0.60:
        return {"rotate": 0.15, "swap": 0.25, "leaf_move": 0.35, "subtree_move": 0.25}
    elif temperature_ratio > 0.20:
        return {"rotate": 0.20, "swap": 0.30, "leaf_move": 0.35, "subtree_move": 0.15}
    else:
        return {"rotate": 0.30, "swap": 0.35, "leaf_move": 0.30, "subtree_move": 0.05}


def adaptive_perturb(blocks: list[Block], nodes: list[TreeNode], root: int, temperature_ratio: float) -> str:
    """全局阶段：按温度自适应选择扰动。"""
    p = get_adaptive_probabilities(temperature_ratio)
    r = random.random()
    if r < p["rotate"]:
        rotate_random_block(blocks); return "rotate"
    elif r < p["rotate"] + p["swap"]:
        swap_random_blocks(nodes); return "swap"
    elif r < p["rotate"] + p["swap"] + p["leaf_move"]:
        if move_random_leaf(nodes, root):
            return "leaf_move"
        swap_random_blocks(nodes); return "swap"
    else:
        if move_random_subtree(nodes, root):
            return "subtree_move"
        if move_random_leaf(nodes, root):
            return "leaf_move"
        swap_random_blocks(nodes); return "swap"


def local_perturb(blocks: list[Block], nodes: list[TreeNode], root: int, temperature_ratio: float = None) -> str:
    """局部精修：关闭 Subtree Move（Rotate25% / Swap35% / Leaf40%）。"""
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
# 10. 模拟退火
# ============================================================

def _sa_core(blocks, nodes, root, seed, T0, min_temp, cooling_rate, moves_per_temp, perturb_fun, tag):
    """全局/局部 SA 共用主循环；perturb_fun(cand_blocks,cand_nodes,root,temperature_ratio)。"""
    random.seed(seed)
    current_blocks, current_nodes = copy.deepcopy(blocks), copy.deepcopy(nodes)
    current_metrics = evaluate_layout(current_blocks, current_nodes, root)
    best_blocks, best_nodes = copy.deepcopy(current_blocks), copy.deepcopy(current_nodes)
    best_metrics = current_metrics.copy()
    T, iteration, level = T0, 0, 0
    history, accepted_count, improved_count = [], 0, 0
    while T > min_temp:
        level += 1
        level_accepted = 0
        temperature_ratio = T / T0
        for _ in range(moves_per_temp):
            iteration += 1
            cand_blocks, cand_nodes = copy.deepcopy(current_blocks), copy.deepcopy(current_nodes)
            operation = perturb_fun(cand_blocks, cand_nodes, root, temperature_ratio)
            cand_metrics = evaluate_layout(cand_blocks, cand_nodes, root)
            delta = cand_metrics["energy"] - current_metrics["energy"]
            if delta <= 0 or random.random() < math.exp(-delta / T):
                current_blocks, current_nodes, current_metrics = cand_blocks, cand_nodes, cand_metrics
                accepted_count += 1
                level_accepted += 1
            if (cand_metrics["area"], cand_metrics["aspect_ratio"]) < (best_metrics["area"], best_metrics["aspect_ratio"]):
                best_blocks, best_nodes, best_metrics = copy.deepcopy(cand_blocks), copy.deepcopy(cand_nodes), cand_metrics.copy()
                improved_count += 1
            history.append({"iteration": iteration, "temperature": T, "best_area": best_metrics["area"],
                            "best_aspect_ratio": best_metrics["aspect_ratio"], "operation": operation})
        if level == 1 or level % 10 == 0:
            print(f"[{tag}] 温度层 {level:3d} | T={T:10.4f} | 当前面积={current_metrics['area']:7d} | "
                  f"最优面积={best_metrics['area']:7d} | ratio={best_metrics['aspect_ratio']:.4f} | "
                  f"接受率={level_accepted/moves_per_temp:.3f}")
        T *= cooling_rate
    return best_blocks, best_nodes, best_metrics, history, iteration, accepted_count, improved_count


def global_simulated_annealing(blocks, nodes, root, seed, initial_temp_ratio=0.02, cooling_rate=0.92,
                               min_temp=3.0, moves_per_temp=300):
    T0 = max(1.0, initial_temp_ratio * evaluate_layout(blocks, nodes, root)["area"])
    return _sa_core(blocks, nodes, root, seed, T0, min_temp, cooling_rate, moves_per_temp,
                    adaptive_perturb, "Global")


def local_polish(blocks, nodes, root, seed, initial_temp_ratio=0.002, cooling_rate=0.95,
                 min_temp=1.0, moves_per_temp=500):
    """以全局最优为起点低温精修；结果不会比全局更差。"""
    local_seed = seed + 100000
    T0 = max(1.0, initial_temp_ratio * evaluate_layout(blocks, nodes, root)["area"])
    print(f"\n开始局部精修 | local_seed={local_seed} | T0={T0:.4f}")
    return _sa_core(blocks, nodes, root, local_seed, T0, min_temp, cooling_rate, moves_per_temp,
                    local_perturb, "Polish")


# ============================================================
# 11. 绘图
# ============================================================

def plot_layout(blocks, W, H, save_path, title):
    fig, ax = plt.subplots(figsize=(9, 9))
    for b in blocks:
        ax.add_patch(Rectangle((b.x, b.y), b.w, b.h, fill=False, linewidth=0.8))
        ax.text(b.x + b.w/2, b.y + b.h/2, b.name, ha="center", va="center", fontsize=5)
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(save_path, dpi=220); plt.close()


def plot_convergence(history, save_path, title):
    plt.figure(figsize=(10, 6))
    plt.plot([item["iteration"] for item in history],
             [item["best_area"] for item in history], linewidth=1.2)
    plt.xlabel("Iteration"); plt.ylabel("Best Outline Area"); plt.title(title)
    plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(save_path, dpi=220); plt.close()


# ============================================================
# 12. 单个 Seed：全局 + 局部精修
# ============================================================

def run_one_seed(base_blocks, seed, output_dir, dataset, random_tree_trials=20):
    blocks = copy.deepcopy(base_blocks)
    nodes, root, start_metrics = select_sa_initial_tree(blocks, seed=seed, trials=random_tree_trials)
    print(f"\n{'='*90}\nSeed={seed} | SA 起点 A={start_metrics['area']} | ratio={start_metrics['aspect_ratio']:.4f}\n{'='*90}")
    global_blocks, global_nodes, global_metrics, global_history, gi, ga, gim = global_simulated_annealing(
        blocks, nodes, root, seed=seed)
    global_W, global_H, global_A = pack_bstar_tree(global_blocks, global_nodes, root)
    gm = calculate_metrics(global_blocks, global_W, global_H)
    print(f"\nGlobal 完成 | A={gm['outline_area']} | W={global_W} | H={global_H} | "
          f"ratio={gm['aspect_ratio']:.4f} | U={gm['utilization']:.4%}")
    polished_blocks, polished_nodes, _, polish_history, pi, pa, pim = local_polish(
        global_blocks, global_nodes, root, seed=seed)
    final_W, final_H, final_A = pack_bstar_tree(polished_blocks, polished_nodes, root)
    neg, ovp = validate_layout(polished_blocks)
    fm = calculate_metrics(polished_blocks, final_W, final_H)
    polish_improvement = (gm["outline_area"] - fm["outline_area"]) / gm["outline_area"]
    print(f"\nPolish 完成 | A={fm['outline_area']} | W={final_W} | H={final_H} | "
          f"ratio={fm['aspect_ratio']:.4f} | U={fm['utilization']:.4%} | 较Global改善={polish_improvement:.4%}")
    # 保存图
    plot_layout(global_blocks, global_W, global_H, str(output_dir / f"问题1_{dataset}_seed{seed}_全局布局.png"),
                f"Global Adaptive 4-Neighborhood SA - Seed {seed}")
    plot_layout(polished_blocks, final_W, final_H, str(output_dir / f"问题1_{dataset}_seed{seed}_精修布局.png"),
                f"Global SA + Local Polish - Seed {seed}")
    plot_convergence(global_history, str(output_dir / f"问题1_{dataset}_seed{seed}_全局收敛.png"),
                     f"Global SA Convergence - Seed {seed}")
    plot_convergence(polish_history, str(output_dir / f"问题1_{dataset}_seed{seed}_精修收敛.png"),
                     f"Local Polish Convergence - Seed {seed}")
    return {"seed": seed, "start_area": start_metrics["area"], "start_aspect_ratio": start_metrics["aspect_ratio"],
            "global_area": gm["outline_area"], "global_width": global_W, "global_height": global_H,
            "global_aspect_ratio": gm["aspect_ratio"], "global_utilization": gm["utilization"],
            "final_area": fm["outline_area"], "final_width": final_W, "final_height": final_H,
            "final_aspect_ratio": fm["aspect_ratio"], "final_utilization": fm["utilization"],
            "final_dead_space_ratio": fm["dead_space_ratio"], "polish_improvement": polish_improvement,
            "negative_count": len(neg), "overlap_count": len(ovp),
            "global_iterations": gi, "polish_iterations": pi,
            "global_accepted": ga, "polish_accepted": pa,
            "global_improved": gim, "polish_improved": pim,
            "blocks": polished_blocks, "nodes": polished_nodes, "root": root}


# ============================================================
# 13. 保存汇总 CSV
# ============================================================

def save_summary_csv(results, save_path):
    fieldnames = ["seed", "start_area", "start_aspect_ratio", "global_area", "global_width",
                  "global_height", "global_aspect_ratio", "global_utilization", "final_area",
                  "final_width", "final_height", "final_aspect_ratio", "final_utilization",
                  "final_dead_space_ratio", "polish_improvement", "negative_count", "overlap_count",
                  "global_iterations", "polish_iterations", "global_accepted", "polish_accepted",
                  "global_improved", "polish_improved"]
    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow({key: item[key] for key in fieldnames})


# ============================================================
# 14. 主程序：5 Seed + Global + Polish
# ============================================================

def run_dataset(dataset, script_dir, output_dir, seeds, random_tree_trials=20):
    """对单个数据集跑：左链基线 + 5种子全局SA+精修，返回最优结果与汇总。"""
    blocks_file = script_dir / "data" / f"{dataset}.blocks"
    if not blocks_file.exists():
        raise FileNotFoundError(f"没有找到文件：{blocks_file}")
    base_blocks = read_blocks(str(blocks_file))
    print(f"\n{'='*100}\n数据集 {dataset}：读取 HardBlock {len(base_blocks)} 个\n{'='*100}")
    # 左链基线
    baseline_blocks = copy.deepcopy(base_blocks)
    baseline_nodes, baseline_root = build_initial_tree(baseline_blocks)
    baseline_metrics = evaluate_layout(baseline_blocks, baseline_nodes, baseline_root)
    print(f"【左链基线】面积：{baseline_metrics['area']} 长宽比：{baseline_metrics['aspect_ratio']:.6f}")
    # 5 个种子
    results = []
    for run_idx, seed in enumerate(seeds, start=1):
        print(f"\n{'='*100}\n第 {run_idx}/{len(seeds)} 次实验 | Seed = {seed}\n{'='*100}")
        results.append(run_one_seed(base_blocks=base_blocks, seed=seed, output_dir=output_dir,
                                    dataset=dataset, random_tree_trials=random_tree_trials))
    # 按题意选最终总体最优
    best_result = min(results, key=lambda r: (r["final_area"], r["final_aspect_ratio"]))
    # 稳定性统计
    areas = [r["final_area"] for r in results]
    ratios = [r["final_aspect_ratio"] for r in results]
    area_mean, area_std = statistics.mean(areas), statistics.pstdev(areas)
    area_cv = area_std / area_mean if area_mean != 0 else 0.0
    ratio_mean = statistics.mean(ratios)
    # 汇总
    print(f"\n{'='*100}\n【{dataset}：5 个随机种子 Global + Polish 汇总】\n{'='*100}")
    print(f"{'Seed':>8} {'Global A':>10} {'Final A':>10} {'W':>6} {'H':>6} {'Ratio':>9} {'Util':>10} {'Polish':>10}")
    for r in results:
        print(f"{r['seed']:>8} {r['global_area']:>10} {r['final_area']:>10} {r['final_width']:>6} "
              f"{r['final_height']:>6} {r['final_aspect_ratio']:>9.4f} {r['final_utilization']:>9.4%} "
              f"{r['polish_improvement']:>9.4%}")
    print(f"【稳定性】面积均值：{area_mean:.2f} 标准差：{area_std:.2f} CV：{area_cv:.4%} 长宽比均值：{ratio_mean:.6f}")
    print(f"【最优】Seed={best_result['seed']} Global={best_result['global_area']} 精修后={best_result['final_area']} "
          f"W={best_result['final_width']} H={best_result['final_height']} "
          f"ratio={best_result['final_aspect_ratio']:.6f} util={best_result['final_utilization']:.4%} "
          f"overlap={best_result['overlap_count']}")
    # 保存 CSV 与总体最优布局
    save_summary_csv(results, output_dir / f"问题1_{dataset}_5种子汇总.csv")
    plot_layout(best_result["blocks"], best_result["final_width"], best_result["final_height"],
                str(output_dir / f"问题1_{dataset}_总体最优布局.png"),
                f"Question 1 - Global Adaptive SA + Local Polish (Best Seed {best_result['seed']})")
    return {"dataset": dataset, "num_blocks": len(base_blocks), "baseline_area": baseline_metrics["area"],
            "baseline_ratio": baseline_metrics["aspect_ratio"], "best": best_result,
            "area_mean": area_mean, "area_cv": area_cv, "ratio_mean": ratio_mean}


def main():
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "output" / "问题1_结果"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 100)
    print("A题问题1：B*-Tree + 自适应四邻域全局SA + 三邻域低温局部精修 + 5随机种子")
    print(f"处理数据集：n100 / n200 / n300")
    print("=" * 100)
    seeds = [2026, 2027, 2028, 42, 1107]
    datasets = ["n100", "n200", "n300"]
    all_results = []
    for dataset in datasets:
        all_results.append(run_dataset(dataset, script_dir, output_dir, seeds, random_tree_trials=20))
    # 三组数据汇总表
    print(f"\n{'='*100}\n【三组芯片问题1结果汇总】\n{'='*100}")
    print(f"{'数据集':>6} {'模块数':>6} {'基线面积':>10} {'最优面积':>10} {'W':>7} {'H':>7} {'长宽比':>9} {'利用率':>9} {'面积CV':>9}")
    for item in all_results:
        b = item["best"]
        print(f"{item['dataset']:>6} {item['num_blocks']:>6} {item['baseline_area']:>10} "
              f"{b['final_area']:>10} {b['final_width']:>7} {b['final_height']:>7} "
              f"{b['final_aspect_ratio']:>9.4f} {b['final_utilization']:>8.4%} {item['area_cv']:>8.4%}")
    print(f"\n结果已保存到：{output_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()

