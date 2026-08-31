# ============================================================
# A题问题4：异形 HardBlock 布图
# 方法：问题1 B*-Tree + 形状感知 Skyline
#      + 自适应模拟退火 + 低温局部精修
#      + 4个模块 4^4=256 旋转组合精修
# ============================================================

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
class RectPart:
    x: int
    y: int
    w: int
    h: int


@dataclass
class Block:
    name: str
    base_parts: list[RectPart]
    rotation: int = 0
    x: int = 0
    y: int = 0

    @property
    def area(self) -> int:
        return sum(p.w*p.h for p in self.base_parts)


@dataclass
class TreeNode:
    block_idx: int
    parent: Optional[int] = None
    left: Optional[int] = None
    right: Optional[int] = None


# ============================================================
# 2. 图3四个模块
# ============================================================

def build_blocks():
    # b1：T型
    # 上横条 4×1，中央竖条 2×2
    b1 = Block(
        "b1",
        [
            RectPart(0, 2, 4, 1),
            RectPart(1, 0, 2, 2),
        ]
    )

    # b2：L型
    # 左竖条 1×4，右下补块 1×2
    b2 = Block(
        "b2",
        [
            RectPart(0, 0, 1, 4),
            RectPart(1, 0, 1, 2),
        ]
    )

    # b3：2×1矩形
    b3 = Block(
        "b3",
        [RectPart(0, 0, 2, 1)]
    )

    # b4：1×4矩形
    b4 = Block(
        "b4",
        [RectPart(0, 0, 1, 4)]
    )

    return [b1, b2, b3, b4]


# ============================================================
# 3. 旋转与几何信息
# ============================================================

def rotate_parts_90(parts):
    rotated = []

    for p in parts:
        # 矩形绕原点逆时针90°：
        # [x,x+w]×[y,y+h] -> [-y-h,-y]×[x,x+w]
        rotated.append(
            RectPart(
                -p.y-p.h,
                p.x,
                p.h,
                p.w
            )
        )

    min_x = min(p.x for p in rotated)
    min_y = min(p.y for p in rotated)

    return [
        RectPart(
            p.x-min_x,
            p.y-min_y,
            p.w,
            p.h
        )
        for p in rotated
    ]


def oriented_parts(block):
    parts = copy.deepcopy(block.base_parts)

    for _ in range(block.rotation % 4):
        parts = rotate_parts_90(parts)

    return parts


def block_bbox(block):
    parts = oriented_parts(block)

    W = max(p.x+p.w for p in parts)
    H = max(p.y+p.h for p in parts)

    return W, H


def block_profiles(block):
    """构造当前方向下的下轮廓 bottom[u] 和上轮廓 top[u]。"""
    parts = oriented_parts(block)
    W, H = block_bbox(block)

    inf = 10**9
    bottom = [inf]*W
    top = [-inf]*W

    for p in parts:
        for u in range(p.x, p.x+p.w):
            bottom[u] = min(bottom[u], p.y)
            top[u] = max(top[u], p.y+p.h)

    if any(v == inf for v in bottom):
        raise RuntimeError(f"{block.name} 存在未覆盖列，无法构造轮廓")

    return W, H, bottom, top


def absolute_parts(block):
    return [
        RectPart(
            block.x+p.x,
            block.y+p.y,
            p.w,
            p.h
        )
        for p in oriented_parts(block)
    ]


# ============================================================
# 4. B*-Tree 初始树
# ============================================================

def build_random_tree(blocks, rng):
    order = list(range(len(blocks)))
    rng.shuffle(order)

    nodes = [
        TreeNode(block_idx=i)
        for i in order
    ]

    root = 0
    slots = [(root, "left"), (root, "right")]

    for idx in range(1, len(nodes)):
        parent, side = slots.pop(
            rng.randrange(len(slots))
        )

        if side == "left":
            nodes[parent].left = idx
        else:
            nodes[parent].right = idx

        nodes[idx].parent = parent

        slots.append((idx, "left"))
        slots.append((idx, "right"))

    return nodes, root


# ============================================================
# 5. 修正后的 B*-Tree + Shape-aware Skyline
# ============================================================

def pack_bstar_tree(blocks, nodes, root):
    for b in blocks:
        b.x = 0
        b.y = 0

    max_width = (
        sum(
            max(block_bbox(b))
            for b in blocks
        )
        + 10
    )

    skyline = [0]*max_width

    stack = [root]
    placed = 0

    while stack:
        node_idx = stack.pop()
        node = nodes[node_idx]
        block = blocks[node.block_idx]

        w, h, bottom, top = block_profiles(block)

        if node.parent is None:
            x = 0
        else:
            pnode = nodes[node.parent]
            pblock = blocks[pnode.block_idx]
            pw, _ = block_bbox(pblock)

            if pnode.left == node_idx:
                x = pblock.x+pw
            elif pnode.right == node_idx:
                x = pblock.x
            else:
                raise RuntimeError("B*-Tree 父子关系异常")

        # 原问题1：y=max(skyline)
        # 问题4：考虑异形块下轮廓
        y = max(
            skyline[x+u]-bottom[u]
            for u in range(w)
        )

        block.x = x
        block.y = y

        # 按真实上轮廓更新 Skyline
        for u in range(w):
            skyline[x+u] = max(
                skyline[x+u],
                y+top[u]
            )

        placed += 1

        if node.right is not None:
            stack.append(node.right)

        if node.left is not None:
            stack.append(node.left)

    if placed != len(blocks):
        raise RuntimeError(
            f"只摆放了 {placed}/{len(blocks)} 个模块"
        )

    W = max(
        b.x+block_bbox(b)[0]
        for b in blocks
    )

    H = max(
        b.y+block_bbox(b)[1]
        for b in blocks
    )

    return W, H, W*H


# ============================================================
# 6. 精确合法性核验
# ============================================================

def rect_overlap(a, b):
    return not (
        a.x+a.w <= b.x
        or b.x+b.w <= a.x
        or a.y+a.h <= b.y
        or b.y+b.h <= a.y
    )


def blocks_overlap(a, b):
    for pa in absolute_parts(a):
        for pb in absolute_parts(b):
            if rect_overlap(pa, pb):
                return True

    return False


def validate_layout(blocks):
    negative = [
        b.name
        for b in blocks
        if b.x < 0 or b.y < 0
    ]

    overlap = []

    for i in range(len(blocks)):
        for j in range(i+1, len(blocks)):
            if blocks_overlap(
                blocks[i],
                blocks[j]
            ):
                overlap.append(
                    (
                        blocks[i].name,
                        blocks[j].name
                    )
                )

    return negative, overlap


# ============================================================
# 7. 指标与目标函数
# ============================================================

def calculate_metrics(blocks, W, H):
    total_area = sum(
        b.area
        for b in blocks
    )

    outline_area = W*H
    ratio = max(W, H)/min(W, H)

    return {
        "block_area": total_area,
        "W": W,
        "H": H,
        "area": outline_area,
        "aspect_ratio": ratio,
        "utilization": total_area/outline_area,
    }


def objective_energy(area, ratio):
    # 与问题1一致：面积严格优先，长宽比仅作次级指标
    shape_penalty = (
        (ratio-1.0)/(ratio+1.0)
    )

    return area+0.5*shape_penalty


def evaluate_layout(blocks, nodes, root):
    W, H, A = pack_bstar_tree(
        blocks,
        nodes,
        root
    )

    ratio = max(W, H)/min(W, H)

    return {
        "W": W,
        "H": H,
        "area": A,
        "aspect_ratio": ratio,
        "energy": objective_energy(
            A,
            ratio
        )
    }


# ============================================================
# 8. SA起点
# ============================================================

def select_initial_tree(
    blocks,
    seed,
    trials=30
):
    rng = random.Random(seed)

    best_nodes = None
    best_root = None
    best_metrics = None

    for _ in range(trials):
        work = copy.deepcopy(blocks)

        # 初始方向也随机化
        for b in work:
            b.rotation = rng.randrange(4)

        nodes, root = build_random_tree(
            work,
            rng
        )

        metrics = evaluate_layout(
            work,
            nodes,
            root
        )

        score = (
            metrics["area"],
            metrics["aspect_ratio"]
        )

        if (
            best_metrics is None
            or score
            <
            (
                best_metrics["area"],
                best_metrics["aspect_ratio"]
            )
        ):
            best_nodes = copy.deepcopy(nodes)
            best_root = root
            best_metrics = metrics.copy()
            best_rotations = [
                b.rotation
                for b in work
            ]

    for b, r in zip(
        blocks,
        best_rotations
    ):
        b.rotation = r

    evaluate_layout(
        blocks,
        best_nodes,
        best_root
    )

    return (
        best_nodes,
        best_root,
        best_metrics
    )


# ============================================================
# 9. 邻域操作
# ============================================================

def rotate_random_block(blocks):
    idx = random.randrange(
        len(blocks)
    )

    old = blocks[idx].rotation

    choices = [
        r
        for r in range(4)
        if r != old
    ]

    blocks[idx].rotation = random.choice(
        choices
    )


def swap_random_blocks(nodes):
    i, j = random.sample(
        range(len(nodes)),
        2
    )

    nodes[i].block_idx, nodes[j].block_idx = (
        nodes[j].block_idx,
        nodes[i].block_idx
    )


def get_subtree_nodes(
    nodes,
    subtree_root
):
    result = set()
    stack = [subtree_root]

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


def _detach(
    nodes,
    moved
):
    parent = nodes[moved].parent

    if parent is None:
        return None, None

    if nodes[parent].left == moved:
        side = "left"
        nodes[parent].left = None

    elif nodes[parent].right == moved:
        side = "right"
        nodes[parent].right = None

    else:
        raise RuntimeError(
            "父子关系异常"
        )

    nodes[moved].parent = None

    return parent, side


def _reattach(
    nodes,
    moved,
    old_parent,
    old_side,
    excluded=None
):
    excluded = excluded or set()
    slots = []

    for u in range(len(nodes)):
        if (
            u == moved
            or u in excluded
        ):
            continue

        if nodes[u].left is None:
            slots.append(
                (u, "left")
            )

        if nodes[u].right is None:
            slots.append(
                (u, "right")
            )

    if not slots:
        if old_side == "left":
            nodes[old_parent].left = moved
        else:
            nodes[old_parent].right = moved

        nodes[moved].parent = old_parent

        return False

    parent, side = random.choice(
        slots
    )

    if side == "left":
        nodes[parent].left = moved
    else:
        nodes[parent].right = moved

    nodes[moved].parent = parent

    return True


def move_random_leaf(
    nodes,
    root
):
    leaves = [
        i
        for i in range(len(nodes))
        if (
            i != root
            and nodes[i].left is None
            and nodes[i].right is None
        )
    ]

    if not leaves:
        return False

    moved = random.choice(leaves)

    old_parent, old_side = _detach(
        nodes,
        moved
    )

    if old_parent is None:
        return False

    return _reattach(
        nodes,
        moved,
        old_parent,
        old_side
    )


def move_random_subtree(
    nodes,
    root
):
    movable = [
        i
        for i in range(len(nodes))
        if i != root
    ]

    if not movable:
        return False

    moved = random.choice(
        movable
    )

    subtree = get_subtree_nodes(
        nodes,
        moved
    )

    old_parent, old_side = _detach(
        nodes,
        moved
    )

    if old_parent is None:
        return False

    return _reattach(
        nodes,
        moved,
        old_parent,
        old_side,
        subtree
    )


# ============================================================
# 10. 邻域选择
# ============================================================

def adaptive_perturb(
    blocks,
    nodes,
    root,
    temperature_ratio
):
    if temperature_ratio > 0.60:
        probs = (
            0.25,
            0.25,
            0.30,
            0.20
        )

    elif temperature_ratio > 0.20:
        probs = (
            0.30,
            0.30,
            0.30,
            0.10
        )

    else:
        probs = (
            0.40,
            0.35,
            0.22,
            0.03
        )

    r = random.random()

    if r < probs[0]:
        rotate_random_block(blocks)
        return "rotate"

    if r < probs[0]+probs[1]:
        swap_random_blocks(nodes)
        return "swap"

    if r < probs[0]+probs[1]+probs[2]:
        if move_random_leaf(
            nodes,
            root
        ):
            return "leaf_move"

        swap_random_blocks(nodes)
        return "swap"

    if move_random_subtree(
        nodes,
        root
    ):
        return "subtree_move"

    swap_random_blocks(nodes)

    return "swap"


def local_perturb(
    blocks,
    nodes,
    root,
    temperature_ratio=None
):
    r = random.random()

    if r < 0.40:
        rotate_random_block(blocks)
        return "rotate"

    if r < 0.70:
        swap_random_blocks(nodes)
        return "swap"

    if move_random_leaf(
        nodes,
        root
    ):
        return "leaf_move"

    swap_random_blocks(nodes)
    return "swap"


# ============================================================
# 11. 模拟退火
# ============================================================

def sa_core(
    blocks,
    nodes,
    root,
    seed,
    T0,
    min_temp,
    cooling_rate,
    moves_per_temp,
    perturb_fun,
    tag,
    area_lower_bound
):
    random.seed(seed)

    current_blocks = copy.deepcopy(blocks)
    current_nodes = copy.deepcopy(nodes)

    current = evaluate_layout(
        current_blocks,
        current_nodes,
        root
    )

    best_blocks = copy.deepcopy(
        current_blocks
    )

    best_nodes = copy.deepcopy(
        current_nodes
    )

    best = current.copy()

    history = []
    T = T0
    iteration = 0
    level = 0

    # 若起点已经达到面积理论下界，直接结束
    if best["area"] == area_lower_bound:
        return (
            best_blocks,
            best_nodes,
            best,
            history
        )

    while T > min_temp:
        level += 1
        accepted = 0

        for _ in range(
            moves_per_temp
        ):
            iteration += 1

            cand_blocks = copy.deepcopy(
                current_blocks
            )

            cand_nodes = copy.deepcopy(
                current_nodes
            )

            operation = perturb_fun(
                cand_blocks,
                cand_nodes,
                root,
                T/T0
            )

            cand = evaluate_layout(
                cand_blocks,
                cand_nodes,
                root
            )

            delta = (
                cand["energy"]
                - current["energy"]
            )

            if (
                delta <= 0
                or
                random.random()
                < math.exp(-delta/T)
            ):
                current_blocks = cand_blocks
                current_nodes = cand_nodes
                current = cand
                accepted += 1

            if (
                cand["area"],
                cand["aspect_ratio"]
            ) < (
                best["area"],
                best["aspect_ratio"]
            ):
                best_blocks = copy.deepcopy(
                    cand_blocks
                )

                best_nodes = copy.deepcopy(
                    cand_nodes
                )

                best = cand.copy()

            history.append(
                {
                    "iteration": iteration,
                    "best_area": best["area"],
                    "best_ratio": best["aspect_ratio"],
                    "operation": operation,
                }
            )

            if best["area"] == area_lower_bound:
                return (
                    best_blocks,
                    best_nodes,
                    best,
                    history
                )

        if (
            level == 1
            or level % 10 == 0
        ):
            print(
                f"[{tag}] level={level:3d} | "
                f"T={T:8.4f} | "
                f"best A={best['area']:3d} | "
                f"W×H={best['W']}×{best['H']} | "
                f"ratio={best['aspect_ratio']:.4f} | "
                f"accept={accepted/moves_per_temp:.3f}"
            )

        T *= cooling_rate

    return (
        best_blocks,
        best_nodes,
        best,
        history
    )


# ============================================================
# 12. 4^4旋转组合精修
# ============================================================

def orientation_polish(
    blocks,
    nodes,
    root
):
    best_blocks = None
    best_metrics = None

    for r0 in range(4):
        for r1 in range(4):
            for r2 in range(4):
                for r3 in range(4):
                    cand = copy.deepcopy(blocks)

                    rotations = [
                        r0,
                        r1,
                        r2,
                        r3
                    ]

                    for b, r in zip(
                        cand,
                        rotations
                    ):
                        b.rotation = r

                    metrics = evaluate_layout(
                        cand,
                        nodes,
                        root
                    )

                    score = (
                        metrics["area"],
                        metrics["aspect_ratio"]
                    )

                    if (
                        best_metrics is None
                        or
                        score
                        <
                        (
                            best_metrics["area"],
                            best_metrics["aspect_ratio"]
                        )
                    ):
                        best_blocks = copy.deepcopy(
                            cand
                        )

                        best_metrics = metrics.copy()

    return (
        best_blocks,
        best_metrics
    )


# ============================================================
# 13. 绘图
# ============================================================

def plot_layout(
    blocks,
    W,
    H,
    save_path,
    title
):
    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    for b in blocks:
        parts = absolute_parts(b)

        for p in parts:
            ax.add_patch(
                Rectangle(
                    (p.x, p.y),
                    p.w,
                    p.h,
                    fill=False,
                    linewidth=2.0
                )
            )

        bw, bh = block_bbox(b)

        ax.text(
            b.x+bw/2,
            b.y+bh/2,
            f"{b.name}\n{b.rotation*90}°",
            ha="center",
            va="center",
            fontsize=11
        )

    ax.add_patch(
        Rectangle(
            (0, 0),
            W,
            H,
            fill=False,
            linewidth=2.5,
            linestyle="--"
        )
    )

    ax.set_xlim(
        -0.3,
        W+0.3
    )

    ax.set_ylim(
        -0.3,
        H+0.3
    )

    ax.set_xticks(
        range(W+1)
    )

    ax.set_yticks(
        range(H+1)
    )

    ax.grid(
        alpha=0.2
    )

    ax.set_aspect(
        "equal",
        adjustable="box"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=220
    )

    plt.close()


def plot_convergence(
    history,
    save_path
):
    if not history:
        return

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        [
            item["iteration"]
            for item in history
        ],
        [
            item["best_area"]
            for item in history
        ],
        linewidth=1.2
    )

    plt.xlabel("Iteration")
    plt.ylabel("Best Outline Area")
    plt.title("Question 4 SA Convergence")

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=220
    )

    plt.close()


# ============================================================
# 14. 单个随机种子
# ============================================================

def run_one_seed(
    base_blocks,
    seed,
    area_lower_bound
):
    blocks = copy.deepcopy(
        base_blocks
    )

    nodes, root, start = select_initial_tree(
        blocks,
        seed,
        trials=30
    )

    print(
        f"\nSeed={seed} | "
        f"起点 A={start['area']} | "
        f"W×H={start['W']}×{start['H']}"
    )

    T0 = max(
        1.0,
        0.08*start["area"]
    )

    (
        g_blocks,
        g_nodes,
        g_metrics,
        g_history
    ) = sa_core(
        blocks,
        nodes,
        root,
        seed,
        T0=T0,
        min_temp=0.05,
        cooling_rate=0.90,
        moves_per_temp=120,
        perturb_fun=adaptive_perturb,
        tag="Global",
        area_lower_bound=area_lower_bound
    )

    # 对全局阶段得到的拓扑做256种旋转组合精修
    (
        r_blocks,
        r_metrics
    ) = orientation_polish(
        g_blocks,
        g_nodes,
        root
    )

    if (
        r_metrics["area"],
        r_metrics["aspect_ratio"]
    ) < (
        g_metrics["area"],
        g_metrics["aspect_ratio"]
    ):
        g_blocks = r_blocks
        g_metrics = r_metrics

    # 如果已经达到面积理论下界，不再浪费计算
    if g_metrics["area"] == area_lower_bound:
        final_blocks = g_blocks
        final_nodes = g_nodes
        final_metrics = g_metrics
        polish_history = []

    else:
        T1 = max(
            0.5,
            0.02*g_metrics["area"]
        )

        (
            final_blocks,
            final_nodes,
            final_metrics,
            polish_history
        ) = sa_core(
            g_blocks,
            g_nodes,
            root,
            seed+100000,
            T0=T1,
            min_temp=0.01,
            cooling_rate=0.92,
            moves_per_temp=160,
            perturb_fun=local_perturb,
            tag="Polish",
            area_lower_bound=area_lower_bound
        )

    W, H, A = pack_bstar_tree(
        final_blocks,
        final_nodes,
        root
    )

    neg, overlap = validate_layout(
        final_blocks
    )

    return {
        "seed": seed,
        "W": W,
        "H": H,
        "area": A,
        "aspect_ratio": max(W, H)/min(W, H),
        "negative_count": len(neg),
        "overlap_count": len(overlap),
        "blocks": final_blocks,
        "nodes": final_nodes,
        "root": root,
        "history": g_history+polish_history,
    }


# ============================================================
# 15. 主程序
# ============================================================

def main():
    script_dir = Path(
        __file__
    ).resolve().parent

    output_dir = (
        script_dir
        / "output"
        / "问题4_异形模块"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    blocks = build_blocks()

    total_block_area = sum(
        b.area
        for b in blocks
    )

    # 任意合法布局的外包矩形面积不能小于模块真实总面积
    area_lower_bound = total_block_area

    print("="*96)
    print("A题问题4：B*-Tree + Shape-aware Skyline + SA")
    print("="*96)
    print(f"b1(T型)面积：{blocks[0].area}")
    print(f"b2(L型)面积：{blocks[1].area}")
    print(f"b3(矩形)面积：{blocks[2].area}")
    print(f"b4(矩形)面积：{blocks[3].area}")
    print(f"模块总面积：{total_block_area}")
    print(f"轮廓面积理论下界：{area_lower_bound}")
    print("="*96)

    seeds = [
        2026,
        2027,
        2028,
        42,
        1107
    ]

    results = []

    for seed in seeds:
        result = run_one_seed(
            blocks,
            seed,
            area_lower_bound
        )

        results.append(result)

        print(
            f"Seed={seed} 完成 | "
            f"A={result['area']} | "
            f"W×H={result['W']}×{result['H']} | "
            f"ratio={result['aspect_ratio']:.4f} | "
            f"overlap={result['overlap_count']}"
        )

    best = min(
        results,
        key=lambda r: (
            r["area"],
            r["aspect_ratio"]
        )
    )

    areas = [
        r["area"]
        for r in results
    ]

    print("\n"+"="*96)
    print("【问题4最终结果】")
    print("="*96)
    print(
        f"理论下界       ：{area_lower_bound}"
    )
    print(
        f"最优轮廓       ：{best['W']}×{best['H']}"
    )
    print(
        f"最小轮廓面积   ：{best['area']}"
    )
    print(
        f"长宽比         ：{best['aspect_ratio']:.6f}"
    )
    print(
        f"模块面积利用率 ：{total_block_area/best['area']:.4%}"
    )
    print(
        f"重叠模块对数   ：{best['overlap_count']}"
    )
    print(
        f"面积均值       ：{statistics.mean(areas):.3f}"
    )
    print(
        f"面积标准差     ：{statistics.pstdev(areas):.3f}"
    )

    if best["area"] == area_lower_bound:
        print(
            "全局最优性核验 ：PASS "
            "(所得面积等于模块总面积理论下界)"
        )
    else:
        print(
            "全局最优性核验 ：未达到面积理论下界"
        )

    # 最优布局坐标
    print("\n【最优模块状态】")

    for b in best["blocks"]:
        print(
            f"{b.name}: "
            f"x={b.x}, y={b.y}, "
            f"rotation={b.rotation*90}°"
        )

    # 保存汇总
    csv_path = (
        output_dir
        / "问题4_五随机种子汇总.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "seed",
                "W",
                "H",
                "area",
                "aspect_ratio",
                "overlap_count"
            ]
        )

        for r in results:
            writer.writerow(
                [
                    r["seed"],
                    r["W"],
                    r["H"],
                    r["area"],
                    r["aspect_ratio"],
                    r["overlap_count"]
                ]
            )

    layout_path = (
        output_dir
        / "问题4_最优布局.png"
    )

    plot_layout(
        best["blocks"],
        best["W"],
        best["H"],
        layout_path,
        (
            "Question 4 - Irregular HardBlock Layout "
            f"(Area={best['area']})"
        )
    )

    curve_path = (
        output_dir
        / "问题4_最优Seed收敛曲线.png"
    )

    plot_convergence(
        best["history"],
        curve_path
    )

    print("\n结果已保存：")
    print(csv_path)
    print(layout_path)
    print(curve_path)
    print("="*96)


if __name__ == "__main__":
    main()
