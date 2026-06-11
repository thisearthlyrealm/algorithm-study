import numpy as np
import pandas as pd
def forward_matrix(matrix,cost_indices=None,interval_indices=None,interval_ranges=None):
    """
        指标正向化
        输入：
            matrix: 原始决策矩阵 (n x m) numpy数组或DataFrame
            cost_indices: 成本型指标索引列表（列索引，从0开始）
            interval_indices: 区间型指标索引列表
            interval_ranges: 区间型指标的最优区间列表，如 [(low1, high1), (low2, high2)]
        返回：
            正向化后的矩阵（所有指标越大越好）
        """
    matrix=matrix.astype(float)
    if cost_indices is not None:
        for idx in cost_indices:
            matrix[:,idx]=1/(matrix[:,idx]+1e-10)
    if interval_indices is not None and interval_ranges is not None:
        for idx,(low,high) in zip(interval_indices,interval_ranges):
            col=matrix[:,idx]
            min_c,max_c=col.min(),col.max()
            diff_low=abs(low-min_c)
            diff_high=abs(high-max_c)
            M=max(diff_low,diff_high)
            new_col=np.where(
                (col>=low) & (col<=high),
                1,
                1-(np.minimum(np.abs(col-low),np.abs(col-high))/M)if M!=0 else 0
            )
            matrix[:,idx]=new_col
    return matrix
def standardize_matrix(matrix,weighted=None):
    """
        向量归一化标准化，同时可引入权重
        输入：
            matrix: 正向化后的决策矩阵 (n x m)
            weighted: 权重向量（长度m），若为None则等权重
        返回：
            加权标准化后的矩阵
    """
    norm=np.sqrt((matrix**2).sum(axis=0))
    norm[norm==0]=1
    normal_matrix=matrix/norm
    if weighted is not None:
        weighted=np.array(weighted)
        normal_matrix=normal_matrix*weighted
    return normal_matrix
def topsis(matrix_after_norm):
    """
        对加权标准化矩阵计算正理想解、负理想解、距离和贴近度
        输入：
            matrix_after_norm: 加权标准化后的决策矩阵 (n x m)
        返回：
            result DataFrame 包含每个方案的得分 (distance_plus, distance_minus, closeness)
            ideal_pos: 正理想解向量 (m)
            ideal_neg: 负理想解向量 (m)
    """
    n,m=matrix_after_norm.shape
    ideal_pos=matrix_after_norm.max(axis=0)
    ideal_neg=matrix_after_norm.min(axis=0)
    d_plus=np.sqrt(((matrix_after_norm-ideal_pos)**2).sum(axis=1))
    d_minus=np.sqrt(((matrix_after_norm-ideal_neg)**2).sum(axis=1))
    closeness=d_minus/(d_plus+d_minus)
    result=pd.DataFrame({
        '方案':[f'方案{i+1}' for i in range(n)],
        'D+':d_plus,
        'D-':d_minus,
        '贴近度':closeness
    })
    return result.sort_values('贴近度',ascending=False),ideal_pos,ideal_neg
def topsis_analysis(data,weights=None,cost_indices=None,interval_indices=None,interval_ranges=None):
    """
        完整TOPSIS分析入口
        参数：
            data: 原始决策矩阵 (n x m)，可以是DataFrame或numpy数组
            weights: 权重向量（长度m），默认等权重
            cost_indices: 成本型指标索引列表
            interval_indices: 区间型指标索引列表
            interval_ranges: 区间型最优区间列表
        返回：
            result: 排序后的结果DataFrame
            details: 包含正向化后矩阵、标准化矩阵等中间数据（可选）
    """
    if isinstance(data,pd.DataFrame):
        original=data.values.copy()
        col_names=data.columns.tolist()
    else:
        original=data.copy()
        col_names=[f'指标{j+1}' for j in range(data.shape[1])]
    matrix_forward=forward_matrix(original.copy(),cost_indices,interval_indices,interval_ranges)
    matrix_norm=standardize_matrix(matrix_forward,weights)
    result,ideal_pos,ideal_neg=topsis(matrix_norm)
    result['原始序号']=range(1,len(result)+1)
    print("正理想解：",ideal_pos)
    print("负理想解：",ideal_neg)
    return result
"""
if __name__ == '__main__':
    data = np.array([
        [5, 1200, 8, 500],
        [6, 800, 7, 300],
        [7, 1000, 6, 600],
        [8, 900, 5, 400],
        [9, 1100, 4, 200]
    ])
    result = topsis_analysis(data, cost_indices=[1], interval_indices=[3], interval_ranges=[[300, 500]])
    print(result)
"""