import numpy as np
import pandas as pd
def entropy_weight_matrix(matrix,positive_indices=None,negative_indices=None,add_small=1e-10):
    """
        熵权法计算权重
        参数：
            matrix: 原始决策矩阵 (n x m)，可以是numpy数组或DataFrame
            positive_indices: 正向指标索引列表（越大越好），默认所有指标
            negative_indices: 负向指标索引列表（越小越好）
            add_small: 极小值，避免log(0)
        返回：
            weights: 各指标权重向量 (m,)
            entropy: 各指标熵值 (m,)
            diff_coeff: 差异系数 (m,)
            norm_matrix: 标准化后的概率矩阵 (n x m)
    """
    if isinstance(matrix,pd.DataFrame):
        matrix=matrix.values
    n,m=matrix.shape
    if positive_indices is None:
        positive_indices=list(range(m))
    if negative_indices is None:
        negative_indices=[]
    standard=np.zeros_like(matrix,dtype=float)
    for j in range(m):
        col=matrix[:,j].astype(float)
        min_j=col.min()
        max_j=col.max()
        if max_j-min_j==0:
            standard[:,j]=(col-min_j)/(max_j-min_j)+add_small
        elif j in negative_indices:
            standard[:,j]=(max_j-col)/(max_j-min_j)+add_small
        else:
            standard[:,j]=(col-min_j)/(max_j-min_j)+add_small
    prob=standard/standard.sum(axis=0,keepdims=True)
    ln_n=np.log(n)
    entropy=np.zeros(m)
    for j in range(m):
        p=prob[:,j]
        entropy[j]=-np.sum(p*np.log(p))/ln_n
    diff_coeff=1-entropy
    weights=diff_coeff/diff_coeff.sum()
    return weights,entropy,diff_coeff,prob
def print_entropy_results(weights,entropy,diff_coeff,col_names=None):
    """打印熵权法结果"""
    m=len(weights)
    if col_names is None:
        col_names=[f'指标{i+1}' for i in range(m)]
    df=pd.DataFrame({
        '指标':col_names,
        '熵值':np.round(entropy,4),
        '差异系数':np.round(diff_coeff,4),
        '权重':np.round(weights,4)
    })
    print(df.to_string(index=False))
"""
if __name__ == '__main__':
    # 示例数据：5个方案，4个指标（前2个正向，后2个负向）
    data = np.array([
        [5, 1200, 8, 500],
        [6, 800, 7, 300],
        [7, 1000, 6, 600],
        [8, 900, 5, 400],
        [9, 1100, 4, 200]
    ])
    # 假设指标3（索引2）和指标4（索引3）是负向指标（越小越好）
    weights, entropy, diff_coeff, prob = entropy_weight_matrix(
        data,
        positive_indices=[0, 1],  # 指标1、2正向
        negative_indices=[2, 3]  # 指标3、4负向
    )
    print_entropy_results(weights, entropy, diff_coeff, ['房价', '面积', '环境', '通勤'])
    print("\n标准化概率矩阵（前5行）:\n", np.round(prob, 4))
"""