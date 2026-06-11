import numpy as np
import pandas as pd
def grey_relation_analysis(reference,sequences,method='initial',rho=0.5):
    """
        灰色关联分析主函数
        参数：
            reference: 参考序列（母序列），形状为 (n,) 的一维数组
            sequences: 比较序列（子序列），形状为 (m, n) 的二维数组
                       m = 比较序列个数，n = 每个序列的长度
            method: 无量纲化方法
                'initial'  — 初值化（除以第一个值）
                'mean'     — 均值化（除以均值）
                'range'    — 极差标准化（推荐，结果在[0,1]）
            rho: 分辨系数，默认0.5，通常取[0,1]
        返回：
            result: DataFrame，包含各序列的关联度和排名
            gamma: 关联度数组
            xi_matrix: 关联系数矩阵 (m x n)
            processed_data: 无量纲化后的数据（第一行为参考序列）
    """
    ref=np.array(reference,dtype=float).ravel()
    seqs=np.array(sequences,dtype=float)
    if seqs.ndim==1:
        seqs=seqs.reshape(1,-1)
    m,n=seqs.shape
    all_data=np.vstack([ref,seqs])
    if method=='initial':
        processed=all_data/all_data[:,[0]]
    elif method=='mean':
        processed=all_data/all_data.mean(axis=1,keepdims=True)
    elif method=='range':
        min_vals=all_data.min(axis=1,keepdims=True)
        max_vals=all_data.max(axis=1,keepdims=True)
        range_vals=max_vals-min_vals
        range_vals[range_vals==0]=1
        processed=(all_data-min_vals)/range_vals
    else:
        raise ValueError(f"未知的无量纲化方法:{method}")
    ref_processed=processed[0]
    seqs_processed=processed[1:]
    abs_diff=np.abs(seqs_processed-ref_processed)
    delta_min=abs_diff.min()
    delta_max=abs_diff.max()
    xi_matrix=(delta_min+rho*delta_max)/(abs_diff+rho*delta_max)
    gamma=xi_matrix.mean(axis=1)
    rank=np.argsort(-gamma)
    result=pd.DataFrame({
        '比较序列':[f'序列{i+1}' for i in range(m)],
        '关联度':np.round(gamma,4),
        '排名':np.zeros(m,dtype=int)
    })
    result.loc[rank,'排名']=np.arange(1,m+1)
    result=result.sort_values('排名')
    return result,gamma,xi_matrix,processed
def grey_relation_evaluation(data,reference_type='max',method='range',rho=0.5):
    """
        从指标数据直接进行灰色关联评价
        参数：
            data: 决策矩阵 (n_schemes x n_indicators)
            reference_type: 参考序列构建方式
                'max' — 每个指标取最大值（越大越好）
                'min' — 每个指标取最小值（越小越好）
                'specified' — 需自行提供reference
            method, rho: 同 grey_relation_analysis
        返回：
            result: 排序后的结果DataFrame
            gamma: 关联度数组
    """
    data=np.array(data,dtype=float)
    n,m=data.shape
    if reference_type=='max':
        reference=data.max(axis=0)
    elif reference_type=='min':
        reference=data.min(axis=0)
    else:
        raise ValueError("reference_type只能为'max'或'min'")
    result,gamma,xi_matrix,processed=grey_relation_analysis(
        reference,data,method=method,rho=rho
    )
    result['原始序号']=range(1,n+1)
    result=result.rename(columns={'比较序列':'方案'})
    return result,gamma
"""
def plot_grey_relation(reference, sequences, labels=None, title='灰色关联分析'):
    import matplotlib.pyplot as plt
    ref = np.array(reference)
    seqs = np.array(sequences)
    n_points = len(ref)
    plt.figure(figsize=(10, 6))
    x = np.arange(1, n_points + 1)
    # 绘参考序列
    plt.plot(x, ref, 'k-', linewidth=3, label='参考序列', marker='o')
    # 绘比较序列
    if labels is None:
        labels = [f'序列{i+1}' for i in range(len(seqs))]
    for i, seq in enumerate(seqs):
        plt.plot(x, seq, '--', linewidth=1.5, label=labels[i], marker='s')
    plt.xlabel('评价指标', fontsize=12)
    plt.ylabel('指标值', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
"""
"""
if __name__ == '__main__':
    print("=" * 60)
    print("灰色关联分析示例")
    print("=" * 60)
    # === 示例1：基础用法 ===
    print("\n--- 示例1：基础灰色关联分析 ---")
    ref = [1, 2, 3, 4, 5]
    seqs = np.array([
        [1.1, 2.2, 2.9, 4.1, 5.0],
        [1.0, 1.0, 2.0, 3.0, 4.0],
        [5.0, 4.0, 3.0, 2.0, 1.0],
    ])
    result, gamma, xi, processed = grey_relation_analysis(ref, seqs, method='initial', rho=0.5)
    print("关联度结果：")
    print(result)
    print("\n关联系数矩阵：")
    print(np.round(xi, 4))
"""