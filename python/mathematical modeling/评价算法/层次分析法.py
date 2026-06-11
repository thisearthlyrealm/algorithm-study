import numpy as np
def weight_geometric_mean(matrix):
    #几何平均法：行元素乘积的 n 次方根，再归一化
    n=matrix.shape[0]
    geometric=np.prod(matrix,axis=1)**(1/n)
    weights=geometric/geometric.sum()
    return weights
def weight_arithmetic_mean(matrix):
    #算术平均法：行和除以所有元素总和
    weights=matrix.sum(axis=1)/matrix.sum()
    return weights
def weight_eigenvector(matrix):
    #特征值法：最大特征值对应的特征向量，归一化
    eigvals,eigvecs=np.linalg.eig(matrix)
    max_idx=np.argmax(eigvals.real)
    weights=eigvecs[:,max_idx].real
    weights=weights/weights.sum()
    return weights
def weight_sum_product(matrix):
    #和积法：列归一化后行平均
    col_sum=matrix.sum(axis=0)
    col_norm=matrix/col_sum
    weights=col_norm.mean(axis=1)
    return weights
def compute_lambda_max(matrix,weights):
    #根据判断矩阵和权重向量计算最大特征值
    aw=matrix@weights
    lambda_max=(aw/weights).mean()
    return lambda_max
def consistency_test(matrix,weights,n):
    """
        一致性检验：计算 CI, RI, CR
        返回 (CI, RI, CR) 和是否通过
    """
    lambda_max=compute_lambda_max(matrix,weights)
    CI=(lambda_max-n)/(n-1)
    RI_dict={
        # 随机一致性指标 RI (Saaty 表)
        1:0,2:0,3:0.58,4:0.9,5:1.12,
        6:1.24,7:1.32,8:1.41,9:1.45,10:1.49
    }
    RI=RI_dict.get(n,1.5)
    CR=CI/RI if RI!=0 else 0
    pass_flag=CR<0.1 if n>2 else True
    return CI,RI,CR,pass_flag
def print_consistency(matrix,weights,name):
    #打印一致性检验结果
    n=matrix.shape[0]
    CI,RI,CR,pass_flag=consistency_test(matrix,weights,n)
    print(f"{name} 方法：λ_max={compute_lambda_max(matrix,weights):.4f} "
          f"CI={CI:.4f}, RI={RI:.4f}, CR={CR:.4f} {'✓通过' if pass_flag else '✗不通过'}")
def ahp_full_analysis(criteria_matrix,scheme_matrices,method="geometric"):
    weight_funcs={
        'geometric':weight_geometric_mean,
        'arithmetic':weight_arithmetic_mean,
        'eigenvector':weight_eigenvector,
        'sum_product':weight_sum_product,
    }
    func=weight_funcs.get(method)
    if func is None:
        raise ValueError(f"未知方法：{method},可选：{list(weight_funcs.keys())}")
    w_criteria=func(criteria_matrix)
    n_criteria=len(w_criteria)
    n_scheme=scheme_matrices[0].shape[0]
    w_scheme=np.zeros((n_criteria,n_scheme))
    for i,mat in enumerate(scheme_matrices):
        w=func(mat)
        w_scheme[i]=w
    total_score=w_criteria@w_scheme
    return w_criteria,w_scheme,total_score
"""
if __name__ == '__main__':
    # 旅行地选择案例
    # 准则层：风景、费用、住宿
    criteria = np.array([
        [1,   3,   7],
        [1/3, 1,   5],
        [1/7, 1/5, 1]
    ])
    # 方案层判断矩阵（风景准则下）
    scheme_scenery = np.array([
        [1,    1/2,  2],
        [2,    1,    3],
        [1/2,  1/3,  1]
    ])
    # 费用准则下
    scheme_cost = np.array([
        [1,    1/3,  1/2],
        [3,    1,    2],
        [2,    1/2,  1]
    ])
    # 住宿准则下
    scheme_lodging = np.array([
        [1,    2,    3],
        [1/2,  1,    2],
        [1/3,  1/2,  1]
    ])
    scheme_matrices = [scheme_scenery, scheme_cost, scheme_lodging]
    # 对每种方法运行并比较
    methods = ['geometric', 'arithmetic', 'eigenvector', 'sum_product']
    for meth in methods:
        print(f"\n========== 方法：{meth} ==========")
        w_crit, w_sch, total = ahp_full_analysis(criteria, scheme_matrices, meth)
        print("准则层权重:", np.round(w_crit, 4))
        print("方案权重矩阵:")
        print(np.round(w_sch, 4))
        print("综合得分:", np.round(total, 4))
        print("最佳方案:", ["A","B","C"][np.argmax(total)])
        # 额外：准则层一致性检验
        print("准则层一致性检验:")
        print_consistency(criteria, w_crit, meth)
        for i, mat in enumerate(scheme_matrices):
            print(f"  准则{i+1}方案层:")
            print_consistency(mat, w_sch[i], meth)
"""