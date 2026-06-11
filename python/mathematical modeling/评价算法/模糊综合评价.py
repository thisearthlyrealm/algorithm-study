import numpy as np
import pandas as pd
def trapezoidal_membership(x,a,b,c,d):
    """
        梯形隶属度函数
        参数：
            x: 输入值（标量或数组）
            a, b, c, d: 梯形四个顶点，a≤b≤c≤d
        返回：
            隶属度值（0~1）
    """
    x=np.array(x,dtype=float)
    y=np.zeros_like(x)
    mask1=(x>=a)&(x<b)
    y[mask1]=(x[mask1]-a)/(b-a+1e-10)
    mask2=(x>=b)&(x<=c)
    y[mask2]=1.0
    mask3=(x>c)&(x<=d)
    y[mask3]=(d-x[mask3])/(d-c+1e-10)
    return y
def triangular_membership(x,a,b,c):
    """
        三角形隶属度函数
        参数：
            x: 输入值
            a, b, c: 三角形三个顶点，a≤b≤c
        返回：
            隶属度值（0~1）
    """
    x=np.array(x,dtype=float)
    y=np.zeros_like(x)
    mask1=(x>=a)&(x<b)
    y[mask1]=(x[mask1]-a)/(b-a+1e-10)
    mask2=(x>=b)&(x<=c)
    y[mask2]=(c-x[mask2])/(c-b+1e-10)
    return y
def gaussian_membership(x,mean,sigma):
    """
        高斯型隶属度函数
        参数：
            x: 输入值
            mean: 均值（中心位置）
            sigma: 标准差（宽度）
        返回：
            隶属度值（0~1）
    """
    return np.exp(-((x-mean)**2/(2*sigma**2)))
def build_relation_matrix(data,criteria_ranges):
    """
        构建模糊关系矩阵 R (m x n) 参数：
            data: 各方案的指标值 (n_schemes x m)
            criteria_ranges: 列表，每个元素是 [评语等级数, 各等级的隶属度参数]
                格式示例：
                [['低', '中', '高'],         # 等级名称（仅用于说明）
                    [[0, 0, 25, 50],         # '低'的梯形参数
                        [25, 50, 50, 75],       # '中'的梯形参数
                        [50, 75, 100, 100]      # '高'的梯形参数
                    ]]
        返回：R_list: 列表，每个元素是 (m x n) 的模糊关系矩阵，对应一个方案
    """
    data=np.array(data,dtype=float)
    n_schemes,m=data.shape
    R_list=[]
    for k in range(n_schemes):
        R=np.zeros((m,len(criteria_ranges)))
        for i in range(m):
            for j,params in enumerate(criteria_ranges):
                if len(params)==4:
                    R[i,j]=trapezoidal_membership(data[k,i],*params)
                elif len(params)==3:
                    R[i,j]=triangular_membership(data[k,i],*params)
                elif len(params)==2:
                    R[i,j]=gaussian_membership(data[k,i],*params)
        R_list.append(R)
    return R_list
def fuzzy_composition(weights,R,operator='weighted'):
    """
        模糊合成运算
        参数：
            weights: 权重向量 (m,)
            R: 模糊关系矩阵 (m, n)
            operator: 算子类型
                'weighted' — M(·,⊕) 加权平均型（推荐）
                'maxmin'   — M(∧,∨) 主因素突出型
                'maxprod'  — M(·,∨) 加权主因素型
                'minplus'  — M(∧,⊕) 全面制约型
        返回：
            B: 综合评判向量 (n,)
    """
    w=np.array(weights)
    if operator=='weighted':
        B=w@R
    elif operator=='maxmin':
        B=np.max(np.minimum(w[:,None],R),axis=0)
    elif operator=='maxprod':
        B=np.max(w[:,None],axis=0)
    elif operator=='minplus':
        B=np.minimum(1,np.sum(np.minimum(w[:,None],R),axis=0))
    else:
        raise ValueError(f"未知算子：{operator}")
    return B
def defuzzify_max_membership(B,labels=None):
    """
        最大隶属度原则：取隶属度最大的评语
        参数：
            B: 综合评判向量 (n,)
            labels: 评语标签列表
        返回：
            best_label: 最佳评语标签
            best_index: 最佳评语索引
            max_value: 最大隶属度值
    """
    best_index=np.argmax(B)
    max_value=B[best_index]
    if labels:
        best_label=labels[best_index]
    else:
        best_label=f'等级{best_index+1}'
    return best_label,best_index,max_value
def defuzzify_weighted_average(B,scores=None):
    """
        加权平均原则：对评语赋值后计算总分
        参数：
            B: 综合评判向量 (n,)
            scores: 各评语对应的分数，默认等距赋值
        返回：
            total_score: 综合得分
    """
    if scores is None:
        n=len(B)
        scores=np.linspace(0,100,n)
    total_score=B@scores
    total_score=total_score/(B.sum(+1e-10))
    return total_score
def fuzzy_comprehensive_evaluation(data,weights,criteria_ranges,operator='weighted',method='max'):
    """
        完整模糊综合评价
        参数：
            data: 原始数据 (n_schemes x m)
            weights: 权重向量 (m,)
            criteria_ranges: 隶属度参数列表，格式见 build_relation_matrix
            operator: 模糊算子
            method: 结果解析方式
                'max' — 最大隶属度原则
                'weighted' — 加权平均原则
        返回：
            result_df: 包含各方案评价结果的DataFrame
            B_mat: 所有方案的模糊综合评判矩阵 (n_schemes x n)
    """
    R_list=build_relation_matrix(data,criteria_ranges)
    n_schemes=len(criteria_ranges)
    n_labels=len(criteria_ranges)
    B_mat=np.zeros((n_schemes,n_labels))
    for k in range(n_schemes):
        B_mat[k]=fuzzy_composition(weights,R_list[k],operator)
    label_names=[f'等级{j+1}' for j in range(n_labels)]
    results=[]
    for k in range(n_schemes):
        B=B_mat[k]
        if method=='max':
            label,idx,val=defuzzify_max_membership(B,label_names)
            results.append({
                '方案': f'方案{k+1}',
                '综合评判向量':np.round(B,4),
                '最佳评语':label,
                '隶属度':val
            })
        elif method=='weighted':
            score=defuzzify_weighted_average(B)
            results.append({
                '方案':f'方案{k+1}',
                '综合评判向量':np.round(B,4),
                '综合得分':np.round(score,4)
            })
    result_df=pd.DataFrame(results)
    if method=='weighted' and '综合得分' in result_df.columns:
        result_df=result_df.sort_values('综合得分',ascending=False)
    return result_df,B_mat