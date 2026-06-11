import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
import statsmodels.api as sm
def linear_regression_analysis(X,Y,plot=True):
    """
        一元线性回归完整分析函数
        参数：
            X: 自变量数组（一维）
            Y: 因变量数组（一维）
            plot: 是否绘制回归图
        返回：
            intercept: 截距
            slope: 斜率
            R_squared: 决定系数
            rmse: 均方根误差
            Y_pred: 预测值数组
    """
    #1.最小二乘法拟合
    X_bar=X.mean()
    Y_bar=Y.mean()
    numerator=((X-X_bar)*(Y-Y_bar)).sum()
    denominator=((X-X_bar)**2).sum()
    slope=numerator/denominator
    intercept=Y_bar-slope*X_bar
    #2.预测
    Y_pred=intercept+slope*X
    SS_res=((Y-Y_pred)**2).sum()
    SS_tot=((Y-Y.mean())**2).sum()
    R_squared=1-SS_res/SS_tot
    mse=((Y-Y_pred)**2).mean()
    rmse=np.sqrt(mse)
    #4.打印结果
    print(f"回归方程：Y={intercept:.4f}+{slope:.4f}*X")
    print(f"R²={R_squared:.4f}")
    print(f"RMSE={rmse:.4f}")
    return intercept,slope,R_squared,rmse,Y_pred
def sklearn_linear_regression(X,Y,plot=True):
    """
        使用 sklearn 进行一元线性回归的封装函数
        参数：
            X: 自变量，一维数组或二维数组 (n,1)
            Y: 因变量，一维数组
            plot: 是否绘制散点图与回归线，默认为 True
        返回：
            intercept: 截距
            slope: 斜率
            R_squared: 决定系数
            rmse: 均方根误差
            Y_pred: 预测值数组
            model: 训练好的 LinearRegression 模型对象
    """
    if X.ndim==1:
        X=X.reshape(-1,1)
    model=LinearRegression()
    model.fit(X,Y)
    intercept=model.intercept_
    slope=model.coef_(0)
    Y_pred=model.predict(X)
    R_squared=r2_score(Y,Y_pred)
    rmse=np.sqrt(mean_squared_error(Y,Y_pred))
    print(f"回归方程：Y={intercept:.4f}+{slope:.4f}*X")
    print(f"R²={R_squared:.4f}")
    print(f"RMSE={rmse:.4f}")
    return intercept,slope,R_squared,rmse,Y_pred,model
def statsmodels_linear_regression(X,Y,print_summary=True,print_params=True,plot=True):
    """
        使用 statsmodels 进行带统计推断的线性回归分析
        参数：
            X: 自变量，一维或二维数组
            Y: 因变量，一维数组
            print_summary: 是否打印完整模型摘要（默认 True）
            print_params: 是否打印截距、斜率及其 p 值（默认 True）
        返回：
            model: 拟合好的 OLS 模型对象（包含 .summary(), .params, .pvalues, .rsquared 等）
            X_with_intercept: 添加截距后的设计矩阵
    """
    if X.ndim==2 and X.shape[1]==1:
        X=X.ravel()
    X_with_intercept=sm.add_constant(X)
    model=sm.OLS(Y,X_with_intercept).fit()
    if print_summary:
        print(model.summary())
    if print_params:
        print(f"截距:{model.params[0]:.4f},p值:{model.pvalues[0]:.6f}")
        print(f"斜率:{model.params[1]:.4f},p值:{model.pvalues[1]:.6f}")
        print(f"R²:{model.rsquared:.4f}")
    return model,X_with_intercept