'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-10-25 15:42:56
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-25 16:05:08
FilePath: /2024_tjjm/src/utils/visualization.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
def scatter_plot(file_path):
    """
    该函数读取一个 Excel 文件，迭代遍历每一个工作表，并为每一个包含缺失值（0）的列
    创建一个散点图来对比实际值和预测值，这里只查看第一个省的第一个包含缺失值的属性列。

    参数:
    file_path (str): 指向包含数据的 Excel 文件的路径。

    返回:
    None. 该函数显示散点图，不返回任何值。
    """

    # 读取 Excel 文件
    xls = pd.ExcelFile(file_path)

    # 迭代遍历 Excel 文件中的每一个工作表
    for sheet_name in xls.sheet_names:
        # 从当前工作表中读取数据
        df = pd.read_excel(file_path, sheet_name=sheet_name)

        # 将缺失值（NaN）替换为 0
        df.replace(np.nan, 0, inplace=True)

        # 获取包含缺失值（0）的列
        cols_with_missing_values = df.columns[(df == 0).any()].tolist()

        # 迭代遍历每一个包含缺失值（0）的列
        for i, col in enumerate(cols_with_missing_values):
            # 提取特征列（年份）和目标列（值）
            X = df.loc[df[col]!= 0, '年份'].values.reshape(-1, 1)
            y = df.loc[df[col]!= 0, col].values

            # 创建一个散点图来对比实际值和预测值
            plt.figure(figsize=(10, 6))
            plt.scatter(X, y, color='#467F79', label='数据')
            plt.plot(X,y, color='#2E4F4A', label='数据线')

            # 设置散点图的标题、标签和图例
            plt.title(f'各个年份的{col}值对比图')
            plt.xlabel('年份')
            plt.ylabel(f'{col}')
            plt.legend()
            
            # 显示散点图
            plt.show()
            break

        break

def check_effect_of_poly_regression_Imputation(input_file_path):
    """
    该函数读取一个 Excel 文件，对每一个子表进行迭代，对每一个选定的列执行多项式回归并绘制结果。在这里我们只查看了选择了两个属性列的拟合情况
    它还预测了缺失值（0）并在图上标记它们。

    参数:
    input_file_path (str): 输入 Excel 文件的路径，包含原始数据。

    返回:
    None. 该函数显示绘制的图表，不返回任何值。
    """
    # 读取 Excel 文件
    file_path = input_file_path
    xls = pd.ExcelFile(file_path)

    # 循环遍历每个子表
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # 将空数据替换为 0
        df.replace(np.nan, 0, inplace=True)
        
        # 创建图表
        plt.figure(figsize=(10, 6))
        
        # 配置颜色
        Line_colors = ['#2E4F4A', '#467F79']
        origin_dot_colors = ['#2E4F4A', '#467F79']
        filled_dot_colors = ['#2E4F4A', '#467F79']
        interval_colors = ['#2E4F4A', '#B9D5BA']

        # 选择 y 值相差不多的两个属性列
        cols_to_plot = ['城乡居民消费水平（元/人）', '居民人均消费支出（元/人）']
        
        # 循环处理每个选定的属性列
        for i, col in enumerate(cols_to_plot):
            # 提取特征列和目标列
            X = df.loc[df[col] != 0, '年份'].values.reshape(-1, 1)
            y = df.loc[df[col] != 0, col].values
            
            # 使用多项式特征转换
            poly_features = PolynomialFeatures(degree=2)
            X_poly = poly_features.fit_transform(X)
            
            # 建立多项式回归模型
            poly_model = LinearRegression()
            poly_model.fit(X_poly, y)
            
            # 预测缺失值为 0 的情况下的属性值
            X_pred = df.loc[df[col] == 0, '年份'].values.reshape(-1, 1)
            X_pred_poly = poly_features.transform(X_pred)
            y_pred = poly_model.predict(X_pred_poly)
            
            # 更新缺失值为 0 的属性列
            df.loc[df[col] == 0, col] = y_pred
            
            # 绘制回归结果
            X_all = df['年份'].values.reshape(-1, 1)
            plt.plot(X_all, poly_model.predict(poly_features.transform(X_all)), label=f'{col}', color=Line_colors[i])
            plt.scatter(X, y, color=origin_dot_colors[i])
            plt.scatter(X_pred, y_pred, color=filled_dot_colors[i], marker='x', label=f"{col}预测点")
            
            # 计算每个X值对应的置信区间
            y_pred_interval = np.zeros((len(X_all), 2))
            for j, x_val in enumerate(X_all):
                # 对于每个X值，计算预测值的标准误差
                y_pred_poly = poly_model.predict(poly_features.transform([x_val]))
                y_pred_samples = []
                for _ in range(100):
                    # 随机抽样100次
                    y_pred_sample = np.random.normal(y_pred_poly[0], np.sqrt(np.mean((y - poly_model.predict(X_poly))**2)))
                    y_pred_samples.append(y_pred_sample)
                # 计算置信区间的上下限
                y_pred_interval[j, 0] = np.percentile(y_pred_samples, 2.5)
                y_pred_interval[j, 1] = np.percentile(y_pred_samples, 97.5)
            plt.fill_between(np.squeeze(X_all), y_pred_interval[:, 0], y_pred_interval[:, 1], alpha=0.2, color=interval_colors[i], label=f"{col}误差区间")


        plt.title(f'{sheet_name}')
        plt.xlabel('年份')
        plt.legend()
        plt.grid(False)

        plt.show()