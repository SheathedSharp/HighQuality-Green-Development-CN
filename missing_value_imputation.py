'''
Author: zixian zhu <21zxzhu@stu.edu.cn>
Date: 2024-04-29 22:34:40
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-05-15 02:03:49
Description: 缺失值填充
env: Python 3.10
'''

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def check_missing_value(file_path):
    """
    该函数读取一个 Excel 文件，识别出缺失值，并用 NaN 代替。
    然后，它会计算并打印出 DataFrame 中每一列的缺失值计数。

    参数:
    file_path (str): 要处理的 Excel 文件的路径。

    返回:
    None. 该函数会打印出每一列的缺失值计数。
    """

    # 读取 Excel 文件
    df = pd.read_excel(file_path)

    # 定义缺失值
    missing_values = [0, 'N/A', 'NaN', '']

    # 填充缺失值
    df.replace(missing_values, np.nan, inplace=True)

    # 统计每列的缺失值
    missing_values_count = df.isnull().sum()

    # 打印每列的缺失值计数
    print(missing_values_count)

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


def split_data_by_province(input_file_path, output_file_path):
    """
    该函数读取一个 Excel 文件，提取出每一个省份的数据，并将其保存到一个新的 Excel 文件中，
    其中每一个省份的数据都保存在一个单独的工作表中。

    参数:
    input_file_path (str): 输入 Excel 文件的路径，包含原始数据。
    output_file_path (str): 输出 Excel 文件的路径，将提取出的数据保存到此文件中。

    返回:
    None. 该函数将数据保存到一个新的 Excel 文件中，不返回任何值。
    """

    # 读取 Excel 文件
    df = pd.read_excel(input_file_path)

    # 获取地区列表
    regions = df['地区名称'].unique()

    # 创建一个 ExcelWriter 对象来保存子表
    writer = pd.ExcelWriter(output_file_path, engine='xlsxwriter')

    # 遍历地区，筛选数据并保存到不同的子表中
    for region in regions:
        # 筛选出当前省份的数据
        region_data = df[df['地区名称'] == region]

        # 将筛选出的数据保存到一个单独的工作表中
        region_data.to_excel(writer, sheet_name=region, index=False)

    # 关闭 ExcelWriter 对象
    writer.close()

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

def missing_values_Imputation_poly_regression(input_file_path, output_file_path):
    """
    该函数读取 Excel 文件，对每一个子表进行迭代，并对每一列中包含的缺失值（0）进行填充。
    它首先使用多项式回归来创建散点图，并对比实际值和预测值。
    如果 R-squared 值小于 0.35，则使用正态分布采样填充数据
    如果 R-squared 值大于或等于 0.35，，则使用多项式回归模型的预测值来填充缺失值。

    参数:
    input_file_path (str): 包含数据的 Excel 文件的路径。
    output_file_path(str): 输出 Excel 文件的路径，将填充了缺失值的新表格保存为新 Excel 文件。

    返回:
    None. 打印填充了缺失值的新表格保存为新 Excel 文件的文件路径。
    """
    # 读取 Excel 文件
    file_path = input_file_path
    xls = pd.ExcelFile(file_path)

    # 循环遍历每个子表
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # 将空数据替换为 0
        df.replace(np.nan, 0, inplace=True)

        # 获取具有缺失值 0 的属性列
        cols_with_missing_values = df.columns[(df == 0).any()].tolist()
        
        # 循环处理每个选定的属性列
        for i, col in enumerate(cols_with_missing_values):
            # 提取特征列和目标列
            X = df.loc[df[col] != 0, '年份'].values.reshape(-1, 1)
            y = df.loc[df[col] != 0, col].values
            
            # 检查样本数量是否大于等于2，以确保至少有一个样本
            if len(X) >= 2:
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

                # 将所有的负预测值进行指数变换
                # y_pred = np.where(y_pred < 0, 0, y_pred)

                # 建立多项式回归模型
                poly_model = LinearRegression()
                poly_model.fit(X_poly, y)
                
                # 计算 R-squared (coefficient of determination)
                r_squared = poly_model.score(X_poly, y)

                if r_squared < 0.35:
                    print(f'R-squared for {col}: {r_squared}')
                    # 创建图表
                    plt.figure(figsize=(10, 6)) 

                    plt.scatter(X, y, color='#2E4F4A', label='实际数据')
                    plt.plot(X, poly_model.predict(poly_features.transform(X)), label=f'{col}多项式拟合曲线', color='#2E4F4A')

                    # 计算预测值与实际值之间的残差
                    residuals = y - poly_model.predict(poly_features.transform(X))
                    residual_std = np.std(residuals)

                    # 计算预测误差区间的上下限
                    upper_bound = poly_model.predict(poly_features.transform(X)) + residual_std
                    lower_bound = poly_model.predict(poly_features.transform(X)) - residual_std

                    # 添加预测误差区间的阴影部分
                    plt.fill_between(X.flatten(), lower_bound, upper_bound, color='#B9D5BA', alpha=0.3, label='预测误差区间')

                    plt.xlabel('年份')
                    plt.ylabel(col)
                    plt.title(f'{sheet_name}的{col}')
                    plt.legend()
                    plt.show()

                    # 为 X 建立正态分布，然后更新缺失值为正态分布中置信区间为 80% 的随机数据
                    mean, std = np.mean(y), np.std(y)
                    X_norm = np.linspace(mean - 3*std, mean + 3*std, 100)
                    y_norm = norm.pdf(X_norm, loc=mean, scale=std)

                    y_zeros = [0] * len(X)
                    # 计算对应 X 值的正态分布的 Y 值
                    y_values = norm.pdf(y, loc=mean, scale=std)
                    # 创建图表
                    plt.figure(figsize=(10, 6)) 
                    plt.plot(X_norm, y_norm, color='#2E4F4A', label=f'{col}的正态分布曲线')
                    confidence_interval = norm.interval(0.80, loc=mean, scale=std)
                    # 确定在置信区间内的索引范围
                    index_start = np.argmax(X_norm >= confidence_interval[0])
                    index_end = np.argmax(X_norm >= confidence_interval[1])

                    # 筛选出在置信区间内的 Y 的取值范围
                    y_confidence_interval = y_norm[index_start:index_end]

                    # 确定对应的 X 的取值范围
                    X_confidence_interval = X_norm[index_start:index_end]

                    plt.scatter(y, y_values, color='#2E4F4A', label='实际数据')
                    # 添加置信区间的阴影部分
                    plt.fill_between(X_confidence_interval, 0, y_confidence_interval, color='#B9D5BA', alpha=0.3, label='80%可信度的置信区间')
                    # print("confidence_interval", confidence_interval)
                    df.loc[df[col] == 0, col] = np.random.uniform(confidence_interval[0], confidence_interval[1], size=np.sum(df[col] == 0))    
                    plt.legend()  # 添加图例
                    plt.xlabel(f'{col}')
                    plt.title(f'{sheet_name}的{col}正态分布表')
                    plt.show()
                else:
                    # 更新缺失值为 0 的属性列
                    df.loc[df[col] == 0, col] = y_pred
                    
            # 如果样本数量等于1，则用样本填充缺失值
            elif len(X) == 1:
                df.loc[df[col] == 0, col] = y[0]
                
            # 如果样本数量为0，则用0填充
            else:
                df.loc[df[col] == 0, col] = 0     
            
        # 保存更新后的子表
        df.to_excel(f'./data/statistical/filled_NaN_xlsx/{sheet_name}_NaN_filled_polynomial.xlsx', index=False)

    # 文件夹路径
    folder_path = './data/statistical/filled_NaN_xlsx'

    # 读取文件夹中的所有 Excel 文件并合并
    dfs = []
    for file in os.listdir(folder_path):
        if file.endswith('.xlsx'):
            file_path = os.path.join(folder_path, file)
            df = pd.read_excel(file_path)
            dfs.append(df)

    # 合并所有 DataFrame
    merged_df = pd.concat(dfs, ignore_index=True)

    # 保存合并后的 DataFrame 到一个 Excel 文件中
    merged_df.to_excel(output_file_path, index=False)

    print(f'Merged DataFrame saved to {output_file_path}')


if __name__ == '__main__':
    """正常的顺序如下
    1. 先查看缺失值情况
    2. 将数据集按照省份分割
    3. 查看散点图
    4. 查看多项式回归填充效果
    5. 使用多项式回归填充缺失值
    """

    # # 缺失值查看
    # check_missing_value('./data/statistical/data_origin.xlsx')

    # # 将数据分割成若干个省份的数据
    # split_data_by_province('./data/statistical/data_origin','./data/statistical/data_region.xlsx' )
    
    # # 散点查看
    # scatter_plot('./data/statistical/data_region.xlsx')

    # # 查看多项式回归的填充效果
    # check_effect_of_poly_regression_Imputation('./data/statistical/data_region.xlsx')
    
    # # 使用多项式回归填充缺失值
    # missing_values_Imputation_poly_regression('./data/statistical/data_region.xlsx', './data/statistical/data_region_merged.xlsx')


