'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-10-25 15:42:56
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-25 23:46:43
FilePath: /2024_tjjm/src/utils/visualization.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE%E5%8E%9F%E5%9B%9E
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False


def scatter_plot(file_path):
    """
    This function reads an Excel file, iterates through each worksheet, and creates a scatter plot
    for each column containing missing values (0) to compare actual values and predicted values.
    It only examines the first attribute column with missing values for the first province.

    Parameters:
    file_path (str): Path to the Excel file containing the data.

    Returns:
    None. The function displays scatter plots and doesn't return any value.
    -----------
    该函数读取一个 Excel 文件，迭代遍历每一个工作表，并为每一个包含缺失值（0）的列
    创建一个散点图来对比实际值和预测值，这里只查看第一个省的第一个包含缺失值的属性列。

    参数:
    file_path (str): 指向包含数据的 Excel 文件的路径。

    返回:
    None. 该函数显示散点图，不返回任何值。
    """

    # Read Excel file
    xls = pd.ExcelFile(file_path)

    # Iterate through each worksheet in the Excel file
    for sheet_name in xls.sheet_names:
        # Read data from the current worksheet
        df = pd.read_excel(file_path, sheet_name=sheet_name)

        # Replace missing values (NaN) with 0
        df.replace(np.nan, 0, inplace=True)

        # Get columns containing missing values (0)
        cols_with_missing_values = df.columns[(df == 0).any()].tolist()

        # Iterate through each column containing missing values (0)
        for i, col in enumerate(cols_with_missing_values):
            # Extract feature column (year) and target column (value)
            X = df.loc[df[col]!= 0, '年份'].values.reshape(-1, 1)
            y = df.loc[df[col]!= 0, col].values

            # Create a scatter plot to compare actual values and predicted values
            plt.figure(figsize=(10, 6))
            plt.scatter(X, y, color='#467F79', label='数据')
            plt.plot(X,y, color='#2E4F4A', label='数据线')

            # Set scatter plot title, labels, and legend
            plt.title(f'各个年份的{col}值对比图')
            plt.xlabel('年份')
            plt.ylabel(f'{col}')
            plt.legend()
            
            # Display scatter plot
            plt.show()
            break

        break

def check_effect_of_poly_regression_Imputation(input_file_path):
    """
    This function reads an Excel file, iterates through each subsheet, performs polynomial regression
    on each selected column, and plots the results. Here we only examine the fitting situation for two selected attribute columns.
    It also predicts missing values (0) and marks them on the plot.

    Parameters:
    input_file_path (str): Path to the input Excel file containing the original data.

    Returns:
    None. The function displays the plotted charts and doesn't return any value.
    ----------
    该函数读取一个 Excel 文件，对每一个子表进行迭代，对每一个选定的列执行多项式回归并绘制结果。在这里我们只查看了选择了两个属性列的拟合情况
    它还预测了缺失值（0）并在图上标记它们。

    参数:
    input_file_path (str): 输入 Excel 文件的路径，包含原始数据。

    返回:
    None. 该函数显示绘制的图表，不返回任何值。
    """
    # Read Excel file
    file_path = input_file_path
    xls = pd.ExcelFile(file_path)

    # Loop through each subsheet
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # Replace empty data with 0
        df.replace(np.nan, 0, inplace=True)
        
        # Create chart
        plt.figure(figsize=(10, 6))
        
        # Configure colors
        Line_colors = ['#2E4F4A', '#467F79']
        origin_dot_colors = ['#2E4F4A', '#467F79']
        filled_dot_colors = ['#2E4F4A', '#467F79']
        interval_colors = ['#2E4F4A', '#B9D5BA']

        # Select two attribute columns with similar y values
        cols_to_plot = ['城乡居民消费水平（元/人）', '居民人均消费支出（元/人）']
        
        # Process each selected attribute column
        for i, col in enumerate(cols_to_plot):
            # Extract feature column and target column
            X = df.loc[df[col] != 0, '年份'].values.reshape(-1, 1)
            y = df.loc[df[col] != 0, col].values
            
            # Use polynomial feature transformation
            poly_features = PolynomialFeatures(degree=2)
            X_poly = poly_features.fit_transform(X)
            
            # Build polynomial regression model
            poly_model = LinearRegression()
            poly_model.fit(X_poly, y)
            
            # Predict attribute values for cases where missing value is 0
            X_pred = df.loc[df[col] == 0, '年份'].values.reshape(-1, 1)
            X_pred_poly = poly_features.transform(X_pred)
            y_pred = poly_model.predict(X_pred_poly)
            
            # Update attribute column where missing value is 0
            df.loc[df[col] == 0, col] = y_pred
            
            # Plot regression results
            X_all = df['年份'].values.reshape(-1, 1)
            plt.plot(X_all, poly_model.predict(poly_features.transform(X_all)), label=f'{col}', color=Line_colors[i])
            plt.scatter(X, y, color=origin_dot_colors[i])
            plt.scatter(X_pred, y_pred, color=filled_dot_colors[i], marker='x', label=f"{col}预测点")
            
            # Calculate confidence interval for each X value
            y_pred_interval = np.zeros((len(X_all), 2))
            for j, x_val in enumerate(X_all):
                # For each X value, calculate the standard error of the prediction
                y_pred_poly = poly_model.predict(poly_features.transform([x_val]))
                y_pred_samples = []
                for _ in range(100):
                    # Random sampling 100 times
                    y_pred_sample = np.random.normal(y_pred_poly[0], np.sqrt(np.mean((y - poly_model.predict(X_poly))**2)))
                    y_pred_samples.append(y_pred_sample)
                # Calculate upper and lower bounds of confidence interval
                y_pred_interval[j, 0] = np.percentile(y_pred_samples, 2.5)
                y_pred_interval[j, 1] = np.percentile(y_pred_samples, 97.5)
            plt.fill_between(np.squeeze(X_all), y_pred_interval[:, 0], y_pred_interval[:, 1], alpha=0.2, color=interval_colors[i], label=f"{col}误差区间")


        plt.title(f'{sheet_name}')
        plt.xlabel('年份')
        plt.legend()
        plt.grid(False)

        plt.show()
