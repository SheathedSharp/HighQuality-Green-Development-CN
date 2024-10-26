'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-10-25 15:41:49
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-26 11:20:30
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import os
from scipy.stats import norm

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False


def missing_values_Imputation_poly_regression(input_file_path, output_file_path):
    """
    This function reads an Excel file, iterates through each subsheet, and fills in missing values (0) in each column.
    It first uses polynomial regression to create scatter plots and compares actual values with predicted values.
    If the R-squared value is less than 0.35, it uses normal distribution sampling to fill the data.
    If the R-squared value is greater than or equal to 0.35, it uses the predicted values from the polynomial regression model to fill in missing values.

    Parameters:
    input_file_path (str): Path to the Excel file containing the data.
    output_file_path (str): Path to the output Excel file where the new table with filled missing values will be saved.

    Returns:
    None. Prints the file path of the new Excel file with filled missing values.
    ---------------
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
    # Read Excel file
    file_path = input_file_path
    xls = pd.ExcelFile(file_path)

    # Loop through each subsheet
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # Replace empty data with 0
        df.replace(np.nan, 0, inplace=True)

        # Get attribute columns with missing values 0
        cols_with_missing_values = df.columns[(df == 0).any()].tolist()
        
        # Process each selected attribute column
        for i, col in enumerate(cols_with_missing_values):
            # Extract feature column and target column
            X = df.loc[df[col] != 0, '年份'].values.reshape(-1, 1)
            y = df.loc[df[col] != 0, col].values
            
            # Check if the sample size is greater than or equal to 2 to ensure at least one sample
            if len(X) >= 2:
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

                # Transform all negative predictions using exponential function
                # y_pred = np.where(y_pred < 0, 0, y_pred)

                # Build polynomial regression model
                poly_model = LinearRegression()
                poly_model.fit(X_poly, y)
                
                # Calculate R-squared (coefficient of determination)
                r_squared = poly_model.score(X_poly, y)

                if r_squared < 0.35:
                    print(f'R-squared for {col}: {r_squared}')
                    # Create chart
                    plt.figure(figsize=(10, 6)) 

                    plt.scatter(X, y, color='#2E4F4A', label='实际数据')
                    plt.plot(X, poly_model.predict(poly_features.transform(X)), label=f'{col}多项式拟合曲线', color='#2E4F4A')

                    # Calculate residuals between predicted and actual values
                    residuals = y - poly_model.predict(poly_features.transform(X))
                    residual_std = np.std(residuals)

                    # Calculate upper and lower bounds of prediction error interval
                    upper_bound = poly_model.predict(poly_features.transform(X)) + residual_std
                    lower_bound = poly_model.predict(poly_features.transform(X)) - residual_std

                    # Add shaded area for prediction error interval
                    plt.fill_between(X.flatten(), lower_bound, upper_bound, color='#B9D5BA', alpha=0.3, label='预测误差区间')

                    plt.xlabel('年份')
                    plt.ylabel(col)
                    plt.title(f'{sheet_name}的{col}')
                    plt.legend()
                    plt.show()

                    # Establish normal distribution for X, then update missing values with random data within 80% confidence interval of normal distribution
                    mean, std = np.mean(y), np.std(y)
                    X_norm = np.linspace(mean - 3*std, mean + 3*std, 100)
                    y_norm = norm.pdf(X_norm, loc=mean, scale=std)

                    y_zeros = [0] * len(X)
                    # Calculate Y values of normal distribution corresponding to X values
                    y_values = norm.pdf(y, loc=mean, scale=std)
                    # Create chart
                    plt.figure(figsize=(10, 6)) 
                    plt.plot(X_norm, y_norm, color='#2E4F4A', label=f'{col}的正态分布曲线')
                    confidence_interval = norm.interval(0.80, loc=mean, scale=std)
                    # Determine index range within confidence interval
                    index_start = np.argmax(X_norm >= confidence_interval[0])
                    index_end = np.argmax(X_norm >= confidence_interval[1])

                    # Filter Y value range within confidence interval
                    y_confidence_interval = y_norm[index_start:index_end]

                    # Determine corresponding X value range
                    X_confidence_interval = X_norm[index_start:index_end]

                    plt.scatter(y, y_values, color='#2E4F4A', label='实际数据')
                    # Add shaded area for confidence interval
                    plt.fill_between(X_confidence_interval, 0, y_confidence_interval, color='#B9D5BA', alpha=0.3, label='80%可信度的置信区间')
                    # print("confidence_interval", confidence_interval)
                    df.loc[df[col] == 0, col] = np.random.uniform(confidence_interval[0], confidence_interval[1], size=np.sum(df[col] == 0))    
                    plt.legend()  # Add legend
                    plt.xlabel(f'{col}')
                    plt.title(f'{sheet_name}的{col}正态分布表')
                    plt.show()
                else:
                    # Update attribute column where missing value is 0
                    df.loc[df[col] == 0, col] = y_pred
                    
            # If sample size is 1, fill missing values with the sample
            elif len(X) == 1:
                df.loc[df[col] == 0, col] = y[0]
                
            # If sample size is 0, fill with 0
            else:
                df.loc[df[col] == 0, col] = 0     
            
        # Save updated subsheet
        df.to_excel(f'./data/statistical/filled_NaN_xlsx/{sheet_name}_NaN_filled_polynomial.xlsx', index=False)

    # Folder path
    folder_path = './data/statistical/filled_NaN_xlsx'

    # Read and merge all Excel files in the folder
    dfs = []
    for file in os.listdir(folder_path):
        if file.endswith('.xlsx'):
            file_path = os.path.join(folder_path, file)
            df = pd.read_excel(file_path)
            dfs.append(df)

    # Merge all DataFrames
    merged_df = pd.concat(dfs, ignore_index=True)

    # Save merged DataFrame to an Excel file
    merged_df.to_excel(output_file_path, index=False)

    print(f'Merged DataFrame saved to {output_file_path}')
