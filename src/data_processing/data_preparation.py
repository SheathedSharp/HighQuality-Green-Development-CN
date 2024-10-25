'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-10-25 15:41:23
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-25 23:39:41
FilePath: /2024_tjjm/src/data_processing/data_preparation.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pandas as pd
import numpy as np
from sklearn import preprocessing

def data_prepare(input_file_path):
    """
    This function prepares data for machine learning models.
    It reads data from an Excel file, performs data preprocessing, and splits the data into training and test sets.
    Here, we set the test set as the last three years for each province, and the training set as the remaining years.
    Then it normalizes and reshapes the data to fit LSTM input.

    Parameters:
    input_file_path (str): Path to the Excel file to be processed.

    Returns:
    X_train : numpy.ndarray
        Input features for the training set, reshaped for LSTM input format.
    y_train : pandas.Series
        Target variable for the training set.
    X_test : numpy.ndarray
        Input features for the test set, reshaped for LSTM input format.
    y_test : pandas.Series
        Target variable for the test set.

    -----------------
    该函数为机器学习模型准备数据。
    它从 Excel 文件中读取数据，执行数据预处理，并将数据拆分为训练集和测试集。在这里我们将测试集化为每个省的后三个年份，训练集为剩余年份。
    然后对数据进行归一化和重塑以适用于 LSTM 输入。

    参数:
    input_file_path (str): 要处理的 Excel 文件的路径。


    返回:
    X_train : numpy.ndarray
        用于训练集的输入特征，已重塑为 LSTM 输入格式。
    y_train : pandas.Series
        训练集的目标变量。
    X_test : numpy.ndarray
        用于测试集的输入特征，已重塑为 LSTM 输入格式。
    y_test : pandas.Series
        测试集的目标变量。
    """

    # Read data from Excel file
    data = pd.read_excel(input_file_path,0) # Read data

    # Group by region name and sort by year
    grouped = data.groupby("地区名称", group_keys=False).apply(lambda x: x.sort_values(by="年份"))

    # Split into test and training sets
    test_indices = grouped.groupby("地区名称").tail(3).index
    train_indices = grouped.drop(test_indices).index

    test_data = data.loc[test_indices]
    train_data = data.loc[train_indices]

    # Perform one-hot encoding on the region column
    test_data_one_hot_encoded = pd.get_dummies(test_data['地区名称'], prefix="地区")
    train_data_one_hot_encoded = pd.get_dummies(train_data['地区名称'], prefix="地区") 

    # Concatenate the one-hot encoded columns back to the original DataFrame
    test_data = pd.concat([test_data, test_data_one_hot_encoded], axis=1)
    train_data = pd.concat([train_data, train_data_one_hot_encoded], axis=1)
        
    # Remove the original attribute column if needed
    test_data.drop('地区名称', axis=1, inplace=True)
    train_data.drop('地区名称', axis=1, inplace=True)

    # Perform normalization
    min_max_scaler = preprocessing.MinMaxScaler()
    train_data_scaler=min_max_scaler.fit_transform(train_data)
    test_data_scaler=min_max_scaler.fit_transform(test_data)

    train_data_scaler = pd.DataFrame(train_data_scaler, columns=train_data.columns)
    test_data_scaler = pd.DataFrame(test_data_scaler, columns=test_data.columns)

    X_train = train_data_scaler.drop('得分', axis=1)
    y_train = train_data_scaler['得分']
    X_test = test_data_scaler.drop('得分', axis=1)
    y_test = test_data_scaler['得分']

    # Reshape input data to [samples, timesteps, features]
    X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

    return X_train, y_train, X_test, y_test

def add_future_three_years_data(input_file_path, output_file_path):
    """
    This function adds data for the years 2024, 2025, and 2026 to the input file and initializes them to 0.

    Parameters:
    input_file_path (str): Path to the Excel file to be processed.
    output_file_path (str): Path to save the resulting Excel file.

    Returns:
    None
    -----------
    该函数将在输入文件中新增2024、2025和2026年的数据，并将其初始化为0。

    参数:
    input_file_path (str): 要处理的 Excel 文件的路径。
    output_file_path (str): 要保存结果的 Excel 文件的路径。

    返回:
    None
    """
   # Read table data
    data = pd.read_excel(input_file_path)

    # Get list of provinces
    provinces = data["地区名称"].unique()

    # Add data for 2024, 2025, and 2026
    new_data = []
    for province in provinces:
        for year in [2024, 2025, 2026]:
            new_row = {"年份": year, "地区名称": province}
            for column in data.columns[2:]:
                new_row[column] = 0
            new_data.append(new_row)

    # Convert to DataFrame and merge
    new_data_df = pd.DataFrame(new_data)
    data = pd.concat([data, new_data_df], ignore_index=True)

    # Save the result to a new file
    data.to_excel(output_file_path, index=False)

def split_data_by_province(input_file_path, output_file_path):
    """
    This function reads an Excel file, extracts data for each province, and saves it to a new Excel file,
    where data for each province is saved in a separate worksheet.

    Parameters:
    input_file_path (str): Path to the input Excel file containing the original data.
    output_file_path (str): Path to the output Excel file where the extracted data will be saved.

    Returns:
    None. The function saves the data to a new Excel file and doesn't return any value.
    -----------------
    该函数读取一个 Excel 文件，提取出每一个省份的数据，并将其保存到一个新的 Excel 文件中，
    其中每一个省份的数据都保存在一个单独的工作表中。

    参数:
    input_file_path (str): 输入 Excel 文件的路径，包含原始数据。
    output_file_path (str): 输出 Excel 文件的路径，将提取出的数据保存到此文件中。

    返回:
    None. 该函数将数据保存到一个新的 Excel 文件中，不返回任何值。
    """

    # Read Excel file
    df = pd.read_excel(input_file_path)

    # Get list of regions
    regions = df['地区名称'].unique()

    # Create an ExcelWriter object to save subsheets
    writer = pd.ExcelWriter(output_file_path, engine='xlsxwriter')

    # Iterate through regions, filter data and save to different subsheets
    for region in regions:
        # Filter data for the current province
        region_data = df[df['地区名称'] == region]

        # Save the filtered data to a separate worksheet
        region_data.to_excel(writer, sheet_name=region, index=False)

    # Close the ExcelWriter object
    writer.close()
