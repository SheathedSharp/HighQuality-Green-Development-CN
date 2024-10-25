'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-10-25 15:41:23
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-25 15:47:13
FilePath: /2024_tjjm/src/data_processing/data_preparation.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pandas as pd
import numpy as np
from sklearn import preprocessing

def data_prepare(input_file_path):
    """
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

    # 读取 Excel 文件中的数据
    data = pd.read_excel(input_file_path,0) #读取数据

    # 按地区名称分组，并按年份排序
    grouped = data.groupby("地区名称", group_keys=False).apply(lambda x: x.sort_values(by="年份"))

    # 划分测试集和训练集
    test_indices = grouped.groupby("地区名称").tail(3).index
    train_indices = grouped.drop(test_indices).index

    test_data = data.loc[test_indices]
    train_data = data.loc[train_indices]

    # 对地区列执行 one-hot 编码
    test_data_one_hot_encoded = pd.get_dummies(test_data['地区名称'], prefix="地区")
    train_data_one_hot_encoded = pd.get_dummies(train_data['地区名称'], prefix="地区") 

    # 将独热编码后的列连接回原始的 DataFrame
    test_data = pd.concat([test_data, test_data_one_hot_encoded], axis=1)
    train_data = pd.concat([train_data, train_data_one_hot_encoded], axis=1)
        

    # 删除原始的属性列，如果需要的话
    test_data.drop('地区名称', axis=1, inplace=True)
    train_data.drop('地区名称', axis=1, inplace=True)

    #进行归一化操作
    min_max_scaler = preprocessing.MinMaxScaler()
    train_data_scaler=min_max_scaler.fit_transform(train_data)
    test_data_scaler=min_max_scaler.fit_transform(test_data)

    train_data_scaler = pd.DataFrame(train_data_scaler, columns=train_data.columns)
    test_data_scaler = pd.DataFrame(test_data_scaler, columns=test_data.columns)


    X_train = train_data_scaler.drop('得分', axis=1)
    y_train = train_data_scaler['得分']
    X_test = test_data_scaler.drop('得分', axis=1)
    y_test = test_data_scaler['得分']

    # 重塑输入数据为 [samples, timesteps, features]
    X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

    return X_train, y_train, X_test, y_test

def add_future_three_years_data(input_file_path, output_file_path):
    """
    该函数将在输入文件中新增2024、2025和2026年的数据，并将其初始化为0。

    参数:
    input_file_path (str): 要处理的 Excel 文件的路径。
    output_file_path (str): 要保存结果的 Excel 文件的路径。

    返回:
    None
    """
   # 读取表格数据
    data = pd.read_excel(input_file_path)

    # 获取省份列表
    provinces = data["地区名称"].unique()

    # 新增2024、2025、2026年的数据
    new_data = []
    for province in provinces:
        for year in [2024, 2025, 2026]:
            new_row = {"年份": year, "地区名称": province}
            for column in data.columns[2:]:
                new_row[column] = 0
            new_data.append(new_row)

    # 转换为DataFrame并合并
    new_data_df = pd.DataFrame(new_data)
    data = pd.concat([data, new_data_df], ignore_index=True)

    # 将结果保存到新文件
    data.to_excel(output_file_path, index=False)

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