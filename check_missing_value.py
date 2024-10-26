'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-10-25 16:03:08
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-26 11:22:23
'''

import pandas as pd
import numpy as np
import sys
import argparse

def check_missing_value(file_path):
    """
    This function reads an Excel file, identifies missing values, and replaces them with NaN.
    Then, it calculates and prints the count of missing values for each column.

    Parameters:
    file_path (str): The path to the Excel file to be processed.

    Returns:
    None. The function prints the count of missing values for each column.
    --------------
    该函数读取一个 Excel 文件，识别出缺失值，并用 NaN 代替。
    然后，它会计算并打印出 DataFrame 中每一列的缺失值计数。

    参数:
    file_path (str): 要处理的 Excel 文件的路径。

    返回:
    None. 该函数会打印出每一列的缺失值计数。
    """

    # read the Excel file
    df = pd.read_excel(file_path)

    # define missing values
    missing_values = [0, 'N/A', 'NaN', '']

    # replace missing values
    df.replace(missing_values, np.nan, inplace=True)

    # count the missing values of each column
    missing_values_count = df.isnull().sum()

    # print the count of missing values for each column
    print(missing_values_count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check missing values in an Excel file.")
    parser.add_argument("file_path", help="Path to the Excel file")
    args = parser.parse_args()
    
    check_missing_value(args.file_path)
