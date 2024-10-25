'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-10-25 16:03:08
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-25 22:12:38
FilePath: /2024_tjjm/src/utils/check_missing_value.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pandas as pd
import numpy as np
import sys
import argparse

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check missing values in an Excel file.")
    parser.add_argument("file_path", help="Path to the Excel file")
    args = parser.parse_args()
    
    check_missing_value(args.file_path)
