'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-10-25 22:03:11
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-25 22:10:00
FilePath: /2024_tjjm/src/utils/run_split_data_by_province.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import argparse
from src.data_processing.data_preparation import split_data_by_province

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data by province from an Excel file.")
    parser.add_argument("input_file", help="Path to the input Excel file")
    parser.add_argument("output_file", help="Path to save the output Excel file")
    
    args = parser.parse_args()
    
    split_data_by_province(args.input_file, args.output_file)
    print(f"Data has been split and saved to {args.output_file}")