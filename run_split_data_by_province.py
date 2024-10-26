'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-10-25 22:03:11
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-26 11:21:07
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