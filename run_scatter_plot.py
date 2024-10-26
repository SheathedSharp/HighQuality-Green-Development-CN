'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-10-25 22:30:40
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-26 11:21:01
'''

import argparse
from src.utils.visualization import scatter_plot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate scatter plots for data visualization.")
    parser.add_argument("input_file", help="Path to the input Excel file")
    
    args = parser.parse_args()
    
    scatter_plot(args.input_file)
    print(f"Scatter plots have been generated for the data in {args.input_file}")

