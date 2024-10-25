'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-10-25 22:30:40
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-25 22:33:13
FilePath: /2024_tjjm/run_scatter_plot.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import argparse
from src.utils.visualization import scatter_plot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate scatter plots for data visualization.")
    parser.add_argument("input_file", help="Path to the input Excel file")
    
    args = parser.parse_args()
    
    scatter_plot(args.input_file)
    print(f"Scatter plots have been generated for the data in {args.input_file}")

