'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-10-25 22:41:13
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-25 22:41:20
FilePath: /2024_tjjm/run_check_poly_regression.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import argparse
from src.utils.visualization import check_effect_of_poly_regression_Imputation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the effect of polynomial regression imputation.")
    parser.add_argument("input_file", help="Path to the input Excel file")
    
    args = parser.parse_args()
    
    check_effect_of_poly_regression_Imputation(args.input_file)
    print(f"Polynomial regression imputation effect has been checked for the data in {args.input_file}")
