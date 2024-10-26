'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-10-25 22:41:13
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-26 11:20:44
'''

import argparse
from src.utils.visualization import check_effect_of_poly_regression_Imputation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the effect of polynomial regression imputation.")
    parser.add_argument("input_file", help="Path to the input Excel file")
    
    args = parser.parse_args()
    
    check_effect_of_poly_regression_Imputation(args.input_file)
    print(f"Polynomial regression imputation effect has been checked for the data in {args.input_file}")
