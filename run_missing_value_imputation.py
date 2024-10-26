'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-10-25 22:49:49
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-26 11:20:55
'''

import argparse
from src.data_processing.missing_value_imputation import missing_values_Imputation_poly_regression

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Impute missing values using polynomial regression.")
    parser.add_argument("input_file", help="Path to the input Excel file")
    parser.add_argument("output_file", help="Path to save the output Excel file with imputed values")
    
    args = parser.parse_args()
    
    missing_values_Imputation_poly_regression(args.input_file, args.output_file)
    print(f"Missing values have been imputed. Results saved to {args.output_file}")
