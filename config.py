'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-10-25 15:43:49
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-25 23:17:51
FilePath: /2024_tjjm/config.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# 配置文件路径
RAW_DATA_PATH = './data/statistical/raw/data_origin.xlsx'
PROCESSED_DATA_PATH = './data/statistical/processed/data_region_merged_with_score.xlsx'
MERGED_DATA_PATH = './data/statistical/processed/data_region_merged.xlsx'
OUTPUT_DATA0_PATH = './data/statistical/output/future_three_years_data_0.xlsx'
OUTPUT_DATA1_PATH = './data/statistical/output/future_three_years_data.xlsx'
OUTPUT_DATA2_PATH = './data/statistical/output/finish.xlsx'

# 模型配置
BEST_MODEL_PATH = './models/best_model.h5'

# 其他配置参数
RANDOM_SEED = 42
TRAIN_TEST_SPLIT_RATIO = 0.8