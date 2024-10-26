<!--
 * @Author: hiddenSharp429 z404878860@163.com
 * @Date: 2024-10-25 23:27:56
 * @LastEditors: hiddenSharp429 z404878860@163.com
 * @LastEditTime: 2024-10-26 14:25:18
-->

# HQ_development_score_prediction

## 1. Project Overview
This project is a three-year prediction of the green high-quality development indicators for each province in China based on three LSTM models.


## 2. Data Processing Steps

The normal sequence is as follows:

1. Check for missing values
2. Split the dataset by province
3. View scatter plots
4. Check the effect of polynomial regression imputation
5. Use polynomial regression to impute missing values

For help with all file parameters, use `python <file_name.py> -h`.

### 2.1 Check Missing Values
```bash
python check_missing_value.py ./data/statistical/raw/data_origin.xlsx
```

### 2.2 Split Data by Province
```bash
python run_split_data_by_province.py ./data/statistical/raw/data_origin.xlsx ./data/statistical/processed/data_region.xlsx
```


### 2.3 View Scatter Plots
```bash
python run_scatter_plot.py ./data/statistical/processed/data_region.xlsx
```


### 2.4 Check Polynomial Regression Imputation Effect
```bash
python run_check_poly_regression.py ./data/statistical/processed/data_region.xlsx
```

### 2.5 Impute Missing Values Using Polynomial Regression
```bash
python run_missing_value_imputation.py ./data/statistical/processed/data_region.xlsx ./data/statistical/processed/data_region_merged.xlsx
```


## 3. Model Training and Prediction Steps

### 3.1 Part One

To validate the model's prediction performance from 2002 to 2023, follow these steps:

1. Split features and labels into training and test sets
2. Input the features of the training and test sets into three different LSTM models
3. Validate the model's prediction performance
4. Select the best model and save it for future use in predicting green economy scores for the next three years

use `train_model.py` to complete the above steps
```bash
python train_model.py
```

### 3.2 Part Two

To add data for 2024 to 2026 and predict their green economy scores, follow these steps:

1. Add data for 2024 to 2026 to the table and initialize it to 0
2. Use ARIMA to fill in the attribute columns for 2024 to 2026
3. Use the filled data for the next three years, load the appropriate model, and predict the green high-quality development indicators for the next three years

use `predict_future_score.py` to complete the above steps
```bash
python predict_future_score.py
```


## 4. Install Dependencies
Install the required dependencies:
```bash
pip install -r requirements.txt
```

## 5. Notes

- Ensure all necessary data files are placed in the correct directories before running the scripts.
- It is recommended to use a virtual environment to run this project to avoid dependency conflicts.