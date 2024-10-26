'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-10-26 13:25:29
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-26 14:05:20
'''

import pandas as pd
from src.data_processing.data_preparation import data_prepare, add_future_three_years_data, split_data_by_province
from src.data_processing.missing_value_imputation import missing_values_Imputation_poly_regression
from src.models.lstm_models import dynamic_residuals_stack_LSTM_model, stack_LSTM_model, traditional_LSTM_model
from src.models.model_evaluation import evaluate_model_performance, save_model, prediction_future_three_years_HQ_score
from src.utils.visualization import scatter_plot, check_effect_of_poly_regression_Imputation
import config

def train_model():
    # First part: train model
    ## Data preparation
    X_train, y_train, X_test, y_test = data_prepare(config.PROCESSED_DATA_PATH)
    ## print the shape of the data
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    ## Model training
    # choose the model
    model_choice = input("Please choose the model: \n"
                         "1. Traditional LSTM model, \n"
                         "2. Stacked LSTM model, \n"
                         "3. Dynamic residuals stack LSTM model\n")
    if model_choice == "1":
        ### Traditional LSTM model
        model_name = "Traditional_LSTM_model"
        model = traditional_LSTM_model(X_train, y_train, X_test, y_test)
    elif model_choice == "2":
        ### Stacked LSTM model  
        model_name = "Stacked_LSTM_model"
        model = stack_LSTM_model(X_train, y_train, X_test, y_test)
    else:
        ### Dynamic residuals stack LSTM model
        model_name = "DR-Stacked_LSTM_model"
        model = dynamic_residuals_stack_LSTM_model(X_train, y_train, X_test, y_test)

    ## Model evaluation
    evaluate_model_performance(X_test, y_test, model=model)

    ## Save model
    save_model(model, config.MODEL_PATH + "/" + model_name + ".pth")

if __name__ == "__main__":
    train_model()