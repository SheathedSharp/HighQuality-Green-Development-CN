'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-10-25 15:42:33
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-26 14:06:55
'''

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import TensorDataset, DataLoader

import config

def evaluate_model_performance(X_test, y_test, model=None, pre_model_url=''):
    """
    This function evaluates the performance of a trained model using MSE, MAE, R-squared (R2), and RMSE.
    It also plots a comparison graph between actual and predicted values.

    Parameters:
    X_test (numpy.ndarray): The input features for testing.
    y_test (numpy.ndarray): The target values for testing.
    model (keras.models.Sequential, optional): The trained model to evaluate.
    pre_model_url (str, optional): The URL of the pre-trained model to load.

    Returns:
    None
    -----------
    评估一个训练好的模型的性能，使用均方误差 (MSE)、平均绝对误差 (MAE)、R-平方 (R2) 和均方根误差 (RMSE) 进行评估。
    还会绘制一个对比图，对比实际值和预测值。

    参数:
    X_test (numpy.ndarray): 用于测试的输入特征。
    y_test (numpy.ndarray): 用于测试的目标值。
    model (keras.models.Sequential): 要评估的训练好的模型。
    pre_model_url (str, optional): 要加载的预训练模型的 URL。默认为 ''。

    返回:
    None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # If the URL of the pre-trained model is provided, load the optimal trained model
    if pre_model_url:
        model = torch.load(pre_model_url)
    
    model.to(device)
    model.eval()

    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)

    # Make predictions on the test set
    with torch.no_grad():
        # Adjust the shape of the prediction results to convert them to a one-dimensional array
        y_pred = model(X_test_tensor).cpu().numpy().squeeze()

    y_test = y_test_tensor.cpu().numpy()


    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    print('MSE: ', mse)
    mae = mean_absolute_error(y_test, y_pred)
    print('MAE: ', mae)
    r2 = r2_score(y_test, y_pred)
    print('R2: ', r2)
    rmse = np.sqrt(mse)
    print('RMSE: ', rmse)

    region_columns_name = ['上海市', '云南省', '内蒙古自治区', '北京市', '吉林省', '四川省', '天津市', '宁夏回族自治区', '安徽省', '山东省', '山西省', '广东省', '广西壮族自治区', '新疆维吾尔自治区', '江苏省', '江西省', '河北省', '河南省', '浙江省', '海南省', '湖北省', '湖南省', '甘肃省', '福建省', '西藏自治区', '贵州省', '辽宁省', '重庆市', '陕西省', '青海省', '黑龙江省']

    # Plot a comparison graph between actual and predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, color='#C8D5B9', label='实际值', linewidth=1)
    plt.plot(y_pred, color='#467F79', label='预测值', linewidth=1)

    # Add a shaded area to represent the prediction error
    plt.fill_between(np.arange(len(y_test)), y_test.squeeze(), y_pred.squeeze(), color='#2E4F4A', alpha=0.5, label='预测误差')

    # Define the index of the region columns
    region_columns_index = list(range(0, 94, 3))
    region_columns = list(range(2, 94, 3))


    # Loop to add dashed vertical lines
    for i in region_columns_index:
        plt.axvline(x=i, color='k', linestyle='--', alpha=0.1)


    plt.xticks(region_columns, region_columns_name, rotation=90)  


    plt.title('三十一省份绿色经济得分预测结果对比图')
    plt.ylabel('绿色经济得分')
    plt.legend()

    # random picture name
    picture_name = str(np.random.randint(1000000)) + '.png'

    # don't show the figure, just save it
    plt.savefig(config.OUTPUT_GRAPH_PATH + '/' + picture_name)
    
    # print the picture name and the path
    print('The picture name is: ', picture_name, 'and the path is: ', config.OUTPUT_GRAPH_PATH + '/' + picture_name)

def save_model(model, filename):
    """
    This function saves a trained model to disk.

    Parameters:
    model (keras.models.Sequential): The trained model to save.
    filename (str): The path to save the model.

    Returns:
    None
    -----------
    该函数将训练好的模型保存到磁盘。
    """
    torch.save(model, filename)

def prediction_future_three_years_HQ_score(input_file_path, output_file_path, pre_model_url=''):
    """
    This function predicts the green economic score for each province for the next three years (2024-2026).
    It reads data from an input file, preprocesses the data, loads a trained model, makes predictions, and saves the results to an output file.

    Parameters:
    input_file_path (str): The path to the input Excel file containing data for prediction.
    output_file_path (str): The path to save the output Excel file with predicted results.
    pre_model_url (str, optional): The URL of the pre-trained model to load. Defaults to ''.

    Returns:
    None
    -----------

    该函数用于预测未来三年（2024-2026年）的各省份的绿色经济得分。
    它从输入文件中读取数据，对数据进行预处理，加载训练好的模型，
    并进行预测。然后将预测结果保存到输出文件中。

    参数：
    input_file_path (str): 输入 Excel 文件的路径，包含用于预测的数据。
    output_file_path (str): 输出 Excel 文件的路径，将在此处保存预测结果。
    pre_model_url (str, optional): 用于预测的预训练模型的 URL。默认为 ''。

    返回：
    None
    """

    # Read data from the input file
    data = pd.read_excel(input_file_path,0)

    # Group data by region and sort by year
    grouped = data.groupby("地区名称", group_keys=False).apply(lambda x: x.sort_values(by="年份"))

    # Get the index of the last three rows for each region
    predict_indices = grouped.groupby("地区名称").tail(3).index

    # Extract the data for prediction
    predict_data = data.loc[predict_indices]

    # Use Pandas' get_dummies function for one-hot encoding
    predict_data_one_hot_encoded = pd.get_dummies(predict_data['地区名称'], prefix="地区")

    # Concatenate the one-hot encoded columns back to the original DataFrame
    predict_data = pd.concat([predict_data, predict_data_one_hot_encoded], axis=1)

    # Drop the original attribute column if needed
    predict_data_region = predict_data['地区名称']
    predict_data.drop('地区名称', axis=1, inplace=True)

    # Perform normalization
    min_max_scaler = preprocessing.MinMaxScaler()
    predict_data_scaler=min_max_scaler.fit_transform(predict_data)

    # Convert the normalized data to a DataFrame
    predict_data_scaler = pd.DataFrame(predict_data_scaler, columns=predict_data.columns)

    # Get the features for prediction
    X = predict_data_scaler.drop('得分', axis=1)

    # Reshape the input data to [samples, timesteps, features]
    X = X.values.reshape((X.shape[0], 1, X.shape[1]))

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(pre_model_url)
    model.to(device)
    model.eval()

    # Convert X to PyTorch tensor
    X_tensor = torch.FloatTensor(X).to(device)

    # Make predictions on the data
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy()

    # Add the prediction results to the original data
    predict_data['得分'] = y_pred.squeeze()

    # Drop unnecessary columns
    predict_data.drop(predict_data.columns[34:], axis=1, inplace=True)

    # Add the region name column
    predict_data['地区名称'] = predict_data_region

    # Save the prediction results to the output file
    predict_data.to_excel(output_file_path, index=False)
