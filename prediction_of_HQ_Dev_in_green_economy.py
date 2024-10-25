'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-05-02 00:10:42
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-05-15 02:07:09
Description: 绿色高质量经济指标预测
env: Python 3.10
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from keras.models import Sequential, Model, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from keras.layers import LSTM, Dropout, Dense, Input, Add, Multiply
from math import sqrt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def data_prepare(input_file_path):
    """
    该函数为机器学习模型准备数据。
    它从 Excel 文件中读取数据，执行数据预处理，并将数据拆分为训练集和测试集。在这里我们将测试集化为每个省的后三个年份，训练集为剩余年份。
    然后对数据进行归一化和重塑以适用于 LSTM 输入。

    参数:
    input_file_path (str): 要处理的 Excel 文件的路径。


    返回:
    X_train : numpy.ndarray
        用于训练集的输入特征，已重塑为 LSTM 输入格式。
    y_train : pandas.Series
        训练集的目标变量。
    X_test : numpy.ndarray
        用于测试集的输入特征，已重塑为 LSTM 输入格式。
    y_test : pandas.Series
        测试集的目标变量。
    """

    # 读取 Excel 文件中的数据
    data = pd.read_excel(input_file_path,0) #读取数据

    # 按地区名称分组，并按年份排序
    grouped = data.groupby("地区名称", group_keys=False).apply(lambda x: x.sort_values(by="年份"))

    # 划分测试集和训练集
    test_indices = grouped.groupby("地区名称").tail(3).index
    train_indices = grouped.drop(test_indices).index

    test_data = data.loc[test_indices]
    train_data = data.loc[train_indices]

    # 对地区列执行 one-hot 编码
    test_data_one_hot_encoded = pd.get_dummies(test_data['地区名称'], prefix="地区")
    train_data_one_hot_encoded = pd.get_dummies(train_data['地区名称'], prefix="地区") 

    # 将独热编码后的列连接回原始的 DataFrame
    test_data = pd.concat([test_data, test_data_one_hot_encoded], axis=1)
    train_data = pd.concat([train_data, train_data_one_hot_encoded], axis=1)
        

    # 删除原始的属性列，如果需要的话
    test_data.drop('地区名称', axis=1, inplace=True)
    train_data.drop('地区名称', axis=1, inplace=True)

    #进行归一化操作
    min_max_scaler = preprocessing.MinMaxScaler()
    train_data_scaler=min_max_scaler.fit_transform(train_data)
    test_data_scaler=min_max_scaler.fit_transform(test_data)

    train_data_scaler = pd.DataFrame(train_data_scaler, columns=train_data.columns)
    test_data_scaler = pd.DataFrame(test_data_scaler, columns=test_data.columns)


    X_train = train_data_scaler.drop('得分', axis=1)
    y_train = train_data_scaler['得分']
    X_test = test_data_scaler.drop('得分', axis=1)
    y_test = test_data_scaler['得分']

    # 重塑输入数据为 [samples, timesteps, features]
    X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

    return X_train, y_train, X_test, y_test

def traditional_LSTM_model(X_train, y_train, X_test, y_test):
    """
    该函数创建一个并训练一个传统的LSTM模型来预测绿色经济得分。

    参数:
    X_train (numpy.ndarray): 用于训练集的输入特征，已按LSTM输入格式重塑。
    y_train (pandas.Series): 训练集的目标变量。
    X_test (numpy.ndarray): 用于测试集的输入特征，已按LSTM输入格式重塑。
    y_test (pandas.Series): 测试集的目标变量。

    返回:
    model (keras.models.Sequential): 用于预测绿色经济得分的经过训练的LSTM模型。
    """

    d = 0.15  # 丢弃率

    # 创建LSTM模型
    model = Sequential()
    model.add(LSTM(16, activation='relu', input_shape=(1, 64)))
    model.add(Dropout(d))  # 建立的遗忘层
    model.add(Dense(1))  # 输出层

    # 创建一个新的Adam优化器，学习率为0.01
    optimizer = Adam(learning_rate=0.01)

    # 在模型的compile方法中使用新的优化器和均方误差损失函数
    model.compile(optimizer=optimizer, loss='mse')

    # 训练模型
    model.fit(X_train, y_train, epochs=100, batch_size=18, verbose=1, validation_data=(X_test, y_test))

    return model

def stack_LSTM_model(X_train, y_train, X_test, y_test):
    """
    该函数创建一个并训练一个堆叠式LSTM模型来预测绿色经济得分。

    参数:
    X_train (numpy.ndarray): 用于训练集的输入特征，已按LSTM输入格式重塑。
    y_train (pandas.Series): 训练集的目标变量。
    X_test (numpy.ndarray): 用于测试集的输入特征，已按LSTM输入格式重塑。
    y_test (pandas.Series): 测试集的目标变量。

    返回:
    model_stack (keras.models.Sequential): 用于预测绿色经济得分的经过训练的LSTM模型。
    """

    # 定义一个学习率衰减的回调函数
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.000001)

    d = 0.15
    model_stack = Sequential()  # 建立层次模型

    # 添加第一个LSTM层
    model_stack.add(LSTM(128, input_shape=(1, 64), return_sequences=True))
    model_stack.add(Dropout(d))  # 添加一个丢弃层

    # 添加第二个LSTM层
    model_stack.add(LSTM(64, return_sequences=True))
    model_stack.add(Dropout(d))  # 添加一个丢弃层

    # 添加第三个LSTM层
    model_stack.add(LSTM(32, return_sequences=True))
    model_stack.add(Dropout(d))  # 添加一个丢弃层

    # 添加第四个LSTM层
    model_stack.add(LSTM(16, return_sequences=False))
    model_stack.add(Dropout(d))  # 添加一个丢弃层

    model_stack.add(Dense(1))  # 添加一个输出层

    # 创建一个新的Adam优化器，学习率为0.01
    optimizer = Adam(learning_rate=0.01)

    model_stack.compile(optimizer=optimizer, loss='mse')  # 编译模型

    # 训练模型
    model_stack.fit(X_train, y_train, epochs=100, batch_size=18, verbose=1, validation_data=(X_test, y_test), callbacks=[reduce_lr])

    return model_stack

def dynamic_residuals_stack_LSTM_model(X_train, y_train, X_test, y_test):
    """
    该函数创建一个并训练一个动态残差堆叠式LSTM模型来预测绿色经济得分。

    参数:
    X_train (numpy.ndarray): 用于训练集的输入特征，已按LSTM输入格式重塑。
    y_train (pandas.Series): 训练集的目标变量。
    X_test (numpy.ndarray): 用于测试集的输入特征，已按LSTM输入格式重塑。
    y_test (pandas.Series): 测试集的目标变量。

    该函数首先定义一个学习率衰减的回调函数，然后添加丢弃率、LSTM层和残差连接。
    接着添加动态残差权重学习层，并使用动态残差调整原始残差。
    最后，添加输出层并编译模型，然后在模型中使用新的优化器和均方误差损失函数进行训练。

    该函数返回一个经过训练的LSTM模型。

    注：这里已经保存好了最优的模型"best_model.h5"，在model文件中
    """

    # 定义一个学习率衰减的回调函数
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=0.0000001)

    d = 0.15

    # 使用Input层定义输入
    inputs = Input(shape=(1, 64))

    # 第一个LSTM层
    lstm1 = LSTM(128, return_sequences=True)(inputs)
    dropout1 = Dropout(d)(lstm1)
    dense1 = Dense(64)(dropout1) # 添加一个全连接层以改变形状

    # 添加残差连接
    residual1 = Add()([inputs, dense1])

    # 添加动态残差权重学习层
    dynamic_residual1 = Dense(64, activation='sigmoid')(residual1)

    # 使用动态残差调整原始残差
    adjusted_residual1 = Multiply()([residual1, dynamic_residual1])

    # 第二个LSTM层
    lstm2 = LSTM(64, return_sequences=True)(adjusted_residual1)
    dropout2 = Dropout(d)(lstm2)

    # 添加残差连接
    residual2 = Add()([residual1, dropout2])

    # 添加动态残差权重学习层
    dynamic_residual2 = Dense(64, activation='sigmoid')(residual2)

    # 使用动态残差调整原始残差
    adjusted_residual2 = Multiply()([residual2, dynamic_residual2])

    # 第三个LSTM层
    lstm3 = LSTM(32, return_sequences=True)(adjusted_residual2)
    dropout3 = Dropout(d)(lstm3)
    dense2 = Dense(64)(dropout3) # 添加一个全连接层以改变形状

    # 添加残差连接
    residual3 = Add()([residual2, dense2])

    # 添加动态残差权重学习层
    dynamic_residual3 = Dense(64, activation='sigmoid')(residual3)

    # 使用动态残差调整原始残差
    adjusted_residual3 = Multiply()([residual3, dynamic_residual3])

    # 第四个LSTM层
    lstm4 = LSTM(16, return_sequences=True)(adjusted_residual3)
    dropout4 = Dropout(d)(lstm4)
    dense3 = Dense(64)(dropout4) # 添加一个全连接层以改变形状

    # 添加残差连接
    residual4 = Add()([residual3, dense3])

    # 添加动态残差权重学习层
    dynamic_residual4 = Dense(64, activation='sigmoid')(residual4)

    # 使用动态残差调整原始残差
    adjusted_residual4 = Multiply()([residual4, dynamic_residual4])

    # 第五个LSTM层
    lstm5 = LSTM(8, return_sequences=False)(adjusted_residual4)
    dropout5 = Dropout(d)(lstm5)
    dense4 = Dense(64)(dropout5) # 添加一个全连接层以改变形状

    # 添加残差连接
    residual5 = Add()([residual4, dense4])

    # 添加动态残差权重学习层
    dynamic_residual5 = Dense(64, activation='sigmoid')(residual5)

    # 使用动态残差调整原始残差
    adjusted_residual5 = Multiply()([residual5, dynamic_residual5])

    # 输出层
    outputs = Dense(1)(adjusted_residual5)

    # 创建模型
    dynamic_model = Model(inputs=inputs, outputs=outputs)

    # 创建一个新的Adam优化器，学习率为0.01
    optimizer = Adam(learning_rate=0.001)

    dynamic_model.compile(optimizer=optimizer, loss='mae')

    # 训练模型
    dynamic_model.fit(X_train, y_train, epochs=200, batch_size=18, verbose=1, validation_data=(X_test, y_test), callbacks=[reduce_lr])

    return dynamic_model

def evaluate_model_performance(X_test, y_test, model=None, pre_model_url=''):
    """
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

    region_columns_name = ['上海市', '云南省', '内蒙古自治区', '北京市', '吉林省', '四川省', '天津市', '宁夏回族自治区', '安徽省', '山东省', '山西省', '广东省', '广西壮族自治区', '新疆维吾尔自治区', '江苏省', '江西省', '河北省', '河南省', '浙江省', '海南省', '湖北省', '湖南省', '甘肃省', '福建省', '西藏自治区', '贵州省', '辽宁省', '重庆市', '陕西省', '青海省', '黑龙江省']

    # 如果传入了预训练后的模型地址则加载最优训练模型
    if pre_model_url!= '':
        model = load_model(pre_model_url)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 调整预测结果的形状，将其转换为一维数组
    y_pred = y_pred.squeeze()

    # 计算性能指标
    mse = mean_squared_error(y_test, y_pred)
    print('均方误差: ', mse)
    mae = mean_absolute_error(y_test, y_pred)
    print('平均绝对误差: ', mae)
    r2 = r2_score(y_test, y_pred)
    print('R方值: ', r2)
    rmse = sqrt(mse)
    print('均方根误差: ', rmse)

    # 绘制实际值和预测值的对比图
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, color='#C8D5B9', label='实际值', linewidth=1)
    plt.plot(y_pred, color='#467F79', label='预测值', linewidth=1)

    # 添加阴影面积表示预测误差
    plt.fill_between(np.arange(len(y_test)), y_test.squeeze(), y_pred.squeeze(), color='#2E4F4A', alpha=0.5, label='预测误差')

    # 定义区域列
    region_columns_index = list(range(0, 94, 3))
    region_columns = list(range(2, 94, 3))


    # 循环添加虚线
    for i in region_columns_index:
        plt.axvline(x=i, color='k', linestyle='--', alpha=0.1)


    plt.xticks(region_columns, region_columns_name, rotation=90)  


    plt.title('三十一省份绿色经济得分预测结果对比图')
    plt.ylabel('绿色经济得分')
    plt.legend()
    plt.show()

def save_model(model, filename):
    """
    该函数将训练好的模型保存到磁盘。
    """
    model.save(filename)

def add_future_three_years_data(input_file_path, output_file_path):
    """
    该函数将在输入文件中新增2024、2025和2026年的数据，并将其初始化为0。

    参数:
    input_file_path (str): 要处理的 Excel 文件的路径。
    output_file_path (str): 要保存结果的 Excel 文件的路径。

    返回:
    None
    """
   # 读取表格数据
    data = pd.read_excel(input_file_path)

    # 获取省份列表
    provinces = data["地区名称"].unique()

    # 新增2024、2025、2026年的数据
    new_data = []
    for province in provinces:
        for year in [2024, 2025, 2026]:
            new_row = {"年份": year, "地区名称": province}
            for column in data.columns[2:]:
                new_row[column] = 0
            new_data.append(new_row)

    # 转换为DataFrame并合并
    new_data_df = pd.DataFrame(new_data)
    data = pd.concat([data, new_data_df], ignore_index=True)

    # 将结果保存到新文件
    data.to_excel(output_file_path, index=False)

def prediction_future_three_years_HQ_score(input_file_path, output_file_path, pre_model_url=''):
    """
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

    # 从输入文件中读取数据
    data = pd.read_excel(input_file_path,0)

    # 对数据按地区进行分组并按年份排序
    grouped = data.groupby("地区名称", group_keys=False).apply(lambda x: x.sort_values(by="年份"))

    # 获取每一地区最后三年的行的索引
    predict_indices = grouped.groupby("地区名称").tail(3).index

    # 取出用于预测的数据
    predict_data = data.loc[predict_indices]

    # 使用 Pandas 的 get_dummies 函数进行独热编码
    predict_data_one_hot_encoded = pd.get_dummies(predict_data['地区名称'], prefix="地区")

    # 将独热编码后的列连接回原始的 DataFrame
    predict_data = pd.concat([predict_data, predict_data_one_hot_encoded], axis=1)

    # 删除原始的属性列，如果需要的话
    predict_data_region = predict_data['地区名称']
    predict_data.drop('地区名称', axis=1, inplace=True)

    #进行归一化操作
    min_max_scaler = preprocessing.MinMaxScaler()
    predict_data_scaler=min_max_scaler.fit_transform(predict_data)

    # 将归一化后的数据转换为 DataFrame
    predict_data_scaler = pd.DataFrame(predict_data_scaler, columns=predict_data.columns)

    # 获取用于预测的特征
    X = predict_data_scaler.drop('得分', axis=1)

    # 重塑输入数据为 [samples, timesteps, features]
    X = X.values.reshape((X.shape[0], 1, X.shape[1]))

    # 加载模型
    model = load_model(pre_model_url)

    # 对数据进行预测
    y_pred = model.predict(X)

    # 将预测结果添加到原始数据中
    predict_data['得分'] = y_pred.squeeze()

    # 删除不需要的列
    predict_data.drop(predict_data.columns[34:], axis=1, inplace=True)

    # 添加地区名称列 
    predict_data['地区名称'] = predict_data_region

    # 将预测结果保存到输出文件中
    predict_data.to_excel(output_file_path, index=False)


if __name__ == '__main__':
    """正常的顺序如下
    第一部分，在2002到2023年中验证模型的预测的性能，将会按照如下步骤进行：
        1. 划分训练集和测试集的特征和标签
        2. 将训练集和测试集的特征输入到三种不同的LSTM模型中
        3. 验证模型的预测性能
        4. 选择最佳模型并保存，以备后续预测后三年的绿色经济得分的使用
    """
    # 1. 划分训练集和测试集的特征和标签
    X_train, y_train, X_test, y_test =  data_prepare('./data/statistical/data_region_merged_with_score.xlsx')


    # 2. 将训练集和测试集的特征输入到三种不同的LSTM模型中
    # model = traditional_LSTM_model(X_train, y_train, X_test, y_test)
    # model = stack_LSTM_model(X_train, y_train, X_test, y_test)
    model = dynamic_residuals_stack_LSTM_model(X_train, y_train, X_test, y_test)


    # 3. 验证模型的预测性能
    # 如果有传入预训练模型地址则加载最优训练模型，下面第二行代码
    # evaluate_model_performance(X_test, y_test, model=model)
    evaluate_model_performance(X_test, y_test, model=None, pre_model_url='./model/best_model.h5')

    # 4. 选择最佳模型并保存，以备后续预测后三年的绿色经济得分的使用
    # save_model(model, './model/best_model.h5')


    """
    第二部分，新增2024到2026年的数据，并且预测其绿色经济得分，将会按照如下步骤进行：
        1. 往表中新增的2024到2026年的数据，并且初始化为0
        2. 使用ARIMA进行2024到2026年的各个属性列的填充
        3. 利用填充好的后三年数据新表，加载相应的模型后进行后三年的绿色高质量发展指标的预测
    """
    # 1. 往表中新增的2024到2026年的数据，并且初始化为0，保存为future_three_years_data_0.xlsx
    add_future_three_years_data('./data/statistical/data_region_merged_with_score.xlsx', './data/statistical/future_three_years_data_0.xlsx')

    # 2. 使用ARIMA进行2024到2026年的各个属性列的填充，填充后的文件保存为future_three_years_data.xlsx

    # 3. 利用填充好的后三年数据新表（future_three_years_data.xlsx），加载相应的模型后进行后三年的绿色高质量发展指标的预测
    prediction_future_three_years_HQ_score('./data/statistical/future_three_years_data.xlsx', './data/statistical/finish.xlsx', pre_model_url='./model/best_model.h5')





