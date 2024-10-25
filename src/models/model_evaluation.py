'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-10-25 15:42:33
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-25 15:51:59
FilePath: /2024_tjjm/src/models/model_evaluation.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import load_model
import numpy as np

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