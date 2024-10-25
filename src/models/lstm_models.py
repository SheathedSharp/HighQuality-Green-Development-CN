'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-10-25 15:42:19
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-25 15:50:27
FilePath: /2024_tjjm/src/models/lstm_models.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, Input, Add, Multiply
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

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