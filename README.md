<!--
 * @Author: hiddenSharp429 z404878860@163.com
 * @Date: 2024-10-25 15:58:16
 * @LastEditors: hiddenSharp429 z404878860@163.com
 * @LastEditTime: 2024-10-25 16:01:06
 * @FilePath: /2024_tjjm/README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
# HQ_development_score_prediction

## 数据处理步骤
```python
"""正常的顺序如下
1. 先查看缺失值情况
2. 将数据集按照省份分割
3. 查看散点图
4. 查看多项式回归填充效果
5. 使用多项式回归填充缺失值
"""

# # 缺失值查看
# check_missing_value('./data/statistical/data_origin.xlsx')

# # 将数据分割成若干个省份的数据
# split_data_by_province('./data/statistical/data_origin','./data/statistical/data_region.xlsx' )

# # 散点查看
# scatter_plot('./data/statistical/data_region.xlsx')

# # 查看多项式回归的填充效果
# check_effect_of_poly_regression_Imputation('./data/statistical/data_region.xlsx')

# # 使用多项式回归填充缺失值
# missing_values_Imputation_poly_regression('./data/statistical/data_region.xlsx', './data/statistical/data_region_merged.xlsx')
```

## 模型训练步骤
```python
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
```