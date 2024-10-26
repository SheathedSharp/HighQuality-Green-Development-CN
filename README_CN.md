<!--
 * @Author: hiddenSharp429 z404878860@163.com
 * @Date: 2024-10-25 15:58:16
 * @LastEditors: hiddenSharp429 z404878860@163.com
 * @LastEditTime: 2024-10-26 11:21:14
-->

# HQ_development_score_prediction
## 1. 项目结构
```
HighQuality-Green-Development-CN/
├── data/
│ └── statistical/
│ ├── raw/
│ │ └── data_origin.xlsx
│ ├── processed/
│ │ ├── data_region.xlsx
│ │ └── data_region_merged.xlsx
│ └── output/
│ └── future_predictions.xlsx
│
├── src/
│ ├── data_processing/
│ │ ├── init.py
│ │ ├── data_preparation.py
│ │ └── missing_value_imputation.py
│ ├── models/
│ │ ├── init.py
│ │ ├── lstm_models.py
│ │ └── model_evaluation.py
│ └── utils/
│ ├── init.py
│ └── visualization.py
│
├── models/
│ └── best_model.h5
│
├── config.py
├── main.py
├── requirements.txt
├── run_check_poly_regression.py
├── run_missing_value_imputation.py
├── run_scatter_plot.py
└── run_split_data_by_province.py
```

## 2. 数据处理步骤
正常的顺序如下
1. 先查看缺失值情况
2. 将数据集按照省份分割
3. 查看散点图
4. 查看多项式回归填充效果
5. 使用多项式回归填充缺失值

所有文件的传参可以使用`python <file_name.py> -h`来获取帮助
### 2.1 缺失值查看
```bash
python check_missing_value.py ./data/statistical/raw/data_origin.xlsx
```

### 2.2 将数据分割成若干个省份的数据
```bash
python run_split_data_by_province.py ./data/statistical/raw/data_origin.xlsx ./data/statistical/processed/data_region.xlsx
```
### 2.3 散点查看
```bash
python run_scatter_plot.py ./data/statistical/processed/data_region.xlsx
```

### 2.4 查看多项式回归的填充效果
```bash 
python run_check_poly_regression.py ./data/statistical/processed/data_region.xlsx
```

### 2.5 使用多项式回归填充缺失值
```bash
python run_missing_value_imputation.py ./data/statistical/processed/data_region.xlsx ./data/statistical/processed/data_region_merged.xlsx
```

## 3. 模型训练以及预测步骤
### 3.1 第一部分
在2002到2023年中验证模型的预测的性能，将会按照如下步骤进行：
1. 划分训练集和测试集的特征和标签
2. 将训练集和测试集的特征输入到三种不同的LSTM模型中
3. 验证模型的预测性能
4. 选择最佳模型并保存，以备后续预测后三年的绿色经济得分的使用

### 3.2 第二部分
新增2024到2026年的数据，并且预测其绿色经济得分，将会按照如下步骤进行：
1. 往表中新增的2024到2026年的数据，并且初始化为0
2. 使用ARIMA进行2024到2026年的各个属性列的填充
3. 利用填充好的后三年数据新表，加载相应的模型后进行后三年的绿色高质量发展指标的预测


### 3.3 运行
```bash
python main.py
```

## 4. 依赖安装
```bash
pip install -r requirements.txt
```

## 5. 注意事项
- 确保在运行脚本之前，所有必要的数据文件都已经放置在正确的目录中。
- 建议使用conda虚拟环境来运行此项目，以避免依赖冲突。