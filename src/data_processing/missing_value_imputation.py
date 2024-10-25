def missing_values_Imputation_poly_regression(input_file_path, output_file_path):
    """
    该函数读取 Excel 文件，对每一个子表进行迭代，并对每一列中包含的缺失值（0）进行填充。
    它首先使用多项式回归来创建散点图，并对比实际值和预测值。
    如果 R-squared 值小于 0.35，则使用正态分布采样填充数据
    如果 R-squared 值大于或等于 0.35，，则使用多项式回归模型的预测值来填充缺失值。

    参数:
    input_file_path (str): 包含数据的 Excel 文件的路径。
    output_file_path(str): 输出 Excel 文件的路径，将填充了缺失值的新表格保存为新 Excel 文件。

    返回:
    None. 打印填充了缺失值的新表格保存为新 Excel 文件的文件路径。
    """
    # 读取 Excel 文件
    file_path = input_file_path
    xls = pd.ExcelFile(file_path)

    # 循环遍历每个子表
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # 将空数据替换为 0
        df.replace(np.nan, 0, inplace=True)

        # 获取具有缺失值 0 的属性列
        cols_with_missing_values = df.columns[(df == 0).any()].tolist()
        
        # 循环处理每个选定的属性列
        for i, col in enumerate(cols_with_missing_values):
            # 提取特征列和目标列
            X = df.loc[df[col] != 0, '年份'].values.reshape(-1, 1)
            y = df.loc[df[col] != 0, col].values
            
            # 检查样本数量是否大于等于2，以确保至少有一个样本
            if len(X) >= 2:
                # 使用多项式特征转换
                poly_features = PolynomialFeatures(degree=2)
                X_poly = poly_features.fit_transform(X)
                
                # 建立多项式回归模型
                poly_model = LinearRegression()
                poly_model.fit(X_poly, y)
                
                # 预测缺失值为 0 的情况下的属性值
                X_pred = df.loc[df[col] == 0, '年份'].values.reshape(-1, 1)
                X_pred_poly = poly_features.transform(X_pred)
                y_pred = poly_model.predict(X_pred_poly)

                # 将所有的负预测值进行指数变换
                # y_pred = np.where(y_pred < 0, 0, y_pred)

                # 建立多项式回归模型
                poly_model = LinearRegression()
                poly_model.fit(X_poly, y)
                
                # 计算 R-squared (coefficient of determination)
                r_squared = poly_model.score(X_poly, y)

                if r_squared < 0.35:
                    print(f'R-squared for {col}: {r_squared}')
                    # 创建图表
                    plt.figure(figsize=(10, 6)) 

                    plt.scatter(X, y, color='#2E4F4A', label='实际数据')
                    plt.plot(X, poly_model.predict(poly_features.transform(X)), label=f'{col}多项式拟合曲线', color='#2E4F4A')

                    # 计算预测值与实际值之间的残差
                    residuals = y - poly_model.predict(poly_features.transform(X))
                    residual_std = np.std(residuals)

                    # 计算预测误差区间的上下限
                    upper_bound = poly_model.predict(poly_features.transform(X)) + residual_std
                    lower_bound = poly_model.predict(poly_features.transform(X)) - residual_std

                    # 添加预测误差区间的阴影部分
                    plt.fill_between(X.flatten(), lower_bound, upper_bound, color='#B9D5BA', alpha=0.3, label='预测误差区间')

                    plt.xlabel('年份')
                    plt.ylabel(col)
                    plt.title(f'{sheet_name}的{col}')
                    plt.legend()
                    plt.show()

                    # 为 X 建立正态分布，然后更新缺失值为正态分布中置信区间为 80% 的随机数据
                    mean, std = np.mean(y), np.std(y)
                    X_norm = np.linspace(mean - 3*std, mean + 3*std, 100)
                    y_norm = norm.pdf(X_norm, loc=mean, scale=std)

                    y_zeros = [0] * len(X)
                    # 计算对应 X 值的正态分布的 Y 值
                    y_values = norm.pdf(y, loc=mean, scale=std)
                    # 创建图表
                    plt.figure(figsize=(10, 6)) 
                    plt.plot(X_norm, y_norm, color='#2E4F4A', label=f'{col}的正态分布曲线')
                    confidence_interval = norm.interval(0.80, loc=mean, scale=std)
                    # 确定在置信区间内的索引范围
                    index_start = np.argmax(X_norm >= confidence_interval[0])
                    index_end = np.argmax(X_norm >= confidence_interval[1])

                    # 筛选出在置信区间内的 Y 的取值范围
                    y_confidence_interval = y_norm[index_start:index_end]

                    # 确定对应的 X 的取值范围
                    X_confidence_interval = X_norm[index_start:index_end]

                    plt.scatter(y, y_values, color='#2E4F4A', label='实际数据')
                    # 添加置信区间的阴影部分
                    plt.fill_between(X_confidence_interval, 0, y_confidence_interval, color='#B9D5BA', alpha=0.3, label='80%可信度的置信区间')
                    # print("confidence_interval", confidence_interval)
                    df.loc[df[col] == 0, col] = np.random.uniform(confidence_interval[0], confidence_interval[1], size=np.sum(df[col] == 0))    
                    plt.legend()  # 添加图例
                    plt.xlabel(f'{col}')
                    plt.title(f'{sheet_name}的{col}正态分布表')
                    plt.show()
                else:
                    # 更新缺失值为 0 的属性列
                    df.loc[df[col] == 0, col] = y_pred
                    
            # 如果样本数量等于1，则用样本填充缺失值
            elif len(X) == 1:
                df.loc[df[col] == 0, col] = y[0]
                
            # 如果样本数量为0，则用0填充
            else:
                df.loc[df[col] == 0, col] = 0     
            
        # 保存更新后的子表
        df.to_excel(f'./data/statistical/filled_NaN_xlsx/{sheet_name}_NaN_filled_polynomial.xlsx', index=False)

    # 文件夹路径
    folder_path = './data/statistical/filled_NaN_xlsx'

    # 读取文件夹中的所有 Excel 文件并合并
    dfs = []
    for file in os.listdir(folder_path):
        if file.endswith('.xlsx'):
            file_path = os.path.join(folder_path, file)
            df = pd.read_excel(file_path)
            dfs.append(df)

    # 合并所有 DataFrame
    merged_df = pd.concat(dfs, ignore_index=True)

    # 保存合并后的 DataFrame 到一个 Excel 文件中
    merged_df.to_excel(output_file_path, index=False)

    print(f'Merged DataFrame saved to {output_file_path}')