def check_missing_value(file_path):
    """
    该函数读取一个 Excel 文件，识别出缺失值，并用 NaN 代替。
    然后，它会计算并打印出 DataFrame 中每一列的缺失值计数。

    参数:
    file_path (str): 要处理的 Excel 文件的路径。

    返回:
    None. 该函数会打印出每一列的缺失值计数。
    """

    # 读取 Excel 文件
    df = pd.read_excel(file_path)

    # 定义缺失值
    missing_values = [0, 'N/A', 'NaN', '']

    # 填充缺失值
    df.replace(missing_values, np.nan, inplace=True)

    # 统计每列的缺失值
    missing_values_count = df.isnull().sum()

    # 打印每列的缺失值计数
    print(missing_values_count)