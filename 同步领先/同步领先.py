import numpy as np
import pandas as pd
from pcalg import estimate_skeleton, estimate_cpdag
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from pingouin import partial_corr

def clean_data(df):
    # 清洗数据，排除为0的值或空值
    df = df.replace(0, np.nan)  # 将0替换为NaN
    df = df.dropna()  # 删除包含NaN的行
    return df

def calculate_nrmse(actual, predicted):
    """
    计算标准化均方根误差（NRMSE）
    """
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    range_y = np.max(actual) - np.min(actual)
    nrmse = rmse / range_y
    return nrmse


# 读取数据
file_path1 = '同步领先-基本面数据集组合1.xlsx'
file_path_target = '同步领先-标的数据集.xlsx'
df1 = pd.read_excel(file_path1, header=0, skiprows=0)
df_target = pd.read_excel(file_path_target, header=0, skiprows=0)

# 将 '指标名称' 列转换为日期类型
df1['指标名称'] = pd.to_datetime(df1['指标名称'])
df_target['指标名称'] = pd.to_datetime(df_target['指标名称'])

# 将日度数据按照月份汇总
df_target_monthly_avg = df_target.resample('M', on='指标名称').mean()

# 将两个数据框按照日期合并
merged_df = pd.merge(df1, df_target_monthly_avg, how='left', on='指标名称')

# 清洗数据
cleaned_merged_df = clean_data(merged_df)

# print(cleaned_merged_df.head(5))

# 基本面因子列名为 Y1-Y10，A50 是目标列
basic_factors = cleaned_merged_df[['Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 'Y10']]
target_asset = 'A50'

data_array = cleaned_merged_df.to_numpy()

def partial_corr_test(data_matrix, i, j, k):
    columns = [f'Column_{idx}' for idx in range(data_matrix.shape[1])]
    data = pd.DataFrame(data_matrix, columns=columns)
    data.dropna()
    print(data.info())
    for col in columns[1:]:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    partial_corr_result = partial_corr(data, x=columns[i], y=columns[j], covar=[columns[idx] for idx in k])

    p_value = partial_corr_result['p-val'].values[0]
    return p_value > 0.05

# 使用 pcalg 库中的 estimate_skeleton 函数估计图的骨架
skel_graph = estimate_skeleton(data_matrix=data_array, alpha=0.05,indep_test_func=partial_corr_test)

# 使用 pcalg 库中的 PC 算法估计 CPDAG
estimated_cpdag = estimate_cpdag(data_array,sep_set={})

# 打印估计的 CPDAG
print("Estimated CPDAG:")
print(estimated_cpdag)

# 提取 'A50' 的父节点（因果因子）
causal_factors = list(estimated_cpdag.parents(target_asset))

# 获取历史数据
historical_data = cleaned_merged_df[basic_factors + [target_asset]]

# 拟合多元线性回归模型
model = LinearRegression()
model.fit(historical_data[basic_factors], historical_data[target_asset])

# 获取当前时间点的因子值
current_factors = cleaned_merged_df[causal_factors]

# 预测 'A50' 的值
predicted_values = model.predict(current_factors)

# 将预测结果添加到数据框中
cleaned_merged_df['A50_predicted'] = predicted_values

# 计算 NRMSE（规范化均方根误差）
rmse = np.sqrt(mean_squared_error(cleaned_merged_df[target_asset], cleaned_merged_df['A50_predicted']))
nrmse = rmse / (cleaned_merged_df[target_asset].max() - cleaned_merged_df[target_asset].min())

# 打印预测结果和 NRMSE
print(cleaned_merged_df[['时间', target_asset, 'A50_predicted']])
print(f'NRMSE: {nrmse:.4f}')

# 可视化预测结果
plt.plot(cleaned_merged_df['指标名称'], cleaned_merged_df[target_asset], label='Actual')
plt.plot(cleaned_merged_df['指标名称'], cleaned_merged_df['A50_predicted'], label='Predicted')
plt.xlabel('时间')
plt.ylabel(target_asset)
plt.legend()
plt.show()