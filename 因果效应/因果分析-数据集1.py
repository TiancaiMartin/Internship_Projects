import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd

def clean_data(df):
    # 清洗数据，排除为0的值或空值
    df = df.replace(0, np.nan)  # 将0替换为NaN
    df = df.dropna()  # 删除包含NaN的行
    return df

def match_time_column(df, result_df):
    # 匹配时间列
    result_df.insert(0, '时间', df.index[:len(result_df)])

def generate_signal(result_df):
    # 创建一个新列'Signal'，并初始化为0，表示逻辑未转变
    result_df['Signal'] = 0

    # 遍历每一行以识别因果模式
    for i in range(1, len(result_df)):
        current_row = result_df.iloc[i]
        previous_row = result_df.iloc[i - 1]

        # 第一种模式（四个因素严格一致）
        strict_consistency = set(current_row[1:].nlargest(5).index) == set(previous_row[1:].nlargest(5).index)
        if strict_consistency and len(set(current_row[1:].nlargest(5).index).intersection(set(previous_row[1:].nlargest(5).index))) >= 4:
            result_df.at[i, 'Signal'] = 1

        # 第二种模式（四个因素宽松一致）
        loose_consistency = len(set(current_row[1:].nlargest(5).index).intersection(set(previous_row[1:].nlargest(5).index))) >= 4
        if loose_consistency:
            result_df.at[i, 'Signal'] = 2

        # 第三种模式（两个因素严格一致）
        two_factors_strict_consistency = len(set(current_row[1:].nlargest(3).index).intersection(set(previous_row[1:].nlargest(3).index))) >= 2
        if two_factors_strict_consistency:
            result_df.at[i, 'Signal'] = 3

        # 第四种模式（两个因素宽松一致）
        two_factors_loose_consistency = len(set(current_row[1:].nlargest(3).index).intersection(set(previous_row[1:].nlargest(3).index))) >= 2
        if two_factors_loose_consistency and set(current_row[1:].nlargest(3).index).intersection(set(previous_row[1:].nlargest(3).index)) == set(current_row[1:].nlargest(3).index):
            result_df.at[i, 'Signal'] = 4

        # 第五种模式（第一个因素改变）
        first_factor_change = current_row[1:].idxmax() != previous_row[1:].idxmax()
        if first_factor_change:
            result_df.at[i, 'Signal'] = 5

    return result_df

def calculate_causal_effects(df, window_size, output_folder='./'):
    print(df.head())  # 打印数据框的前几行
    for target_asset in df.columns:
        # 如果目标资产是 '指标名称'，则跳过
        if target_asset == '指标名称':
            continue

        # 获取所有资产的列名（除目标资产）
        all_assets = [col for col in df.columns if (col != target_asset and col != '指标名称')]

        # 初始化一个字典来记录因果效应
        causal_effects_dict = {}

        data = df

        for i in range(len(data) - window_size + 1):
            # 选择当前窗口的数据
            window_data = data.iloc[i:i + window_size]

            for asset in all_assets:
                # 提取目标资产和当前资产的价格
                target_asset_prices = window_data[target_asset].values.astype(float)
                current_asset_prices = window_data[asset].values.astype(float)

                # 使用 Gaussian KDE 估计联合概率密度函数
                kde_joint = gaussian_kde(np.vstack([current_asset_prices, target_asset_prices]), bw_method='silverman')

                # 使用 Gaussian KDE 估计条件概率密度函数
                kde_conditional = gaussian_kde(current_asset_prices, bw_method='silverman')

                # 计算因果效应
                causal_effect = kde_joint.pdf(np.vstack([current_asset_prices, target_asset_prices])) / kde_conditional.pdf(
                    current_asset_prices)

                # 记录结果
                column_name = f"{asset}_{target_asset}"
                if column_name not in causal_effects_dict:
                    causal_effects_dict[column_name] = []
                causal_effects_dict[column_name].append(causal_effect[0])  # 注意这里取标量值

        # 将因果效应字典转为 DataFrame
        result_df = pd.DataFrame(causal_effects_dict)

        # 生成信号列
        result_df_with_signal = generate_signal(result_df)

        # 匹配时间列
        match_time_column(df, result_df_with_signal)

        # 将 DataFrame 写入 Excel 文件
        output_file = f"{output_folder}因果效应结果_{target_asset}.xlsx"
        result_df_with_signal.to_excel(output_file, index=False)
        print(f"成功完成{output_folder}因果效应结果_{target_asset}.xlsx！")

# 使用示例
# 读取数据
file_path = '因果分析-数据集1.xlsx'
df = pd.read_excel(file_path, header=0, skiprows=0)

# 清洗数据
cleaned_df = clean_data(df)

# 将日度数据按照月份汇总
df_target_monthly_avg = cleaned_df.resample('M', on='指标名称').mean()

# 滚动窗口大小
window_size = 6

# 计算因果效应
calculate_causal_effects(df_target_monthly_avg, window_size)
