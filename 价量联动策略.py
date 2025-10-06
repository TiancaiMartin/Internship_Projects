import pandas as pd
import numpy as np
import empyrical as ep
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# 读取数据
df = pd.read_excel('/Users/tiancaixiaohuoban/Desktop/实习/买方实习/中信建投期货（金融工程）/研报复现/国信证券/价量联动择时策略/成交量_上证综合指数.xlsx', index_col='Date', parse_dates=True)
df = df.head(8100)

# 成交量取对数后标准化
# 取对数
df['log_Vol'] = np.log(df['Volume'])

# 使用 StandardScaler 进行标准化
scaler = StandardScaler()
df['std_log_Vol'] = scaler.fit_transform(df['log_Vol'].values.reshape(-1, 1))

# 收盘价取对数后标准化
# 取对数
df['log_Close'] = np.log(df['Close'])
# 使用 StandardScaler 进行标准化
scaler = StandardScaler()
df['std_log_Close'] = scaler.fit_transform(df['log_Close'].values.reshape(-1, 1))


def brownian_correlation(prices1, volumes1, prices2, volumes2):
    # 计算价格序列的Pearson相关系数
    price_correlation, _ = pearsonr(prices1, prices2)
    # 计算成交量序列的Pearson相关系数
    volume_correlation, _ = pearsonr(volumes1, volumes2)
    # 计算Brownian相关系数
    brownian_corr = np.sqrt(price_correlation ** 2 + volume_correlation ** 2)

    return brownian_corr


# 展示前30条数据
# print(df.head(30))
# 轮动取时间窗口为30的二维序列（取两个series出来）
# 价格序列
prices = df['std_log_Close'].values
# 成交量序列
volumes = df['std_log_Vol'].values
# 交易信号序列
signals = []

# 遍历两个序列，取时间窗口为30的二维序列
# 时间窗口大小
window = 30
# 价格涨跌幅阈值
price_change_threshold = 0.01
# 成交量涨跌幅阈值
volume_change_threshold = 0.01
# 三维序列，添加signal列
price_volume_pairs = []
# 遍历序列
for i in range(0, len(prices), window):
    # 取时间窗口为30的二维序列
    price_volume_pairs.append((prices[i:i + window], volumes[i:i + window], signals[i:i + window]))


# 遍历二维序列，选择每一个窗口序列，计算其与其他所有同长度序列的Brownian相关系数
# 选择典型序列
def gnerate_signal_for_pairs(typical_price, typical_volume, typical_signal):
    # 计算Brownian相关系数
    brownian_corr = []
    for price, volume, signal in price_volume_pairs:
        brownian_corr.append(brownian_correlation(typical_price, typical_volume, price, volume))
    # 选择相关系数最高的7个窗口，不含典型序列自身
    # 相关系数最高的7个窗口
    top7 = np.argsort(brownian_corr)[-8:-1]
    # 计算相关系数最高的7个窗口的平均价格序列和平均成交量序列
    # 平均价格序列
    mean_price = np.mean([price_volume_pairs[i][0] for i in top7], axis=0)
    # 平均成交量序列
    mean_volume = np.mean([price_volume_pairs[i][1] for i in top7], axis=0)
    # 计算平均价格序列和平均成交量序列的涨跌幅
    # 平均价格序列的涨跌幅
    mean_price_changes = np.diff(mean_price)
    # 平均成交量序列的涨跌幅
    mean_volume_changes = np.diff(mean_volume)
    # 将得到的平均价格序列涨跌幅和平均成交量序列涨跌幅作为典型序列的预测价格涨跌幅和预测成交量涨跌幅
    # 根据所得预测价格涨跌幅和预测成交量涨跌幅，设定交易信号
    # 交易信号
    # 遍历预测价格涨跌幅和预测成交量涨跌幅
    for price_change, volume_change in zip(mean_price_changes, mean_volume_changes):
        # 如果预测价格涨跌幅和预测成交量涨跌幅均大于0，买入
        if price_change > 0 and volume_change > 0:
            typical_signal.append(1)
        # 如果预测价格涨跌幅和预测成交量涨跌幅均小于0，卖出
        elif price_change < 0 and volume_change < 0:
            typical_signal.append(-1)
        # 其他情况不操作
        else:
            typical_signal.append(0)
    typical_signal.append(0)
    return typical_signal


from tqdm import tqdm

for index, item in enumerate(tqdm(price_volume_pairs)):
    typical_price, typical_volume, typical_signal = item
    # item[2] = gnerate_signal_for_pairs(typical_price, typical_volume, typical_signal)
    price_volume_pairs[index] = (
    (typical_price, typical_volume, gnerate_signal_for_pairs(typical_price, typical_volume, typical_signal)))

# 展示前pairs的前5条数据
# print(price_volume_pairs[:5])

df['Signal'] = np.concatenate([item[2] for item in price_volume_pairs])


def backtest_strategy(data):
    # Assume initial capital is 10000
    capital = 10000

    # Calculate daily returns based on price and signals
    data['returns'] = data['Close'].pct_change() * data['Signal'].shift(1)

    # Calculate cumulative returns
    data['cumulative_returns'] = (1 + data['returns']).cumprod()

    # Calculate performance metrics
    total_return = ep.cum_returns_final(data['returns'])
    annualized_return = ep.annual_return(data['returns'])
    max_drawdown = ep.max_drawdown(data['returns'])
    sharpe_ratio = ep.sharpe_ratio(data['returns'])

    # Calculate and print win rate
    positive_returns = data['returns'][data['returns'] > 0]
    total_signals = data['Signal'][data['Signal'] != 0]
    win_rate = len(positive_returns) / len(total_signals)
    print(f"Win Rate: {win_rate:.2%}")

    # Print performance metrics
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")

    # Plot cumulative returns
    data['cumulative_returns'].plot()

trade = backtest_strategy(df)