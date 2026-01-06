# main.py —— 复现天风五资产 CTA + 资产配置完整框架
import warnings
warnings.filterwarnings("ignore")
from utils import *
from strategy import *


print("Loading data...")

# =============================
# 0. 读取数据（Wind 本地 CSV）
# =============================
gold   = load_local("data/gold.csv")
comm   = load_local("data/comm.csv")
stock  = load_local("data/stock.csv")
cash   = load_local("data/cash.csv")     # CBA02201.CS
credit = load_local("data/credit.csv")   # CBA02041.CS

# 价格序列
price_dict = {
    "gold":  gold["open"],
    "comm":  comm["open"],
    "stock": stock["open"],
    "cash":  cash["close"],
    "credit": credit["close"],
}

# =============================
# 1. 低风险资产轮动（货币↔信用债）
# =============================
ret_low, credit_weight = low_risk_rotation(
    cash["close"], credit["close"],
    k=20, m=5, fee=0.0005
)

nav_low = (1 + ret_low).cumprod()


# =============================
# 2. 高风险趋势信号
# =============================
sig_ma = {
    "gold":  trend_filter_ma(price_dict["gold"], 120),
    "comm":  trend_filter_ma(price_dict["comm"], 120),
    "stock": trend_filter_ma(price_dict["stock"], 120),
}

sig_dc = {
    "gold":  trend_filter_dc(price_dict["gold"], 20),
    "comm":  trend_filter_dc(price_dict["comm"], 20),
    "stock": trend_filter_dc(price_dict["stock"], 20),
}

sig_dcma = {
    "gold":  trend_filter_dc_ma(price_dict["gold"], 20, 120),
    "comm":  trend_filter_dc_ma(price_dict["comm"], 20, 120),
    "stock": trend_filter_dc_ma(price_dict["stock"], 20, 120),
}

# 高风险收益序列字典
rets = {
    "gold":  price_dict["gold"].pct_change().fillna(0),
    "comm":  price_dict["comm"].pct_change().fillna(0),
    "stock": price_dict["stock"].pct_change().fillna(0),
}


# =============================
# 3. 高风险子组合 NAV（波动率加权）
# =============================
ret_risk_ma,   w_ma   = build_risk_asset_portfolio(rets, sig_ma)
ret_risk_dc,   w_dc   = build_risk_asset_portfolio(rets, sig_dc)
ret_risk_dcma, w_dcma = build_risk_asset_portfolio(rets, sig_dcma)

nav_risk_ma   = (1 + ret_risk_ma).cumprod()
nav_risk_dc   = (1 + ret_risk_dc).cumprod()
nav_risk_dcma = (1 + ret_risk_dcma).cumprod()

# ================================================================
# 4. 最终五资产组合（稳健优化版：控制回撤）
# ================================================================

# --- 核心修改：调低目标，限制上限 ---
TARGET_VOL = 0.12  # 降为 12% (既能提升收益，又不会回撤过大)
MAX_LEVERAGE = 2.0  # 降为 2.0 倍 (封顶，防止低波时仓位过重)
VOL_WINDOW = 20  # 波动率计算窗口

# 五个资产分别的日收益
df_all_ret = pd.DataFrame({
    "gold": rets["gold"],
    "comm": rets["comm"],
    "stock": rets["stock"],
    "credit": credit["close"].pct_change().fillna(0),
    "cash": cash["close"].pct_change().fillna(0),
})

sig_sets = {
    "MA": sig_ma,
    "DC": sig_dc,
    "DCMA": sig_dcma,
}

nav_risk_map = {
    "MA": nav_risk_ma,
    "DC": nav_risk_dc,
    "DCMA": nav_risk_dcma,
}

nav_total = {}
weights_total = {}

print(f"\n[策略调整] 目标波动率: {TARGET_VOL * 100}% | 最大杠杆: {MAX_LEVERAGE}x | 波动窗口: {VOL_WINDOW}D")

for name, sig_dic in sig_sets.items():
    # ① 获取该策略的高风险子组合 NAV
    nav_r = nav_risk_map[name]

    # ② 计算滚动波动率 (更平滑)
    rolling_vol = nav_r.pct_change().rolling(VOL_WINDOW).std() * np.sqrt(252)
    # 设定一个波动率地板(8%)，防止市场极其平静时杠杆算出来太大
    rolling_vol = rolling_vol.shift(1).fillna(0.08).clip(lower=0.08)

    # ③ 计算杠杆
    leverage = TARGET_VOL / rolling_vol
    leverage = leverage.clip(upper=MAX_LEVERAGE)  # 严格封顶

    # ④ 获取高风险资产内部权重 (基础权重)
    _, w_risk = build_risk_asset_portfolio(rets, sig_dic)

    # ⑤ 构建组合
    w_all = pd.DataFrame(index=w_risk.index, columns=df_all_ret.columns)

    # --- 权重分配逻辑 (Risk Budgeting 思想) ---
    # 高风险仓位 = 基础权重 * 杠杆
    # 注意：这里我们引入 0.4 的基础风险预算，即默认只用 40% 的钱做 CTA，
    # 只有当波动率非常低且趋势极好时，才会动用杠杆超过 100%。
    RISK_BUDGET_BASE = 0.4

    final_risk_weight = w_risk.mul(leverage * RISK_BUDGET_BASE, axis=0)
    w_all[["gold", "comm", "stock"]] = final_risk_weight

    # 剩余仓位给低风险 (永远保留至少 10-20% 的现金/债券作为安全垫，不建议满仓干)
    total_risky = final_risk_weight.sum(axis=1)
    remain = (1 - total_risky).clip(lower=0)  # 允许为0，但不允许为负(不融资)

    w_all["credit"] = credit_weight * remain
    w_all["cash"] = (1 - credit_weight) * remain

    weights_total[name] = w_all.copy()

    # ⑥ 计算收益
    daily_ret = (w_all.shift(1) * df_all_ret).sum(axis=1)
    nav_total[name] = (1 + daily_ret).cumprod()

nav_total["Low-Risk"] = nav_low

# =============================
# 5. NAV 曲线（图13）
# =============================
plot_nav({
    "Low-Risk Rotation": nav_total["Low-Risk"],
    "MA": nav_total["MA"],
    "DC": nav_total["DC"],
    "DC+MA Final": nav_total["DCMA"],
})


# =============================
# 6. 输出绩效指标
# =============================
def show_report(name, nav):
    ret = nav.pct_change().fillna(0)
    rpt = performance_report(ret, nav)
    print(f"\n===== {name} =====")
    for k, v in rpt.items():
        print(f"{k}: {v}")


show_report("Low-Risk Rotation", nav_total["Low-Risk"])
show_report("MA", nav_total["MA"])
show_report("DC", nav_total["DC"])
show_report("DC+MA（最终策略）", nav_total["DCMA"])


# =================================
# 7. 输出逐年表现 (新增模块)
# =================================
print("\n===== 逐年表现详细数据 =====")
for name in nav_total.keys():

    # 1. 提取最终策略的日收益序列
    final_daily_ret = nav_total[name].pct_change().fillna(0)

    # 2. 调用 utils.py 里的函数计算年化数据
    df_yearly = yearly_performance(final_daily_ret)

    # 3. 格式化输出（保留2位小数，百分比显示）
    # 这样打印出来直接截图就可以发给 Mentor
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    format_dict = {
        'Return': '{:.2%}',
        'Vol': '{:.2%}',
        'Sharpe': '{:.2f}',
        'MaxDD': '{:.2%}'
    }
    print(f"\n--- {name} ---")
    print(df_yearly.style.format(format_dict).to_string())
# ================================================================
# 8. 参数敏感性分析 (DC+MA 策略参数遍历)
#    复现天风研报 表15 (年化收益) 和 表16 (夏普比)
# ================================================================
import seaborn as sns  # 如果报错 No module named 'seaborn'，请在终端运行: pip install seaborn
def run_dcma_parameter_sweep(ma_range, dc_range):
    print(f"\n正在进行参数遍历 (共 {len(ma_range) * len(dc_range)} 组参数)，请稍候...")

    # 初始化结果矩阵 (行索引=MA长周期, 列索引=DC短周期)
    results_ret = pd.DataFrame(index=ma_range, columns=dc_range)
    results_sharpe = pd.DataFrame(index=ma_range, columns=dc_range)

    # --- 保持与“最终策略”完全一致的配置 ---
    TARGET_VOL = 0.12  # 目标波动率 10%
    MAX_LEVERAGE = 2.0  # 最大杠杆 2.0x
    VOL_WINDOW = 20  # 波动率窗口 60天
    RISK_BUDGET_BASE = 0.4  # 基础风险预算 40%

    # 双重循环遍历
    for k_ma in ma_range:
        for k_dc in dc_range:
            # 1. 重新计算三类资产的 DC+MA 信号
            # 注意：gold/comm/stock 使用相同的 k_dc 和 k_ma
            curr_sigs = {
                "gold": trend_filter_dc_ma(price_dict["gold"], k_dc=k_dc, k_ma=k_ma),
                "comm": trend_filter_dc_ma(price_dict["comm"], k_dc=k_dc, k_ma=k_ma),
                "stock": trend_filter_dc_ma(price_dict["stock"], k_dc=k_dc, k_ma=k_ma),
            }

            # 2. 构建高风险子组合 (波动率倒数加权)
            # 使用默认 fee (或者你可以显式传入 fee=0.0005)
            _, w_risk = build_risk_asset_portfolio(rets, curr_sigs)

            # 3. 计算高风险子组合的基础净值 (用于算波动率)
            # 这是一个“虚拟净值”，仅用于计算 leverage
            high_risk_raw_ret = (w_risk.shift(1) * pd.DataFrame(rets)).sum(axis=1)

            # 4. 计算动态杠杆 (Target Vol 逻辑)
            rolling_vol = high_risk_raw_ret.rolling(VOL_WINDOW).std() * np.sqrt(252)
            rolling_vol = rolling_vol.shift(1).fillna(0.08).clip(lower=0.08)  # 地板波动率8%

            leverage = TARGET_VOL / rolling_vol
            leverage = leverage.clip(upper=MAX_LEVERAGE)  # 封顶

            # 5. 分配最终权重
            # 高风险部分 = 基础权重 * 杠杆 * 风险预算(0.4)
            w_final_risk = w_risk.mul(leverage * RISK_BUDGET_BASE, axis=0)

            # 低风险部分 = 1 - 高风险总仓位
            total_risky_weight = w_final_risk.sum(axis=1)
            remain_weight = (1 - total_risky_weight).clip(lower=0)

            # 6. 组合总收益
            # High Risk 贡献
            ret_part_risk = (w_final_risk.shift(1) * pd.DataFrame(rets)).sum(axis=1)
            # Low Risk 贡献 (直接利用之前算好的 credit_weight)
            ret_part_credit = (credit_weight * remain_weight).shift(1) * df_all_ret["credit"]
            ret_part_cash = ((1 - credit_weight) * remain_weight).shift(1) * df_all_ret["cash"]

            port_ret = ret_part_risk + ret_part_credit + ret_part_cash

            # 7. 记录指标
            ann_ret = port_ret.mean() * 252
            ann_vol = port_ret.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

            results_ret.loc[k_ma, k_dc] = ann_ret
            results_sharpe.loc[k_ma, k_dc] = sharpe

    return results_ret, results_sharpe

# --- 定义参数范围 (根据你的截图) ---
ma_params = [80, 90, 100, 110, 120, 130, 140]  # 表格的行
dc_params = [10, 15, 20, 25, 30]  # 表格的列

# --- 执行遍历 ---
df_table15, df_table16 = run_dcma_parameter_sweep(ma_params, dc_params)


# --- 绘图函数 ---
def plot_heatmap(df, title, fmt_str):
    df = df.astype(float)
    plt.figure(figsize=(10, 6))
    # 使用 RdYlGn (红-黄-绿) 色系，绿色代表高收益/高夏普
    sns.heatmap(df, annot=True, fmt=fmt_str, cmap="RdYlGn",
                linewidths=0.5, linecolor='gray')
    plt.title(title, fontsize=14)
    plt.xlabel("Donchian Window (K2)", fontsize=12)
    plt.ylabel("MA Window (K1)", fontsize=12)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


# --- 输出结果 ---
print("\n===== 表15: 不同K1, K2取值下策略年化收益 =====")
print(df_table15)
plot_heatmap(df_table15, "Table 15: Annual Return by Parameters", ".2%")

print("\n===== 表16: 不同K1, K2取值下策略夏普比 =====")
print(df_table16)
plot_heatmap(df_table16, "Table 16: Sharpe Ratio by Parameters", ".2f")

print("\n回测结束。")