# strategy.py —— 五资产统一配置版（天风 CTA + 资产配置复现）

import pandas as pd
import numpy as np


# =========================
# 1. MA 长周期趋势过滤
# =========================
def trend_filter_ma(price: pd.Series, k: int = 120):
    price = pd.Series(price)
    ma = price.rolling(k).mean()
    sig = (price > ma).astype(int)
    return sig.shift(1).fillna(0).astype(int)  # T+1 调仓


# =========================
# 2. Donchian 唐安奇过滤
# =========================
def trend_filter_dc(price: pd.Series, k: int = 20):
    price = pd.Series(price)
    high_k = price.rolling(k).max().shift(1)
    low_k  = price.rolling(k).min().shift(1)

    sig = pd.Series(0, index=price.index)

    for i in range(1, len(price)):
        p = price.iloc[i]
        if p > high_k.iloc[i]:
            sig.iloc[i] = 1
        elif p < low_k.iloc[i]:
            sig.iloc[i] = 0
        else:
            sig.iloc[i] = sig.iloc[i - 1]

    return sig.shift(1).fillna(0).astype(int)


# ================================
# 3. 唐安奇 + MA 的双重趋势过滤 (最终增强版：双重止损 + 阳线确认)
# ================================
def trend_filter_dc_ma(price, k_dc=20, k_ma=120):
    price = pd.Series(price)

    # 计算收益率用于判断当日是否上涨
    ret = price.pct_change().fillna(0)

    # 计算指标
    don_high = price.rolling(k_dc).max().shift(1)  # 唐安奇上轨 (昨日及之前)
    don_low = price.rolling(k_dc).min().shift(1)  # 唐安奇下轨 (昨日及之前)
    ma = price.rolling(k_ma).mean()  # 长周期均线

    sig = pd.Series(0, index=price.index)

    for i in range(1, len(price)):
        p = price.iloc[i]
        r = ret.iloc[i]

        # --- 信号判断逻辑 ---

        # Buy (开仓条件)：
        # 1. 价格突破过去20天最高价 (趋势突破)
        # 2. 价格在120日均线之上 (大趋势向上)
        # 3. 当日收益率 > 0 (新增：必须是阳线，拒绝假突破)
        if (p > don_high.iloc[i]) and (p > ma.iloc[i]) and (r > 0):
            sig.iloc[i] = 1

        # Sell (双重止损)：
        # 条件A：价格跌破 120 日均线 (长期趋势破坏)
        # 条件B：价格跌破 20 日新低 (短期趋势反转)
        # 满足任一条件即清仓
        elif (p < ma.iloc[i]) or (p < don_low.iloc[i]):
            sig.iloc[i] = 0

        else:
            # 既没触发买入，也没触发止损，维持原有仓位
            sig.iloc[i] = sig.iloc[i - 1]

    return sig.shift(1).fillna(0).astype(int)


# ============================================
# 4. 低风险资产轮动：货币 ↔ 信用债
# ============================================
def low_risk_rotation(cash_close, credit_close, k=20, m=5, fee=0.0005):

    df = pd.DataFrame(index=cash_close.index)
    df["credit"] = credit_close
    df["cash"] = cash_close
    df["rel"] = df["credit"] / df["cash"]

    # 唐安奇信号
    df["don_up"] = df["rel"].rolling(k).max().shift(1)
    df["don_low"] = df["rel"].rolling(k).min().shift(1)

    df["up"] = (df["rel"] > df["don_up"]).astype(int)
    df["down"] = (df["rel"] < df["don_low"]).astype(int)

    df["up"] = df["up"].replace({0: np.nan}).ffill(limit=m - 1).fillna(0)
    df["down"] = df["down"].replace({0: np.nan}).ffill(limit=m - 1).fillna(0)

    # 信号累计天数
    def block_count(s):
        c = 0
        out = []
        for v in s:
            c = c + 1 if v else 0
            out.append(c)
        return out

    df["up_cnt"] = block_count(df["up"])
    df["down_cnt"] = block_count(df["down"])

    hold = pd.DataFrame(0, index=df.index, columns=["credit", "cash"])
    hold.iloc[0] = [0, 1]

    for i in range(1, len(df)):
        prev = hold.iloc[i - 1].copy()
        up = df["up_cnt"].iloc[i]
        down = df["down_cnt"].iloc[i]

        # 向信用债加仓
        if up > 0 and down == 0:
            prev["credit"] = min(1, prev["credit"] + 1 / m)
            prev["cash"] = 1 - prev["credit"]

        # 向货币加仓
        elif down > 0 and up == 0:
            prev["cash"] = min(1, prev["cash"] + 1 / m)
            prev["credit"] = 1 - prev["cash"]

        hold.iloc[i] = prev

    # 计算收益
    ret_credit = df["credit"].pct_change().fillna(0)
    ret_cash = df["cash"].pct_change().fillna(0)

    pnl = hold["credit"].shift(1) * ret_credit + hold["cash"].shift(1) * ret_cash
    cost = fee * hold.diff().abs().sum(axis=1).fillna(0)

    ret_total = pnl - cost
    return ret_total, hold["credit"]   # credit 比例


# =========================================================
# 5. 高风险子组合（黄金、商品、股票）—— 使用波动率权重
# =========================================================
def build_risk_asset_portfolio(rets: dict, sigs: dict, fee=0.003):

    df_ret = pd.DataFrame(rets).fillna(0)
    df_sig = pd.DataFrame(sigs).fillna(0).astype(int)

    weights = pd.DataFrame(0, index=df_ret.index, columns=df_ret.columns)
    prev_w = pd.Series(0, index=df_ret.columns)

    for i in range(1, len(df_ret)):
        t_prev = df_ret.index[i - 1]
        t = df_ret.index[i]

        # 使用当天信号决定权重（T+1 调仓）
        active = df_sig.loc[t] == 1

        if active.sum() == 0:
            w = prev_w * 0
        else:
            sigma = df_ret.loc[:t_prev, active].rolling(20).std().iloc[-1]
            inv = 1 / sigma.replace(0, np.nan)
            w0 = inv / inv.sum()
            w = w0.reindex(df_ret.columns).fillna(0)

        weights.loc[t] = w
        prev_w = w

    # NAV
    nav = pd.Series(1.0, index=df_ret.index)
    for i in range(1, len(df_ret)):
        t_prev = df_ret.index[i - 1]
        t = df_ret.index[i]

        pnl = (weights.loc[t_prev] * df_ret.loc[t]).sum()
        cost = fee * abs(weights.loc[t] - weights.loc[t_prev]).sum()
        nav.iloc[i] = nav.iloc[i - 1] * (1 + pnl - cost)

    return nav.pct_change().fillna(0), weights