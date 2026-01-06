# utils.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ------------ 中文显示（Mac 优先）------------
plt.rcParams["font.family"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# =========================
# 数据读取
# =========================
def load_local(path: str):
    """
    读取本地 CSV（来自 Wind 下载结果）
    兼容两种格式：
    1) 有 'date' 列
    2) 第一列就是日期（无列名或非 date）
    """
    df = pd.read_csv(path)

    # 统一列名为小写
    df.columns = [str(c).lower() for c in df.columns]

    # 处理日期列
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    else:
        # 默认第一列是日期
        first_col = df.columns[0]
        df[first_col] = pd.to_datetime(df[first_col])
        df = df.set_index(first_col)

    df = df.sort_index()

    # 关键列兜底
    if "close" not in df.columns:
        raise KeyError(f"{path} 缺少 close 列")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    # 收益率列
    if "ret" not in df.columns:
        if "pct_chg" in df.columns:
            df["ret"] = df["pct_chg"] / 100.0
        else:
            df["ret"] = df["close"].pct_change()

    df["ret"] = df["ret"].fillna(0.0)

    return df


# =========================
# 绩效指标（按你要求：mean*252）
# =========================
def annual_return(ret):
    return (1 + ret).prod() ** (252/len(ret)) - 1

def annual_vol(ret):
    return ret.std() * np.sqrt(252)

def sharpe(ret):
    if ret.std() == 0:
        return 0
    return annual_return(ret) / annual_vol(ret)

def max_drawdown(nav):
    nav = nav / nav.iloc[0]
    roll_max = nav.cummax()
    dd = nav / roll_max - 1
    return dd.min()

def plot_nav(nav_dict: dict):
    plt.figure(figsize=(12,6))
    for name, series in nav_dict.items():
        plt.plot(series.index, series.values, label=name)
    plt.legend()
    plt.title("NAV Curves")
    plt.grid(True)
    plt.show()

def performance_report(ret, nav):
    ann_ret = ret.mean() * 252
    ann_vol = ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    peak = nav.cummax()
    mdd = ((nav - peak) / peak).min()

    calmar = -ann_ret / mdd if mdd < 0 else np.nan
    win_rate = (ret > 0).mean()

    return {
        "Annual Return": ann_ret,
        "Annual Vol": ann_vol,
        "Sharpe": sharpe,
        "MaxDD": mdd,
        "Calmar": calmar,
        "Win Rate": win_rate
    }

def yearly_performance(ret):
    df = pd.DataFrame({"ret": ret})
    df["year"] = df.index.year

    out = []
    for y in sorted(df["year"].unique()):
        sub = df[df["year"] == y]["ret"]
        if len(sub) < 20:
            continue

        nav = (1 + sub).cumprod()
        peak = nav.cummax()
        mdd = ((nav - peak) / peak).min()

        out.append({
            "Year": y,
            "Return": sub.mean()*252,
            "Vol": sub.std()*np.sqrt(252),
            "Sharpe": sub.mean()*252 / (sub.std()*np.sqrt(252)+1e-8),
            "MaxDD": mdd
        })

    return pd.DataFrame(out).set_index("Year")