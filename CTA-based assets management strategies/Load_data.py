# Load_data.py -- Wind 数据下载 (严谨回测版)

from WindPy import w
import pandas as pd
import os

# 启动 Wind API
if not w.isconnected():
    w.start()

START = "2007-01-01"
END = "2019-10-30"  # 如需更新到最新日期，可改为 datetime.today()

SAVE_DIR = "data"
os.makedirs(SAVE_DIR, exist_ok=True)

# === 修改点 1: 增加 'open' 字段，这是严谨回测(T+1成交)的核心 ===
FIELDS = ["open", "high", "low", "close", "volume", "pct_chg"]

# 资产代码映射
# 注意：债券指数建议确认是“总财富(Total Wealth)”指数，包含利息再投资收益
CODE_MAP = {
    "gold": "S0035819",  # 黄金 (上海金交所 Au99.99 或类似现货)
    "comm": "NH0100.NHF",  # 南华商品指数 (监控换月跳空的最佳指数)
    "stock": "000300.SH",  # 沪深300

    # 债券类通常只有日频收盘价(Fixing)，Open/High/Low 可能与 Close 相同或为空，脚本会自动处理
    "cash": "CBA02201.CS",  # 中债-国债总财富(总值)指数
    "credit": "CBA02041.CS"  # 中债-信用债总财富(总值)指数
}


def download_one(name, code):
    print(f"\n=== Downloading {name} ({code}) ===")

    # 1. 先下载 close 来确定标准的交易日期索引
    #    (Wind 有时不同指标返回长度不一，以 Close 为准最安全)
    base = w.wsd(code, "close", START, END, "")
    if base.ErrorCode != 0:
        print(f"Error fetching base data for {name}: {base.ErrorCode}")
        return

    dates = pd.to_datetime(base.Times)
    df = pd.DataFrame(index=dates)

    # 2. 逐一下载字段
    for field in FIELDS:
        print(f"  downloading {field} ...")
        # option="" 留空使用默认设置
        data = w.wsd(code, field, START, END, "")

        # 检查数据有效性
        if data.ErrorCode == 0 and data.Data and len(data.Data[0]) == len(dates):
            df[field] = data.Data[0]
        else:
            print(f"  Warning: {field} 数据缺失或长度不匹配，尝试填充...")
            # 如果是 Open/High/Low 缺失（常见于某些债券指数），用 Close 填充
            if field in ['open', 'high', 'low'] and 'close' in df.columns:
                print(f"  -> 用 Close 填充 {field}")
                df[field] = df['close']
            else:
                df[field] = pd.Series([None] * len(dates), index=dates)

    # 3. 数据清洗与计算

    # 确保数值类型
    df = df.apply(pd.to_numeric, errors='coerce')

    # 处理收益率 (ret)
    # 优先使用 pct_chg (Wind提供)，如果没有则用 close 计算
    if "pct_chg" in df.columns and df["pct_chg"].notna().sum() > 10:
        df["ret"] = df["pct_chg"] / 100.0
    else:
        print("  Info: pct_chg 不可用或数据过少，使用 close 计算收益")
        df["ret"] = df["close"].pct_change().fillna(0.0)

    # 填补 Volume 空值 (指数可能没有成交量)
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0.0)

    # 最终空值检查 (向前填充，处理停牌)
    df = df.fillna(method='ffill').fillna(0.0)

    # 保存
    file_path = f"{SAVE_DIR}/{name}.csv"
    df.to_csv(file_path)
    print(f"Saved → {file_path}")


# 执行下载
if __name__ == "__main__":
    print(f"Start downloading data from {START} to {END}...")
    for name, code in CODE_MAP.items():
        download_one(name, code)
    print("\nAll data downloaded successfully.")