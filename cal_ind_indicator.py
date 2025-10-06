# encoding: utf-8
import pandas as pd
import os
from typing import Dict, Optional


class IndustryMetricsCalculator:
    def __init__(
            self,
            basicinfo_path: str = 'basicinfo.pkl',
            balsheet_path: str = 'balsheet.pkl',
            cf_path: str = 'cf.pkl',
            profit_path: str = 'profit.pkl',
            sw_ind_level2_path: str = 'SW_IND_LEVEL2_NEW.pkl',
            start_year: int = None
    ):
        """
        初始化 IndustryMetricsCalculator 类，读取所需的五个数据文件。

        参数:
        - basicinfo_path: basicinfo.pkl 文件路径
        - balsheet_path: balsheet.pkl 文件路径
        - cf_path: cf.pkl 文件路径
        - profit_path: profit.pkl 文件路径
        - sw_ind_level2_path: SW_IND_LEVEL2_NEW.pkl 文件路径
        - start_year: 起始年份（可选）
        """
        # 检查文件是否存在
        for path in [basicinfo_path, balsheet_path, cf_path, profit_path, sw_ind_level2_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

        # 读取基本信息
        self.start_year = start_year
        self.basicinfo = pd.read_pickle(basicinfo_path)
        self.basicinfo = self.basicinfo[self.basicinfo['SETYPE'] == '101']
        self.compcode_to_symbol = self.basicinfo.reset_index().set_index('COMPCODE')['SYMBOL'].to_dict()
        self.symbol_to_compcode = self.basicinfo.set_index('SYMBOL')['COMPCODE'].to_dict()

        self.sw_ind_level2 = pd.read_pickle(sw_ind_level2_path)

        # 定义需要加载的财务表及其路径
        financial_paths = {
            'balsheet': balsheet_path,
            'cf': cf_path,
            'profit': profit_path
        }

        # 加载并过滤财务数据
        for table_name, path in financial_paths.items():
            df = self.load_and_filter_data(path)
            setattr(self, table_name, df)

        # 数据预处理：确保日期字段为 datetime 类型
        for df, date_cols in [
            (self.balsheet, ['PUBLISHDATE', 'ENDDATE']),
            (self.profit, ['PUBLISHDATE', 'BEGINDATE', 'ENDDATE']),
            (self.cf, ['PUBLISHDATE', 'BEGINDATE', 'ENDDATE']),
        ]:
            for col in date_cols:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # 过滤起始年份
        if self.start_year is not None:
            start_date = pd.Timestamp(year=self.start_year, month=1, day=1)
            for table in ['balsheet', 'cf', 'profit']:
                table_df = getattr(self, table)
                setattr(self, table, table_df[table_df['ENDDATE'] >= start_date].reset_index(drop=True))

        # 设置索引以加快查询速度
        self.basicinfo.set_index('SYMBOL', inplace=True)
        for table_name in financial_paths.keys():
            df = getattr(self, table_name)
            df = df.set_index(['PUBLISHDATE', 'SYMBOL', 'ENDDATE']).sort_index()
            setattr(self, table_name, df)
        self.sw_ind_level2.set_index(['INDUSTRYCODE'], inplace=True)

        # 仅存储财务表的名称
        self.financial_tables = list(financial_paths.keys())

        # 初始化缓存：按 calc_type, PIT 和 freq 分类
        self.stock_metrics_cache = {
            'quarter': {
                True: {'D': {}, 'W': {}, 'M': {}, 'Q': {}},
                False: {'D': {}, 'W': {}, 'M': {}, 'Q': {}}
            },
            'ttm': {
                True: {'D': {}, 'W': {}, 'M': {}, 'Q': {}},
                False: {'D': {}, 'W': {}, 'M': {}, 'Q': {}}
            }
        }

    def load_and_filter_data(self, path: str) -> pd.DataFrame:
        """
        加载并过滤财务数据。

        参数:
        - path: 数据文件路径

        返回:
        - 过滤后的 DataFrame
        """
        df = pd.read_pickle(path)
        mask = (df['COMPCODE'].isin(self.basicinfo['COMPCODE'])) & (df['REPORTTYPE'] == '1')
        df_filtered = df[mask].reset_index(drop=True)
        df_filtered['SYMBOL'] = df_filtered['COMPCODE'].map(self.compcode_to_symbol)
        return df_filtered

    def get_stock_data(
            self,
            expression: str,
            calc_type: str = 'quarter',  # 'quarter' 或 'ttm'
            PIT: bool = True,
            freq: str = 'M',  # 'D','W','M' 'Q'
            field_table_map: Optional[Dict[str, str]] = None  # 新增参数
    ) -> pd.DataFrame:
        """
        获取个股层面的某个字段或表达式计算后的全部历史数据。

        参数:
        - expression: 表达式字符串，例如 'NETPROFIT', 'NETPROFIT + SALES'
        - calc_type: 计算口径，'quarter' 或 'ttm'
        - PIT: 是否为点时间数据
        - freq: 频率，例如 'D','W','M','Q'
        - field_table_map: 字典，键为表达式中的变量，值为对应的表名（可选）

        返回:
        - DataFrame，索引为 PUBLISHDATE 或 ENDDATE，列为 COMPCODE
        """
        # 表达式安全性检查
        allowed_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_+-*/().= "
        if not all(char in allowed_chars for char in expression):
            raise ValueError("Expression contains invalid characters.")

        # 确保 calc_type 合法
        if calc_type not in ['quarter', 'ttm']:
            raise ValueError("calc_type must be 'quarter' or 'ttm'.")

        # 确保 freq 合法
        if freq not in ['D', 'W', 'M', 'Q']:
            raise ValueError("freq must be 'D', 'W', 'M', or 'Q'.")

        # 提取表达式中的字段
        fields = set()
        tokens = expression.split('=')[-1].replace('(', ' ').replace(')', ' ').replace('+', ' ').replace('-', ' ') \
            .replace('*', ' ').replace('/', ' ').split()
        for token in tokens:
            if token.isidentifier():
                fields.add(token)

        today = pd.to_datetime('today').strftime('%Y-%m-%d')
        if self.start_year is not None:
            date_list = pd.date_range(start=str(self.start_year) + '-01-01', end=today, freq=freq)
        else:
            date_list = pd.date_range(start='2010-01-01', end=today, freq=freq)

        # 收集需要的字段数据
        field_data = []
        for field in fields:
            if field_table_map and field in field_table_map:
                # 使用用户指定的表名
                table = field_table_map[field]
                if table not in self.financial_tables:
                    raise ValueError(f"Specified table '{table}' for field '{field}' is not a valid financial table.")
                df = getattr(self, table)
                if field in df.columns:
                    # 打印使用的表名
                    print(f"Using table '{table}' for field '{field}'.")
                else:
                    raise ValueError(f"Field '{field}' not found in specified table '{table}'.")
            else:
                # 自动查找表名
                found_tables = []
                for table in self.financial_tables:
                    df = getattr(self, table)
                    if field in df.columns:
                        found_tables.append(table)
                if len(found_tables) == 0:
                    raise ValueError(f"Field '{field}' not found in any financial table.")
                elif len(found_tables) == 1:
                    table = found_tables[0]
                else:
                    # 选择 NaN 最少的表
                    nan_counts = {table: getattr(self, table)[field].isna().sum() for table in found_tables}
                    selected_table = min(nan_counts, key=nan_counts.get)
                    print(
                        f"Field '{field}' found in multiple tables {found_tables}. Using table '{selected_table}' with least NaNs ({nan_counts[selected_table]} NaNs).")
                    table = selected_table
                df = getattr(self, table)

            if self.stock_metrics_cache[calc_type][PIT][freq].get(field) is not None:
                df_format = self.stock_metrics_cache[calc_type][PIT][freq][field]
                field_data.append(df_format)
                print(f"Using cached data for field '{calc_type, 'PIT', PIT, freq, field}'.")
                continue

            # 判断是否为 'balsheet'
            is_balsheet = (table == 'balsheet')

            if PIT:
                if not is_balsheet:
                    # 对于 cf 和 profit，使用区间数据求差
                    df_grouped = df.groupby(level=(0, 1, 2)).last()[field].unstack().groupby(level=1).ffill()
                    dfq = df_grouped.T.diff()
                    dfq.loc[dfq.index.month == 3] = df_grouped.T
                    df_period = dfq
                else:
                    # 对于 balsheet，不求差
                    df_period = df.groupby(level=(0, 1, 2)).last()[field].unstack().groupby(level=1).ffill().T

                if calc_type == 'ttm':
                    if not is_balsheet:
                        df_period = df_period.rolling(4).sum().T.stack().groupby(level=(0, 1)).last().unstack().ffill()
                    else:
                        df_period = df_period.rolling(4).mean().T.stack().groupby(level=(0, 1)).last().unstack().ffill()
                else:
                    df_period = df_period.T.stack().groupby(level=(0, 1)).last().unstack().ffill()

            else:
                # PIT=False 的情况
                df_grouped = df.groupby(level=(1, 2)).last()[field].unstack().groupby(level=0).ffill()
                if not is_balsheet:
                    dfq = df_grouped.T.diff()
                    dfq.loc[dfq.index.month == 3] = df_grouped.T
                    df_period = dfq
                else:
                    df_period = df_grouped

                if calc_type == 'ttm':
                    if not is_balsheet:
                        df_period = df_period.rolling(4).sum()
                    else:
                        df_period = df_period.rolling(4).mean()

            df_period = df_period.sort_index()
            df_format = df_period.reindex(date_list, method='ffill').stack().to_frame(field)
            field_data.append(df_format)
            self.stock_metrics_cache[calc_type][PIT][freq][field] = df_format

        if not field_data:
            raise ValueError("No fields were processed. Please check your expression and field_table_map.")

        # 创建一个 DataFrame，包含所有字段
        combined_df = pd.concat(field_data, axis=1)

        # 计算表达式
        try:
            if '=' in expression:
                combined_df.eval(expression, inplace=True)
            else:
                combined_df['metric'] = combined_df.eval(expression)
        except Exception as e:
            raise ValueError(f"Error evaluating expression '{expression}': {e}")

        return combined_df

    def get_industry_indicator(
            self,
            expression: str,
            industry: str,
            **variables  # 传入的指标变量，如 PROFIT=PROFIT
    ) -> pd.DataFrame:
        """
        计算指定行业的某个基本面指标的全部历史变化情况，包括各种变化算子和聚合方式。

        参数:
        - expression: 表达式字符串，例如 'sum(PROFIT)', 'mean(ROE)'
        - industry: 行业代码，例如 '0001'
        - variables: 传入的指标变量，均为 pd.DataFrame 或 pd.Series，索引为日期，列为 SYMBOL

        返回:
        - DataFrame，索引为日期，包含计算后的指标
        """
        # 定义支持的聚合函数
        def agg_sum(df):
            return df.sum(axis=1)

        def agg_mean(df):
            return df.mean(axis=1)

        def weighted_mean(df, weights):
            return (df * weights).sum(axis=1) / weights.sum(axis=1)

        # 定义支持的计算函数
        def yoy(series):
            return change_rate(series)

        def mom(series):
            return change_rate(series,periods=1)

        def change_rate(series, periods=12):
            return series.pct_change(periods=periods)

        def change_diff(series, periods=1):
            return series.diff(periods=periods)

        def pct_change_time(series: pd.Series, offset: str) -> pd.Series:
            """
            计算基于时间间隔的百分比变化。
            参数:
            - series: pd.Series，索引为 datetime 类型
            - offset: 时间偏移量字符串，例如 '365D' 表示365天，'12M' 表示12个月
            返回:
            - pd.Series，表示基于时间间隔的百分比变化
            """
            series = series.sort_index()
            shifted = series.shift(freq=offset)
            pct_change = (series - shifted) / shifted
            return pct_change

        # 添加函数到本地字典
        allowed_functions = {
            'sum': agg_sum,
            'mean': agg_mean,
            'weighted_mean': weighted_mean,
            'yoy': yoy,
            'mom': mom,
            'change_rate': change_rate,
            'change_diff': change_diff,
            'pct_change_time': pct_change_time
        }

        # 安全的表达式评估环境
        safe_dict = {name: func for name, func in allowed_functions.items()}

        # 获取行业成分股
        try:
            sw_ind_stock_code = self.sw_ind_level2.loc[industry]['SYMBOL']
        except KeyError:
            raise ValueError(f"Industry code '{industry}' not found in sw_ind_level2.")

        # 确保 sw_ind_stock_code 是列表
        if isinstance(sw_ind_stock_code, pd.Series):
            symbols = sw_ind_stock_code.tolist()
        elif isinstance(sw_ind_stock_code, str):
            symbols = [sw_ind_stock_code]
        else:
            symbols = list(sw_ind_stock_code)

        # 检查传入的变量是否为 DataFrame 或 Series
        for var_name, var_data in variables.items():
            if not isinstance(var_data, (pd.Series, pd.DataFrame)):
                raise ValueError(f"Variable '{var_name}' must be a pandas Series or DataFrame.")

        # 提取行业内的指标数据
        industry_data = {}
        for var_name, var_data in variables.items():
            if isinstance(var_data, pd.DataFrame):
                # 选择行业内的 SYMBOL 列
                if not set(symbols).issubset(var_data.columns):
                    missing = set(symbols) - set(var_data.columns)
                    raise ValueError(f"Variable '{var_name}' is missing symbols: {missing}")
                industry_data[var_name] = var_data[symbols]
            elif isinstance(var_data, pd.Series):
                index_level1 = var_data.index.get_level_values(1)
                # 如果是 Series，假设索引是 (date, symbol)
                if not all(symbol in index_level1 for symbol in symbols):
                    missing = set(symbols) - set(var_data.index.levels[1])
                    raise ValueError(f"Variable '{var_name}' is missing symbols: {missing}")
                industry_data[var_name] = var_data[index_level1.isin(symbols)].unstack()

        # 添加指标数据到安全字典
        for var_name, data in industry_data.items():
            safe_dict[var_name] = data

        # 评估表达式
        try:
            # 使用 pandas eval 进行矢量化计算
            industry_metric = pd.eval(expression, local_dict=safe_dict, engine='python')
        except Exception as e:
            raise ValueError(f"Error evaluating industry expression '{expression}': {e}")

        # 将结果转换为 DataFrame
        if isinstance(industry_metric, pd.Series):
            result = industry_metric.to_frame(name='Metric')
        else:
            result = pd.DataFrame(industry_metric, columns=['Metric'])

        return result

# 示例用法
if __name__ == "__main__":
    calculator = IndustryMetricsCalculator(
        basicinfo_path='basicinfo.pkl',
        balsheet_path='balsheet.pkl',
        cf_path='cf.pkl',
        profit_path='profit.pkl',
        sw_ind_level2_path='SW_IND_LEVEL2_NEW.pkl',
        start_year=2008
    )

    # 获取个股数据可以跨三张表获取，通过pd.dataframe.eval计算带括号的加减乘除、传入多个表达式、给计算结果命名等
    # 假定资产负债表是时点值，现金流量表和利润表是区间值。
    # 如果计算季度值calc_type='quarter'，则需要将现金流量表和利润表转换为季度值，资产负债表不需要。
    # 如果计算TTM值calc_type='ttm'，则需要将现金流量表和利润表计算为过去四个季度的季度值求和，资产负债表则计算为过去四个季度的季度值求平均。
    # PIT=True表示可获得最新数据，False表示按照报告期获取
    # freq='M'表示按月计算，可以传入'D','W','M' 'Q'
    # field_table_map={'NETPROFIT': 'profit'}表示使用profit表中的NETPROFIT字段。因为三张表中含有相同字段，所以需要指定表名，如果不指定，
    # 则会自动选择含有该字段的表，如果有多个表含有该字段，则选择缺失值最少的表。
    TEST_DATA = calculator.get_stock_data(
        expression='TOTLIAB + BIZCASHINFL + NETPROFIT',
        calc_type='quarter',
        PIT=True,
        field_table_map={'NETPROFIT': 'profit'}
    )

    # 个股层面的 ROE（TTM）
    ROE = calculator.get_stock_data(
        expression='ROE=NETPROFIT/PARESHARRIGH',
        calc_type='ttm',
        PIT=True,
    )

    # 个股层面的净利润（TTM）
    NETPROFIT = calculator.get_stock_data(
        expression='NETPROFIT=NETPROFIT',
        calc_type='ttm',
        PIT=True,
    )

    # 个股层面的净利润（TTM）
    ASSET = calculator.get_stock_data(
        expression='TOTASSET=TOTASSET',
        calc_type='quarter',
        PIT=True,
    )

    # 计算某行业的净利润同比变化
    yoy_profit_110100 = calculator.get_industry_indicator(
        expression='yoy(sum(NETPROFIT))',
        industry='110100',
        NETPROFIT=NETPROFIT['NETPROFIT']
    )

    # 计算某行业的 ROE 加权平均
    roe_weight = calculator.get_industry_indicator(
        expression='weighted_mean(ROE, ASSET)',
        industry='110100',
        ROE=ROE['ROE'],
        ASSET=ASSET['TOTASSET']
    )
