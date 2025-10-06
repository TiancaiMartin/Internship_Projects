# encoding: utf-8
import pandas as pd
from get_db_funs import get_db_data

# ---------------------------------1. 获取股票代码-------------------------------------------
table_name = 'AShareDescription'
AShareDescription = get_db_data(table_name)
kc = AShareDescription[AShareDescription['S_INFO_WINDCODE'].str.startswith('68')]

end_date = '20241031'
start_list_date = (pd.to_datetime(end_date) - pd.DateOffset(months=6)).strftime('%Y%m%d')
start_cal_date = (pd.to_datetime(end_date) - pd.DateOffset(months=12) + pd.DateOffset(days=1)).strftime('%Y%m%d')
kc = kc[kc['S_INFO_LISTDATE'] < start_list_date]
# ---------------------------------2. 获取基础数据-------------------------------------------

table = 'AShareEODPrices'
additional = {'S_INFO_WINDCODE': kc['S_INFO_WINDCODE'].tolist(),
              'TRADE_DT': [start_cal_date, end_date]}
keys = ['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_AMOUNT', 'S_DQ_CLOSE']
AShareEODPrices = get_db_data(table, keywords=keys, additional_conditions=additional)
S_DQ_AMOUNT_AVG = AShareEODPrices.groupby('S_INFO_WINDCODE')['S_DQ_AMOUNT'].mean().reset_index()
S_DQ_AMOUNT_AVG = S_DQ_AMOUNT_AVG.sort_values(by='S_DQ_AMOUNT', ascending=False)
S_DQ_AMOUNT_AVG = S_DQ_AMOUNT_AVG.iloc[:int(len(S_DQ_AMOUNT_AVG) * 0.9)]  # 剔除成交额低10%的股票
AShareEODPrices = AShareEODPrices[AShareEODPrices['S_INFO_WINDCODE'].isin(S_DQ_AMOUNT_AVG['S_INFO_WINDCODE'])]

keys = ['S_INFO_WINDCODE', 'CHANGE_DT', 'S_SHARE_TOTALA', 'FLOAT_A_SHR']
additional = {'S_INFO_WINDCODE': S_DQ_AMOUNT_AVG['S_INFO_WINDCODE'].tolist(), }
TOT_SHR = get_db_data('AShareCapitalization', additional_conditions=additional, keywords=keys)
TOT_SHR = TOT_SHR.rename(columns={'CHANGE_DT': 'TRADE_DT'})
keys = ['S_INFO_WINDCODE', 'CHANGE_DT1', 'S_SHARE_FREESHARES']
AShareFreeFloat = get_db_data('AShareFreeFloat', additional_conditions=additional, keywords=keys)
AShareFreeFloat = AShareFreeFloat.rename(columns={'CHANGE_DT1': 'TRADE_DT', 'S_SHARE_FREESHARES': 'FREEFLOAT_A_SHR'})

AShareEODPrices = AShareEODPrices.merge(TOT_SHR, on=['S_INFO_WINDCODE', 'TRADE_DT'], how='outer')
AShareEODPrices = AShareEODPrices.merge(AShareFreeFloat, on=['S_INFO_WINDCODE', 'TRADE_DT'], how='outer')
AShareEODPrices = AShareEODPrices.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])
AShareEODPrices[['S_SHARE_TOTALA', 'FLOAT_A_SHR', 'FREEFLOAT_A_SHR']] = AShareEODPrices.groupby('S_INFO_WINDCODE')[
    ['S_SHARE_TOTALA', 'FLOAT_A_SHR', 'FREEFLOAT_A_SHR']].fillna(method='ffill')
AShareEODPrices['TOTAL_MV'] = AShareEODPrices['S_DQ_CLOSE'] * AShareEODPrices['S_SHARE_TOTALA']
AShareEODPrices['FLOAT_MV'] = AShareEODPrices['S_DQ_CLOSE'] * AShareEODPrices['FLOAT_A_SHR']
AShareEODPrices['FREEFLOAT_MV'] = AShareEODPrices['S_DQ_CLOSE'] * AShareEODPrices['FREEFLOAT_A_SHR']
AShareEODPrices = AShareEODPrices.dropna(subset=['S_DQ_CLOSE'])
AShareEODPrices = AShareEODPrices.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])
TOTAL_MV_AVG = AShareEODPrices.groupby('S_INFO_WINDCODE')['TOTAL_MV'].mean().reset_index()
TOTAL_MV_AVG = TOTAL_MV_AVG.sort_values(by='TOTAL_MV', ascending=False)
FREEFLOAT_last = AShareEODPrices.groupby('S_INFO_WINDCODE')['FREEFLOAT_MV'].last().astype(float).reset_index()

kc_stock = TOTAL_MV_AVG.merge(AShareDescription[['S_INFO_WINDCODE', 'S_INFO_NAME', 'S_INFO_LISTDATE']],
                              on='S_INFO_WINDCODE')
kc_stock = kc_stock.merge(FREEFLOAT_last, on='S_INFO_WINDCODE')
kc_stock['rank'] = kc_stock['TOTAL_MV'].rank(ascending=False)
# ---------------------------------3. 获取已有成份股-------------------------------------------
table = 'AIndexMembers'
additional = {'S_INFO_WINDCODE': ['000698.SH', '000688.SH', '000699.SH']}
keys = ['S_CON_WINDCODE', 'S_CON_INDATE', 'S_CON_OUTDATE', 'S_INFO_WINDCODE']
kc = get_db_data(table, keywords=keys, additional_conditions=additional)
kc['S_CON_OUTDATE'] = kc['S_CON_OUTDATE'].fillna('20500101')
kc = kc[(kc['S_CON_INDATE'] <= end_date) & (kc['S_CON_OUTDATE'] > end_date)]
kc['NAME'] = kc['S_CON_WINDCODE'].map(dict(zip(AShareDescription['S_INFO_WINDCODE'], AShareDescription['S_INFO_NAME'])))
old_kc_50 = kc[kc['S_INFO_WINDCODE'] == '000688.SH']
old_kc_100 = kc[kc['S_INFO_WINDCODE'] == '000698.SH']
old_kc_200 = kc[kc['S_INFO_WINDCODE'] == '000699.SH']

# ---------------------------------4. 计算kc50-------------------------------------------
old_kc_50 = old_kc_50.merge(AShareDescription[['S_INFO_WINDCODE', 'S_INFO_NAME', 'S_INFO_LISTDATE']],
                            left_on='S_CON_WINDCODE', right_on='S_INFO_WINDCODE')
old_kc_50_new_rank = kc_stock[kc_stock['S_INFO_WINDCODE'].isin(old_kc_50['S_CON_WINDCODE'])]
new_rank_excl_old_50 = kc_stock[~(kc_stock['S_INFO_WINDCODE'].isin(old_kc_50['S_CON_WINDCODE']))]
# print(old_kc_50_new_rank)
# print(new_rank_excl_old_50)
del_kc50 = old_kc_50_new_rank[old_kc_50_new_rank['rank'] > 60]
print('kc50删除')
print(del_kc50)
add_kc50 = new_rank_excl_old_50.iloc[:len(del_kc50)]
new_kc_50 = pd.concat([old_kc_50_new_rank[old_kc_50_new_rank['rank'] <= 60], add_kc50])
new_kc_50_float_mv = AShareEODPrices[AShareEODPrices['S_INFO_WINDCODE'].isin(new_kc_50['S_INFO_WINDCODE'])]
new_kc_50_float_mv = new_kc_50_float_mv.groupby('S_INFO_WINDCODE')['FREEFLOAT_MV'].last().astype(float)
wgt = (new_kc_50_float_mv / new_kc_50_float_mv.sum()).rename('Weight').sort_values(ascending=False).reset_index()
new_kc_50 = new_kc_50.merge(wgt, on='S_INFO_WINDCODE')
new_kc_50 = new_kc_50.sort_values('Weight', ascending=False)
add_kc50 = new_kc_50[new_kc_50['S_INFO_WINDCODE'].isin(add_kc50['S_INFO_WINDCODE'])]

print('kc50新增')
print(add_kc50)

# ---------------------------------4. 计算kc100-------------------------------------------
kc_rank_for_100 = kc_stock[kc_stock['rank'] > 40]
kc_rank_for_100 = kc_rank_for_100[~kc_rank_for_100['S_INFO_WINDCODE'].isin(new_kc_50['S_INFO_WINDCODE'])]

kc_rank_for_100 = kc_rank_for_100.sort_values('TOTAL_MV', ascending=False)
kc_rank_for_100['rank'] = kc_rank_for_100['TOTAL_MV'].rank(ascending=False)

old_kc_100_new_rank = kc_rank_for_100[kc_rank_for_100['S_INFO_WINDCODE'].isin(old_kc_100['S_CON_WINDCODE'])]
new_rank_excl_old_100 = kc_rank_for_100[~(kc_rank_for_100['S_INFO_WINDCODE'].isin(old_kc_100['S_CON_WINDCODE']))]

# print(old_kc_100_new_rank)
# print(new_rank_excl_old_100)
print('kc100删除')
old_kc100_remain = old_kc_100_new_rank[old_kc_100_new_rank['rank'] <= 120]
del_kc100 = old_kc_100[~(old_kc_100['S_CON_WINDCODE'].isin(old_kc100_remain['S_INFO_WINDCODE']))]
print(del_kc100)
add_kc100 = new_rank_excl_old_100.iloc[:(100 - len(old_kc100_remain))]
new_kc_100 = pd.concat([old_kc100_remain, add_kc100])
new_kc_100_float_mv = AShareEODPrices[AShareEODPrices['S_INFO_WINDCODE'].isin(new_kc_100['S_INFO_WINDCODE'])]
new_kc_100_float_mv = new_kc_100_float_mv.groupby('S_INFO_WINDCODE')['FREEFLOAT_MV'].last().astype(float)
wgt = (new_kc_100_float_mv / new_kc_100_float_mv.sum()).rename('Weight').sort_values(ascending=False).reset_index()
new_kc_100 = new_kc_100.merge(wgt, on='S_INFO_WINDCODE')
new_kc_100 = new_kc_100.sort_values('Weight', ascending=False)
add_kc100 = new_kc_100[new_kc_100['S_INFO_WINDCODE'].isin(add_kc100['S_INFO_WINDCODE'])]
print('kc100新增')
print(add_kc100)

# ---------------------------------4. 计算kc200-------------------------------------------
kc_rank_for_200 = kc_stock[kc_stock['rank'] > 130]
kc_rank_for_200 = kc_rank_for_200[~kc_rank_for_200['S_INFO_WINDCODE'].isin(
    new_kc_50['S_INFO_WINDCODE'].tolist() + new_kc_100['S_INFO_WINDCODE'].tolist())]
kc_rank_for_200 = kc_rank_for_200.sort_values('TOTAL_MV', ascending=False)
kc_rank_for_200['rank'] = kc_rank_for_200['TOTAL_MV'].rank(ascending=False)

old_kc_200_new_rank = kc_rank_for_200[kc_rank_for_200['S_INFO_WINDCODE'].isin(old_kc_200['S_CON_WINDCODE'])]
new_rank_excl_old_200 = kc_rank_for_200[~(kc_rank_for_200['S_INFO_WINDCODE'].isin(old_kc_200['S_CON_WINDCODE']))]

# print(old_kc_100_new_rank)
# print(new_rank_excl_old_100)
print('kc200删除')
old_kc200_remain = old_kc_200_new_rank[old_kc_200_new_rank['rank'] <= 240]
del_kc200 = old_kc_200[~(old_kc_200['S_CON_WINDCODE'].isin(old_kc200_remain['S_INFO_WINDCODE']))]
print(del_kc200)
add_kc200 = new_rank_excl_old_200.iloc[:(200 - len(old_kc200_remain))]
new_kc_200 = pd.concat([old_kc200_remain, add_kc200])
new_kc_200_float_mv = AShareEODPrices[AShareEODPrices['S_INFO_WINDCODE'].isin(new_kc_200['S_INFO_WINDCODE'])]
new_kc_200_float_mv = new_kc_200_float_mv.groupby('S_INFO_WINDCODE')['FREEFLOAT_MV'].last().astype(float)
wgt = (new_kc_200_float_mv / new_kc_200_float_mv.sum()).rename('Weight').sort_values(ascending=False).reset_index()
new_kc_200 = new_kc_200.merge(wgt, on='S_INFO_WINDCODE')
new_kc_200 = new_kc_200.sort_values('Weight', ascending=False)
add_kc200 = new_kc_200[new_kc_200['S_INFO_WINDCODE'].isin(add_kc200['S_INFO_WINDCODE'])]
print('kc200新增')
print(add_kc200)
