from get_db_funs import get_db_data
# add_conditions
add_conditions = {'TRADE_DT': ('>', '20201031'), 'TRADE_DT': ('<', '20250501')}

# Keys
AShareDescription_keys = ['S_INFO_WINDCODE','S_INFO_NAME','S_INFO_LISTBOARDNAME', 'S_INFO_LISTDATE']
AIndexCSI500Weight_keys = ['TRADE_DT','S_CON_WINDCODE','S_INFO_WINDCODE','WEIGHT']
AIndexCSI1000Weight_keys = ['TRADE_DT','S_CON_WINDCODE','S_INFO_WINDCODE','WEIGHT']
AIndexHS300CloseWeight_keys = ['TRADE_DT','S_CON_WINDCODE','S_INFO_WINDCODE','I_WEIGHT']
AShareEODPrices_keys = ['TRADE_DT','S_INFO_WINDCODE','S_DQ_AMOUNT']
AShareEODDerivativeIndicator_keys = ['TRADE_DT','S_INFO_WINDCODE','S_VAL_MV']
# Load Data
AShareDescription = get_db_data('AShareDescription', keywords=AShareDescription_keys)
AIndexCSI500Weight = get_db_data('AIndexCSI500Weight', keywords=AIndexCSI500Weight_keys, additional_conditions=add_conditions)
print("Done!")
AIndexCSI1000Weight = get_db_data('AIndexCSI1000Weight', keywords=AIndexCSI1000Weight_keys,additional_conditions= add_conditions)
AIndexHS300CloseWeight = get_db_data('AIndexHS300CloseWeight', keywords=AIndexHS300CloseWeight_keys,additional_conditions= add_conditions)
print("Done!")
AShareEODPrices = get_db_data('AShareEODPrices', keywords=AShareEODPrices_keys, additional_conditions=add_conditions)
AShareEODDerivativeIndicator = get_db_data('AShareEODDerivativeIndicator', keywords=AShareEODDerivativeIndicator_keys, additional_conditions=add_conditions)
print("All done!")