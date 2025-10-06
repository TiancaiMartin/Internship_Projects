# -*- coding: utf-8 -*-


import pymysql
import pandas as pd


def db_connect():
    wind_db = {
        'host': '192.168.64.57',
        'port': 3306,
        'user': 'inforesdep01',
        'password': 'tfyfInfo@1602',
        'db': 'wind',
        'charset': 'utf8'
    }
    return pymysql.connect(**wind_db)


def get_db_data(table_name, conn=db_connect(), keywords=None, additional_conditions=None):
    import pymysql
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    if keywords is None:
        keys = '*'
    else:
        keys = ','.join(['`' + key + '`' for key in keywords])
    sql = f"SELECT {keys} FROM `{table_name}`"

    params = []
    conditions = []

    if additional_conditions:
        for field, value in additional_conditions.items():
            if value is None:
                continue
            if isinstance(value, tuple) and len(value) == 2:
                operator, val = value
                if operator in [">", "<", ">=", "<=", "=", "!=", "LIKE"]:
                    conditions.append(f"`{field}` {operator} %s")
                    params.append(val)
                elif operator == "IS NULL":
                    conditions.append(f"`{field}` IS NULL")
                elif operator == "IS NOT NULL":
                    conditions.append(f"`{field}` IS NOT NULL")
                else:
                    raise ValueError(f"Invalid operator: {operator}")
            elif isinstance(value, list):
                if len(value) == 1:
                    conditions.append(f"`{field}` = %s")
                    params.append(value[0])
                elif len(value) == 2:
                    conditions.append(f"`{field}` BETWEEN %s AND %s")
                    params.extend(value)
                else:
                    placeholders = ', '.join(['%s'] * len(value))
                    conditions.append(f"`{field}` IN ({placeholders})")
                    params.extend(value)
            else:
                conditions.append(f"`{field}` = %s")
                params.append(value)

    if conditions:
        sql += ' WHERE ' + ' AND '.join(conditions)

    cursor.execute(sql, tuple(params))
    results = cursor.fetchall()
    df = pd.DataFrame(results)

    cursor.close()

    return df


# def db_connect():
#     import cx_Oracle
#     host = "172.16.6.7"  # wind数据库
#     port = "1521"  # 端口
#     sid = "wind"  # 数据库名称
#     username = 'QUANT'  # 用这个只读账号的SQL必须加表空间的名字,即WINDDATA
#     password = 'QUANT#2016'
#     dsn = cx_Oracle.makedsn(host, port, sid)
#     return cx_Oracle.connect(username, password, dsn)


# def get_db_data(table_name, conn=db_connect(), keywords=None, additional_conditions=None):
#     def split_fun(conditions):
#         """将条件分割成多个子条件，每个子条件最多包含1000个元素。"""
#         split_conditions = [[]]
#         for field, value in conditions.items():
#             if isinstance(value, list) and len(value) > 1000:
#                 chunks = [value[i:i + 1000] for i in range(0, len(value), 1000)]
#                 split_conditions = [current + [(field, chunk)] for current in split_conditions for chunk in chunks]
#             else:
#                 split_conditions = [current + [(field, value)] for current in split_conditions]
#             # print(split_conditions)
#         return split_conditions, [dict(condition) for condition in split_conditions]

#     if additional_conditions != None:
#         _, split_conditions_list = split_fun(additional_conditions)
#         dfs = []
#         for conditions in split_conditions_list:
#             df = sub_get_db_data(table_name, conn, keywords, conditions)
#             dfs.append(df)
#         return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
#     else:
#         df = sub_get_db_data(table_name, conn, keywords)
#         return df


# def sub_get_db_data(table_name, conn=db_connect(), keywords=None, additional_conditions=None):
#     cursor = conn.cursor()

#     if keywords is None:
#         keys = '*'
#     else:
#         keys = ','.join([f'"{key}"' for key in keywords])

#     sql = f"SELECT {keys} FROM {table_name}"
#     params = []

#     conditions = []

#     if additional_conditions:
#         for field, value in additional_conditions.items():
#             if value is None:
#                 continue
#             if isinstance(value, list):
#                 if len(value) == 1:
#                     conditions.append(f'"{field}" = :{len(params) + 1}')
#                     params.append(value[0])
#                 elif len(value) == 2:
#                     conditions.append(f'"{field}" BETWEEN :{len(params) + 1} AND :{len(params) + 2}')
#                     params.extend(value)
#                 else:
#                     placeholders = ', '.join([':' + str(len(params) + i + 1) for i in range(len(value))])
#                     conditions.append(f'"{field}" IN ({placeholders})')
#                     params.extend(value)
#             else:
#                 conditions.append(f'"{field}" = :{len(params) + 1}')
#                 params.append(value)

#     if conditions:
#         sql += ' WHERE ' + ' AND '.join(conditions)

#     cursor.execute(sql, params)
#     results = cursor.fetchall()
#     columns = [col[0] for col in cursor.description]
#     df = pd.DataFrame(results, columns=columns)

#     cursor.close()

#     return df


# # if __name__ == '__main__':
# #     # additional_conditions = {
# #     #     'age': ('>', 25),
# #     #     'salary': ('<=', 5000),
# #     #     'status': ['single', 'married', 'divorced'],
# #     #     'height': [170, 180],
# #     #     'name': ('LIKE', 'John%'),
# #     #     'birthdate': ('IS NOT NULL', None)
# #     # }
# #     add_conditions = {'TRADE_DT': ('>=', '20240717'), 'TRADE_DT': ('<', '20240719')}
# #     keys = ['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_CLOSE']
# #     tmp = get_db_data('ashareeodprices', conn=db_connect(),keywords=keys, additional_conditions=add_conditions)
