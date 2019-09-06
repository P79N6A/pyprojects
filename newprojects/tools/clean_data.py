import pymysql



#api表内的task_id进行修改
#  旧数据库
conn_old = pymysql.connect(host='10.6.19.35', user='performance_w', passwd='qEE18Sq5wYRNk4Ja_119Jmj859UfNwSB', db='performance', port=3307, charset='utf8')
#  新数据库
conn_new = pymysql.connect(host='10.3.20.205', user='performance_w', passwd='Kfyy0vzCLUveGGV_E21w6HmzSyCsdA9C',
                           db='performance', port=4018, charset='utf8')
# 1. 灌task数据
cur_old_1 = conn_old.cursor()
cur_old_1.execute('SELECT  * FROM task')
tasks = cur_old_1.fetchall()
# values_list = []
# for task in tasks:
#     temp = task[2], task[3], task[6], task[7], task[8], task[9], task[10], task[11], task[12]
#     values_list.append((temp))
#
# cur_new_1 = conn_new.cursor()
# cur_new_1.executemany('''insert into task(task_name, hosts,  status, url, deleted, deleted_time, deleted_author, create_time, update_time)
#                       values(%s, %s, %s, %s, %s, %s, %s, %s, %s)''', values_list)
# cur_new_1.commit()

# 保存旧数据库中task_id的数据和新表中id的映射
cur_new_2 = conn_new.cursor()
cur_new_2.execute('SELECT  * FROM task')
tasks_new = cur_new_2.fetchall()
id_maps = dict()
for i in range(len(tasks)):
    id_maps[tasks[i][1]] = tasks_new[i][0]

# 保存旧数据库中api_id的数据和新表中id的映射
cur_old_2 = conn_old.cursor()
cur_old_2.execute('SELECT  * FROM api,param where api.api_id = param.api_id')
api_param_datas = cur_old_2.fetchall()
old_api_index = []
for api_param_data in api_param_datas:
    if (api_param_data[2] in id_maps.keys()):
        old_api_index.append(api_param_data[1])

cur_new_2 = conn_new.cursor()
cur_new_2.execute('SELECT  * FROM api')
api_index_new = cur_new_2.fetchall()
api_id_maps = dict()
for i in range(len(api_index_new)):
    api_id_maps[old_api_index[i]] = api_index_new[i][0]


#2. 灌api数据
# cur_old_2 = conn_old.cursor()
# cur_old_2.execute('SELECT  * FROM api,param where api.api_id = param.api_id')
# api_param_datas = cur_old_2.fetchall()
# api_list = []
# param_list = []
# for api_param_data in api_param_datas:
#     if not (api_param_data[2] in id_maps.keys()):
#         continue
#     checks = []
#     random_infos = []
#     body_parameter = {}
#     header = {}
#     raw = ""
#     if api_param_data[9]:
#         data = api_param_data[9].split("#check#")
#         check = {}
#         check["key"] = data[0]
#         if data[0] == "Satuts":
#             check["in"] = "header"
#         else:
#             check["in"] = "body"
#         check["type"] = "random"
#         checks.append(check)
#
#     if api_param_data[16]:
#         datas = []
#         if "#parameter#" in api_param_data[16]:
#             datas = api_param_data[16].split("#parameter#")
#         else:
#             datas.append(api_param_data[16])
#         for data in datas:
#             header_list = data.split("#header#")
#             header[header_list[0]] = header_list[1]
#
#     if api_param_data[17]:
#         if '{' in api_param_data[17]:
#             raw = api_param_data[17]
#         else:
#             datas = []
#             if "#parameter#" in api_param_data[17]:
#                 datas = api_param_data[17].split("#parameter#")
#             else:
#                 datas.append(api_param_data[17])
#             for data in datas:
#                 body_list = data.split("#body#")
#                 body_parameter[body_list[0]] = body_list[1]
#
#     if api_param_data[18] and api_param_data[18] != 'null':
#         datas = []
#         if "#parameter#" in api_param_data[18]:
#             datas = api_param_data[18].split("#parameter#")
#         else:
#             datas.append(api_param_data[18])
#         for data in datas:
#             if "#stragy#" not in data:
#                 continue
#             pos = data.split("#randomer#")[0]
#             key = data.split("#stragy#")[0].split("#randomer#")[1]
#             random_info = {}
#             if pos == "hdparam":
#                 random_info["in"] = "header"
#             elif pos == "bdparam":
#                 random_info["in"] = "body_parameter"
#             elif pos == "ddparam":
#                 random_info["in"] = "domain"
#             elif pos == "pdparam":
#                 random_info["in"] = "port"
#             elif pos == "cdparam":
#                 random_info["in"] = "cookie"
#             random_info["key"] = key
#             random_info["type"] = "random"
#             random_infos.append(random_info)
#
#
# #    api = id_maps[api_param_data[2]], api_param_data[4],api_param_data[5],api_param_data[6],api_param_data[7],\
# #         api_param_data[8],api_param_data[12],api_param_data[13],str(header),str(body_parameter),'NULL',str(check),api_param_data[3],\
# #          str(raw),str(random_infos)
#     param = api_param_data[26],api_param_data[23],api_param_data[24],api_param_data[25],id_maps[api_param_data[2]]
# #    api_list.append((api))
#     param_list.append((param))
# cur_new_2 = conn_new.cursor()
#
# #cur_new_2.executemany('''insert into api(task_id, domain,  api, port, method, scene_type, create_time, update_time, header,body_parameter,contexts,checks,protocol,raw,random_info)
#  #                     values(%s, %s, %s, %s, %s, %s, %s, %s, %s,%s, %s, %s, %s, %s, %s)''', api_list)
# print(param_list[0])
# cur_new_3 = conn_new.cursor()
# cur_new_3.executemany('''insert into param(file_name, file_path,  create_time, update_time, task_id)
#                           values(%s, %s, %s, %s, %s)''', param_list)
# conn_new.commit()


# 灌history
# cur_old_3 = conn_old.cursor()
# cur_old_3.execute('SELECT  * FROM history')
# history_datas = cur_old_3.fetchall()
# history_data_new = []
# for data in history_datas:
#     if not (data[1] in id_maps.keys()):
#         continue
#     data_new = id_maps[data[1]],data[2],data[3],data[4],data[5],data[6],data[7],data[8],data[9],data[10],data[11],data[12],data[13],data[14],data[15],data[16],data[17],data[18],data[19]
#     history_data_new.append((data_new))
# cur_new_3 = conn_new.cursor()
# cur_new_3.executemany('''insert into history(task_id, task_name,  resc_qps, resc_total, resc_ok, resc_ko, resc_mean, resc_max, resc_min,pct_99,pct_95,pct_75,pct_50,resc_std,remain_1,remain_2,remain_3,create_time,update_time)
#                      values(%s, %s, %s, %s, %s, %s, %s, %s, %s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''', history_data_new)
# conn_new.commit()


# 灌result
# cur_old_4 = conn_old.cursor()
# cur_old_4.execute('SELECT  * FROM result')
# result_datas = cur_old_4.fetchall()
# result_data_new = []
# for data in result_datas:
#     if not (data[1] in id_maps.keys()):
#         continue
#     data_new = id_maps[data[1]],data[2],data[3],data[4],data[5],data[6]
#     result_data_new.append((data_new))
# cur_new_3 = conn_new.cursor()
# cur_new_3.executemany('''insert into result(task_id, file_name,  file_path, status, create_time, update_time)
#                      values(%s, %s, %s, %s, %s, %s)''', result_data_new)
# conn_new.commit()

# 灌request_count
# cur_old_5 = conn_old.cursor()
# cur_old_5.execute('SELECT  * FROM request_count')
# request_count_datas = cur_old_5.fetchall()
# request_count_data_new = []
# api_id_maps[''] = ''
# for data in request_count_datas:
#     if not (data[1] in id_maps.keys() and data[3] in api_id_maps.keys()):
#         continue
#     data_new = id_maps[data[1]],data[2],api_id_maps[data[3]],data[4],data[5],data[6]
#     request_count_data_new.append((data_new))
# cur_new_3 = conn_new.cursor()
# cur_new_3.executemany('''insert into request_count(task_id, task_name,  api_id, counts, create_time, update_time)
#                      values(%s, %s, %s, %s, %s, %s)''', request_count_data_new)
# conn_new.commit()

# 灌product
cur_old_5 = conn_old.cursor()
cur_old_5.execute('SELECT  * FROM product')
product_datas = cur_old_5.fetchall()
product_data_new = []
for data in product_datas:
    if not (data[2] in id_maps.keys()):
        continue
    data_new = data[1],id_maps[data[2]],data[3],data[4],data[5]
    product_data_new.append((data_new))
cur_new_5 = conn_new.cursor()
cur_new_5.executemany('''insert into product(product_name, task_id,  owner, create_time, update_time)
                     values(%s, %s, %s, %s, %s)''', product_data_new)
conn_new.commit()

conn_old.close()
conn_new.close()
