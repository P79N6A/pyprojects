import pymysql

#  三期迭代数据库迁移：先把目前线上的数据迁到测试库，然后用脚本处理数据，处理后再进行线上数据验证
#  有变更的表
#    task、api、
#  线上数据库
conn_old = pymysql.connect(host='10.3.20.205', user='performance_w', passwd='Kfyy0vzCLUveGGV_E21w6HmzSyCsdA9C', db='performance', port=4018, charset='utf8')
#  测试数据库
conn_new = pymysql.connect(host='10.224.4.27', user='root', passwd='123', db='performance', port=4018, charset='utf8')

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
