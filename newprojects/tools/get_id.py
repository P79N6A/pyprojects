import csv
import json

'''
json_file = '/Users/withheart/Documents/stress_million_du/stress_android.json'
txt_file = '/Users/withheart/Documents/stress_million_du/android_did.txt'
with open(json_file, 'r') as json_file:
    ids = []
    lines = json_file.readlines()
    for line in lines:
        dict_lines = json.loads(line)
        id = str(dict_lines['id'])
        ids.append(id)
    ids = '\n'.join(ids)
    with open(txt_file,'w') as txt_file:
        txt_file.write(ids)
'''

map_file = '/Users/withheart/Documents/stress_million_du/android_did.txt'
stress_user = '/Users/withheart/Documents/stress_million_du/stress_user.tsv'
did_uid = '/Users/withheart/Documents/stress_million_du/did-uid.txt'
txt_file = '/Users/withheart/Documents/stress_million_du/android_did.txt'
sess_file = '/Users/withheart/Documents/stress_million_dus/stress_user_session.tsv'
with open(txt_file,'r') as did:
    with open(sess_file,'r') as sess:
        lines = sess.readlines()
        for i in range(len(lines)):
            if i == 0:
                continue
            else:
                data = lines[i].strip("\n").split("\t")

    with open(stress_user,'r') as uid:
        dids = did.readlines()
        uids = uid.readlines()
        maps = []
        for i in range(1,10001):
            did = dids[i].strip("\n")
            uid = uids[i]
            map = did + "\t" + uid
            maps.append(map)
        maps_str = ''.join(maps)
        with open(did_uid,'w') as data:
            data.write(maps_str)
