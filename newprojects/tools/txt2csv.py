#-*-coding:utf-8 -*-
import csv
import random
import json

txt_file = "/Users/withheart/Desktop/32-did-odin.token"
csv_file = "/Users/withheart/Desktop/did_odin_tt.csv"
stress_did = '/Users/withheart/Documents/stress_million_du/stress_user.tsv'
with open(csv_file, 'w') as csvfile:
    spam_writer = csv.writer(csvfile, dialect='excel')
    spam_writer.writerow(["odin_uid","did","odin_tt"])
    with open(txt_file) as filein:
        lines = filein.readlines()
        for i in range(1, len(lines)):
            datas = lines[i].strip("\n").split(" ")
            line = []
            line.append(datas[0])
            line.append(datas[1])
            line.append(datas[2])
            # line.append(lines[i].strip("\n"))
            spam_writer.writerow(line)


# ios_did = "/Users/withheart/Documents/stress_million_du/did-uid-csv.csv"
# uid = "/Users/withheart/Documents/stress_million_du/did-uid-csv.csv"
# ios_did_uid = "/Users/withheart/Documents/stress_million_du/did-uid-csv.csv"
# with open(ios_did_uid) as csv_file:
#     spam_writer = csv.writer(csv_file,dialect='excel')
#     spam_writer.writerow(['did','row'])
#     dids = []
#     uids = []
#     with open(ios_did,'r') as ios_did:
#         lines = ios_did.readlines()
#         for line in lines:
#             json_line = json.loads(line)
#             dids.append(str(json_line['id']))
#     with open(uid,'r') as uid:
#         lines = uid.readlines()
#         for i in range(len(lines)):
#             if i == 0:
#                 continue
#             else:
#                 uids.append(line)
#     uids_len = len(uids)
#     uid_dids = []
#     for i in range(uid_dids):
#         uid_did = []
#         uid_did.append(uids[i])
#         uid_did.append(dids[i])
#         uid_dids.append(uid_did)
#     spam_writer.writerows(uid_dids)
