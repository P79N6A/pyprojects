#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Author: (hexue@bytedance.com)
#########################################################################
# Created Time: 2016-12-21 20:23:09
# File Name: protobufhelper.py
# Description: ProtocolBuffer类
#########################################################################
import sys
import os
import glob2
from google.protobuf.json_format import MessageToJson, Parse
import re
import logging
import importlib

logger = logging.getLogger('main')

CWD_PATH = os.getcwd()    # 获取当前工作目录
PROTOC_DIR = CWD_PATH + '/protoc'   # 按照idl库存放proto源文件
PB_GEN_DIR = CWD_PATH + '/protoc/pb_gen'   # 存放proto生成的py文件，按项目存放
# protoc 命令路径
# if sys.platform == 'darwin':
#     PROTOC_EXEC = 'bin/protoc_mac'
# else:
#     PROTOC_EXEC = 'bin/protoc'
PROTOC_EXEC = "/usr/local/bin/protoc"
USERNAME = 'ci_testing'                   # testing拉取代码的用户名
ACCESS_TOKEN = '-kqWMyVXSuFJ5Cd2UwaQ'     # testing拉取代码access token


regex = r"(.*)\/(.*)\.git"
# 管理proto文件，进行pb库的拉取，更新
def manage_proto_repo(module):
    print "hello"
    # temp = str(module['pb_repo']).split('https://code.byted.org/')[1]
    # match = re.search(regex, temp)
    # if match:
    #     repo_name = '_'.join(match.groups())
    # else:
    #     repo_name = temp.split('.git')[0].replace('/', '_')
    #
    # pb_file = str(module['pb_file'])
    # pb_repo_path = os.path.join(PROTOC_DIR, repo_name)   # pb项目路径
    # pb_import_path = module['pb_import_path']
    #
    # # 如果项目已存在，更新仓库，否则clone仓库
    # try:
    #     if not os.path.exists(pb_repo_path):
    #         git_path = 'https://ci_testing:{access_token}@{git_addr}'.format(access_token=ACCESS_TOKEN, git_addr=module['pb_repo'].split('//')[-1])
    #         git.Repo.clone_from(url=git_path, to_path=pb_repo_path)
    #     else:
    #         repo = git.Repo(pb_repo_path)
    #         repo.git.pull()
    #
    # except git.exc.InvalidGitRepositoryError as ex:
    #     ex_info = 'Repository is invalid, ex: %s' % ex
    #     logging.error(ex_info)
    #     raise Exception(ex_info)
    # except git.GitCommandError as ex:
    #     logger.error('manage_proto_repo error, ex: %s' % ex)
    #     ex_info = "git command error, please make sure 'ci_testing' is the member of repository"
    #     raise Exception(ex_info)
    # except Exception as ex:
    #     raise Exception(ex)
    #
    # return generate_py_file(module, pb_repo_path, pb_file, repo_name, pb_import_path)


# 根据proto文件生成对应的py文件
def generate_py_file(pb_repo_path, pb_file, repo_name, pb_import_path):
    pb_dir = os.path.join(PB_GEN_DIR, repo_name)    # proto对应pb文件目录

    if pb_import_path:
        pb_dir = os.path.join(pb_dir, pb_import_path)

    if not os.path.exists(pb_dir):
        os.makedirs(pb_dir, 0o755)

    # 查找对应的proto文件
    if pb_import_path:
        include_path = os.path.join(pb_repo_path, pb_import_path)
    else:
        include_path = pb_repo_path
    all_files = glob2.glob(os.path.join(include_path, '**', '*.proto'))
    for file_path in all_files:
        proto_gen_command = '{proto_cmd} --proto_path={pb_include_path} --python_out={pb_gen_dir} {pb_file}'.format(
            proto_cmd = PROTOC_EXEC,
            pb_include_path = include_path,
            pb_gen_dir = pb_dir,
            pb_file = file_path)
        try:
            os.system(proto_gen_command)
        except Exception as ex:
            logger.error('protoc cmd error, %s' % ex)
            continue

        pb_relative_file = file_path[len(include_path):].strip('/')
        pb_relative_modules = pb_relative_file.split('/')[:-1]

        cwd = pb_dir
        for rm in pb_relative_modules:
            cwd = os.path.join(cwd, rm)
            os.system("touch " + cwd + "/__init__.py")

    os.system("touch " + pb_dir + "/__init__.py")
    pb_check_file = pb_file.replace('.proto', '_pb2.py')    # 需要校验的proto生成的python文件

    return pb_check_file, pb_dir


# 根据输入的proto 仓库信息，拉取proto文件，反序列化response
def protobuf_response(module, response):
    '''
    获取模块描述信息
    '''
    try:
        # pb_gen_file, pb_gen_path = manage_proto_repo(module)
        repo_name = "person"
        pb_repo_path = os.path.join(PROTOC_DIR, repo_name)
        pb_file = "protoc/person/person_student.proto"
        pb_import_path = "/Users/withheart/Documents/studys/tools/protoc/person"
        pb_gen_file, pb_gen_pat = generate_py_file(pb_repo_path, pb_file, repo_name, pb_import_path)
    except Exception as ex:
        return False, ex
    sys.path.append(pb_gen_file)
    MODULE_NAME = pb_gen_file.replace('.py', '').replace('/', '.')
    MODULE_RESPONSE = module['pb_resp_name']
    name = importlib.import_module(MODULE_NAME)


    try:
        target = getattr(name, MODULE_RESPONSE)()
        target.ParseFromString(response)
        json_resp = MessageToJson(target, including_default_value_fields=True)
    except Exception as ex:
        return False, '反序列化proto出错，%s' % ex
    return True, json_resp


# 将对应的json数据转成message，然后序列化成二进制返回
def json_to_pb_message(module, str_response):
    '''
    获取模块描述信息
    '''
    try:
        pb_gen_file, pb_gen_path = manage_proto_repo(module)
    except Exception as ex:
        return False, '拉取代码仓库错误， %s' % ex

    sys.path.append(pb_gen_path)
    MODULE_NAME = pb_gen_file.replace('.py', '').replace('/', '.')
    MODULE_RESPONSE = module['pb_resp_name']
    name = importlib.import_module(MODULE_NAME)
    target = getattr(name, MODULE_RESPONSE)()

    parsed_pb = Parse(str_response, target, ignore_unknown_fields=False)
    protobuf_str = parsed_pb.SerializeToString()
    return protobuf_str


if __name__ == '__main__':
    module = {
        # 'pb_repo': 'https://code.byted.org/rocket/common.git',
        # 'pb_repo': 'https://code.byted.org/ez/idl.git',
        # 'pb_file': 'client/index.proto',
        # 'pb_import_path': 'proto',
        'pb_resp_name': 'Student',
    }

    flag, json_resp = protobuf_response(module, '')
    print json_resp

    url = 'http://api.openlanguage.com/ez/studentapp/v15/home?iid=41421991999&device_id=41024141561&ac=wifi&channel=local_test&aid=1335&app_name=open_language&version_code=203&version_name=2.0.3&device_platform=android&ssmix=a&device_type=OPPO+R11+Plus&device_brand=OPPO&language=zh&os_api=25&os_version=7.1.1&openudid=d20fc7a44e1423b&manifest_version_code=203&resolution=1080*1920&dpi=480&update_version_code=2030&_rticket=1534747599027'

    import requests
    resp = requests.get(url)
    if resp.status_code == 200:
        flag, json_resp = protobuf_response(module, resp.content)
        # print json_resp
        # assert 'err_no' in json_resp, 'errno not in json'
        # print json_resp
        f = open('text.json', 'w')
        f.write(json_resp)
        f.close()

        binary_data = json_to_pb_message(module, json_resp)
        bf = open('test_pb', 'wb')
        bf.write(binary_data)
        bf.close()


