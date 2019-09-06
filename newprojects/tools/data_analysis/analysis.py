import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
import requests
import json

dataInfoUrl = "http://10.2.206.30:8001/perf/data/task/all?start=%s&end=%s"


def data_info():
    # plt.figure(1)  # 创建图表1
    # plt.figure(2)  # 创建图表2
    # ax1 = plt.subplot(211)  # 在图表2中创建子图1
    # ax2 = plt.subplot(212)  # 在图表2中创建子图2
    #
    # x = np.linspace(0, 3, 100)
    # for i in range(5):
    #     plt.figure(1)  # ❶ # 选择图表1
    #     plt.plot(x, np.exp(i * x / 3))
    #     plt.sca(ax1)  # ❷ # 选择图表2的子图1
    #     plt.plot(x, np.sin(i * x))
    #     plt.sca(ax2)  # 选择图表2的子图2
    #     plt.plot(x, np.cos(i * x))
    #
    # plt.show()

    resp = requests.get(dataInfoUrl % ("2019-05-17 15:04:05", "2019-05-24 15:04:05"))
    dict = json.loads(resp.content)["data"]

    labels = dict["labels"]

    for i,value in enumerate(dict["datas"]):
        data = dict["datas"][i]["data"]
        fig = plt.figure()
        colors = ['orange','yellowgreen', 'lightskyblue']  # 每块颜色定义
        plt.pie(data, labels=labels, autopct='%1.2f%%',colors=colors,)  # 画饼图（数据，数据对应的标签，百分数保留两位小数点）
        plt.title(dict["datas"][i]["title"])
        plt.show()


if __name__ == '__main__':
    data_info()
