
class qps_count:
    def get_cur_qps(self,online_cpu,qps1,cpu1,qps2,cpu2):
        online1 = (qps1 * online_cpu) / (cpu1 - online_cpu)
        online2 = (qps2 * online_cpu) / (cpu2 - online_cpu)
        print("value1:")
        print(online1)
        print("value2:")
        print(online2)
        cur_qps = (online1 + online2) / 2
        return cur_qps

    def get_cpu_50_qps(self,cur_qps, qps2, cpu2):
        qps = (50 * (cur_qps + qps2)) /cpu2
        return qps

    def get_cluster_qps(self, cpu_50_qps, instances):
        return cpu_50_qps * instances

qps = qps_count()
cur_qps = qps.get_cur_qps(7.30,50,17.10,100,26)
pre_qps = qps.get_cpu_50_qps(cur_qps, 100,26)
print(pre_qps)
cluser_qps = qps.get_cluster_qps(pre_qps, 1040)
print(cluser_qps)