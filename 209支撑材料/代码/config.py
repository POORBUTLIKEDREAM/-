# config.py
"""
存储所有全局配置和仿真参数。
"""
import numpy as np

# -------------------- 仿真核心参数 --------------------
# 网格与时间
M, N = 20, 30             # 网格大小 (M列, N行)
T = 90                    # 系统运行时间(s)

# 数据流参数
flow_size_mb = 10.0       # 每条流总数据量(Mb)
flow_rate_req = 5.0       # 每条流申请速率(Mb/s)
flows_per_sensor_times = [0, 30, 60]  # 每传感器产生新流的时刻列表 (s)

# 车辆参数
y_pairs = [(5,6), (15,16), (25,26)]   # 三辆车各自覆盖的两行
num_cars = len(y_pairs)

# 网络容量与时延
B_sensor = 10.0           # 传感器间单链路容量(Mbps)
B_receive = 100.0         # 车辆聚合接收上限(Mbps)
hop_delay_s = 0.05        # 传感器网络每跳时延(s)
car_link_delay_s = 0.05   # 传感器到车的无线链路时延(s)

# 传感器->车 无线链路带宽 b(t) 的参数
period = 10.0             # 带宽变化周期(s)
phi_rng = (0.0, period)   # 随机相位的范围
Bpeak_rng = (30.0, 60.0)  # 峰值带宽的范围(Mbps)

# 拥塞模型参数
cong_th = 0.9             # 触发拥塞丢失的车端聚合利用率阈值
cong_loss_max = 0.08      # 达到100%利用率时的最大额外丢包率

# 算法参数
K_candidates = 6          # 每流每秒最多考察的出口候选数量

# 随机数生成器
rng = np.random.default_rng(20250808)  # 固定种子以保证结果可复现