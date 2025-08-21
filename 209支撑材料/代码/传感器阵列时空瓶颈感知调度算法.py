import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass

# -------------------- 参数区（与题意一致） --------------------
M, N = 20, 30  # 网格大小
T = 90  # 系统运行时间(s)

flow_size_mb = 10.0  # 每条流总数据量(Mb)
flow_rate_req = 5.0  # 每条流申请速率(Mb/s)
flows_per_sensor_times = [0, 30, 60]  # 每传感器产生流的时刻

y_pairs = [(5, 6), (15, 16), (25, 26)]  # 三辆车覆盖的两行
num_cars = len(y_pairs)

B_sensor = 10.0  # 传感器-传感器单链路容量(Mbps)
hop_delay_s = 0.05  # 每跳时延(s)
car_link_delay_s = 0.05  # 传感器-车时延(s)
B_receive = 100.0  # 车辆聚合接收上限(Mbps)

# b(t)：周期/峰值/相位（题面给周期10s、线性变化；这里用三角波近似线性往返）
period = 10.0
phi_rng = (0.0, period)
Bpeak_rng = (30.0, 60.0)  # 每个传感器到车的峰值带宽范围（可按题面给定/数据集替换）

# 拥塞损失（模拟接入口不稳定）：当车端当秒利用率>阈值，线性增至最大丢失
cong_th = 0.9
cong_loss_max = 0.08  # 最多8%额外丢失

K_candidates = 6  # 每流每秒最多考察的出口候选（从"所有车×6覆盖点"中就近取K）

rng = np.random.default_rng(20250808)  # 固定种子可复现


# -------------------- 工具函数 --------------------
def triangular_wave_unit(u: float) -> float:
    """单位三角波：周期1，峰值1。"""
    v = u % 1.0
    return 2 * v if v <= 0.5 else 2 * (1.0 - v)


def b_link(t, Bpeak, phi):
    return Bpeak * triangular_wave_unit((t + phi) / period)


def car_center_x(t):
    start_c, end_c = 2.0, M - 1.0
    v = (end_c - start_c) / 100.0  # 列/秒
    cc = start_c + v * t
    return max(2.0, min(end_c, cc))


def covered_window(cc):
    """由中心列cc计算覆盖列集合（3列，整数坐标，裁剪到[1..M]）。"""
    c = int(math.floor(cc))
    cols = [c - 1, c, c + 1]
    cols = [min(M, max(1, x)) for x in cols]
    return sorted(set(cols))


def manhattan_path(x1, y1, x2, y2):
    links = []
    x, y = x1, y1
    # 水平
    step = 1 if x2 >= x else -1
    while x != x2:
        if step == 1:
            links.append(('H', x, y))  # (x,y)->(x+1,y)
            x += 1
        else:
            links.append(('H', x - 1, y))  # (x-1,y)->(x,y)
            x -= 1
    # 垂直
    step = 1 if y2 >= y else -1
    while y != y2:
        if step == 1:
            links.append(('V', x, y))  # (x,y)->(x,y+1)
            y += 1
        else:
            links.append(('V', x, y - 1))  # (x,y-1)->(x,y)
            y -= 1
    return links


def generate_path_nodes(x1, y1, x2, y2):
    """
    生成从(x1, y1)到(x2, y2)的先横后纵曼哈顿路径上的节点坐标列表。
    """
    nodes = []
    x, y = x1, y1
    # 水平移动
    step_x = 1 if x2 > x else -1 if x2 < x else 0
    while x != x2:
        nodes.append((x, y))
        x += step_x
    # 垂直移动
    step_y = 1 if y2 > y else -1 if y2 < y else 0
    while y != y2:
        nodes.append((x, y))
        y += step_y
    # 添加终点
    nodes.append((x2, y2))
    return nodes


# -------------------- 容量表/覆盖窗口 --------------------
H_shape = (M - 1, N)  # 水平链路矩阵大小
V_shape = (M, N - 1)  # 垂直链路矩阵大小


def cap_tables_init():
    # 每秒的链路剩余容量（初始化为满额）
    cap_H = np.full((T, H_shape[0], H_shape[1]), B_sensor, dtype=float)
    cap_V = np.full((T, V_shape[0], V_shape[1]), B_sensor, dtype=float)
    # 每秒每车聚合剩余
    cap_car = np.full((T, num_cars), B_receive, dtype=float)
    return cap_H, cap_V, cap_car


def cars_cover_map_for_t(t):
    """返回每辆车在时刻t覆盖的网格点坐标列表（长度最多6）。"""
    cover = []
    cc = car_center_x(t)
    cols = covered_window(cc)
    for (_, (y1, y2)) in enumerate(y_pairs):
        win = [(x, y1) for x in cols] + [(x, y2) for x in cols]
        cover.append(win)
    return cover  # list[list[(x,y)]]


# -------------------- 随机化 b(t) 参数（每传感器独立峰值/相位） --------------------
Bpeak = rng.uniform(Bpeak_rng[0], Bpeak_rng[1], size=(M + 1, N + 1))  # 1-indexed
Phi = rng.uniform(phi_rng[0], phi_rng[1], size=(M + 1, N + 1))

# -------------------- 流的生成（每传感器在 0/30/60 分别产生一条） --------------------
flows_by_time = defaultdict(list)
flow_id = 0
for x in range(1, M + 1):
    for y in range(1, N + 1):
        for t0 in flows_per_sensor_times:
            flows_by_time[t0].append((flow_id, x, y))
            flow_id += 1
F = flow_id  # 总流数，= 20*30*3 = 1800


@dataclass
class FlowState:
    fid: int
    x: int
    y: int
    t_start: int
    remaining: float  # Mb
    done: bool = False
    last_arrival_time: float = None
    bytes_scheduled: float = 0.0


# 到达事件：记录 (arrival_time, car_id, amount_Mb, flow_id)
arrival_queue = []

# 车端统计
car_recv_ts = np.zeros((T, num_cars))  # 实际接收(Mb/s)
car_sched_ts = np.zeros((T, num_cars))  # 调度到车(Mb/s)

# 链路累计利用，用于热力图
H_used = np.zeros(H_shape)
V_used = np.zeros(V_shape)

# 初始化容量与状态容器
cap_H, cap_V, cap_car = cap_tables_init()
active_flows = {}
finished_flows = {}

# 初始化调度方案日志
schedule_log = []

# -------------------- 主仿真循环（以秒为粒度） --------------------
# 存储热力图的列表
heatmap_figures = []

for t in range(T):
    # 1) 激活新流
    for (fid, x, y) in flows_by_time.get(t, []):
        active_flows[fid] = FlowState(fid=fid, x=x, y=y, t_start=t, remaining=flow_size_mb)

    # 2) 车辆覆盖候选
    cover = cars_cover_map_for_t(t)
    candidates = []
    for cid, win in enumerate(cover):
        for (xx, yy) in win:
            candidates.append((cid, xx, yy))

    # 3) 遍历活跃流，逐一分配
    for fid, st in list(active_flows.items()):
        if st.done:
            continue
        req = min(flow_rate_req, st.remaining)
        if req <= 1e-9:
            st.done = True
            finished_flows[fid] = st
            del active_flows[fid]
            continue

        # 3.1 选取K个最近出口（按L1距离）
        dists = []
        for (cid, xx, yy) in candidates:
            d = abs(xx - st.x) + abs(yy - st.y)
            dists.append((d, cid, xx, yy))
        dists.sort(key=lambda z: (z[0], z[1]))
        picked = dists[:K_candidates]

        # 3.2 在候选中选择"可获得吞吐最大"的路径
        best_rate = 0.0
        best_links = None
        best_cid = None
        best_arrival = None
        best_ingress_node = None

        for (_, cid, xx, yy) in picked:
            links = manhattan_path(st.x, st.y, xx, yy)

            # 路径本秒瓶颈
            bottleneck = req
            ok = True
            for typ, a, b in links:
                if typ == 'H':
                    if not (1 <= a <= M - 1 and 1 <= b <= N):
                        ok = False;
                        break
                    rcap = cap_H[t, a - 1, b - 1]
                else:
                    if not (1 <= a <= M and 1 <= b <= N - 1):
                        ok = False;
                        break
                    rcap = cap_V[t, a - 1, b - 1]
                bottleneck = min(bottleneck, rcap)
                if bottleneck <= 1e-9:
                    ok = False;
                    break
            if not ok:
                continue

            # 车链路瓶颈（瞬时 b(t) 与 车聚合剩余）
            bt = b_link(t, Bpeak[xx, yy], Phi[xx, yy])
            car_left = cap_car[t, cid]
            car_bottle = min(bt, car_left)

            feasible = min(bottleneck, car_bottle, req)
            if feasible > best_rate + 1e-9:
                hops = len(links)
                last_arrival = (t + 1.0) + hops * hop_delay_s + car_link_delay_s
                best_rate = feasible
                best_links = links
                best_cid = cid
                best_arrival = last_arrival
                best_ingress_node = (xx, yy)
            elif abs(feasible - best_rate) <= 1e-9 and best_links is not None:
                # 平局时选更短路径
                if len(links) < len(best_links):
                    hops = len(links)
                    last_arrival = (t + 1.0) + hops * hop_delay_s + car_link_delay_s
                    best_rate = feasible
                    best_links = links
                    best_cid = cid
                    best_arrival = last_arrival
                    best_ingress_node = (xx, yy)

        if best_links is None or best_rate <= 1e-9:
            continue  # 本秒无法分配

        # 记录调度决策到日志
        path_nodes = generate_path_nodes(st.x, st.y, best_ingress_node[0], best_ingress_node[1])
        schedule_log.append({
            "k_time": t,
            "flow_id": fid,
            "source_sensor": f"({st.x}, {st.y})",
            "path_to_car": f"{path_nodes} -> Car-{best_cid}",
            "rate_mbps": best_rate
        })

        r = best_rate  # Mb/s，本秒持续1s -> Mb
        # 扣减链路容量 + 记录利用
        for typ, a, b in best_links:
            if typ == 'H':
                cap_H[t, a - 1, b - 1] = max(0.0, cap_H[t, a - 1, b - 1] - r)
                H_used[a - 1, b - 1] += r
            else:
                cap_V[t, a - 1, b - 1] = max(0.0, cap_V[t, a - 1, b - 1] - r)
                V_used[a - 1, b - 1] += r

        # 扣减车端聚合并计算拥塞丢失
        car_sched_ts[t, best_cid] += r
        cap_car[t, best_cid] = max(0.0, cap_car[t, best_cid] - r)

        util = car_sched_ts[t, best_cid] / B_receive if B_receive > 0 else 0.0
        loss_ratio = 0.0
        if util > cong_th:
            loss_ratio = min(cong_loss_max, (util - cong_th) / (1.0 - cong_th) * cong_loss_max)

        received = r * (1.0 - loss_ratio)
        car_recv_ts[t, best_cid] += received
        arrival_queue.append((best_arrival, best_cid, received, fid))

        # 更新流状态
        st.remaining -= r
        st.bytes_scheduled += r
        st.last_arrival_time = max(st.last_arrival_time or 0.0, best_arrival)
        if st.remaining <= 1e-9:
            st.done = True
            finished_flows[fid] = st
            del active_flows[fid]

    # 每15秒生成热力图
    if t % 15 == 0 or t == T - 1:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(f"Network Link Utilization at Time t = {t}s", fontsize=20)

        # 水平链路热力图
        ax1 = axes[0]
        # 计算使用率百分比
        H_used_percent = np.zeros_like(H_used)
        for i in range(H_shape[0]):
            for j in range(H_shape[1]):
                total_possible = B_sensor * (t + 1)  # 到当前时间为止的总容量
                if total_possible > 0:
                    H_used_percent[i, j] = min(100, H_used[i, j] / total_possible * 100)

        im1 = ax1.imshow(H_used_percent.T, aspect='auto', origin='lower',
                         cmap='RdYlGn_r', vmin=0, vmax=100)
        ax1.set_title(f"Horizontal Links Utilization (t={t}s)", fontsize=16)
        ax1.set_xlabel("X-coordinate (Link Start)", fontsize=12)
        ax1.set_ylabel("Y-coordinate", fontsize=12)
        plt.colorbar(im1, ax=ax1, label='Utilization (%)')

        # 添加网格线
        ax1.set_xticks(np.arange(-0.5, H_shape[0] - 0.5, 1), minor=True)
        ax1.set_yticks(np.arange(-0.5, H_shape[1] - 0.5, 1), minor=True)
        ax1.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
        ax1.set_xticks(np.arange(0, H_shape[0], 5))
        ax1.set_xticklabels(np.arange(1, H_shape[0] + 1, 5))
        ax1.set_yticks(np.arange(0, H_shape[1], 5))
        ax1.set_yticklabels(np.arange(1, H_shape[1] + 1, 5))

        # 垂直链路热力图
        ax2 = axes[1]
        # 计算使用率百分比
        V_used_percent = np.zeros_like(V_used)
        for i in range(V_shape[0]):
            for j in range(V_shape[1]):
                total_possible = B_sensor * (t + 1)
                if total_possible > 0:
                    V_used_percent[i, j] = min(100, V_used[i, j] / total_possible * 100)

        im2 = ax2.imshow(V_used_percent.T, aspect='auto', origin='lower',
                         cmap='RdYlGn_r', vmin=0, vmax=100)
        ax2.set_title(f"Vertical Links Utilization (t={t}s)", fontsize=16)
        ax2.set_xlabel("X-coordinate", fontsize=12)
        ax2.set_ylabel("Y-coordinate (Link Start)", fontsize=12)
        plt.colorbar(im2, ax=ax2, label='Utilization (%)')

        # 添加网格线
        ax2.set_xticks(np.arange(-0.5, V_shape[0] - 0.5, 1), minor=True)
        ax2.set_yticks(np.arange(-0.5, V_shape[1] - 0.5, 1), minor=True)
        ax2.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
        ax2.set_xticks(np.arange(0, V_shape[0], 5))
        ax2.set_xticklabels(np.arange(1, V_shape[0] + 1, 5))
        ax2.set_yticks(np.arange(0, V_shape[1], 5))
        ax2.set_yticklabels(np.arange(1, V_shape[1] + 1, 5))

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        # 保存热力图
        filename = f"link_utilization_heatmap_t_{t}.png"
        plt.savefig(filename, dpi=150)
        heatmap_figures.append((t, filename))
        plt.close(fig)
        print(f"[热力图] 已保存: {filename}")

# -------------------- 统计与输出 --------------------
# 每流实际接收量
recv_per_flow = defaultdict(float)
for (ta, cid, amt, fid) in arrival_queue:
    recv_per_flow[fid] += amt

Q_total = F * flow_size_mb
Q_recv = sum(recv_per_flow.values())
PLR = 1.0 - (Q_recv / Q_total) if Q_total > 0 else 0.0

delays = []
for fid, st in finished_flows.items():
    if st.last_arrival_time is not None:
        delays.append(st.last_arrival_time - st.t_start)

# 为未完成的流（如果有）估算延迟
if len(delays) < F:
    all_fids_in_sim = set(finished_flows.keys()) | set(active_flows.keys())
    for fid in range(F):
        if fid not in all_fids_in_sim:
            # 推断 t_start
            t_start = 0 if fid < M * N else (30 if fid < 2 * M * N else 60)
            arrs = [ta for (ta, _, _, ff) in arrival_queue if ff == fid]
            if arrs:
                delays.append(max(arrs) - t_start)

avg_delay = float(np.mean(delays)) if delays else float('nan')

print("-------------------- 核心指标 --------------------")
print(f"总流数 F = {F}")
print(f"总应发数据量 Q_total = {Q_total:.2f} Mb")
print(f"车辆接收总量 Σqi = {Q_recv:.2f} Mb")
print(f"丢包率 PLR = {PLR:.6f}")
print(f"传输平均时延 (s) = {avg_delay:.6f}")
print("\n")

# 导出 per-flow 结果
rows = []
for fid in range(F):
    t_start = 0 if fid < M * N else (30 if fid < 2 * M * N else 60)
    recv = recv_per_flow.get(fid, 0.0)
    rows.append({"flow_id": fid, "t_start": t_start, "recv_Mb": recv})
df_flows = pd.DataFrame(rows)
out_csv = "per_flow_reception.csv"
df_flows.to_csv(out_csv, index=False)
print(f"[CSV] 每流接收量已保存：{out_csv}")

# 导出调度方案
df_schedule = pd.DataFrame(schedule_log)
schedule_csv = "scheduling_plan.csv"
df_schedule.to_csv(schedule_csv, index=False)
print(f"[CSV] 详细调度方案已保存：{schedule_csv}")
print("     调度方案文件包含了每个时刻(k_time)对每条流(flow_id)的调度决策：")
print("     - source_sensor: 数据流起源的传感器坐标")
print("     - path_to_car: 数据流在传感器网络中的转发路径，直至接入的车辆")
print("     - rate_mbps: 为该流在该秒分配的传输速率")

# -------------------- 可视化 --------------------
plt.style.use('seaborn-v0_8-whitegrid')

# 每车每秒接收量
plt.figure(figsize=(12, 5))
for cid in range(num_cars):
    plt.plot(range(T), car_recv_ts[:, cid], label=f"Car-{cid + 1}", linewidth=2)
plt.title("Per-second Actual Reception per Car", fontsize=16)
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("Received Rate (Mbps)", fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("car_reception_rates.png", dpi=150)
plt.show()

# 水平链路利用热图
plt.figure(figsize=(12, 5))
H_used_total_percent = (H_used / (B_sensor * T)) * 100
im_h = plt.imshow(H_used_total_percent.T, aspect='auto', origin='lower', cmap='RdYlGn_r',
                  vmin=0, vmax=100)
plt.title("Total Utilization of Horizontal Links (Entire Simulation)", fontsize=16)
plt.xlabel("X-coordinate of Link Start (i.e., sensor (x,y))", fontsize=12)
plt.ylabel("Y-coordinate of Link (i.e., sensor (x,y))", fontsize=12)
plt.colorbar(im_h, label='Utilization (%)')
plt.tight_layout()
plt.savefig("horizontal_links_utilization.png", dpi=150)
plt.show()

# 垂直链路利用热图
plt.figure(figsize=(12, 5))
V_used_total_percent = (V_used / (B_sensor * T)) * 100
im_v = plt.imshow(V_used_total_percent.T, aspect='auto', origin='lower', cmap='RdYlGn_r',
                  vmin=0, vmax=100)
plt.title("Total Utilization of Vertical Links (Entire Simulation)", fontsize=16)
plt.xlabel("X-coordinate of Link (i.e., sensor (x,y))", fontsize=12)
plt.ylabel("Y-coordinate of Link Start (i.e., sensor (x,y))", fontsize=12)
plt.colorbar(im_v, label='Utilization (%)')
plt.tight_layout()
plt.savefig("vertical_links_utilization.png", dpi=150)
plt.show()

# 输出热力图文件列表
print("\n-------------------- 生成的热力图文件 --------------------")
for t, filename in heatmap_figures:
    print(f"时间 t={t}s: {filename}")