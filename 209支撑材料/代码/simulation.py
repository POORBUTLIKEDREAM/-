# simulation.py
"""
定义数据结构，并包含仿真的初始化函数和核心执行循环。
"""
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import config as cfg
import utils


# --- 数据结构定义 ---
@dataclass
class FlowState:
    """用于跟踪单条数据流状态的数据类。"""
    fid: int
    x: int
    y: int
    t_start: int
    remaining: float = cfg.flow_size_mb
    done: bool = False
    last_arrival_time: float = None
    bytes_scheduled: float = 0.0


# --- 初始化函数 ---
def initialize_capacities():
    """初始化所有链路和车辆的每秒剩余容量表。"""
    H_shape = (cfg.M - 1, cfg.N)
    V_shape = (cfg.M, cfg.N - 1)
    cap_H = np.full((cfg.T, H_shape[0], H_shape[1]), cfg.B_sensor, dtype=float)
    cap_V = np.full((cfg.T, V_shape[0], V_shape[1]), cfg.B_sensor, dtype=float)
    cap_car = np.full((cfg.T, cfg.num_cars), cfg.B_receive, dtype=float)
    return cap_H, cap_V, cap_car


def initialize_wireless_params():
    """为每个传感器随机生成b(t)的峰值带宽和相位。"""
    Bpeak = cfg.rng.uniform(cfg.Bpeak_rng[0], cfg.Bpeak_rng[1], size=(cfg.M + 1, cfg.N + 1))
    Phi = cfg.rng.uniform(cfg.phi_rng[0], cfg.phi_rng[1], size=(cfg.M + 1, cfg.N + 1))
    return Bpeak, Phi


def initialize_flows():
    """根据预设时刻，为每个传感器生成数据流，并返回总流数。"""
    flows_by_time = defaultdict(list)
    flow_id = 0
    for x in range(1, cfg.M + 1):
        for y in range(1, cfg.N + 1):
            for t0 in cfg.flows_per_sensor_times:
                flows_by_time[t0].append((flow_id, x, y))
                flow_id += 1
    total_flows = flow_id
    return flows_by_time, total_flows


# --- 主仿真循环 ---
def run_simulation(cap_H, cap_V, cap_car, Bpeak, Phi, flows_by_time):
    """执行完整的逐秒调度仿真。"""
    active_flows, finished_flows = {}, {}
    arrival_queue, schedule_log = [], []
    car_recv_ts = np.zeros((cfg.T, cfg.num_cars))
    car_sched_ts = np.zeros((cfg.T, cfg.num_cars))
    H_used = np.zeros((cfg.M - 1, cfg.N))
    V_used = np.zeros((cfg.M, cfg.N - 1))

    for t in range(cfg.T):
        for (fid, x, y) in flows_by_time.get(t, []):
            active_flows[fid] = FlowState(fid=fid, x=x, y=y, t_start=t)

        cc = utils.car_center_x(t)
        cols = utils.covered_window(cc)
        all_candidates = []
        for cid, (y1, y2) in enumerate(cfg.y_pairs):
            win = [(x, y1) for x in cols] + [(x, y2) for x in cols]
            all_candidates.extend([(cid, xx, yy) for (xx, yy) in win])

        for fid, st in list(active_flows.items()):
            if st.done: continue
            req = min(cfg.flow_rate_req, st.remaining)
            if req <= 1e-9:
                st.done = True
                finished_flows[fid] = st
                del active_flows[fid]
                continue

            dists = [(abs(xx - st.x) + abs(yy - st.y), cid, xx, yy) for (cid, xx, yy) in all_candidates]
            dists.sort(key=lambda z: (z[0], z[1]))
            picked_candidates = dists[:cfg.K_candidates]

            best_rate, best_links, best_cid, best_arrival, best_ingress_node = 0.0, None, None, None, None

            for _, cid, xx, yy in picked_candidates:
                links = utils.manhattan_path(st.x, st.y, xx, yy)
                path_bottleneck = req
                for typ, a, b in links:
                    rcap = cap_H[t, a - 1, b - 1] if typ == 'H' else cap_V[t, a - 1, b - 1]
                    path_bottleneck = min(path_bottleneck, rcap)

                bt = utils.b_link(t, Bpeak[xx, yy], Phi[xx, yy])
                car_bottle = min(bt, cap_car[t, cid])
                feasible_rate = min(path_bottleneck, car_bottle)

                is_better = False
                if feasible_rate > best_rate + 1e-9:
                    is_better = True
                elif abs(feasible_rate - best_rate) <= 1e-9 and best_links is not None and len(links) < len(best_links):
                    is_better = True

                if is_better:
                    best_rate, best_links, best_cid = feasible_rate, links, cid
                    best_arrival = (t + 1.0) + len(links) * cfg.hop_delay_s + cfg.car_link_delay_s
                    best_ingress_node = (xx, yy)

            if best_rate > 1e-9:
                r = best_rate
                path_nodes = utils.generate_path_nodes(st.x, st.y, best_ingress_node[0], best_ingress_node[1])
                schedule_log.append({
                    "k_time": t, "flow_id": fid, "source_sensor": f"({st.x}, {st.y})",
                    "path_to_car": f"{path_nodes} -> Car-{best_cid}", "rate_mbps": r
                })

                for typ, a, b in best_links:
                    if typ == 'H':
                        cap_H[t, a - 1, b - 1] -= r
                        H_used[a - 1, b - 1] += r
                    else:
                        cap_V[t, a - 1, b - 1] -= r
                        V_used[a - 1, b - 1] += r

                car_sched_ts[t, best_cid] += r
                cap_car[t, best_cid] -= r
                util = car_sched_ts[t, best_cid] / cfg.B_receive
                loss_ratio = 0.0
                if util > cfg.cong_th:
                    loss_ratio = min(cfg.cong_loss_max, (util - cfg.cong_th) / (1.0 - cfg.cong_th) * cfg.cong_loss_max)

                received = r * (1.0 - loss_ratio)
                car_recv_ts[t, best_cid] += received
                arrival_queue.append((best_arrival, best_cid, received, fid))

                st.remaining -= r
                st.bytes_scheduled += r
                st.last_arrival_time = max(st.last_arrival_time or 0.0, best_arrival)
                if st.remaining <= 1e-9:
                    st.done = True
                    finished_flows[fid] = st
                    del active_flows[fid]

    return finished_flows, arrival_queue, car_recv_ts, H_used, V_used, schedule_log