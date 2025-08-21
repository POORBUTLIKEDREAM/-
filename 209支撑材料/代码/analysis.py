# analysis.py
"""
负责仿真结果的后处理：统计分析、数据导出和可视化。
"""
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config as cfg


def analyze_and_report_results(finished_flows, arrival_queue, total_flows):
    """计算并打印核心性能指标。"""
    recv_per_flow = defaultdict(float)
    for (_, _, amt, fid) in arrival_queue:
        recv_per_flow[fid] += amt

    Q_total = total_flows * cfg.flow_size_mb
    Q_recv = sum(recv_per_flow.values())
    PLR = 1.0 - (Q_recv / Q_total) if Q_total > 0 else 0.0

    delays = []
    for st in finished_flows.values():
        if st.last_arrival_time is not None:
            delays.append(st.last_arrival_time - st.t_start)

    avg_delay = float(np.mean(delays)) if delays else float('nan')

    print("-------------------- 核心指标 --------------------")
    print(f"总流数 F = {total_flows}")
    print(f"总应发数据量 Q_total = {Q_total:.2f} Mb")
    print(f"车辆接收总量 Σqi = {Q_recv:.2f} Mb")
    print(f"丢包率 PLR = {PLR:.6f}")
    print(f"传输平均时延 (s) = {avg_delay:.6f}\n")

    return recv_per_flow


def export_data(recv_per_flow, schedule_log, total_flows):
    """将仿真结果导出到CSV文件。"""
    rows = []
    for fid in range(total_flows):
        t_start = 0
        if fid >= cfg.M * cfg.N and fid < 2 * cfg.M * cfg.N:
            t_start = 30
        elif fid >= 2 * cfg.M * cfg.N:
            t_start = 60
        rows.append({"flow_id": fid, "t_start": t_start, "recv_Mb": recv_per_flow.get(fid, 0.0)})

    pd.DataFrame(rows).to_csv("per_flow_reception.csv", index=False)
    print("[CSV] 每流接收量已保存: per_flow_reception.csv")

    pd.DataFrame(schedule_log).to_csv("scheduling_plan.csv", index=False)
    print("[CSV] 详细调度方案已保存: scheduling_plan.csv\n")


def create_visualizations(car_recv_ts, H_used, V_used):
    """生成并显示结果图表。"""
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.figure(figsize=(12, 5))
    for cid in range(cfg.num_cars):
        plt.plot(range(cfg.T), car_recv_ts[:, cid], label=f"Car-{cid + 1}")
    plt.title("Per-second Actual Reception per Car", fontsize=16)
    plt.xlabel("Time (s)"), plt.ylabel("Received Rate (Mbps)"), plt.legend(), plt.tight_layout(), plt.show()

    plt.figure(figsize=(12, 5))
    im_h = plt.imshow(H_used.T, aspect='auto', origin='lower', cmap='viridis', extent=[1, cfg.M, 1, cfg.N])
    plt.title("Total Usage of Horizontal Links (Mb)", fontsize=16)
    plt.xlabel("X-coordinate of Link Start"), plt.ylabel("Y-coordinate of Link")
    plt.colorbar(im_h, label='Total data transmitted (Mb)'), plt.tight_layout(), plt.show()

    plt.figure(figsize=(12, 5))
    im_v = plt.imshow(V_used.T, aspect='auto', origin='lower', cmap='viridis', extent=[1, cfg.M + 1, 1, cfg.N])
    plt.title("Total Usage of Vertical Links (Mb)", fontsize=16)
    plt.xlabel("X-coordinate of Link"), plt.ylabel("Y-coordinate of Link Start")
    plt.colorbar(im_v, label='Total data transmitted (Mb)'), plt.tight_layout(), plt.show()