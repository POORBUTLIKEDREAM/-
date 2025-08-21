# main.py
"""
主执行文件。
按顺序调用其他模块的函数来完成整个仿真和分析流程。
"""
import simulation as sim
import analysis


def main():
    """主函数，按顺序执行整个仿真流程。"""
    # 1. 初始化
    print("正在初始化仿真环境...")
    cap_H, cap_V, cap_car = sim.initialize_capacities()
    Bpeak, Phi = sim.initialize_wireless_params()
    flows_by_time, total_flows = sim.initialize_flows()
    print("初始化完成。\n")

    # 2. 运行主仿真循环
    print("开始执行调度仿真...")
    results = sim.run_simulation(cap_H, cap_V, cap_car, Bpeak, Phi, flows_by_time)
    finished_flows, arrival_queue, car_recv_ts, H_used, V_used, schedule_log = results
    print("仿真执行完毕。\n")

    # 3. 分析和报告结果
    recv_per_flow = analysis.analyze_and_report_results(finished_flows, arrival_queue, total_flows)

    # 4. 导出数据
    analysis.export_data(recv_per_flow, schedule_log, total_flows)

    # 5. 可视化
    print("正在生成可视化图表...")
    analysis.create_visualizations(car_recv_ts, H_used, V_used)
    print("所有流程执行完毕。")


if __name__ == "__main__":
    main()