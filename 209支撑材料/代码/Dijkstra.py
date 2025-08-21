import numpy as np
import collections
import heapq

M_ROWS, N_COLS, B_SENSOR, T_SENSOR = 20, 30, 10.0, 0.050
NUM_VEHICLES, B_RECEIVE, T_S2G = 3, 100.0, 0.050
VEHICLE_TRIP_DURATION, VEHICLE_COVERAGE_WIDTH = 100.0, 3
VEHICLE_COVERAGE_ROWS = {1: [5, 6], 2: [15, 16], 3: [25, 26]}
FLOW_SIZE, FLOW_RATE = 10.0, 5.0
T_SIMULATION = 90


def get_s2g_bandwidth(t: float, peak_B: float, phase: float) -> float:
    time_in_period = (t + phase) % 10.0
    if time_in_period <= 5.0:
        return (peak_B / 5.0) * time_in_period
    else:
        return peak_B - (peak_B / 5.0) * (time_in_period - 5.0)


class SchedulerV4:
    def __init__(self):
        self.sensor_grid = np.array([[self.Sensor(x + 1, y + 1) for y in range(N_COLS)] for x in range(M_ROWS)])
        self.data_flows = self._generate_flows()
        self.schedule = collections.defaultdict(dict)
        self.total_data_generated = len(self.data_flows) * FLOW_SIZE
        self.W_WAIT = 1.0;
        self.W_REM = 1.5

    class Sensor:
        def __init__(self, x, y):
            self.x, self.y = x, y;
            np.random.seed(x * M_ROWS + y)
            self.s2g_peak_b = np.random.uniform(8, 15)
            self.s2g_phase = np.random.uniform(0, 10)

    class DataFlow:
        def __init__(self, id, x, y, t):
            self.id, self.start_pos, self.start_time = id, (x, y), t
            self.total_size = FLOW_SIZE;
            self.data_sent = 0.0
            self.completion_time = -1;
            self.last_tx_time = t

        @property
        def is_complete(self): return self.data_sent >= self.total_size

        @property
        def remaining_data(self): return self.total_size - self.data_sent

    def _generate_flows(self):
        flows = [];
        fid = 0
        for start_t in [0, 30, 60]:
            for x in range(M_ROWS):
                for y in range(N_COLS):
                    flows.append(self.DataFlow(fid, x + 1, y + 1, start_t));
                    fid += 1
        return flows

    def get_vehicle_x_position(self, t: int, max_x: int, trip_duration: float) -> int:
        cycle_time = t % (2 * trip_duration);
        path_length = max_x - VEHICLE_COVERAGE_WIDTH
        if cycle_time < trip_duration:
            progress = cycle_time / trip_duration;
            pos = 1 + int(progress * path_length)
        else:
            progress = (cycle_time - trip_duration) / trip_duration;
            pos = (max_x - VEHICLE_COVERAGE_WIDTH + 1) - int(progress * path_length)
        return max(1, pos)

    def get_vehicle_coverage(self, vehicle_id: int, t: int) -> list:
        if vehicle_id not in VEHICLE_COVERAGE_ROWS: return []
        start_x = self.get_vehicle_x_position(t, M_ROWS, VEHICLE_TRIP_DURATION)
        rows = VEHICLE_COVERAGE_ROWS[vehicle_id];
        coverage = []
        for x in range(start_x, start_x + VEHICLE_COVERAGE_WIDTH):
            for y in rows:
                if 1 <= x <= M_ROWS and 1 <= y <= N_COLS: coverage.append((x, y))
        return coverage

    def _get_neighbors(self, x, y):
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 1 <= nx <= M_ROWS and 1 <= ny <= N_COLS: neighbors.append((nx, ny))
        return neighbors

    def find_widest_path(self, start_pos, end_pos, link_bandwidth):
        """ 使用Dijkstra变体寻找最大容量路径 (Widest Path) """
        # 记录到各点的已知最大瓶颈带宽
        capacity = collections.defaultdict(float)
        capacity[start_pos] = float('inf')
        # 记录路径
        parent = {}
        # 优先队列，存储 (-瓶颈带宽, 节点)，用负号实现最大堆
        pq = [(-float('inf'), start_pos)]

        while pq:
            cap, curr_pos = heapq.heappop(pq)
            cap = -cap

            if cap < capacity[curr_pos]:
                continue

            if curr_pos == end_pos:
                break  # 找到了到终点的路径

            for neighbor_pos in self._get_neighbors(*curr_pos):
                link_bw = link_bandwidth[tuple(sorted((curr_pos, neighbor_pos)))]
                # 到达邻居节点的路径瓶颈，取决于当前瓶颈和新链路的瓶颈
                path_bottleneck = min(cap, link_bw)

                if path_bottleneck > capacity[neighbor_pos]:
                    capacity[neighbor_pos] = path_bottleneck
                    parent[neighbor_pos] = curr_pos
                    heapq.heappush(pq, (-path_bottleneck, neighbor_pos))

        if end_pos not in parent:
            return None, 0

        # 回溯路径
        path = [];
        curr = end_pos
        while curr != start_pos:
            path.append(curr)
            curr = parent[curr]
        path.append(start_pos)
        path.reverse()

        return path, capacity[end_pos]

    def run_simulation(self):
        print("开始执行流量调度仿真 (V4 - 带宽感知 & 预见性)...")
        for t in range(T_SIMULATION):
            # --- 初始化当前时间片的状态 ---
            link_bandwidth = collections.defaultdict(lambda: B_SENSOR)
            vehicle_load = collections.defaultdict(float)

            # --- OPTIMIZATION 2: 预见性评估 ---
            exit_potentials = {}
            for vid in range(1, NUM_VEHICLES + 1):
                coverage = self.get_vehicle_coverage(vid, t)
                for pos in coverage:
                    sensor = self.sensor_grid[pos[0] - 1][pos[1] - 1]
                    # 计算未来潜力分
                    b_t0 = get_s2g_bandwidth(t, sensor.s2g_peak_b, sensor.s2g_phase)
                    b_t1 = get_s2g_bandwidth(t + 1, sensor.s2g_peak_b, sensor.s2g_phase)
                    future_score = 0.7 * b_t0 + 0.3 * b_t1  # 简化版，只看下一秒
                    exit_potentials[pos] = {'vid': vid, 'future_score': future_score, 's2g_now': b_t0}

            active_flows = [f for f in self.data_flows if not f.is_complete and t >= f.start_time]

            def get_urgency(flow):
                return self.W_WAIT * (t - flow.last_tx_time) - self.W_REM * (flow.remaining_data / FLOW_SIZE)

            sorted_flows = sorted(active_flows, key=get_urgency, reverse=True)

            for flow in sorted_flows:
                best_option = {'path': None, 'rate': 0, 'vehicle_id': None, 'future_score': -1}

                for exit_pos, potential in exit_potentials.items():
                    # 调用新的寻路算法
                    path, mesh_bottleneck = self.find_widest_path(flow.start_pos, exit_pos, link_bandwidth)

                    if not path or mesh_bottleneck < 1e-6: continue

                    vehicle_id = potential['vid']
                    vehicle_rem_bw = B_RECEIVE - vehicle_load[vehicle_id]

                    # 最终瓶颈由 mesh瓶颈、s2g带宽、车辆剩余容量 共同决定
                    possible_rate = min(FLOW_RATE, mesh_bottleneck, potential['s2g_now'], vehicle_rem_bw)

                    if possible_rate < 1e-6: continue

                    # 使用“未来潜力分”作为主要的决策依据
                    if potential['future_score'] > best_option['future_score']:
                        best_option = {'path': path, 'rate': possible_rate, 'vehicle_id': vehicle_id,
                                       'future_score': potential['future_score']}

                if best_option['path']:
                    rate, path, vehicle_id = best_option['rate'], best_option['path'], best_option['vehicle_id']

                    flow.data_sent += rate;
                    flow.last_tx_time = t
                    if flow.is_complete and flow.completion_time == -1: flow.completion_time = t + 1

                    path_links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                    for link in path_links: link_bandwidth[tuple(sorted(link))] -= rate
                    vehicle_load[vehicle_id] += rate
                    self.schedule[t][flow.id] = (path, rate, vehicle_id)

            if (t + 1) % 10 == 0: print(f"  已完成 {t + 1}/{T_SIMULATION} 秒的调度...")
        print("仿真结束。")
        self.calculate_and_print_metrics()

    def calculate_and_print_metrics(self):
        total_data_received = sum(f.total_size if f.is_complete else f.data_sent for f in self.data_flows)
        success_rate = total_data_received / self.total_data_generated
        packet_loss_rate = 1 - success_rate
        total_delay = 0;
        completed_flows = 0
        for f in self.data_flows:
            if f.is_complete:
                path, _, _ = self.schedule[f.completion_time - 1][f.id]
                hops = len(path) - 1;
                mesh_delay = hops * T_SENSOR;
                s2g_delay = T_S2G
                total_delay += (f.completion_time - f.start_time) + mesh_delay + s2g_delay
                completed_flows += 1
        average_delay = total_delay / completed_flows if completed_flows > 0 else float('inf')

        print("\n--- 最终性能评价 (V4) ---")
        print(f"总生成数据量: {self.total_data_generated:.2f} Mb")
        print(f"总接收数据量: {total_data_received:.2f} Mb")
        print(f"完成传输的流数量: {completed_flows}/{len(self.data_flows)}")
        print(f"评价指标1 - 丢包率: {packet_loss_rate:.4%}")
        print(f"评价指标2 - 传输平均时延: {average_delay:.4f} 秒")


if __name__ == '__main__':
    scheduler = SchedulerV4()
    scheduler.run_simulation()