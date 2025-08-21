# utils.py
"""
存放通用的、与模型相关的辅助函数。
"""
import math
import config as cfg

def triangular_wave_unit(u: float) -> float:
    """单位三角波：周期为1，峰值为1。"""
    v = u % 1.0
    return 2 * v if v <= 0.5 else 2 * (1.0 - v)

def b_link(t, Bpeak, phi):
    """计算传感器到车的瞬时带宽(Mbps)，使用周期性三角波模拟。"""
    return Bpeak * triangular_wave_unit((t + phi) / cfg.period)

def car_center_x(t):
    """计算t时刻车辆覆盖窗口的中心列位置（浮点数）。"""
    start_c, end_c = 2.0, cfg.M - 1.0
    v = (end_c - start_c) / 100.0
    cc = start_c + v * t
    return max(2.0, min(end_c, cc))

def covered_window(cc):
    """根据中心列位置cc，计算车辆覆盖的3列整数坐标集合。"""
    c = int(math.floor(cc))
    cols = [c - 1, c, c + 1]
    cols = [min(cfg.M, max(1, x)) for x in cols]
    return sorted(set(cols))

def manhattan_path(x1, y1, x2, y2):
    """生成先横后纵的曼哈顿路径上的链路序列。"""
    links = []
    x, y = x1, y1
    step_x = 1 if x2 >= x else -1
    while x != x2:
        links.append(('H', x, y) if step_x == 1 else ('H', x - 1, y))
        x += step_x
    step_y = 1 if y2 >= y else -1
    while y != y2:
        links.append(('V', x, y) if step_y == 1 else ('V', x, y - 1))
        y += step_y
    return links

def generate_path_nodes(x1, y1, x2, y2):
    """生成从(x1,y1)到(x2,y2)的先横后纵曼哈顿路径上的节点坐标列表。"""
    nodes = []
    x, y = x1, y1
    step_x = 1 if x2 > x else -1 if x2 < x else 0
    while x != x2:
        nodes.append((x, y))
        x += step_x
    step_y = 1 if y2 > y else -1 if y2 < y else 0
    while y != y2:
        nodes.append((x, y))
        y += step_y
    nodes.append((x2, y2))
    return nodes