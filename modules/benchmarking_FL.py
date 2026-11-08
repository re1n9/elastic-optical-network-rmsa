import numpy as np

NUM_SLOTS = 320
slot_grid = {}

def initialize_slot_grid(G):
    """初始化slot_grid，为每条边分配320个槽位（0表示空闲）"""
    global slot_grid
    slot_grid.clear()  # 清空之前的数据
    
    for u, v in G.edges():
        key = (min(u, v), max(u, v))  # 确保键有序
        slot_grid[key] = [0] * 320
    
    return slot_grid

def find_least_used_slots(path, num_slots):
    best_slot_start, min_usage = None, float('inf')
    for start_slot in range(NUM_SLOTS - num_slots + 1):
        valid = True
        usage_sum = 0
        for u, v in zip(path, path[1:]):
            key = (min(u, v), max(u, v))
            segment = slot_grid[key][start_slot:start_slot + num_slots]
            if any(segment):  # 如果有槽位被占用，则跳过
                valid = False
                break
            usage_sum += sum(segment)  # 统计使用量

        if valid and usage_sum < min_usage:
            min_usage = usage_sum
            best_slot_start = start_slot

    return best_slot_start

def allocate_slots_least_used(G, results):
    """
    根据 "Least Used" 策略，为路径上的所有请求分配频谱槽位。
    对于每个请求，计算所有空闲槽位区间的使用量，选择使用量最少的区间。
    
    参数:
        G: 网络拓扑图
        results: 包含路径、调制格式和所需槽位数量的请求列表
        
    返回:
        更新后的请求列表，包含分配的槽位起始位置
    """
    global slot_grid
    
    # 确保slot_grid已初始化
    if not slot_grid:
        initialize_slot_grid(G)
    
    for i, req in enumerate(results):
        path = req["path"]
        slots_needed = req["slots"]
        
        # 检查每个可能的起始位置
        best_start = None
        min_usage = float('inf')
        
        # 遍历所有可能的起始槽位
        for start_slot in range(320 - slots_needed + 1):
            valid = True
            usage_sum = 0
            
            # 检查路径上每条边的槽位是否可用
            for u, v in zip(path, path[1:]):
                key = (min(u, v), max(u, v))
                
                # 检查该边在 [start_slot:start_slot + slots_needed] 区间是否有被占用的槽位
                for s in range(start_slot, start_slot + slots_needed):
                    if slot_grid[key][s] == 1:
                        valid = False
                        break
                        
                if not valid:
                    break
                    
                # 统计该边的总体使用量（当前使用的槽位数）
                usage_sum += sum(slot_grid[key])
            
            # 如果找到一个有效的起始位置且使用量最少，则更新
            if valid and usage_sum < min_usage:
                min_usage = usage_sum
                best_start = start_slot
        
        # 如果找到可行的起始位置，则分配槽位
        if best_start is not None:
            req["slot_start"] = best_start
            req["slot_end"] = best_start + slots_needed - 1
            
            # 更新 slot_grid
            for u, v in zip(path, path[1:]):
                key = (min(u, v), max(u, v))
                for s in range(best_start, best_start + slots_needed):
                    slot_grid[key][s] = 1
        else:
            # 如果无法分配，设置为 None
            req["slot_start"] = None
            req["slot_end"] = None
            print(f"警告: 请求 {i} 无法分配槽位: {path}, 需求: {slots_needed} 槽位")
    
    return results

def calculate_link_fragmentation(slots):
    """
    计算单条链路的垂直碎片化指标。
    参数:
        slots: 一个列表，表示该链路的槽位状态，0 表示空闲，1 表示已占用。
    返回:
        F: 碎片化指标，取值范围 [0, 1]，值越大表示碎片化越严重。
    """
    # 找到所有空闲槽位的总数
    total_free = slots.count(0)
    if total_free == 0:
        # 如果没有空闲槽位，则认为没有碎片化（或定义为0）
        return 0

    # 遍历槽位，找出最大连续空闲块的长度
    max_contiguous = 0
    current_contiguous = 0
    for s in slots:
        if s == 0:
            current_contiguous += 1
            max_contiguous = max(max_contiguous, current_contiguous)
        else:
            current_contiguous = 0

    # 碎片化指标 F = 1 - (最大连续空闲槽位数 / 总空闲槽位数)
    F = 1 - (max_contiguous / total_free)
    return F


def calculate_rmsf(slots, num_links):
    """
    计算单条链路的 RMSF，并使用论文推荐的归一化方法。
    参数:
        slots: 该链路的槽位状态列表 (0 表示空闲，1 表示已占用)。
        num_links: 网络中链路的总数量，用于归一化。
    返回:
        归一化的 RMSF 值，表示该链路的垂直碎片化程度。
    """
    s_max = max((i for i, s in enumerate(slots) if s == 1), default=0)

    # 统计空闲块
    free_blocks = []
    current_block = 0
    for s in slots:
        if s == 0:
            current_block += 1
        else:
            if current_block > 0:
                free_blocks.append(current_block)
                current_block = 0
    if current_block > 0:
        free_blocks.append(current_block)

    num_free_blocks = len(free_blocks)
    if num_free_blocks == 0 or s_max == 0:
        return 0  # 如果没有空闲块或没有占用槽位，则 RMSF 设为 0

    # 计算 RMSF（论文推荐公式）
    rmsf_value = (s_max * num_free_blocks) / (np.sqrt(sum(x ** 2 for x in free_blocks)) + 1e-5)

    # 归一化处理：除以链路数，确保 RMSF 值在合理范围
    rmsf_value = rmsf_value / num_links

    return rmsf_value


def calculate_max_vertical_fragmentation(slot_grid):
    """
    计算整个网络中的最大垂直碎片化 (最大 RMSF)。
    参数:
        slot_grid: 字典，键为链路 (min(u, v), max(u, v))，值为该链路的槽位状态列表。
    返回:
        网络中的最大 RMSF。
    """
    num_links = len(slot_grid)  # 计算总链路数
    rmsf_values = [calculate_rmsf(slots, num_links) for slots in slot_grid.values()]

    return max(rmsf_values)  # 返回最大 RMSF，即最大垂直碎片化

