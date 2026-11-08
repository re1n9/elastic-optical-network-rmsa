import networkx as nx
import copy
import random
import math
from .modulation import choose_modulation, MODULATION_FORMATS
from .k_shortest_path import k_shortest_paths
import numpy as np

TOTAL_SLOTS = 320  # 总槽位数

class Link:
    def __init__(self, u, v):
        self.u = u
        self.v = v
        self.slots = [0] * TOTAL_SLOTS  # 0表示空闲，1表示已分配

    def assign_slots(self, start, length):
        """检查并分配连续槽位，若槽位已占用返回 False"""
        for i in range(start, start + length):
            if self.slots[i] == 1:
                return False
        for i in range(start, start + length):
            self.slots[i] = 1
        return True

    def remove_slots(self, start, length):
        """释放槽位"""
        for i in range(start, start + length):
            self.slots[i] = 0

    def compute_fragmentation(self, num_links):
        """
        计算 RMSF 碎片化指标（基于论文公式）
        参数:
            num_links: 网络的总链路数，用于归一化 RMSF 值
        返回:
            归一化的 RMSF 碎片化指标
        """
        # 找到最高已使用槽索引
        s_max = max((i for i, s in enumerate(self.slots) if s == 1), default=0)

        # 统计空闲块
        free_blocks = []
        current_block = 0

        for s in self.slots:
            if s == 0:
                current_block += 1
            else:
                if current_block > 0:
                    free_blocks.append(current_block)
                    current_block = 0
        if current_block > 0:
            free_blocks.append(current_block)  # 处理最后一个空闲块

        num_free_blocks = len(free_blocks)  # 空闲块总数

        if num_free_blocks == 0 or s_max == 0:
            return 0  # 如果没有空闲槽，RMSF 设为 0

        # 计算 RMSF（基于论文公式）
        rmsf_value = (s_max * num_free_blocks) / (np.sqrt(sum(x ** 2 for x in free_blocks)) + 1e-5)

        # 归一化处理（防止 RMSF 过大）
        rmsf_value = rmsf_value / num_links

        return rmsf_value


def get_edges_from_path(path):
    """
    将由节点构成的路径转换为边列表，边以 (min(u,v), max(u,v)) 表示
    """
    return [tuple(sorted((path[i], path[i + 1]))) for i in range(len(path) - 1)]


# ------------------ Network 类（基于 networkx 图构建） ------------------
class Network:
    def __init__(self, graph):
        """
        根据 networkx 图构造网络，每个边创建一个 Link 对象
        """
        self.graph = graph
        self.links = {}
        for u, v in graph.edges():
            edge = tuple(sorted((u, v)))
            self.links[edge] = Link(u, v)

    def assign_spectrum_on_path_least_used(self, path, required_slots):
        """
        根据 "Least Used" 策略，在给定路径上分配频谱。
        1. 检查所有可能的连续槽位区间是否空闲。
        2. 统计每个可行区间的使用量 (usage_sum)。
        3. 选择使用量最少的区间并进行分配。
        如果无法找到可行区间，则返回 None。
        """
        edges = get_edges_from_path(path)
        best_slot_start, min_usage = None, float('inf')

        # 1. 遍历所有可能的起始槽位
        for start_slot in range(TOTAL_SLOTS - required_slots + 1):
            valid = True
            usage_sum = 0

            # 2. 检查路径上每条边在 [start_slot:start_slot + required_slots] 区间是否空闲，并统计使用量
            for edge in edges:
                segment = self.links[edge].slots[start_slot:start_slot + required_slots]
                if any(segment):  # 如果有槽位被占用，则该区间不可用
                    valid = False
                    break
                usage_sum += sum(segment)  # 累加该区间的使用量

            # 3. 如果该区间有效且使用量比当前最小值还小，则更新最优分配
            if valid and usage_sum < min_usage:
                min_usage = usage_sum
                best_slot_start = start_slot

        # 4. 如果没有找到可行区间，则返回 None
        if best_slot_start is None:
            return None

        # 5. 分配最佳区间：在路径上的每条边上分配槽位
        for edge in edges:
            self.links[edge].assign_slots(best_slot_start, required_slots)

        # 6. 返回分配的起始槽位
        return best_slot_start

    def remove_assignment(self, path, start, required_slots):
        """在路径对应的所有边上释放槽位"""
        edges = get_edges_from_path(path)
        for edge in edges:
            self.links[edge].remove_slots(start, required_slots)

    def compute_network_fragmentation(self):
        """
        计算整个网络的最大 RMSF（最大垂直碎片化）
        """
        num_links = len(self.links)  # 计算总链路数
        max_rmsf = 0
        for link in self.links.values():
            rmsf = link.compute_fragmentation(num_links)
            max_rmsf = max(max_rmsf, rmsf)
        return max_rmsf

# ------------------ Request 定义 ------------------
class Request:
    def __init__(self, request_id, candidate_info, source=None, target=None, original_demand=None):
        """
        candidate_paths 为候选路径列表，每个路径由节点序列构成
        """
        self.request_id = request_id
        self.candidate_info = candidate_info
        self.source = source
        self.target = target
        self.original_demand = original_demand  # 添加原始流量需求属性

    @staticmethod
    def generate_requests_from_matrix(G, traffic_matrix, k=5, order = "descending"):
        traffic_matrix = sorted(traffic_matrix, key=lambda x: x[2], reverse=(order == "descending"))
        requests = []
        req_id = 0
        
        for source, target, demand_gbps in traffic_matrix:
            # 确定此链路上可用的调制格式和最大速率
            paths = k_shortest_paths(G, source, target, k=k)
            if not paths:
                print(f"无法为 {source}->{target} 找到路径，跳过此请求")
                continue
                
            shortest_path = paths[0]
            path_length = sum(G[u][v]['weight'] for u, v in zip(shortest_path, shortest_path[1:]))
            
            # 找出可用的最高速率调制格式
            max_rate = 0
            for mod, props in MODULATION_FORMATS.items():
                if path_length <= props["max_length"] and props["rate"] > max_rate:
                    max_rate = props["rate"]
            
            if max_rate == 0:
                print(f"路径 {source}->{target} 长度为 {path_length}，超出所有调制格式支持的最大距离，跳过此请求")
                continue
            
            # 计算需要的光路数量
            num_lightpaths = int(-(-demand_gbps // max_rate))  # 向上取整并转换为整数
            
            if num_lightpaths > 1:
                print(f"请求 {source}->{target} 需求 {demand_gbps} Gbps 分割为 {num_lightpaths} 个并行请求，每个 {max_rate} Gbps")
                
                # 分割为多个并行请求
                for i in range(num_lightpaths - 1):
                    sub_demand = max_rate  # 前n-1个请求都是最大速率
                    candidate_info = []
                    
                    for path in paths:
                        path_length = sum(G[u][v]['weight'] for u, v in zip(path, path[1:]))
                        mod, total_slots, total_cost = choose_modulation(path_length, sub_demand)
                        if mod:  # 确保有可用的调制格式
                            candidate_info.append({
                                "path": path,
                                "mod": mod,
                                "slots": total_slots,
                                "cost": total_cost
                            })
                    
                    if candidate_info:  # 如果有可用路径
                        req_obj = Request(req_id, candidate_info, source=source, target=target, original_demand=demand_gbps)
                        requests.append(req_obj)
                        req_id += 1
                
                # 最后一个请求处理剩余的需求
                last_demand = demand_gbps - max_rate * (num_lightpaths - 1)
                candidate_info = []
                for path in paths:
                    path_length = sum(G[u][v]['weight'] for u, v in zip(path, path[1:]))
                    mod, total_slots, total_cost = choose_modulation(path_length, last_demand)
                    if mod:  # 确保有可用的调制格式
                        candidate_info.append({
                            "path": path,
                            "mod": mod,
                            "slots": total_slots,
                            "cost": total_cost
                        })
                
                if candidate_info:  # 如果有可用路径
                    req_obj = Request(req_id, candidate_info, source=source, target=target, original_demand=demand_gbps)
                    requests.append(req_obj)
                    req_id += 1
            else:
                # 不需要分割，直接处理原始请求
                candidate_info = []
                for path in paths:
                    path_length = sum(G[u][v]['weight'] for u, v in zip(path, path[1:]))
                    mod, total_slots, total_cost = choose_modulation(path_length, demand_gbps)
                    if mod:  # 确保有可用的调制格式
                        candidate_info.append({
                            "path": path,
                            "mod": mod,
                            "slots": total_slots,
                            "cost": total_cost
                        })

                if candidate_info:  # 如果有可用路径
                    req_obj = Request(req_id, candidate_info, source=source, target=target, original_demand=demand_gbps)
                    requests.append(req_obj)
                    req_id += 1
        
        print(f"总共生成 {len(requests)} 个请求 (含分割后的并行请求)")
        return requests


def simulated_annealing(requests, network, max_iter=10000, initial_T=1.0, cooling_rate=0.80):
    """
    针对每个请求选择候选路径并分配频谱，目标是最小化网络最大垂直碎片。
    solution 记录每个请求的选择：(候选路径索引, 分配槽位起始位置)。
    如果请求无法分配，则在初始阶段打印信息并忽略该请求。
    """
    # ✅ 初始解：为每个请求分配第一个候选路径
    solution = {}
    successful_requests = []  # ✅ 记录分配成功的请求

    print("正在生成初始解...")
    for req in requests:
        chosen = req.candidate_info[0]
        chosen_path = chosen['path']
        chosen_slots = chosen['slots']

        start_slot = network.assign_spectrum_on_path_least_used(chosen_path, chosen_slots)
        if start_slot is None:
            print(f"❌ 初始分配失败：请求 {req.request_id} 路径 {chosen_path} 无法分配频谱")
        else:
            solution[req.request_id] = (0, start_slot)
            successful_requests.append(req)  # ✅ 仅保留分配成功的请求

    if not successful_requests:
        print("⚠️ 所有请求均分配失败，无法进行模拟退火优化。")
        return {}, network.compute_network_fragmentation()

    current_frag = network.compute_network_fragmentation()
    best_frag = current_frag
    best_solution = copy.deepcopy(solution)
    T = initial_T

    print(f"初始解生成完成，开始模拟退火优化 (最大迭代次数: {max_iter})...")
    progress_step = max(1, max_iter // 10)  # 每完成约10%的迭代就打印一次进度
    improved_count = 0  # 记录改进次数
    
    # ✅ 模拟退火过程：仅迭代分配成功的请求
    for it in range(max_iter):
        # 打印进度
        if (it + 1) % progress_step == 0:
            print(f"模拟退火进度: {(it + 1) / max_iter * 100:.1f}%, 当前温度: {T:.6f}, 当前最佳碎片化: {best_frag:.6f}")
        
        # 如果连续多次迭代没有改进且温度已经很低，提前结束
        if T < 1e-6 and it > max_iter // 2:
            print(f"温度已接近0，提前结束模拟退火，完成迭代次数: {it+1}/{max_iter}")
            break
            
        req = random.choice(successful_requests)
        if req.request_id not in solution:
            continue  # ✅ 如果请求未分配成功，则跳过

        current_candidate_index, current_start = solution[req.request_id]
        available_indices = [i for i in range(len(req.candidate_info)) if i != current_candidate_index]
        if not available_indices:
            continue

        new_candidate_index = random.choice(available_indices)
        new_candidate = req.candidate_info[new_candidate_index]
        new_path = new_candidate['path']
        new_slots = new_candidate['slots']

        # ✅ 备份当前网络状态 - 使用浅拷贝以提高性能
        network_backup = copy.deepcopy(network)  # 使用深拷贝确保状态正确恢复
        old_path = req.candidate_info[current_candidate_index]['path']
        old_slots = req.candidate_info[current_candidate_index]['slots']
        old_start = current_start

        # ✅ 移除旧分配
        network.remove_assignment(old_path, current_start, old_slots)

        # ✅ 尝试新分配
        new_start = network.assign_spectrum_on_path_least_used(new_path, new_slots)
        if new_start is None:
            # 如果新分配失败，恢复旧分配
            network = network_backup  # 使用备份的网络状态恢复
            continue

        # ✅ 如果新分配成功，评估碎片化
        new_frag = network.compute_network_fragmentation()
        delta = new_frag - current_frag

        # ✅ Metropolis 准则决定是否接受新解
        if delta < 0 or random.random() < math.exp(-delta / T):
            solution[req.request_id] = (new_candidate_index, new_start)
            current_frag = new_frag
            if new_frag < best_frag:
                best_frag = new_frag
                best_solution = copy.deepcopy(solution)
                improved_count += 1
        else:
            # ✅ 不接受新解，恢复旧分配
            network = network_backup  # 使用备份的网络状态恢复
            solution[req.request_id] = (current_candidate_index, current_start)

        T *= cooling_rate  # 降温

    print(f"✅ 模拟退火完成：最佳碎片化度量 = {best_frag}，总改进次数 = {improved_count}")
    return best_solution, best_frag




# ------------------ 1+1专用路径保护相关函数 ------------------

def check_disjoint_paths(path1, path2):
    """
    检查两条路径是否链路不相交(链路分离)
    
    参数:
        path1: 第一条路径的节点序列
        path2: 第二条路径的节点序列
        
    返回:
        True如果两条路径完全链路不相交，否则False
    """
    # 将路径转换为边的集合
    edges1 = set(tuple(sorted((path1[i], path1[i+1]))) for i in range(len(path1) - 1))
    edges2 = set(tuple(sorted((path2[i], path2[i+1]))) for i in range(len(path2) - 1))
    
    # 检查两个集合是否有交集
    return len(edges1.intersection(edges2)) == 0


def find_disjoint_path_pairs(paths, max_pairs=3):
    """
    从k条路径中找出所有不相交的路径对
    
    参数:
        paths: 候选路径列表
        max_pairs: 返回的最大路径对数量
        
    返回:
        不相交路径对的列表[(path_idx1, path_idx2), ...]
    """
    disjoint_pairs = []
    
    for i in range(len(paths)):
        for j in range(i+1, len(paths)):
            if check_disjoint_paths(paths[i], paths[j]):
                disjoint_pairs.append((i, j))
                if len(disjoint_pairs) >= max_pairs:
                    return disjoint_pairs
    
    return disjoint_pairs


class ProtectedRequest:
    """为1+1保护设计的请求类"""
    
    def __init__(self, request_id, source, target, candidate_pairs=None, original_demand=None):
        """
        初始化具有保护的请求
        
        参数:
            request_id: 请求ID
            source: 源节点
            target: 目标节点
            candidate_pairs: 候选路径对的列表，每对包含主路径和保护路径的信息
            original_demand: 原始流量需求（分割前）
        """
        self.request_id = request_id
        self.source = source
        self.target = target
        self.candidate_pairs = candidate_pairs or []
        self.original_demand = original_demand  # 添加原始流量需求属性
    
    @staticmethod
    def generate_protected_requests_from_matrix(G, traffic_matrix, k=5, order="descending", max_pairs=3):
        """
        从流量矩阵生成带有1+1保护的请求列表
        
        参数:
            G: 网络拓扑图
            traffic_matrix: 流量矩阵，每行是(source, target, demand_gbps)
            k: 每对(source, target)找k条最短路径
            order: 排序顺序，"descending"(默认)或"ascending"
            max_pairs: 每个请求的最大路径对数量
            
        返回:
            带有1+1保护的请求列表
        """
        sorted_traffic = sorted(traffic_matrix, key=lambda x: x[2], reverse=(order == "descending"))
        protected_requests = []
        req_id = 0
        
        for source, target, demand_gbps in sorted_traffic:
            # 获取k条最短路径
            paths = k_shortest_paths(G, source, target, k=k)
            
            if not paths:
                print(f"无法为 {source}->{target} 找到路径，跳过此请求")
                continue
            
            # 找出不相交的路径对
            disjoint_pairs = find_disjoint_path_pairs(paths, max_pairs)
            
            # 如果没有找到不相交的路径对，则跳过此请求
            if not disjoint_pairs:
                print(f"警告: 请求 {source}->{target} 无法找到不相交的路径对，无法提供1+1保护")
                continue
            
            # 确定可用的最高速率调制格式
            # 需要同时考虑主路径和备份路径的长度限制
            max_rates = {}
            for primary_idx, backup_idx in disjoint_pairs:
                primary_path = paths[primary_idx]
                backup_path = paths[backup_idx]
                
                primary_length = sum(G[u][v]['weight'] for u, v in zip(primary_path, primary_path[1:]))
                backup_length = sum(G[u][v]['weight'] for u, v in zip(backup_path, backup_path[1:]))
                
                # 找出主路径可用的最高速率
                primary_max_rate = 0
                for mod, props in MODULATION_FORMATS.items():
                    if primary_length <= props["max_length"] and props["rate"] > primary_max_rate:
                        primary_max_rate = props["rate"]
                
                # 找出备份路径可用的最高速率
                backup_max_rate = 0
                for mod, props in MODULATION_FORMATS.items():
                    if backup_length <= props["max_length"] and props["rate"] > backup_max_rate:
                        backup_max_rate = props["rate"]
                
                # 路径对的最高速率受限于主路径和备份路径中速率较低的一个
                pair_max_rate = min(primary_max_rate, backup_max_rate)
                max_rates[(primary_idx, backup_idx)] = pair_max_rate
            
            # 计算所有路径对中的最高速率
            if not max_rates:
                print(f"路径 {source}->{target} 所有候选路径都超出调制格式支持的最大距离，跳过此请求")
                continue
            
            overall_max_rate = max(max_rates.values())
            if overall_max_rate == 0:
                print(f"路径 {source}->{target} 无可用调制格式，跳过此请求")
                continue
            
            # 计算需要的光路数量
            num_lightpaths = int(-(-demand_gbps // overall_max_rate))  # 向上取整并转换为整数
            
            if num_lightpaths > 1:
                print(f"请求 {source}->{target} 需求 {demand_gbps} Gbps 分割为 {num_lightpaths} 个并行请求，每个最多 {overall_max_rate} Gbps")
                
                # 分割为多个并行请求
                for i in range(num_lightpaths - 1):
                    sub_demand = overall_max_rate  # 前n-1个请求都是最大速率
                    
                    # 为每对路径计算调制格式和资源需求
                    candidate_pairs = []
                    for primary_idx, backup_idx in disjoint_pairs:
                        primary_path = paths[primary_idx]
                        backup_path = paths[backup_idx]
                        
                        # 计算主路径长度和资源需求
                        primary_length = sum(G[u][v]['weight'] for u, v in zip(primary_path, primary_path[1:]))
                        primary_mod, primary_slots, primary_cost = choose_modulation(primary_length, sub_demand)
                        
                        # 计算备份路径长度和资源需求
                        backup_length = sum(G[u][v]['weight'] for u, v in zip(backup_path, backup_path[1:]))
                        backup_mod, backup_slots, backup_cost = choose_modulation(backup_length, sub_demand)
                        
                        # 只有当主路径和备份路径都有可用调制格式时才添加
                        if primary_mod and backup_mod:
                            candidate_pairs.append({
                                'primary': {
                                    'path': primary_path,
                                    'mod': primary_mod,
                                    'slots': primary_slots,
                                    'cost': primary_cost,
                                    'distance': primary_length
                                },
                                'backup': {
                                    'path': backup_path,
                                    'mod': backup_mod,
                                    'slots': backup_slots,
                                    'cost': backup_cost,
                                    'distance': backup_length
                                }
                            })
                    
                    # 创建带1+1保护的请求
                    if candidate_pairs:
                        protected_request = ProtectedRequest(req_id, source, target, candidate_pairs, original_demand=demand_gbps)
                        protected_requests.append(protected_request)
                        req_id += 1
                
                # 最后一个请求处理剩余的需求
                last_demand = demand_gbps - overall_max_rate * (num_lightpaths - 1)
                
                # 为每对路径计算调制格式和资源需求
                candidate_pairs = []
                for primary_idx, backup_idx in disjoint_pairs:
                    primary_path = paths[primary_idx]
                    backup_path = paths[backup_idx]
                    
                    # 计算主路径长度和资源需求
                    primary_length = sum(G[u][v]['weight'] for u, v in zip(primary_path, primary_path[1:]))
                    primary_mod, primary_slots, primary_cost = choose_modulation(primary_length, last_demand)
                    
                    # 计算备份路径长度和资源需求
                    backup_length = sum(G[u][v]['weight'] for u, v in zip(backup_path, backup_path[1:]))
                    backup_mod, backup_slots, backup_cost = choose_modulation(backup_length, last_demand)
                    
                    # 只有当主路径和备份路径都有可用调制格式时才添加
                    if primary_mod and backup_mod:
                        candidate_pairs.append({
                            'primary': {
                                'path': primary_path,
                                'mod': primary_mod,
                                'slots': primary_slots,
                                'cost': primary_cost,
                                'distance': primary_length
                            },
                            'backup': {
                                'path': backup_path,
                                'mod': backup_mod,
                                'slots': backup_slots,
                                'cost': backup_cost,
                                'distance': backup_length
                            }
                        })
                
                # 创建带1+1保护的请求
                if candidate_pairs:
                    protected_request = ProtectedRequest(req_id, source, target, candidate_pairs, original_demand=demand_gbps)
                    protected_requests.append(protected_request)
                    req_id += 1
            else:
                # 不需要分割，直接处理原始请求
                # 为每对路径计算调制格式和资源需求
                candidate_pairs = []
                for primary_idx, backup_idx in disjoint_pairs:
                    primary_path = paths[primary_idx]
                    backup_path = paths[backup_idx]
                    
                    # 计算主路径长度和资源需求
                    primary_length = sum(G[u][v]['weight'] for u, v in zip(primary_path, primary_path[1:]))
                    primary_mod, primary_slots, primary_cost = choose_modulation(primary_length, demand_gbps)
                    
                    # 计算备份路径长度和资源需求
                    backup_length = sum(G[u][v]['weight'] for u, v in zip(backup_path, backup_path[1:]))
                    backup_mod, backup_slots, backup_cost = choose_modulation(backup_length, demand_gbps)
                    
                    # 只有当主路径和备份路径都有可用调制格式时才添加
                    if primary_mod and backup_mod:
                        candidate_pairs.append({
                            'primary': {
                                'path': primary_path,
                                'mod': primary_mod,
                                'slots': primary_slots,
                                'cost': primary_cost,
                                'distance': primary_length
                            },
                            'backup': {
                                'path': backup_path,
                                'mod': backup_mod,
                                'slots': backup_slots,
                                'cost': backup_cost,
                                'distance': backup_length
                            }
                        })
                
                # 创建带1+1保护的请求
                if candidate_pairs:
                    protected_request = ProtectedRequest(req_id, source, target, candidate_pairs, original_demand=demand_gbps)
                    protected_requests.append(protected_request)
                    req_id += 1
        
        print(f"总共生成 {len(protected_requests)} 个带1+1保护的请求 (含分割后的并行请求)")
        return protected_requests


class ProtectedNetwork(Network):
    """扩展Network类，添加处理1+1保护的方法"""
    
    def remove_protected_assignment(self, primary_path, primary_start, primary_slots, 
                                    backup_path, backup_start, backup_slots):
        """
        移除主路径和备份路径的频谱分配
        """
        # 移除主路径的频谱分配
        self.remove_assignment(primary_path, primary_start, primary_slots)
        
        # 移除备份路径的频谱分配
        self.remove_assignment(backup_path, backup_start, backup_slots)


def simulated_annealing_protection(protected_requests, network, max_iter=10000, initial_T=1.0, cooling_rate=0.80):
    """
    针对1+1保护的模拟退火算法，为每个请求选择一对不相交路径并分配频谱资源
    
    参数:
        protected_requests: 带有1+1保护的请求列表
        network: 网络对象
        max_iter: 最大迭代次数
        initial_T: 初始温度
        cooling_rate: 冷却速率
        
    返回:
        solution: 每个请求的选择 {req_id: (pair_idx, primary_start, backup_start), ...}
        best_frag: 最佳解的网络最大垂直碎片
    """
    # 初始解：为每个请求选择第一对候选路径
    solution = {}
    successful_requests = []
    
    print("正在生成初始解(带1+1保护)...")
    for req in protected_requests:
        # 选择第一对候选路径
        if not req.candidate_pairs:
            print(f"❌ 请求 {req.request_id} 没有可行的路径对，无法提供1+1保护")
            continue
            
        chosen_pair = req.candidate_pairs[0]
        primary_path = chosen_pair['primary']['path']
        primary_slots = chosen_pair['primary']['slots']
        backup_path = chosen_pair['backup']['path']
        backup_slots = chosen_pair['backup']['slots']
        
        # 尝试为主路径分配频谱
        primary_start = network.assign_spectrum_on_path_least_used(primary_path, primary_slots)
        if primary_start is None:
            print(f"❌ 初始分配失败：请求 {req.request_id} 主路径 {primary_path} 无法分配频谱")
            continue
        
        # 尝试为备份路径分配频谱
        backup_start = network.assign_spectrum_on_path_least_used(backup_path, backup_slots)
        if backup_start is None:
            # 如果备份路径分配失败，需要先释放主路径的资源
            network.remove_assignment(primary_path, primary_start, primary_slots)
            print(f"❌ 初始分配失败：请求 {req.request_id} 备份路径 {backup_path} 无法分配频谱")
            continue
        
        # 主路径和备份路径都分配成功，记录到解中
        solution[req.request_id] = (0, primary_start, backup_start)
        successful_requests.append(req)
    
    if not successful_requests:
        print("⚠️ 所有请求均分配失败，无法进行模拟退火优化。")
        return {}, network.compute_network_fragmentation()
    
    # 记录当前和最佳的碎片情况
    current_frag = network.compute_network_fragmentation()
    best_frag = current_frag
    best_solution = copy.deepcopy(solution)
    T = initial_T
    
    print(f"初始解生成完成，开始模拟退火优化(带1+1保护)，最大迭代次数: {max_iter}...")
    progress_step = max(1, max_iter // 10)
    improved_count = 0
    
    # 模拟退火主循环
    for it in range(max_iter):
        # 打印进度
        if (it + 1) % progress_step == 0:
            print(f"模拟退火进度: {(it + 1) / max_iter * 100:.1f}%, 当前温度: {T:.6f}, 当前最佳碎片化: {best_frag:.6f}")
        
        # 如果温度很低且已运行半数以上迭代，可以提前结束
        if T < 1e-6 and it > max_iter // 2:
            print(f"温度已接近0，提前结束模拟退火，完成迭代次数: {it+1}/{max_iter}")
            break
        
        # 随机选择一个成功分配的请求
        req = random.choice(successful_requests)
        if req.request_id not in solution:
            continue
        
        current_pair_idx, current_primary_start, current_backup_start = solution[req.request_id]
        
        # 决定是更换路径对还是仅重新分配槽位 (50%概率)
        change_pair = random.random() < 0.5
        
        if change_pair and len(req.candidate_pairs) > 1:
            # 选择一个不同的路径对
            available_pair_indices = [i for i in range(len(req.candidate_pairs)) if i != current_pair_idx]
            new_pair_idx = random.choice(available_pair_indices)
            new_pair = req.candidate_pairs[new_pair_idx]
        else:
            # 保持相同的路径对，但尝试重新分配槽位
            new_pair_idx = current_pair_idx
            new_pair = req.candidate_pairs[current_pair_idx]
        
        # 获取新路径对的信息
        new_primary_path = new_pair['primary']['path']
        new_primary_slots = new_pair['primary']['slots']
        new_backup_path = new_pair['backup']['path']
        new_backup_slots = new_pair['backup']['slots']
        
        # 获取当前路径对的信息
        current_pair = req.candidate_pairs[current_pair_idx]
        current_primary_path = current_pair['primary']['path']
        current_primary_slots = current_pair['primary']['slots']
        current_backup_path = current_pair['backup']['path']
        current_backup_slots = current_pair['backup']['slots']
        
        # 备份当前网络状态
        network_backup = copy.deepcopy(network)
        
        # 移除当前分配
        network.remove_assignment(current_primary_path, current_primary_start, current_primary_slots)
        network.remove_assignment(current_backup_path, current_backup_start, current_backup_slots)
        
        # 尝试新的分配
        new_primary_start = network.assign_spectrum_on_path_least_used(new_primary_path, new_primary_slots)
        if new_primary_start is None:
            # 主路径分配失败，回退到原来的状态
            network = network_backup
            continue
        
        new_backup_start = network.assign_spectrum_on_path_least_used(new_backup_path, new_backup_slots)
        if new_backup_start is None:
            # 备份路径分配失败，先删除刚分配的主路径，然后回退到原来的状态
            network.remove_assignment(new_primary_path, new_primary_start, new_primary_slots)
            network = network_backup
            continue
        
        # 计算新解的碎片化程度
        new_frag = network.compute_network_fragmentation()
        delta = new_frag - current_frag
        
        # 根据Metropolis准则决定是否接受新解
        if delta < 0 or random.random() < math.exp(-delta / T):
            # 接受新解
            solution[req.request_id] = (new_pair_idx, new_primary_start, new_backup_start)
            current_frag = new_frag
            if new_frag < best_frag:
                best_frag = new_frag
                best_solution = copy.deepcopy(solution)
                improved_count += 1
        else:
            # 不接受新解，恢复原来的状态
            network = network_backup
            solution[req.request_id] = (current_pair_idx, current_primary_start, current_backup_start)
        
        # 降温
        T *= cooling_rate
    
    print(f"✅ 模拟退火完成(带1+1保护)：最佳碎片化度量 = {best_frag}，总改进次数 = {improved_count}")
    return best_solution, best_frag
