from modules.load_file import load_topology, load_traffic_matrix
from modules.modulation import choose_modulation
from modules.k_shortest_path import k_shortest_paths
from modules.benchmarking_FL import allocate_slots_least_used, calculate_max_vertical_fragmentation, slot_grid
from modules.visualization import  plot_spectrum_heatmap
from modules.optimization_RMSA import Network, Request, simulated_annealing, ProtectedRequest, ProtectedNetwork, simulated_annealing_protection
from modules.metrics import get_all_metrics, print_metrics_comparison

import pandas as pd
import time  # 添加时间模块用于计时
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def run_benchmark(G, traffic_matrix, order="descending"):
    sorted_traffic = sorted(traffic_matrix, key=lambda x: x[2], reverse=(order == "descending"))
    results = []

    for source, target, demand_gbps in sorted_traffic:
        paths = k_shortest_paths(G, source, target, k=5)
        path = paths[0]

        path_length = sum(G[u][v]['weight'] for u, v in zip(path, path[1:]))
        mod, total_slots, total_cost = choose_modulation(path_length, demand_gbps)
        results.append({"source": source, "target": target,  "path": path,
                            "distance": path_length, "modulation": mod, "slots": total_slots, "cost": total_cost})

    results = allocate_slots_least_used(G, results)

    if any(item.get("slot_start") is None for item in results):
        print("有请求的 slot_start 为 None")
    else:
        print("所有请求的 slot_start 都已正确分配")

    entropy = calculate_max_vertical_fragmentation(slot_grid)
    print(f"基准算法 - 最大垂直碎片化: {entropy}")
    return pd.DataFrame(results)


if __name__ == "__main__":
    G7_topoly = load_topology(BASE_DIR / 'data' / 'IT10-topology.txt')
    G7_matrix_1 = load_traffic_matrix(BASE_DIR / 'data' / 'IT10-matrix-4.txt')

    # 运行基准算法
    print("\n===== 正在运行基准算法 =====")
    start_time = time.time()
    df_benchmark = run_benchmark(G7_topoly, G7_matrix_1, order="ascending")
    benchmark_time = time.time() - start_time
    print(f"基准算法运行时间: {benchmark_time:.2f}秒")
    
    plot_spectrum_heatmap(slot_grid, "benchmark_spectrum.png")
    
    # 计算基准算法的指标
    benchmark_slot_grid = slot_grid.copy()
    benchmark_results = df_benchmark.to_dict('records')
    benchmark_metrics = get_all_metrics(benchmark_slot_grid, benchmark_results)
    
    # 输出基准算法的主要指标
    print("\n基准算法性能指标:")
    print(f"最高使用的FSU: {benchmark_metrics['highest_used_fsu']}")
    print(f"FSU总使用量: {benchmark_metrics['total_used_fsus']}")
    print(f"平均链路使用率: {benchmark_metrics['avg_link_usage']:.4f}")
    print(f"频谱利用率熵: {benchmark_metrics['utilization_entropy']:.4f}")
    print(f"香农熵: {benchmark_metrics['shannon_entropy']:.4f}")

    # 运行模拟退火算法
    print("\n===== 正在运行模拟退火算法 =====")
    network = Network(G7_topoly)
    requests = Request.generate_requests_from_matrix(G7_topoly, G7_matrix_1, k=5, order="ascending")

    # 减少迭代次数，提高冷却速率以加快收敛
    start_time = time.time()
    best_solution, best_frag = simulated_annealing(
        requests=requests,
        network=network,
        max_iter=1000,  # 减少迭代次数，从10000降低到1000
        initial_T=1.0,
        cooling_rate=0.88  # 略微提高冷却速率以加快收敛
    )
    sa_time = time.time() - start_time
    print(f"模拟退火算法运行时间: {sa_time:.2f}秒")

    # 创建SA结果列表，用于计算性能指标
    sa_results = []
    sa_slot_grid = {}
    
    # 初始化SA的slot_grid (与基准算法相同的结构)
    for u, v in G7_topoly.edges():
        key = (min(u, v), max(u, v))
        if key not in sa_slot_grid:
            sa_slot_grid[key] = [0] * 320  # 使用与基准算法相同的槽位数
    
    # 填充SA结果和slot_grid
    print("\n模拟退火算法结果:")
    successful_count = 0
    for req in requests:
        if req.request_id in best_solution:
            successful_count += 1
            chosen_index, start_slot = best_solution[req.request_id]
            info = req.candidate_info[chosen_index]
            path = info['path']
            slots_needed = info['slots']
            
            # 更新SA的slot_grid
            for u, v in zip(path, path[1:]):
                key = (min(u, v), max(u, v))
                for i in range(slots_needed):
                    sa_slot_grid[key][start_slot + i] = 1
            
            # 添加到结果列表
            sa_results.append({
                "source": req.source, 
                "target": req.target, 
                "path": path,
                "distance": sum(G7_topoly[u][v]['weight'] for u, v in zip(path, path[1:])),
                "modulation": info['mod'],
                "slots": slots_needed,
                "cost": info.get('cost', 0),  # 如果没有cost，默认为0
                "slot_start": start_slot,
                "slot_end": start_slot + slots_needed - 1
            })
            
            print(f"请求 {req.request_id}: 路径 {path}, 调制 {info['mod']}, 槽位数 {slots_needed}, 起始槽位 {start_slot}")
        else:
            print(f"请求 {req.request_id} 未能成功分配")
    
    print(f"成功分配请求数量: {successful_count}/{len(requests)}")
    print(f"模拟退火算法 - 最大碎片化: {best_frag}")
    
    # 计算SA算法的指标
    sa_metrics = get_all_metrics(sa_slot_grid, sa_results)
    
    # 输出SA算法的主要指标
    print("\n模拟退火算法性能指标:")
    print(f"最高使用的FSU: {sa_metrics['highest_used_fsu']}")
    print(f"FSU总使用量: {sa_metrics['total_used_fsus']}")
    print(f"平均链路使用率: {sa_metrics['avg_link_usage']:.4f}")
    print(f"频谱利用率熵: {sa_metrics['utilization_entropy']:.4f}")
    print(f"香农熵: {sa_metrics['shannon_entropy']:.4f}")
    
    # 绘制SA算法的频谱热图
    plot_spectrum_heatmap(sa_slot_grid, "sa_spectrum.png")
    
    # 比较两种算法的性能指标
    print_metrics_comparison(benchmark_metrics, sa_metrics)

# ----------------- 1+1专用路径保护算法函数 -----------------


def run_sa_with_protection(G, traffic_matrix, benchmark_metrics=None, k=5, order="ascending"):
    """
    运行带有1+1专用路径保护的模拟退火算法
    
    参数:
        G: 网络拓扑图
        traffic_matrix: 流量矩阵
        benchmark_metrics: 基准算法的性能指标（用于比较）
        k: 每对(source, target)找k条最短路径
        order: 流量排序方式
        
    返回:
        保护算法的结果和性能指标
    """
    print("\n===== 正在运行1+1专用路径保护算法 =====")
    
    # 创建保护网络实例
    protected_network = ProtectedNetwork(G)
    
    # 生成带保护的请求列表
    start_time = time.time()
    protected_requests = ProtectedRequest.generate_protected_requests_from_matrix(
        G, traffic_matrix, k=k, order=order, max_pairs=3
    )
    
    print(f"总共生成 {len(protected_requests)} 个带1+1保护的请求")
    
    # 运行带保护的模拟退火算法
    best_solution, best_frag = simulated_annealing_protection(
        protected_requests=protected_requests,
        network=protected_network,
        max_iter=1000,
        initial_T=1.0,
        cooling_rate=0.88
    )
    protection_time = time.time() - start_time
    print(f"1+1保护算法运行时间: {protection_time:.2f}秒")
    
    # 创建保护算法结果列表
    protection_results = []
    protection_slot_grid = {}
    
    # 初始化保护算法的slot_grid
    for u, v in G.edges():
        key = (min(u, v), max(u, v))
        if key not in protection_slot_grid:
            protection_slot_grid[key] = [0] * 320
    
    # 填充保护算法结果和slot_grid
    print("\n1+1保护算法结果:")
    successful_count = 0
    for req in protected_requests:
        if req.request_id in best_solution:
            successful_count += 1
            pair_idx, primary_start, backup_start = best_solution[req.request_id]
            chosen_pair = req.candidate_pairs[pair_idx]
            
            # 主路径信息
            primary_info = chosen_pair['primary']
            primary_path = primary_info['path']
            primary_slots = primary_info['slots']
            
            # 备份路径信息
            backup_info = chosen_pair['backup']
            backup_path = backup_info['path']
            backup_slots = backup_info['slots']
            
            # 更新保护算法的slot_grid（主路径）
            for u, v in zip(primary_path, primary_path[1:]):
                key = (min(u, v), max(u, v))
                for i in range(primary_slots):
                    protection_slot_grid[key][primary_start + i] = 1
            
            # 更新保护算法的slot_grid（备份路径）
            for u, v in zip(backup_path, backup_path[1:]):
                key = (min(u, v), max(u, v))
                for i in range(backup_slots):
                    protection_slot_grid[key][backup_start + i] = 1
            
            # 添加主路径到结果列表
            protection_results.append({
                "source": req.source,
                "target": req.target,
                "path": primary_path,
                "distance": sum(G[u][v]['weight'] for u, v in zip(primary_path, primary_path[1:])),
                "modulation": primary_info['mod'],
                "slots": primary_slots,
                "cost": primary_info['cost'],
                "slot_start": primary_start,
                "slot_end": primary_start + primary_slots - 1,
                "path_type": "primary"
            })
            
            # 添加备份路径到结果列表
            protection_results.append({
                "source": req.source,
                "target": req.target,
                "path": backup_path,
                "distance": sum(G[u][v]['weight'] for u, v in zip(backup_path, backup_path[1:])),
                "modulation": backup_info['mod'],
                "slots": backup_slots,
                "cost": backup_info['cost'],
                "slot_start": backup_start,
                "slot_end": backup_start + backup_slots - 1,
                "path_type": "backup"
            })
            
            print(f"请求 {req.request_id}:")
            print(f"  主路径: {primary_path}, 调制 {primary_info['mod']}, 槽位数 {primary_slots}, 起始槽位 {primary_start}")
            print(f"  备份路径: {backup_path}, 调制 {backup_info['mod']}, 槽位数 {backup_slots}, 起始槽位 {backup_start}")
        else:
            print(f"请求 {req.request_id} 未能成功分配")
    
    print(f"成功分配请求数量: {successful_count}/{len(protected_requests)}")
    print(f"1+1保护算法 - 最大碎片化: {best_frag}")
    
    # 绘制保护算法的频谱热图
    plot_spectrum_heatmap(protection_slot_grid, "protection_spectrum.png")
    
    # 计算保护算法的指标
    protection_metrics = get_all_metrics(protection_slot_grid, protection_results)
    
    # 输出保护算法的主要指标
    print("\n1+1保护算法性能指标:")
    print(f"最高使用的FSU: {protection_metrics['highest_used_fsu']}")
    print(f"FSU总使用量: {protection_metrics['total_used_fsus']}")
    print(f"平均链路使用率: {protection_metrics['avg_link_usage']:.4f}")
    print(f"频谱利用率熵: {protection_metrics['utilization_entropy']:.4f}")
    print(f"香农熵: {protection_metrics['shannon_entropy']:.4f}")
    
    # 如果提供了基准算法的指标，则比较两种算法
    if benchmark_metrics:
        print("\n" + "="*50)
        print("性能指标比较: 基准算法 vs 1+1保护算法")
        print("="*50)
        print_metrics_comparison(benchmark_metrics, protection_metrics)
    
    return protection_results, protection_metrics


# 如果想在主函数中启用1+1保护算法，可以取消以下注释
"""
if __name__ == "__main__":
    # 前面的代码保持不变
    # ...
    
    # 在原有算法运行完后，运行1+1保护算法
    protection_results, protection_metrics = run_sa_with_protection(G7_topoly, G7_matrix_1, benchmark_metrics)
    
    # 比较所有三种算法的性能指标
    print("\n" + "="*50)
    print("性能指标比较: 基准算法 vs 模拟退火算法 vs 1+1保护算法")
    print("="*50)
    
    # 创建一个函数来比较三种算法
    def print_three_algorithms_comparison(benchmark_metrics, sa_metrics, protection_metrics):
        metrics = [
            ("最高使用的FSU", "highest_used_fsu"),
            ("FSU总使用量", "total_used_fsus"),
            ("平均链路使用率", "avg_link_usage"),
            ("平均路径长度", "avg_path_length"),
            ("平均转发器成本", "avg_transponder_cost"),
            ("利用率熵", "utilization_entropy"),
            ("香农熵", "shannon_entropy"),
            ("最大垂直碎片化", "max_fragmentation")
        ]
        
        for metric_name, metric_key in metrics:
            if metric_key in benchmark_metrics and metric_key in sa_metrics and metric_key in protection_metrics:
                benchmark_value = benchmark_metrics[metric_key]
                sa_value = sa_metrics[metric_key]
                protection_value = protection_metrics[metric_key]
                
                print(f"{metric_name}:")
                print(f"  基准算法: {benchmark_value:.4f}")
                print(f"  模拟退火: {sa_value:.4f} ({((sa_value - benchmark_value) / benchmark_value) * 100:.2f}%)")
                print(f"  1+1保护: {protection_value:.4f} ({((protection_value - benchmark_value) / benchmark_value) * 100:.2f}%)")
                print("-"*50)
    
    print_three_algorithms_comparison(benchmark_metrics, sa_metrics, protection_metrics)
"""
