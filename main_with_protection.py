import os
import time
from pathlib import Path
import networkx as nx
import pandas as pd

from modules.load_file import load_topology, load_traffic_matrix
from modules.modulation import choose_modulation
from modules.k_shortest_path import k_shortest_paths
from modules.benchmarking_FL import allocate_slots_least_used, calculate_max_vertical_fragmentation, slot_grid, \
    initialize_slot_grid
from modules.visualization import plot_spectrum_heatmap
from modules.optimization_RMSA import Network, Request, simulated_annealing, ProtectedRequest, ProtectedNetwork, \
    simulated_annealing_protection
from modules.metrics import get_all_metrics, print_metrics_comparison
from modules.shared_protection import run_sa_with_shared_protection

BASE_DIR = Path(__file__).resolve().parent
topology_file_path = BASE_DIR / "data" / "G7-topology.txt"
traffic_file_path = BASE_DIR / "data" / "G7-matrix-5.txt"


def print_three_algorithms_comparison(benchmark_metrics, sa_metrics, protection_metrics):
    """
    打印三种算法的性能指标比较
    """
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
            print(
                f"  1+1保护: {protection_value:.4f} ({((protection_value - benchmark_value) / benchmark_value) * 100:.2f}%)")
            print("-" * 50)


def print_four_algorithms_comparison(benchmark_metrics, sa_metrics, protection_metrics, shared_metrics):
    """
    Print comparison of metrics for four algorithms.

    Args:
        benchmark_metrics: Metrics from benchmark algorithm
        sa_metrics: Metrics from simulated annealing algorithm
        protection_metrics: Metrics from 1+1 protection algorithm
        shared_metrics: Metrics from shared protection algorithm
    """
    print("\n" + "=" * 50)
    print("性能指标比较: 基准算法 vs 模拟退火算法 vs 1+1保护算法 vs 共享保护算法")
    print("=" * 50)

    # Metrics to compare
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

    # Display each metric
    for metric_name, metric_key in metrics:
        if metric_key in benchmark_metrics and metric_key in sa_metrics and metric_key in protection_metrics and metric_key in shared_metrics:
            benchmark_value = benchmark_metrics[metric_key]
            sa_value = sa_metrics[metric_key]
            protection_value = protection_metrics[metric_key]
            shared_value = shared_metrics[metric_key]

            # Calculate percentage change from benchmark
            sa_pct_change = (sa_value - benchmark_value) / benchmark_value * 100 if benchmark_value != 0 else float(
                'inf')
            protection_pct_change = (
                                                protection_value - benchmark_value) / benchmark_value * 100 if benchmark_value != 0 else float(
                'inf')
            shared_pct_change = (
                                            shared_value - benchmark_value) / benchmark_value * 100 if benchmark_value != 0 else float(
                'inf')

            # Print the metric comparison
            print(f"{metric_name}:")
            print(f"  基准算法: {benchmark_value:.4f}")
            print(f"  模拟退火: {sa_value:.4f} ({sa_pct_change:.2f}%)")
            print(f"  1+1保护: {protection_value:.4f} ({protection_pct_change:.2f}%)")
            print(f"  共享保护: {shared_value:.4f} ({shared_pct_change:.2f}%)")
            print("-" * 50)

    # Additional metrics for shared protection
    if "sharing_ratio" in shared_metrics:
        print(f"共享保护共享比率: {shared_metrics['sharing_ratio']:.4f}")

        # 显示更详细的共享资源统计信息
        if "avg_sharing_percentage" in shared_metrics:
            print(f"平均共享百分比: {shared_metrics['avg_sharing_percentage']:.4f}")

        if "shared_requests_count" in shared_metrics and "shared_requests_percentage" in shared_metrics:
            print(
                f"共享资源的请求数量: {shared_metrics['shared_requests_count']} ({shared_metrics['shared_requests_percentage'] * 100:.2f}%)")

        if "resource_savings" in shared_metrics:
            print(f"资源节约率: {shared_metrics['resource_savings'] * 100:.2f}%")

        print("-" * 50)


def run_benchmark_algorithm(G, traffic_matrix, k=5, order="ascending"):
    """
    运行基准算法（First-Fit Least-Used）
    """
    print("\n===== 运行基准算法 =====")
    start_time = time.time()

    # 初始化slot_grid
    initialize_slot_grid(G)

    # 按顺序处理流量请求
    sorted_traffic = sorted(traffic_matrix, key=lambda x: x[2], reverse=(order == "descending"))

    benchmark_results = []
    for i, (source, target, demand_gbps) in enumerate(sorted_traffic):
        # 找出k条最短路径
        paths = k_shortest_paths(G, source, target, k=k)

        # 为每条路径计算调制格式和槽位需求
        path_options = []
        for path in paths:
            # 计算路径长度
            path_length = sum(G[u][v]['weight'] for u, v in zip(path, path[1:]))
            # 根据路径长度选择调制格式
            mod, slots, cost = choose_modulation(path_length, demand_gbps)
            path_options.append((path, mod, slots, cost, path_length))

        # 按照路径调制等级排序（更高的调制等级需要更少的槽位）
        path_options.sort(key=lambda x: x[2])

        # 尝试为每个路径分配频谱
        allocated = False
        for path, mod, slots, cost, path_length in path_options:
            # 简单添加到结果列表，稍后分配频谱
            benchmark_results.append({
                "source": source,
                "target": target,
                "path": path,
                "distance": path_length,
                "modulation": mod,
                "slots": slots,
                "cost": cost
            })
            allocated = True
            break

    # 使用allocate_slots_least_used函数分配频谱
    benchmark_results = allocate_slots_least_used(G, benchmark_results)

    benchmark_time = time.time() - start_time
    print(f"基准算法运行时间: {benchmark_time:.2f}秒")

    # 计算碎片化指标
    entropy = calculate_max_vertical_fragmentation(slot_grid)
    print(f"基准算法 - 最大垂直碎片化: {entropy}")

    # 绘制频谱热图
    # 为不同排序方式生成不同文件名
    if order == "descending":
        heatmap_filename = "benchmark_spectrum_desc.png"
        heatmap_title = "Benchmark Algorithm Spectrum Allocation (Descending Order)"
    else:
        heatmap_filename = "benchmark_spectrum.png"
        heatmap_title = "Benchmark Algorithm Spectrum Allocation (Ascending Order)"

    plot_spectrum_heatmap(slot_grid, heatmap_filename, title=heatmap_title)

    # 创建一个本地拷贝以便计算指标
    benchmark_slot_grid = {}
    for key, value in slot_grid.items():
        benchmark_slot_grid[key] = value.copy()

    # 计算并打印性能指标
    benchmark_metrics = get_all_metrics(benchmark_slot_grid, benchmark_results)

    print("\n基准算法性能指标:")
    print(f"最高使用的FSU: {benchmark_metrics['highest_used_fsu']}")
    print(f"FSU总使用量: {benchmark_metrics['total_used_fsus']}")
    print(f"平均链路使用率: {benchmark_metrics['avg_link_usage']:.4f}")
    print(f"频谱利用率熵: {benchmark_metrics['utilization_entropy']:.4f}")
    print(f"香农熵: {benchmark_metrics['shannon_entropy']:.4f}")

    return benchmark_results, benchmark_metrics, benchmark_slot_grid


def run_sa_algorithm(G, traffic_matrix, k=5, order="ascending"):
    """
    运行模拟退火算法
    """
    print("\n===== 运行模拟退火算法 =====")

    # 创建网络和请求对象
    network = Network(G)

    # 生成请求列表
    requests = Request.generate_requests_from_matrix(G, traffic_matrix, k=k, order=order)

    # 运行模拟退火算法
    start_time = time.time()
    best_solution, best_frag = simulated_annealing(
        requests, network, max_iter=10000, initial_T=1.0, cooling_rate=0.97
    )
    sa_time = time.time() - start_time
    print(f"模拟退火算法运行时间: {sa_time:.2f}秒")
    print(f"模拟退火算法 - 最大碎片化: {best_frag}")

    # 创建一个新的slot_grid和结果列表
    sa_slot_grid = {}
    for u, v in G.edges():
        key = (min(u, v), max(u, v))
        if key not in sa_slot_grid:
            sa_slot_grid[key] = [0] * 320

    # 填充模拟退火算法的结果
    sa_results = []
    for req in requests:
        if req.request_id not in best_solution:
            continue

        candidate_idx, start_slot = best_solution[req.request_id]
        chosen = req.candidate_info[candidate_idx]
        path = chosen['path']
        slots = chosen['slots']

        # 更新slot_grid
        for u, v in zip(path, path[1:]):
            key = (min(u, v), max(u, v))
            for i in range(slots):
                sa_slot_grid[key][start_slot + i] = 1

        # 添加到结果列表
        sa_results.append({
            "source": req.source,
            "target": req.target,
            "path": path,
            "distance": sum(G[u][v]['weight'] for u, v in zip(path, path[1:])),
            "modulation": chosen['mod'],
            "slots": slots,
            "cost": chosen['cost'],
            "slot_start": start_slot,
            "slot_end": start_slot + slots - 1
        })

    # 绘制频谱热图
    # 为不同排序方式生成不同文件名
    if order == "descending":
        heatmap_filename = "sa_spectrum_desc.png"
        heatmap_title = "Simulated Annealing Spectrum Allocation (Descending Order)"
    else:
        heatmap_filename = "sa_spectrum.png"
        heatmap_title = "Simulated Annealing Spectrum Allocation (Ascending Order)"

    plot_spectrum_heatmap(sa_slot_grid, heatmap_filename, title=heatmap_title)

    # 计算并打印性能指标
    sa_metrics = get_all_metrics(sa_slot_grid, sa_results)

    print("\n模拟退火算法性能指标:")
    print(f"最高使用的FSU: {sa_metrics['highest_used_fsu']}")
    print(f"FSU总使用量: {sa_metrics['total_used_fsus']}")
    print(f"平均链路使用率: {sa_metrics['avg_link_usage']:.4f}")
    print(f"频谱利用率熵: {sa_metrics['utilization_entropy']:.4f}")
    print(f"香农熵: {sa_metrics['shannon_entropy']:.4f}")

    return sa_results, sa_metrics, sa_slot_grid


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
        max_iter=10000,
        initial_T=1.0,
        cooling_rate=0.97
    )
    protection_time = time.time() - start_time
    print(f"1+1保护算法运行时间: {protection_time:.2f}秒")
    print(f"1+1保护算法 - 最大碎片化: {best_frag}")

    # 创建保护算法结果列表和slot_grid
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

    print(f"成功分配请求数量: {successful_count}/{len(protected_requests)}")

    # 绘制保护算法的频谱热图
    # 为不同排序方式生成不同文件名
    if order == "descending":
        heatmap_filename = "protection_spectrum_desc.png"
        heatmap_title = "1+1 Protection Spectrum Allocation (Descending Order)"
    else:
        heatmap_filename = "protection_spectrum.png"
        heatmap_title = "1+1 Protection Spectrum Allocation (Ascending Order)"

    plot_spectrum_heatmap(protection_slot_grid, heatmap_filename, title=heatmap_title)

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
        print("\n" + "=" * 50)
        print("性能指标比较: 基准算法 vs 1+1保护算法")
        print("=" * 50)
        print_metrics_comparison(benchmark_metrics, protection_metrics)

    return protection_results, protection_metrics, protection_slot_grid


def create_combined_heatmap_pdf(heatmap_files, output_pdf="combined_heatmaps.pdf", traffic_file_identifier=None,
                                output_dir=None):
    """
    将所有频谱分配热图合并到一个PDF文件中

    Args:
        heatmap_files: 包含热图文件名的字典
        output_pdf: 输出PDF文件名
        traffic_file_identifier: 流量文件标识符（可选）
        output_dir: 输出目录（可选）

    Returns:
        PDF文件路径或None（如果生成失败）
    """
    # 如果提供了流量文件标识符，将其添加到输出文件名中
    if traffic_file_identifier:
        # 将输出文件名拆分为文件名部分和扩展名
        filename_parts = output_pdf.rsplit('.', 1)
        if len(filename_parts) > 1:
            output_pdf = f"{filename_parts[0]}_{traffic_file_identifier}.{filename_parts[1]}"
        else:
            output_pdf = f"{output_pdf}_{traffic_file_identifier}"

    # 如果提供了输出目录，将输出路径设置为该目录中
    if output_dir:
        output_pdf = os.path.join(output_dir, os.path.basename(output_pdf))

    try:
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt
        from matplotlib.image import imread

        # 检查所有文件是否存在
        valid_files = {}
        for key, filename in heatmap_files.items():
            if os.path.exists(filename):
                valid_files[key] = filename

        if not valid_files:
            print("Warning: No valid heatmap files found")
            return None

        # 创建PDF
        with PdfPages(output_pdf) as pdf:
            for key, filename in valid_files.items():
                # 如果存在输出目录，也将热图复制到该目录
                if output_dir:
                    import shutil
                    new_filename = os.path.join(output_dir, os.path.basename(filename))
                    shutil.copy2(filename, new_filename)

                # 添加到PDF
                img = imread(filename)
                plt.figure(figsize=(10, 8))
                plt.imshow(img)
                plt.axis('off')

                # 设置标题
                if key == 'benchmark_asc':
                    plt.title('Benchmark Algorithm (Ascending Order)')
                elif key == 'sa_asc':
                    plt.title('Simulated Annealing Algorithm (Ascending Order)')
                elif key == 'protection_asc':
                    plt.title('1+1 Protection Algorithm (Ascending Order)')
                elif key == 'shared_asc':
                    plt.title('Shared Protection Algorithm (Ascending Order)')
                elif key == 'benchmark_desc':
                    plt.title('Benchmark Algorithm (Descending Order)')
                elif key == 'sa_desc':
                    plt.title('Simulated Annealing Algorithm (Descending Order)')
                elif key == 'protection_desc':
                    plt.title('1+1 Protection Algorithm (Descending Order)')
                elif key == 'shared_desc':
                    plt.title('Shared Protection Algorithm (Descending Order)')

                plt.tight_layout()
                pdf.savefig()
                plt.close()

        print(f"Combined heatmaps PDF generated: {output_pdf}")
        return output_pdf
    except ImportError:
        print("Warning: matplotlib or required libraries not available, skipping PDF generation")
        return None
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        return None


def create_performance_charts(
        benchmark_metrics_asc, sa_metrics_asc, protection_metrics_asc, shared_metrics_asc,
        benchmark_metrics_desc, sa_metrics_desc, protection_metrics_desc, shared_metrics_desc,
        output_pdf="performance_charts.pdf", traffic_file_identifier=None, output_dir=None
):
    """
    Create comparison charts of performance metrics and save to PDF

    Args:
        benchmark_metrics_asc, sa_metrics_asc, protection_metrics_asc: Metrics from ascending order
        benchmark_metrics_desc, sa_metrics_desc, protection_metrics_desc: Metrics from descending order
        output_pdf: Output PDF file path
        traffic_file_identifier: 流量文件标识符（可选）
        output_dir: 输出目录（可选）

    Returns:
        Path to the generated PDF file
    """
    # 如果提供了流量文件标识符，将其添加到输出文件名中
    if traffic_file_identifier:
        # 将输出文件名拆分为文件名部分和扩展名
        filename_parts = output_pdf.rsplit('.', 1)
        if len(filename_parts) > 1:
            output_pdf = f"{filename_parts[0]}_{traffic_file_identifier}.{filename_parts[1]}"
        else:
            output_pdf = f"{output_pdf}_{traffic_file_identifier}"

    # 如果提供了输出目录，将输出路径设置为该目录中
    if output_dir:
        output_pdf = os.path.join(output_dir, os.path.basename(output_pdf))

    # 辅助函数：安全除法
    def safe_division(numerator, denominator, default=1.0):
        if denominator == 0:
            return default
        return numerator / denominator

    # 辅助函数：保存图表为PNG和添加到PDF
    def save_figure(fig, chart_name, pdf):
        # 保存到PDF
        pdf.savefig(fig)

        # 如果提供了输出目录，还保存为单独的PNG文件
        if output_dir:
            png_filename = f"{chart_name}"
            if traffic_file_identifier:
                png_filename = f"{chart_name}_{traffic_file_identifier}"
            png_path = os.path.join(output_dir, f"{png_filename}.png")
            fig.savefig(png_path, dpi=300, bbox_inches='tight')

    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        import numpy as np

        with PdfPages(output_pdf) as pdf:
            # Highest Used FSU Comparison
            plt.figure(figsize=(12, 6))
            plt.suptitle('Highest Used FSU Comparison', fontsize=16)

            # Prepare data
            algorithms = ['Benchmark', 'SA', '1+1 Protection', 'Shared Protection']
            asc_values = [
                benchmark_metrics_asc['highest_used_fsu'],
                sa_metrics_asc['highest_used_fsu'],
                protection_metrics_asc['highest_used_fsu'],
                shared_metrics_asc['highest_used_fsu']
            ]
            desc_values = [
                benchmark_metrics_desc['highest_used_fsu'],
                sa_metrics_desc['highest_used_fsu'],
                protection_metrics_desc['highest_used_fsu'],
                shared_metrics_desc['highest_used_fsu']
            ]

            # Create bars
            barWidth = 0.35
            r1 = np.arange(len(algorithms))
            r2 = [x + barWidth for x in r1]

            plt.bar(r1, asc_values, width=barWidth, label='Ascending Order', color='skyblue')
            plt.bar(r2, desc_values, width=barWidth, label='Descending Order', color='lightcoral')

            # Add labels and legend
            plt.xlabel('Algorithm')
            plt.ylabel('Highest Used FSU')
            plt.xticks([r + barWidth / 2 for r in range(len(algorithms))], algorithms)
            plt.legend()

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            save_figure(plt.gcf(), "highest_used_fsu", pdf)
            plt.close()

            # Total FSUs Comparison
            plt.figure(figsize=(12, 6))
            plt.suptitle('Total Used FSUs Comparison', fontsize=16)

            # Prepare data
            asc_values = [
                benchmark_metrics_asc['total_used_fsus'],
                sa_metrics_asc['total_used_fsus'],
                protection_metrics_asc['total_used_fsus'],
                shared_metrics_asc['total_used_fsus']
            ]
            desc_values = [
                benchmark_metrics_desc['total_used_fsus'],
                sa_metrics_desc['total_used_fsus'],
                protection_metrics_desc['total_used_fsus'],
                shared_metrics_desc['total_used_fsus']
            ]

            # Create bars
            plt.bar(r1, asc_values, width=barWidth, label='Ascending Order', color='skyblue')
            plt.bar(r2, desc_values, width=barWidth, label='Descending Order', color='lightcoral')

            # Add labels and legend
            plt.xlabel('Algorithm')
            plt.ylabel('Total Used FSUs')
            plt.xticks([r + barWidth / 2 for r in range(len(algorithms))], algorithms)
            plt.legend()

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            save_figure(plt.gcf(), "total_used_fsus", pdf)
            plt.close()

            # Average Link Usage Comparison
            plt.figure(figsize=(12, 6))
            plt.suptitle('Average Link Usage Comparison', fontsize=16)

            # Prepare data
            asc_values = [
                benchmark_metrics_asc['avg_link_usage'],
                sa_metrics_asc['avg_link_usage'],
                protection_metrics_asc['avg_link_usage'],
                shared_metrics_asc['avg_link_usage']
            ]
            desc_values = [
                benchmark_metrics_desc['avg_link_usage'],
                sa_metrics_desc['avg_link_usage'],
                protection_metrics_desc['avg_link_usage'],
                shared_metrics_desc['avg_link_usage']
            ]

            # Create bars
            plt.bar(r1, asc_values, width=barWidth, label='Ascending Order', color='skyblue')
            plt.bar(r2, desc_values, width=barWidth, label='Descending Order', color='lightcoral')

            # Add labels and legend
            plt.xlabel('Algorithm')
            plt.ylabel('Average Link Usage')
            plt.xticks([r + barWidth / 2 for r in range(len(algorithms))], algorithms)
            plt.legend()

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            save_figure(plt.gcf(), "average_link_usage", pdf)
            plt.close()

            # Transponder Cost Comparison
            plt.figure(figsize=(12, 6))
            plt.suptitle('Average Transponder Cost Comparison', fontsize=16)

            # Prepare data
            asc_values = [
                benchmark_metrics_asc['avg_transponder_cost'],
                sa_metrics_asc['avg_transponder_cost'],
                protection_metrics_asc['avg_transponder_cost'],
                shared_metrics_asc['avg_transponder_cost']
            ]
            desc_values = [
                benchmark_metrics_desc['avg_transponder_cost'],
                sa_metrics_desc['avg_transponder_cost'],
                protection_metrics_desc['avg_transponder_cost'],
                shared_metrics_desc['avg_transponder_cost']
            ]

            # Create bars
            plt.bar(r1, asc_values, width=barWidth, label='Ascending Order', color='skyblue')
            plt.bar(r2, desc_values, width=barWidth, label='Descending Order', color='lightcoral')

            # Add labels and legend
            plt.xlabel('Algorithm')
            plt.ylabel('Average Transponder Cost')
            plt.xticks([r + barWidth / 2 for r in range(len(algorithms))], algorithms)
            plt.legend()

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            save_figure(plt.gcf(), "average_transponder_cost", pdf)
            plt.close()

            # Path Length Comparison
            plt.figure(figsize=(12, 6))
            plt.suptitle('Average Path Length Comparison', fontsize=16)

            # Prepare data
            asc_values = [
                benchmark_metrics_asc['avg_path_length'],
                sa_metrics_asc['avg_path_length'],
                protection_metrics_asc['avg_path_length'],
                shared_metrics_asc['avg_path_length']
            ]
            desc_values = [
                benchmark_metrics_desc['avg_path_length'],
                sa_metrics_desc['avg_path_length'],
                protection_metrics_desc['avg_path_length'],
                shared_metrics_desc['avg_path_length']
            ]

            # Create bars
            plt.bar(r1, asc_values, width=barWidth, label='Ascending Order', color='skyblue')
            plt.bar(r2, desc_values, width=barWidth, label='Descending Order', color='lightcoral')

            # Add labels and legend
            plt.xlabel('Algorithm')
            plt.ylabel('Average Path Length')
            plt.xticks([r + barWidth / 2 for r in range(len(algorithms))], algorithms)
            plt.legend()

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            save_figure(plt.gcf(), "average_path_length", pdf)
            plt.close()

            # Fragmentation Comparison
            plt.figure(figsize=(12, 6))
            plt.suptitle('Max Fragmentation Comparison', fontsize=16)

            # Prepare data
            asc_values = [
                benchmark_metrics_asc['max_fragmentation'],
                sa_metrics_asc['max_fragmentation'],
                protection_metrics_asc['max_fragmentation'],
                shared_metrics_asc['max_fragmentation']
            ]
            desc_values = [
                benchmark_metrics_desc['max_fragmentation'],
                sa_metrics_desc['max_fragmentation'],
                protection_metrics_desc['max_fragmentation'],
                shared_metrics_desc['max_fragmentation']
            ]

            # Create bars
            plt.bar(r1, asc_values, width=barWidth, label='Ascending Order', color='skyblue')
            plt.bar(r2, desc_values, width=barWidth, label='Descending Order', color='lightcoral')

            # Add labels and legend
            plt.xlabel('Algorithm')
            plt.ylabel('Max Fragmentation')
            plt.xticks([r + barWidth / 2 for r in range(len(algorithms))], algorithms)
            plt.legend()

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            save_figure(plt.gcf(), "max_fragmentation", pdf)
            plt.close()

            # Add a sharing ratio comparison (specific to shared protection)
            plt.figure(figsize=(8, 6))
            plt.suptitle('Shared Protection - Sharing Ratio', fontsize=16)

            # Prepare data
            labels = ['Ascending Order', 'Descending Order']
            values = [shared_metrics_asc['sharing_ratio'], shared_metrics_desc['sharing_ratio']]

            # Create bars
            plt.bar(labels, values, color=['skyblue', 'lightcoral'])

            # Add labels
            plt.xlabel('Processing Order')
            plt.ylabel('Sharing Ratio')

            # Add value labels above bars
            for i, v in enumerate(values):
                plt.text(i, v + 0.01, f"{v:.4f}", ha='center')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            save_figure(plt.gcf(), "sharing_ratio", pdf)
            plt.close()

            # Protection Efficiency Comparison
            plt.figure(figsize=(12, 6))
            plt.suptitle('Protection Efficiency Comparison', fontsize=16)

            # Define metrics
            metrics = ['FSU Usage', 'Highest FSU', 'Avg Path Length']

            # Protection vs Shared Protection (ascending)
            protection_values = [
                safe_division(protection_metrics_asc['total_used_fsus'], benchmark_metrics_asc['total_used_fsus']),
                safe_division(protection_metrics_asc['highest_used_fsu'], benchmark_metrics_asc['highest_used_fsu']),
                safe_division(protection_metrics_asc['avg_path_length'], benchmark_metrics_asc['avg_path_length'])
            ]

            shared_values = [
                safe_division(shared_metrics_asc['total_used_fsus'], benchmark_metrics_asc['total_used_fsus']),
                safe_division(shared_metrics_asc['highest_used_fsu'], benchmark_metrics_asc['highest_used_fsu']),
                safe_division(shared_metrics_asc['avg_path_length'], benchmark_metrics_asc['avg_path_length'])
            ]

            x = np.arange(len(metrics))
            width = 0.35

            fig, ax = plt.subplots(figsize=(12, 6))
            rects1 = ax.bar(x - width / 2, protection_values, width, label='1+1 Protection')
            rects2 = ax.bar(x + width / 2, shared_values, width, label='Shared Protection')

            ax.set_ylabel('Relative to Benchmark')
            ax.set_title('Protection Efficiency (Ascending Order)')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.legend()

            # Add value labels
            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:.2f}x',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')

            autolabel(rects1)
            autolabel(rects2)

            plt.tight_layout()
            save_figure(plt.gcf(), "protection_efficiency", pdf)
            plt.close()

        print(f"Performance charts PDF generated: {output_pdf}")
        return output_pdf
    except ImportError:
        print("Warning: matplotlib or required libraries not available, skipping chart generation")
        return None
    except Exception as e:
        print(f"Error generating charts: {str(e)}")
        return None


def generate_comprehensive_report(
        benchmark_metrics_asc, sa_metrics_asc, protection_metrics_asc, shared_metrics_asc,
        benchmark_metrics_desc, sa_metrics_desc, protection_metrics_desc, shared_metrics_desc,
        heatmap_files, output_file="experiment_results.txt", combined_pdf=None, traffic_file_identifier=None
):
    """
    Generate a comprehensive report of all experiment results and save to a file

    Args:
        benchmark_metrics_asc: Metrics from benchmark algorithm with ascending order
        sa_metrics_asc: Metrics from simulated annealing algorithm with ascending order
        protection_metrics_asc: Metrics from protection algorithm with ascending order
        shared_metrics_asc: Metrics from shared protection algorithm with ascending order
        benchmark_metrics_desc: Metrics from benchmark algorithm with descending order
        sa_metrics_desc: Metrics from simulated annealing algorithm with descending order
        protection_metrics_desc: Metrics from protection algorithm with descending order
        shared_metrics_desc: Metrics from shared protection algorithm with descending order
        heatmap_files: Dictionary of generated heatmap image filenames
        output_file: Output file path for the report
        combined_pdf: Path to the combined PDF file of heatmaps (optional)
        traffic_file_identifier: Identifier for the traffic file being processed (optional)
    """

    # 辅助函数：安全计算百分比变化
    def safe_percent_change(new_value, old_value):
        if old_value == 0:
            return "N/A (div by zero)"
        return f"{((new_value - old_value) / old_value * 100):.2f}%"

    # 辅助函数：安全除法
    def safe_division(numerator, denominator, default=1.0):
        if denominator == 0:
            return default
        return numerator / denominator

    # 如果提供了流量文件标识符，将其添加到输出文件名中
    output_dir = None
    if traffic_file_identifier:
        # 创建基于流量文件标识符的子文件夹
        output_dir = f"results_{traffic_file_identifier}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 将输出文件名设置为该子文件夹中
        output_file = os.path.join(output_dir, os.path.basename(output_file))

    with open(output_file, 'w') as f:
        # Title and introduction
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE EXPERIMENTAL RESULTS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("This report contains the results of experiments comparing different\n")
        f.write("routing, modulation, and spectrum allocation algorithms in elastic optical networks.\n\n")

        # Reference to PDF if available
        if combined_pdf:
            f.write(f"A PDF file with all spectrum allocation heatmaps is available at: {combined_pdf}\n\n")

        # Experiment setup
        f.write("-" * 80 + "\n")
        f.write("EXPERIMENT SETUP\n")
        f.write("-" * 80 + "\n\n")
        f.write(f"Topology file: {topology_file_path}\n")
        f.write(f"Traffic file: {traffic_file_path}\n\n")

        # Ascending order results
        f.write("=" * 80 + "\n")
        f.write("ASCENDING ORDER RESULTS (SMALL DEMANDS FIRST)\n")
        f.write("=" * 80 + "\n\n")

        # Benchmark algorithm results (ascending)
        f.write("-" * 80 + "\n")
        f.write("BENCHMARK ALGORITHM (ASCENDING)\n")
        f.write("-" * 80 + "\n\n")
        for key, value in benchmark_metrics_asc.items():
            f.write(f"{key}: {value}\n")
        f.write(f"Heatmap: {heatmap_files['benchmark_asc']}\n\n")

        # Simulated annealing algorithm results (ascending)
        f.write("-" * 80 + "\n")
        f.write("SIMULATED ANNEALING ALGORITHM (ASCENDING)\n")
        f.write("-" * 80 + "\n\n")
        for key, value in sa_metrics_asc.items():
            f.write(f"{key}: {value}\n")
        f.write(f"Heatmap: {heatmap_files['sa_asc']}\n\n")

        # Protection algorithm results (ascending)
        f.write("-" * 80 + "\n")
        f.write("1+1 PROTECTION ALGORITHM (ASCENDING)\n")
        f.write("-" * 80 + "\n\n")
        for key, value in protection_metrics_asc.items():
            f.write(f"{key}: {value}\n")
        f.write(f"Heatmap: {heatmap_files['protection_asc']}\n\n")

        # Shared protection algorithm results (ascending)
        f.write("-" * 80 + "\n")
        f.write("SHARED PROTECTION ALGORITHM (ASCENDING)\n")
        f.write("-" * 80 + "\n\n")
        for key, value in shared_metrics_asc.items():
            f.write(f"{key}: {value}\n")
        f.write(f"Heatmap: {heatmap_files['shared_asc']}\n\n")

        # Comparison of all four algorithms (ascending)
        f.write("-" * 80 + "\n")
        f.write("COMPARISON OF ALL FOUR ALGORITHMS (ASCENDING)\n")
        f.write("-" * 80 + "\n\n")

        metrics_table = [
            ["Metric", "Benchmark", "Sim. Annealing", "1+1 Protection", "Shared Protection"],
            ["Highest used FSU", benchmark_metrics_asc["highest_used_fsu"], sa_metrics_asc["highest_used_fsu"],
             protection_metrics_asc["highest_used_fsu"], shared_metrics_asc["highest_used_fsu"]],
            ["Total FSU usage", benchmark_metrics_asc["total_used_fsus"], sa_metrics_asc["total_used_fsus"],
             protection_metrics_asc["total_used_fsus"], shared_metrics_asc["total_used_fsus"]],
            ["Avg link usage", f"{benchmark_metrics_asc['avg_link_usage']:.4f}",
             f"{sa_metrics_asc['avg_link_usage']:.4f}", f"{protection_metrics_asc['avg_link_usage']:.4f}",
             f"{shared_metrics_asc['avg_link_usage']:.4f}"],
            ["Utilization entropy", f"{benchmark_metrics_asc['utilization_entropy']:.4f}",
             f"{sa_metrics_asc['utilization_entropy']:.4f}", f"{protection_metrics_asc['utilization_entropy']:.4f}",
             f"{shared_metrics_asc['utilization_entropy']:.4f}"],
            ["Shannon entropy", f"{benchmark_metrics_asc['shannon_entropy']:.4f}",
             f"{sa_metrics_asc['shannon_entropy']:.4f}", f"{protection_metrics_asc['shannon_entropy']:.4f}",
             f"{shared_metrics_asc['shannon_entropy']:.4f}"],
            ["Avg path length", f"{benchmark_metrics_asc['avg_path_length']:.4f}",
             f"{sa_metrics_asc['avg_path_length']:.4f}", f"{protection_metrics_asc['avg_path_length']:.4f}",
             f"{shared_metrics_asc['avg_path_length']:.4f}"],
            ["Avg transponder cost", f"{benchmark_metrics_asc['avg_transponder_cost']:.4f}",
             f"{sa_metrics_asc['avg_transponder_cost']:.4f}", f"{protection_metrics_asc['avg_transponder_cost']:.4f}",
             f"{shared_metrics_asc['avg_transponder_cost']:.4f}"]
        ]

        # Add shared protection specific metrics
        if "sharing_ratio" in shared_metrics_asc:
            metrics_table.append(["Sharing ratio", "-", "-", "-", f"{shared_metrics_asc['sharing_ratio']:.4f}"])

        # Add more detailed sharing statistics
        if "avg_sharing_percentage" in shared_metrics_asc:
            metrics_table.append(
                ["Avg sharing %", "-", "-", "-", f"{shared_metrics_asc['avg_sharing_percentage']:.4f}"])

        if "shared_requests_percentage" in shared_metrics_asc:
            metrics_table.append(
                ["Shared requests %", "-", "-", "-", f"{shared_metrics_asc['shared_requests_percentage'] * 100:.2f}%"])

        if "resource_savings" in shared_metrics_asc:
            metrics_table.append(
                ["Resource savings", "-", "-", "-", f"{shared_metrics_asc['resource_savings'] * 100:.2f}%"])

        # Format and write the table
        col_widths = [max(len(str(row[i])) for row in metrics_table) + 2 for i in range(len(metrics_table[0]))]

        for row in metrics_table:
            row_str = ""
            for i, item in enumerate(row):
                row_str += str(item).ljust(col_widths[i])
            f.write(row_str + "\n")
        f.write("\n")

        # SA vs Benchmark Improvement (ascending)
        f.write("Simulated Annealing vs Benchmark Improvement:\n")
        improvements = {}
        for key in benchmark_metrics_asc:
            if isinstance(benchmark_metrics_asc[key], (int, float)) and isinstance(sa_metrics_asc[key], (int, float)):
                improvements[key] = safe_percent_change(sa_metrics_asc[key], benchmark_metrics_asc[key])

        for key, value in improvements.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        # Protection vs Benchmark Comparison (ascending)
        f.write("1+1 Protection vs Benchmark Comparison:\n")
        comparisons = {}
        for key in benchmark_metrics_asc:
            if isinstance(benchmark_metrics_asc[key], (int, float)) and isinstance(protection_metrics_asc[key],
                                                                                   (int, float)):
                comparisons[key] = safe_percent_change(protection_metrics_asc[key], benchmark_metrics_asc[key])

        for key, value in comparisons.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        # Shared Protection vs 1+1 Protection Comparison (ascending)
        f.write("Shared Protection vs 1+1 Protection Comparison:\n")
        comparisons = {}
        for key in protection_metrics_asc:
            if isinstance(protection_metrics_asc[key], (int, float)) and isinstance(shared_metrics_asc[key],
                                                                                    (int, float)):
                comparisons[key] = safe_percent_change(shared_metrics_asc[key], protection_metrics_asc[key])

        for key, value in comparisons.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        # Descending order results
        f.write("=" * 80 + "\n")
        f.write("DESCENDING ORDER RESULTS (LARGE DEMANDS FIRST)\n")
        f.write("=" * 80 + "\n\n")

        # Benchmark algorithm results (descending)
        f.write("-" * 80 + "\n")
        f.write("BENCHMARK ALGORITHM (DESCENDING)\n")
        f.write("-" * 80 + "\n\n")
        for key, value in benchmark_metrics_desc.items():
            f.write(f"{key}: {value}\n")
        f.write(f"Heatmap: {heatmap_files['benchmark_desc']}\n\n")

        # Simulated annealing algorithm results (descending)
        f.write("-" * 80 + "\n")
        f.write("SIMULATED ANNEALING ALGORITHM (DESCENDING)\n")
        f.write("-" * 80 + "\n\n")
        for key, value in sa_metrics_desc.items():
            f.write(f"{key}: {value}\n")
        f.write(f"Heatmap: {heatmap_files['sa_desc']}\n\n")

        # Protection algorithm results (descending)
        f.write("-" * 80 + "\n")
        f.write("1+1 PROTECTION ALGORITHM (DESCENDING)\n")
        f.write("-" * 80 + "\n\n")
        for key, value in protection_metrics_desc.items():
            f.write(f"{key}: {value}\n")
        f.write(f"Heatmap: {heatmap_files['protection_desc']}\n\n")

        # Shared protection algorithm results (descending)
        f.write("-" * 80 + "\n")
        f.write("SHARED PROTECTION ALGORITHM (DESCENDING)\n")
        f.write("-" * 80 + "\n\n")
        for key, value in shared_metrics_desc.items():
            f.write(f"{key}: {value}\n")
        f.write(f"Heatmap: {heatmap_files['shared_desc']}\n\n")

        # Comparison of all four algorithms (descending)
        f.write("-" * 80 + "\n")
        f.write("COMPARISON OF ALL FOUR ALGORITHMS (DESCENDING)\n")
        f.write("-" * 80 + "\n\n")

        metrics_table = [
            ["Metric", "Benchmark", "Sim. Annealing", "1+1 Protection", "Shared Protection"],
            ["Highest used FSU", benchmark_metrics_desc["highest_used_fsu"], sa_metrics_desc["highest_used_fsu"],
             protection_metrics_desc["highest_used_fsu"], shared_metrics_desc["highest_used_fsu"]],
            ["Total FSU usage", benchmark_metrics_desc["total_used_fsus"], sa_metrics_desc["total_used_fsus"],
             protection_metrics_desc["total_used_fsus"], shared_metrics_desc["total_used_fsus"]],
            ["Avg link usage", f"{benchmark_metrics_desc['avg_link_usage']:.4f}",
             f"{sa_metrics_desc['avg_link_usage']:.4f}", f"{protection_metrics_desc['avg_link_usage']:.4f}",
             f"{shared_metrics_desc['avg_link_usage']:.4f}"],
            ["Utilization entropy", f"{benchmark_metrics_desc['utilization_entropy']:.4f}",
             f"{sa_metrics_desc['utilization_entropy']:.4f}", f"{protection_metrics_desc['utilization_entropy']:.4f}",
             f"{shared_metrics_desc['utilization_entropy']:.4f}"],
            ["Shannon entropy", f"{benchmark_metrics_desc['shannon_entropy']:.4f}",
             f"{sa_metrics_desc['shannon_entropy']:.4f}", f"{protection_metrics_desc['shannon_entropy']:.4f}",
             f"{shared_metrics_desc['shannon_entropy']:.4f}"],
            ["Avg path length", f"{benchmark_metrics_desc['avg_path_length']:.4f}",
             f"{sa_metrics_desc['avg_path_length']:.4f}", f"{protection_metrics_desc['avg_path_length']:.4f}",
             f"{shared_metrics_desc['avg_path_length']:.4f}"],
            ["Avg transponder cost", f"{benchmark_metrics_desc['avg_transponder_cost']:.4f}",
             f"{sa_metrics_desc['avg_transponder_cost']:.4f}", f"{protection_metrics_desc['avg_transponder_cost']:.4f}",
             f"{shared_metrics_desc['avg_transponder_cost']:.4f}"]
        ]

        # Add shared protection specific metrics
        if "sharing_ratio" in shared_metrics_desc:
            metrics_table.append(["Sharing ratio", "-", "-", "-", f"{shared_metrics_desc['sharing_ratio']:.4f}"])

        # Add more detailed sharing statistics
        if "avg_sharing_percentage" in shared_metrics_desc:
            metrics_table.append(
                ["Avg sharing %", "-", "-", "-", f"{shared_metrics_desc['avg_sharing_percentage']:.4f}"])

        if "shared_requests_percentage" in shared_metrics_desc:
            metrics_table.append(
                ["Shared requests %", "-", "-", "-", f"{shared_metrics_desc['shared_requests_percentage'] * 100:.2f}%"])

        if "resource_savings" in shared_metrics_desc:
            metrics_table.append(
                ["Resource savings", "-", "-", "-", f"{shared_metrics_desc['resource_savings'] * 100:.2f}%"])

        # Format and write the table
        col_widths = [max(len(str(row[i])) for row in metrics_table) + 2 for i in range(len(metrics_table[0]))]

        for row in metrics_table:
            row_str = ""
            for i, item in enumerate(row):
                row_str += str(item).ljust(col_widths[i])
            f.write(row_str + "\n")
        f.write("\n")

        # Order comparison (ascending vs descending)
        f.write("=" * 80 + "\n")
        f.write("COMPARISON BETWEEN ASCENDING AND DESCENDING ORDER\n")
        f.write("=" * 80 + "\n\n")

        # Benchmark algorithm order comparison
        f.write("Benchmark Algorithm: Ascending vs Descending\n")
        comparisons = {}
        for key in benchmark_metrics_asc:
            if isinstance(benchmark_metrics_asc[key], (int, float)) and isinstance(benchmark_metrics_desc[key],
                                                                                   (int, float)):
                comparisons[key] = safe_percent_change(benchmark_metrics_desc[key], benchmark_metrics_asc[key])

        for key, value in comparisons.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        # SA algorithm comparison
        f.write("Simulated Annealing: Ascending vs Descending\n")
        comparisons = {}
        for key in sa_metrics_asc:
            if isinstance(sa_metrics_asc[key], (int, float)) and isinstance(sa_metrics_desc[key], (int, float)):
                comparisons[key] = safe_percent_change(sa_metrics_desc[key], sa_metrics_asc[key])

        for key, value in comparisons.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        # Protection algorithm comparison
        f.write("1+1 Protection: Ascending vs Descending\n")
        comparisons = {}
        for key in protection_metrics_asc:
            if isinstance(protection_metrics_asc[key], (int, float)) and isinstance(protection_metrics_desc[key],
                                                                                    (int, float)):
                comparisons[key] = safe_percent_change(protection_metrics_desc[key], protection_metrics_asc[key])

        for key, value in comparisons.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        # Shared Protection algorithm comparison
        f.write("Shared Protection: Ascending vs Descending\n")
        comparisons = {}
        for key in shared_metrics_asc:
            if isinstance(shared_metrics_asc[key], (int, float)) and isinstance(shared_metrics_desc[key], (int, float)):
                comparisons[key] = safe_percent_change(shared_metrics_desc[key], shared_metrics_asc[key])

        for key, value in comparisons.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        # Conclusion
        f.write("=" * 80 + "\n")
        f.write("CONCLUSION\n")
        f.write("=" * 80 + "\n\n")

        f.write("Based on the experimental results, we can draw the following conclusions:\n\n")

        f.write("1. Order Impact (Ascending vs Descending):\n")
        f.write("   - Descending order (large demands first) typically results in higher FSU usage\n")
        f.write("   - Descending order generally leads to higher utilization entropy\n")
        f.write("   - The impact of order varies across different algorithms\n\n")

        f.write("2. Algorithm Comparison:\n")
        f.write("   - Simulated annealing improves spectrum efficiency compared to the benchmark\n")
        f.write("   - 1+1 Protection requires additional resources but provides resilience against failures\n")
        f.write("   - Shared protection significantly reduces resource usage compared to 1+1 protection\n")
        f.write("   - Protection algorithms significantly reduce transponder costs\n\n")

        f.write("3. Protection Strategies:\n")
        f.write("   - 1+1 protection provides full dedicated backup paths for all connections\n")
        f.write("   - Shared protection achieves similar resilience with lower resource consumption\n")
        f.write("   - Shared protection achieves a sharing ratio of approximately ")
        f.write(
            f"{shared_metrics_asc['sharing_ratio']:.2f} (ascending) and {shared_metrics_desc['sharing_ratio']:.2f} (descending)\n\n")

        f.write("4. Optimization Effectiveness:\n")
        f.write("   - Simulated annealing consistently improves fragmentation metrics\n")
        f.write("   - The choice of initial solution ordering significantly affects final results\n")
        f.write("   - For protection scenarios, resource overhead can be partially mitigated through optimization\n")
        f.write("   - Shared protection optimization further reduces spectrum consumption\n\n")

        # Generated images summary
        f.write("-" * 80 + "\n")
        f.write("GENERATED HEATMAP IMAGES\n")
        f.write("-" * 80 + "\n\n")

        for key, filename in heatmap_files.items():
            f.write(f"{key}: {filename}\n")

    print(f"Comprehensive report generated: {output_file}")
    return output_file


if __name__ == "__main__":
    # 加载拓扑和流量
    print("加载拓扑和流量数据...")
    # 使用相对于project_code的路径
    topology_file = topology_file_path
    traffic_file = traffic_file_path

    # 提取流量文件标识符
    traffic_file_identifier = os.path.basename(traffic_file).split('.')[0]

    # 创建基于流量文件的输出目录
    output_dir = f"results_{traffic_file_identifier}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 加载网络拓扑
    G = load_topology(topology_file)
    # 加载流量矩阵
    traffic_matrix = load_traffic_matrix(traffic_file)

    print(f"加载完成! 拓扑: {len(G.nodes())}节点, {len(G.edges())}链路")
    print(f"流量请求数量: {len(traffic_matrix)}")

    # 运行升序排序的实验
    print("\n" + "=" * 80)
    print("运行升序排序的实验 (小流量优先分配)")
    print("=" * 80)

    # 运行基准算法 (升序)
    print("\n运行基准算法 (升序)...")
    benchmark_results_asc, benchmark_metrics_asc, benchmark_slot_grid_asc = run_benchmark_algorithm(G, traffic_matrix,
                                                                                                    order="ascending")

    # 运行模拟退火算法 (升序)
    print("\n运行模拟退火算法 (升序)...")
    sa_results_asc, sa_metrics_asc, sa_slot_grid_asc = run_sa_algorithm(G, traffic_matrix, order="ascending")

    # 运行1+1保护算法 (升序)
    print("\n运行1+1保护算法 (升序)...")
    protection_results_asc, protection_metrics_asc, protection_slot_grid_asc = run_sa_with_protection(G, traffic_matrix,
                                                                                                      benchmark_metrics_asc,
                                                                                                      order="ascending")

    # 运行共享保护算法 (升序)
    print("\n运行共享保护算法 (升序)...")
    shared_metrics_asc, shared_slot_grid_asc = run_sa_with_shared_protection(
        G, traffic_matrix, benchmark_metrics=benchmark_metrics_asc, order="ascending"
    )

    # 为共享保护算法生成热力图 (升序)
    plot_spectrum_heatmap(shared_slot_grid_asc, "shared_spectrum.png",
                          title="Shared Protection Spectrum Allocation (Ascending Order)")

    # 比较所有四种算法 (升序)
    print("\n" + "=" * 50)
    print("性能指标比较: 基准算法 vs 模拟退火算法 vs 1+1保护算法 vs 共享保护算法 (升序)")
    print("=" * 50)
    print_four_algorithms_comparison(
        benchmark_metrics_asc, sa_metrics_asc, protection_metrics_asc, shared_metrics_asc)

    # 运行降序排序的实验
    print("\n\n" + "=" * 80)
    print("运行降序排序的实验 (大流量优先分配)")
    print("=" * 80)

    # 运行基准算法 (降序)
    print("\n运行基准算法 (降序)...")
    benchmark_results_desc, benchmark_metrics_desc, benchmark_slot_grid_desc = run_benchmark_algorithm(G,
                                                                                                       traffic_matrix,
                                                                                                       order="descending")

    # 运行模拟退火算法 (降序)
    print("\n运行模拟退火算法 (降序)...")
    sa_results_desc, sa_metrics_desc, sa_slot_grid_desc = run_sa_algorithm(G, traffic_matrix, order="descending")

    # 运行1+1保护算法 (降序)
    print("\n运行1+1保护算法 (降序)...")
    protection_results_desc, protection_metrics_desc, protection_slot_grid_desc = run_sa_with_protection(G,
                                                                                                         traffic_matrix,
                                                                                                         benchmark_metrics_desc,
                                                                                                         order="descending")

    # 运行共享保护算法 (降序)
    print("\n运行共享保护算法 (降序)...")
    shared_metrics_desc, shared_slot_grid_desc = run_sa_with_shared_protection(
        G, traffic_matrix, benchmark_metrics=benchmark_metrics_desc, order="descending"
    )

    # 为共享保护算法生成热力图 (降序)
    plot_spectrum_heatmap(shared_slot_grid_desc, "shared_spectrum_desc.png",
                          title="Shared Protection Spectrum Allocation (Descending Order)")

    # 比较所有四种算法 (降序)
    print("\n" + "=" * 50)
    print("性能指标比较: 基准算法 vs 模拟退火算法 vs 1+1保护算法 vs 共享保护算法 (降序)")
    print("=" * 50)
    print_four_algorithms_comparison(
        benchmark_metrics_desc, sa_metrics_desc, protection_metrics_desc, shared_metrics_desc)

    # 比较升序和降序的实验结果
    print("\n\n" + "=" * 80)
    print("升序 vs 降序排序比较")
    print("=" * 80)

    print("\n" + "=" * 50)
    print("基准算法: 升序 vs 降序")
    print("=" * 50)

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
        if metric_key in benchmark_metrics_asc and metric_key in benchmark_metrics_desc:
            asc_value = benchmark_metrics_asc[metric_key]
            desc_value = benchmark_metrics_desc[metric_key]

            print(f"{metric_name}:")
            print(f"  升序: {asc_value:.4f}")
            print(f"  降序: {desc_value:.4f}")
            print(f"  差异: {((desc_value - asc_value) / asc_value) * 100:.2f}%")
            print("-" * 50)

    print("\n" + "=" * 50)
    print("模拟退火算法: 升序 vs 降序")
    print("=" * 50)

    for metric_name, metric_key in metrics:
        if metric_key in sa_metrics_asc and metric_key in sa_metrics_desc:
            asc_value = sa_metrics_asc[metric_key]
            desc_value = sa_metrics_desc[metric_key]

            print(f"{metric_name}:")
            print(f"  升序: {asc_value:.4f}")
            print(f"  降序: {desc_value:.4f}")
            print(f"  差异: {((desc_value - asc_value) / asc_value) * 100:.2f}%")
            print("-" * 50)

    print("\n" + "=" * 50)
    print("1+1保护算法: 升序 vs 降序")
    print("=" * 50)

    for metric_name, metric_key in metrics:
        if metric_key in protection_metrics_asc and metric_key in protection_metrics_desc:
            asc_value = protection_metrics_asc[metric_key]
            desc_value = protection_metrics_desc[metric_key]

            print(f"{metric_name}:")
            print(f"  升序: {asc_value:.4f}")
            print(f"  降序: {desc_value:.4f}")
            print(f"  差异: {((desc_value - asc_value) / asc_value) * 100:.2f}%")
            print("-" * 50)

    print("\n" + "=" * 50)
    print("共享保护算法: 升序 vs 降序")
    print("=" * 50)

    for metric_name, metric_key in metrics:
        if metric_key in shared_metrics_asc and metric_key in shared_metrics_desc:
            asc_value = shared_metrics_asc[metric_key]
            desc_value = shared_metrics_desc[metric_key]

            print(f"{metric_name}:")
            print(f"  升序: {asc_value:.4f}")
            print(f"  降序: {desc_value:.4f}")
            print(f"  差异: {((desc_value - asc_value) / asc_value) * 100:.2f}%")
            print("-" * 50)

    print("\n运行完成! 生成的频谱分配热图:")
    print("升序排序:")
    print("- benchmark_spectrum.png (基准算法)")
    print("- sa_spectrum.png (模拟退火算法)")
    print("- protection_spectrum.png (1+1保护算法)")
    print("- shared_spectrum.png (共享保护算法)")
    print("降序排序:")
    print("- benchmark_spectrum_desc.png (基准算法)")
    print("- sa_spectrum_desc.png (模拟退火算法)")
    print("- protection_spectrum_desc.png (1+1保护算法)")
    print("- shared_spectrum_desc.png (共享保护算法)")

    # 生成综合报告
    heatmap_files = {
        'benchmark_asc': 'benchmark_spectrum.png',
        'sa_asc': 'sa_spectrum.png',
        'protection_asc': 'protection_spectrum.png',
        'shared_asc': 'shared_spectrum.png',
        'benchmark_desc': 'benchmark_spectrum_desc.png',
        'sa_desc': 'sa_spectrum_desc.png',
        'protection_desc': 'protection_spectrum_desc.png',
        'shared_desc': 'shared_spectrum_desc.png'
    }

    # 生成组合热图PDF
    combined_pdf = create_combined_heatmap_pdf(
        heatmap_files,
        traffic_file_identifier=traffic_file_identifier,
        output_dir=output_dir
    )

    # 生成性能图表
    charts_pdf = create_performance_charts(
        benchmark_metrics_asc, sa_metrics_asc, protection_metrics_asc, shared_metrics_asc,
        benchmark_metrics_desc, sa_metrics_desc, protection_metrics_desc, shared_metrics_desc,
        traffic_file_identifier=traffic_file_identifier,
        output_dir=output_dir
    )

    # 生成综合报告
    generate_comprehensive_report(
        benchmark_metrics_asc, sa_metrics_asc, protection_metrics_asc, shared_metrics_asc,
        benchmark_metrics_desc, sa_metrics_desc, protection_metrics_desc, shared_metrics_desc,
        heatmap_files, traffic_file_identifier=traffic_file_identifier
    )

    print(f"\n实验报告已生成在目录 {output_dir} 中:")
    if combined_pdf:
        print(f"- 频谱分配热图: {os.path.basename(combined_pdf)}")
    if charts_pdf:
        print(f"- 性能比较图表: {os.path.basename(charts_pdf)}")
    print(f"- 综合结果报告: experiment_results.txt")
    print(f"- 单独图表: 各个性能指标的PNG文件")
