import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_spectrum_heatmap(slot_grid, filename="spectrum_heatmap.png", cmap="YlGn_r", show_plot=False, title=None):
    """
    绘制频谱分配热图
    
    参数:
        slot_grid: 频谱分配状态字典
        filename: 保存的文件名
        cmap: 颜色映射
        show_plot: 是否显示图形
        title: 图表标题，如果为None则使用默认标题
    """
    spectrum_matrix = np.array([slot_grid[(u, v)] for (u, v) in slot_grid])
    plt.figure(figsize=(12, 6))
    sns.heatmap(spectrum_matrix, cmap=cmap, cbar=True)
    plt.xlabel("Spectrum Slots")
    plt.ylabel("Links (Edges)")
    
    if title:
        plt.title(title)
    else:
        plt.title("Spectrum Utilization Heatmap")
    
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close()  # 关闭图形，避免显示
