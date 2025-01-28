
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 已有的数据读取函数
def read_data():
    targets_file = 'trajectories/all_info.csv'
    radars_file = 'trajectories/radar_info.csv'
    targets_df = pd.read_csv(targets_file)
    radars_df = pd.read_csv(radars_file)
    return targets_df, radars_df

# 绘图函数
def plot_radar_tracking_time(scheduling_results):
    """ 基于调度结果绘制每部雷达的累计跟踪时长柱状图 """
    radar_tracking_duration = scheduling_results.groupby('radar_id')['tracking_duration'].sum()
    radar_tracking_duration.plot(kind='bar', color='skyblue')
    plt.title('Radar Cumulative Tracking Time')
    plt.xlabel('Radar ID')
    plt.ylabel('Total Tracking Time (Time Steps)')
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

def plot_radar_switching_frequency(scheduling_results):
    """ 计算并绘制雷达在不同时间步内的目标切换频率柱状图 """
    switching_counts = scheduling_results.groupby(['radar_id', 'time']).apply(lambda x: x['target_id'].nunique())
    switching_counts = switching_counts.reset_index(name='switching_count')
    switching_frequency = switching_counts.groupby('radar_id')['switching_count'].sum()
    switching_frequency.plot(kind='bar', color='orange')
    plt.title('Radar Switching Frequency Over Time')
    plt.xlabel('Radar ID')
    plt.ylabel('Total Switching Frequency')
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

def plot_algorithm_time_complexity(time_steps, time_complexities):
    """ 绘制算法时间复杂度图 """
    plt.plot(time_steps, time_complexities, marker='o', linestyle='-', color='purple')
    plt.title('Algorithm Time Complexity')
    plt.xlabel('Number of Targets')
    plt.ylabel('Execution Time (s)')
    plt.grid()
    plt.show()

def plot_policy_evolution_heatmap(policy_matrix, radar_ids, time_steps):
    """ 绘制策略演化热图 """
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(policy_matrix, cmap='coolwarm')
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(time_steps)))
    ax.set_xticklabels(time_steps)
    ax.set_yticks(np.arange(len(radar_ids)))
    ax.set_yticklabels(radar_ids)
    plt.title('Policy Evolution Heatmap')
    plt.xlabel('Time Step')
    plt.ylabel('Radar ID')
    plt.show()

# 假设该函数为调度的核心算法，返回一个包含调度结果的 DataFrame
def radar_tracking_scheduling(targets_df, radars_df):
    """ 
    调度算法主函数。假设其输出包含以下字段：radar_id、target_id、time、tracking_duration。
    tracking_duration 表示某个雷达对某目标的总跟踪时长。
    返回：
    - scheduling_results: pandas DataFrame，包含调度结果。
    """
    # 示例调度结果（可根据实际调度算法生成的结果替换）
    scheduling_results = pd.DataFrame({
        'radar_id': np.random.choice(['Radar 1', 'Radar 2', 'Radar 3'], size=100),
        'target_id': np.random.randint(1, 10, size=100),
        'time': np.random.randint(0, 20, size=100),
        'tracking_duration': np.random.randint(1, 5, size=100)
    })
    return scheduling_results

# 主函数入口，进行数据读取、调度与结果绘制
if __name__ == '__main__':
    # 读取输入数据
    targets_df, radars_df = read_data()

    # 执行调度算法，获取调度结果
    scheduling_results = radar_tracking_scheduling(targets_df, radars_df)

    # 绘制每部雷达的累计跟踪时长柱状图
    plot_radar_tracking_time(scheduling_results)

    # 绘制雷达在不同时间步内的目标切换频率柱状图
    plot_radar_switching_frequency(scheduling_results)

    # 示例时间复杂度数据
    time_steps = [10, 20, 30, 40, 50]
    time_complexities = [0.5, 1.2, 3.4, 7.6, 15.0]  # 示例时间复杂度
    plot_algorithm_time_complexity(time_steps, time_complexities)

    # 示例策略演化热图数据
    radar_ids = ['Radar 1', 'Radar 2', 'Radar 3']
    time_steps = [1, 2, 3, 4, 5]
    policy_matrix = np.random.rand(len(radar_ids), len(time_steps))  # 随机生成的策略矩阵
    plot_policy_evolution_heatmap(policy_matrix, radar_ids, time_steps)
