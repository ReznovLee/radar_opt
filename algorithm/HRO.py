import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def read_data():
    """
    读取目标和雷达数据，返回 targets_df 和 radars_df。
    """
    # 读取目标数据
    targets_df = pd.read_csv('trajectories/all_info.csv')
    # 确保目标数据包含必要的列
    required_columns = ['id', 'x', 'y', 'z', 'type', 'time', 'in_radar_range', 'threat_level']
    if not all(col in targets_df.columns for col in required_columns):
        raise ValueError(f"目标数据缺少必要的列，请确保包含以下列：{required_columns}")

    # 读取雷达数据
    radars_df = pd.read_csv('trajectories/radar_info.csv')
    # 确保雷达数据包含必要的列
    required_columns = ['id', 'x', 'y', 'z', 'radii', 'channel']
    if not all(col in radars_df.columns for col in required_columns):
        raise ValueError(f"雷达数据缺少必要的列，请确保包含以下列：{required_columns}")

    # 将 radars_df 的索引设置为 'id'
    radars_df.set_index('id', inplace=True)

    return targets_df, radars_df

def estimate_next_position(current_pos, previous_pos):
    """
    根据当前和上一时刻的位置估计下一时刻的位置。
    """
    # 简单的线性外推
    return current_pos + (current_pos - previous_pos)

def check_in_radar_range(target_pos, radar_pos, radar_radius):
    """
    检查目标是否在雷达的探测范围内。
    """
    distance = np.linalg.norm(target_pos - radar_pos)
    return distance <= radar_radius

def greedy_initial_assignment(targets_df, radars_df, time_step):
    """
    初始时刻的贪心算法分配。
    """
    # 初始化分配结果
    assignments = []
    radar_availability = radars_df['channel'].to_dict()

    # 获取初始时刻的目标
    initial_time = targets_df['time'].min()
    current_targets = targets_df[targets_df['time'] == initial_time]

    # 按照威胁度排序目标
    current_targets = current_targets.sort_values(by='threat_level', ascending=False)

    for _, target_row in current_targets.iterrows():
        target_id = target_row['id']
        threat_level = target_row['threat_level']
        target_pos = target_row[['x', 'y', 'z']].values.astype(float)

        # 查找覆盖该目标的雷达
        candidate_radars = []
        for radar_id, radar_row in radars_df.iterrows():
            radar_pos = radar_row[['x', 'y', 'z']].values.astype(float)
            radar_radius = radar_row['radii']
            if check_in_radar_range(target_pos, radar_pos, radar_radius):
                if radar_availability[radar_id] > 0:
                    candidate_radars.append((radar_id, np.linalg.norm(target_pos - radar_pos)))
        if candidate_radars:
            # 选择最近的雷达
            candidate_radars.sort(key=lambda x: x[1])
            assigned_radar_id = candidate_radars[0][0]

            # 检查通道是否已满
            if radar_availability[assigned_radar_id] > 0:
                total_channels = radars_df.loc[assigned_radar_id, 'channel']
                channel_number = total_channels - radar_availability[assigned_radar_id] + 1
                radar_availability[assigned_radar_id] -= 1

                assignments.append({
                    'time': initial_time,
                    'target_id': target_id,
                    'radar_id': assigned_radar_id,
                    'channel_number': channel_number,
                    'threat_level': threat_level
                })
    return assignments

def neighborhood_search_assignment(targets_df, radars_df, time_step):
    """
    后续时刻的邻域搜索算法分配，并计算累积跟踪时间。
    """
    # 初始化分配结果
    schedule_records = []
    radar_channel_limits = radars_df['channel'].to_dict()

    # 获取所有时间点
    time_points = np.arange(targets_df['time'].min(), targets_df['time'].max() + time_step, time_step)
    num_time_steps = len(time_points)

    # 初始化雷达可用通道信息
    radar_availability = radar_channel_limits.copy()

    # 记录上一时间步的分配
    previous_assignments = {}

    # 初始化累积跟踪时间
    cumulative_tracking_time = []
    cumulative_weighted_tracking_time = []
    total_tracking_time = 0
    total_weighted_tracking_time = 0

    for idx, current_time in enumerate(tqdm(time_points, desc="调度进度")):
        current_targets = targets_df[targets_df['time'] == current_time]

        # 如果没有目标，跳过
        if current_targets.empty:
            cumulative_tracking_time.append((current_time, total_tracking_time))
            cumulative_weighted_tracking_time.append((current_time, total_weighted_tracking_time))
            continue

        # 按照威胁度排序目标
        current_targets = current_targets.sort_values(by='threat_level', ascending=False)

        # 初始化当前时间的分配
        current_assignments = []

        for _, target_row in current_targets.iterrows():
            target_id = target_row['id']
            threat_level = target_row['threat_level']
            target_pos = target_row[['x', 'y', 'z']].values.astype(float)

            # 预测下一时刻位置
            previous_time = current_time - time_step
            if previous_time >= targets_df['time'].min():
                previous_row = targets_df[(targets_df['id'] == target_id) & (targets_df['time'] == previous_time)]
                if not previous_row.empty:
                    previous_pos = previous_row[['x', 'y', 'z']].values.astype(float)[0]
                    next_pos = estimate_next_position(target_pos, previous_pos)
                else:
                    next_pos = target_pos
            else:
                next_pos = target_pos

            # 尝试保持上一时刻的分配
            assigned = False
            if target_id in previous_assignments:
                assigned_radar_id = previous_assignments[target_id]
                radar_row = radars_df.loc[assigned_radar_id]
                radar_pos = radar_row[['x', 'y', 'z']].values.astype(float)
                radar_radius = radar_row['radii']

                if check_in_radar_range(target_pos, radar_pos, radar_radius):
                    if radar_availability[assigned_radar_id] > 0:
                        total_channels = radars_df.loc[assigned_radar_id, 'channel']
                        channel_number = total_channels - radar_availability[assigned_radar_id] + 1
                        radar_availability[assigned_radar_id] -= 1

                        current_assignments.append({
                            'time': current_time,
                            'target_id': target_id,
                            'radar_id': assigned_radar_id,
                            'channel_number': channel_number,
                            'threat_level': threat_level
                        })
                        assigned = True

            if not assigned:
                # 查找覆盖该目标的雷达
                candidate_radars = []
                for radar_id, radar_row in radars_df.iterrows():
                    radar_pos = radar_row[['x', 'y', 'z']].values.astype(float)
                    radar_radius = radar_row['radii']
                    if check_in_radar_range(target_pos, radar_pos, radar_radius):
                        if radar_availability[radar_id] > 0:
                            candidate_radars.append((radar_id, np.linalg.norm(target_pos - radar_pos)))

                if candidate_radars:
                    # 选择最近的雷达
                    candidate_radars.sort(key=lambda x: x[1])
                    assigned_radar_id = candidate_radars[0][0]

                    # 检查通道是否已满
                    if radar_availability[assigned_radar_id] > 0:
                        total_channels = radars_df.loc[assigned_radar_id, 'channel']
                        channel_number = total_channels - radar_availability[assigned_radar_id] + 1
                        radar_availability[assigned_radar_id] -= 1

                        current_assignments.append({
                            'time': current_time,
                            'target_id': target_id,
                            'radar_id': assigned_radar_id,
                            'channel_number': channel_number,
                            'threat_level': threat_level
                        })
                        previous_assignments[target_id] = assigned_radar_id
                else:
                    # 如果存在威胁度为3的目标，移除威胁度为1的目标
                    if threat_level == 3:
                        for radar_id in radar_availability.keys():
                            radar_row = radars_df.loc[radar_id]
                            radar_pos = radar_row[['x', 'y', 'z']].values.astype(float)
                            radar_radius = radar_row['radii']
                            if check_in_radar_range(target_pos, radar_pos, radar_radius):
                                # 从已分配的低威胁度目标中移除一个
                                low_threat_targets = [a for a in current_assignments if a['radar_id'] == radar_id and a['threat_level'] == 1]
                                if low_threat_targets:
                                    removed_target = low_threat_targets[0]
                                    current_assignments.remove(removed_target)
                                    radar_availability[radar_id] += 1  # 释放通道

                                    # 分配高威胁度目标
                                    total_channels = radars_df.loc[radar_id, 'channel']
                                    channel_number = total_channels - radar_availability[radar_id] + 1
                                    radar_availability[radar_id] -= 1

                                    current_assignments.append({
                                        'time': current_time,
                                        'target_id': target_id,
                                        'radar_id': radar_id,
                                        'channel_number': channel_number,
                                        'threat_level': threat_level
                                    })
                                    previous_assignments[target_id] = radar_id
                                    break

        # 计算当前时间的跟踪时间
        total_tracking_time += len(current_assignments) * time_step
        total_weighted_tracking_time += sum([a['threat_level'] for a in current_assignments]) * time_step
        cumulative_tracking_time.append((current_time, total_tracking_time))
        cumulative_weighted_tracking_time.append((current_time, total_weighted_tracking_time))

        # 重置雷达可用通道信息
        radar_availability = radar_channel_limits.copy()
        schedule_records.extend(current_assignments)

    schedule_df = pd.DataFrame(schedule_records)
    convergence_df = pd.DataFrame({
        'time': [t[0] for t in cumulative_tracking_time],
        'cumulative_tracking_time': [t[1] for t in cumulative_tracking_time],
        'cumulative_weighted_tracking_time': [w[1] for w in cumulative_weighted_tracking_time]
    })
    return schedule_df, convergence_df

def plot_convergence_curve(convergence_df, filename):
    """
    绘制并保存算法的收敛曲线。
    """
    plt.figure(figsize=(10, 6))
    plt.plot(convergence_df['time'], convergence_df['cumulative_tracking_time'], label='Cumulative Tracking Time')
    plt.plot(convergence_df['time'], convergence_df['cumulative_weighted_tracking_time'], label='Cumulative Weighted Tracking Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Cumulative Time')
    plt.title('Convergence Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_radar_channel_gantt(schedule_df, radars_df, targets_df, time_step, filename):
    """
    绘制并保存雷达通道的甘特图，以不同目标设置不同颜色。
    """
    if schedule_df.empty:
        print("调度结果为空，无法绘制雷达通道甘特图。")
        return

    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(12, 6))

    # 设置背景为白色
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    yticks = []
    yticklabels = []
    y_base = 0
    y_height = 0.8

    # 为不同目标设置颜色
    target_ids = schedule_df['target_id'].unique()
    num_colors = len(target_ids)
    cmap = plt.get_cmap('tab20', num_colors)
    target_color_map = {target_id: cmap(i) for i, target_id in enumerate(target_ids)}

    radar_ids = radars_df.index.unique()
    for radar_id in radar_ids:
        radar_schedule = schedule_df[schedule_df['radar_id'] == radar_id]
        if not radar_schedule.empty:
            channels = sorted(radar_schedule['channel_number'].unique())
            for channel_num in channels:
                channel_schedule = radar_schedule[radar_schedule['channel_number'] == channel_num]
                # 按照时间和目标ID排序
                channel_schedule = channel_schedule.sort_values(by=['target_id', 'time'])
                # 按照目标ID进行分组
                grouped = channel_schedule.groupby('target_id')
                for target_id, group in grouped:
                    start_time = group['time'].min()
                    end_time = group['time'].max() + time_step
                    duration = end_time - start_time

                    ax.broken_barh(
                        [(start_time, duration)],
                        (y_base, y_height),
                        facecolors=target_color_map[target_id],
                        edgecolors='black',
                        linewidth=0.5
                    )
                yticks.append(y_base + y_height / 2)
                yticklabels.append(f'Radar {int(radar_id)} Channel {int(channel_num)}')
                y_base += y_height + 0.2

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Radar Channels')
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.grid(True)

    # 添加图例
    legend_patches = [mpatches.Patch(color=target_color_map[target_id], label=f'Target {int(target_id)}') for target_id in target_ids[:10]]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', title='Targets')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_target_gantt(schedule_df, targets_df, time_step, filename):
    """
    绘制并保存目标的跟踪甘特图，以不同雷达设置不同颜色。
    """
    if schedule_df.empty:
        print("调度结果为空，无法绘制目标甘特图。")
        return

    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(12, 6))

    # 设置背景为白色
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    yticks = []
    yticklabels = []
    y_base = 0
    y_height = 0.8

    # 为不同雷达设置颜色
    radar_ids = schedule_df['radar_id'].unique()
    num_colors = len(radar_ids)
    cmap = plt.get_cmap('tab20', num_colors)
    radar_color_map = {radar_id: cmap(i) for i, radar_id in enumerate(radar_ids)}

    target_ids = targets_df['id'].unique()
    for target_id in target_ids:
        target_schedule = schedule_df[schedule_df['target_id'] == target_id]
        if not target_schedule.empty:
            # 按照时间和雷达ID排序
            target_schedule = target_schedule.sort_values(by=['radar_id', 'time'])
            # 按照雷达ID进行分组
            grouped = target_schedule.groupby('radar_id')
            for radar_id, group in grouped:
                start_time = group['time'].min()
                end_time = group['time'].max() + time_step
                duration = end_time - start_time

                ax.broken_barh(
                    [(start_time, duration)],
                    (y_base, y_height),
                    facecolors=radar_color_map[radar_id],
                    edgecolors='black',
                    linewidth=0.5
                )
            yticks.append(y_base + y_height / 2)
            yticklabels.append(f'Target {int(target_id)}')
            y_base += y_height + 0.2

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Targets')
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.grid(True)

    # 添加图例
    legend_patches = [mpatches.Patch(color=radar_color_map[radar_id], label=f'Radar {int(radar_id)}') for radar_id in radar_ids[:10]]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', title='Radars')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def example_usage():
    """
    示例函数，演示如何使用调度算法并绘制结果。
    """
    # 读取数据
    targets_df, radars_df = read_data()

    # 设置时间步长
    time_step = 0.0001

    # 初始分配
    print("进行初始分配...")
    initial_assignments = greedy_initial_assignment(targets_df, radars_df, time_step)
    initial_schedule_df = pd.DataFrame(initial_assignments)

    # 后续时刻的分配
    print("进行后续时刻的调度...")
    schedule_df, convergence_df = neighborhood_search_assignment(targets_df, radars_df, time_step)

    # 合并初始分配和后续分配
    schedule_df = pd.concat([initial_schedule_df, schedule_df], ignore_index=True)
    schedule_df = schedule_df.sort_values(by=['time', 'target_id'])

    # 绘制甘特图
    print("绘制雷达通道甘特图...")
    plot_radar_channel_gantt(schedule_df, radars_df, targets_df, time_step, 'radar_channel_gantt.png')
    print("绘制目标甘特图...")
    plot_target_gantt(schedule_df, targets_df, time_step, 'target_gantt.png')

    # 绘制收敛曲线
    print("绘制算法收敛曲线...")
    plot_convergence_curve(convergence_df, 'convergence_curve.png')

    print("调度完成，结果已保存。")

if __name__ == "__main__":
    example_usage()
