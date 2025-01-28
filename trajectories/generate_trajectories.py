import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import splprep, splev

def generate_initial_positions(num_ballistic_missiles=10,
                               num_airplanes=40,
                               num_cruise_missiles=50,
                               x_range=(0, 10000),
                               y_range=(0, 10000),
                               airplane_altitude=10000,
                               ballistic_missile_altitude=30000,
                               cruise_missile_altitude=10000
):
    """
    生成弹道导弹、飞机和巡航导弹的初始位置
    :param num_ballistic_missiles: 弹道导弹数量
    :param num_airplanes: 飞机数量
    :param num_cruise_missiles: 巡航导弹数量
    :param x_range: x坐标范围
    :param y_range: y坐标范围
    :param airplane_altitude: 飞机高度
    :param ballistic_missile_altitude: 弹道导弹高度
    :param cruise_missile_altitude: 巡航导弹高度
    :return: 三类目标的初始位置
    """
    # 生成弹道导弹初始位置
    ballistic_missile_positions = np.column_stack((
        np.random.uniform(low=x_range[0], high=x_range[1], size=num_ballistic_missiles),
        np.random.uniform(low=y_range[0], high=y_range[1], size=num_ballistic_missiles),
        np.full(num_ballistic_missiles, ballistic_missile_altitude)
    ))

    # 生成飞机初始位置
    airplane_positions = np.column_stack((
        np.random.uniform(low=x_range[0], high=x_range[1], size=num_airplanes),
        np.random.uniform(low=y_range[0], high=y_range[1], size=num_airplanes),
        np.full(num_airplanes, airplane_altitude)
    ))

    # 生成巡航导弹初始位置
    cruise_missile_positions = np.column_stack((
        np.random.uniform(low=x_range[0], high=x_range[1], size=num_cruise_missiles),
        np.random.uniform(low=y_range[0], high=y_range[1], size=num_cruise_missiles),
        np.full(num_cruise_missiles, cruise_missile_altitude)
    ))

    # 生成三类目标的初始位置
    positions = {
        'ballistic_missiles': ballistic_missile_positions,
        'airplanes': airplane_positions,
        'cruise_missiles': cruise_missile_positions
    }
    
    return positions

# 生成弹道导弹的轨迹
def generate_ballistic_missile_trajectories(initial_positions,
                                            num_points=500,
                                            target_area_x=(0, 10000),
                                            target_area_y=(-20000, -10000)
):
    """
    生成弹道导弹的轨迹
    :param initial_positions: 弹道导弹的初始位置
    :param num_points: 轨迹点数
    :param target_area_x: 目标区域x坐标范围
    :param target_area_y: 目标区域y坐标范围
    :return: 弹道导弹的轨迹
    """
    
    trajectories = []
    num_missiles = len(initial_positions)

    for i in range(num_missiles):
        initial_position = initial_positions[i]

        # 随机生成目标位置，位于指定的目标区域
        target_x = np.random.uniform(*target_area_x)
        target_y = np.random.uniform(*target_area_y)
        target_z = 0.0  # 目标在地面高度

        target_position = np.array([target_x, target_y, target_z])

        # 计算方向向量
        direction = target_position - initial_position
        distance = np.linalg.norm(direction)
        direction = direction / distance

        # 生成俯冲轨迹
        t = np.linspace(0, 1, num_points)
        trajectory = initial_position + np.outer(t, direction * distance)
        
        trajectories.append(trajectory)

    return trajectories


# 生成弹道导弹的轨迹，考虑速度和加速度
def generate_ballistic_missile_trajectories_with_speed(initial_positions,
                                                       num_points=500,
                                                       initial_velocity=500,
                                                       acceleration=-9.81,
                                                       target_area_x=(0, 10000),
                                                       target_area_y=(-20000, -10000)
):
    """
    生成弹道导弹的轨迹，考虑速度和加速度
    :param initial_positions: 弹道导弹的初始位置
    :param num_points: 轨迹点数
    :param initial_velocity: 初始速度
    :param acceleration: 加速度
    :param target_area_x: 目标区域x坐标范围
    :param target_area_y: 目标区域y坐标范围
    :return: 弹道导弹的轨迹
    """
    trajectories = []  # 存储所有导弹的轨迹
    num_missiles = len(initial_positions)  # 导弹数量

    for i in range(num_missiles):
        initial_position = np.array(initial_positions[i])

        # 随机生成目标位置，位于指定的目标区域
        target_x = np.random.uniform(*target_area_x)
        target_y = np.random.uniform(*target_area_y)
        target_z = 0.0  # 目标在地面高度(假设目标高度为0)

        target_position = np.array([target_x, target_y, target_z])

        # 计算方向向量
        direction = target_position - initial_position
        total_distance = np.linalg.norm(direction)  # 计算到目标的总距离
        direction = direction / total_distance  # 单位化方向向量

        # 初始化速度和位置
        current_position = initial_position
        current_velocity = initial_velocity
        positions = [current_position]  # 存储当前导弹的轨迹

        # 时间间隔：根据初始速度和加速度动态调整
        time_intervals = np.linspace(0, 1, num_points)

        # 模拟每个轨迹点
        for t in time_intervals[1:]:
            # 使用基础运动方程计算新的位置： s = v_0 * t + 0.5 * a * t^2
            displacement = (current_velocity * t) + 0.5 * acceleration * (t ** 2)
            
            # 更新速度（v = v_0 + a * t）
            current_velocity += acceleration * t

            # 新的位移向量（沿 direction 方向）
            new_position = initial_position + direction * displacement
            positions.append(new_position)

        # 保存生成的轨迹
        trajectories.append(np.array(positions))

    return trajectories

    
# 生成飞机的轨迹
def generate_airplane_trajectories(num_planes,
                                   num_points=500,
                                   num_control_points=7,
                                   initial_positions=None,
                                   target_area_y=(-20000, -10000),
                                   target_altitude_range=(8000, 12000)
):
    """
    生成飞机的轨迹，突变点在中间阶段
    :param num_planes: 飞机数量
    :param num_points: 轨迹点数
    :param num_control_points: 控制点数 (奇数，保证中间至少有一个转折点)
    :param initial_positions: 飞机的初始位置，形状为 (n,3)
    :param target_area_y: 目标区域y坐标范围 (min,max)
    :param target_altitude_range: 目标高度范围 (min,max)
    :return: 飞机的轨迹列表
    """

    trajectories = []
    if initial_positions is None:
        raise ValueError("必须提供 initial_positions 参数")

    for i in range(num_planes):
        x0, y0, z0 = initial_positions[i]

        # 目标位置，位于指定的目标区域和高度范围
        target_y = np.random.uniform(*target_area_y)
        target_z = np.random.uniform(*target_altitude_range)
        target_position = np.array([x0, target_y, target_z])

        # 控制点生成
        control_points = []
        control_points.append([x0, y0, z0])
        
        # 前半段飞行
        mid_index = num_control_points // 2
        for j in range(1, mid_index):
            t = j / mid_index
            pos = (1 - t) * np.array([x0, y0, z0]) + t * np.array([x0, (y0 + target_y)/2, (z0 + target_z)/2])
            control_points.append(pos)

        # 中间转折点，随机偏移
        turn_deviation = np.random.uniform(-5000, 5000)
        mid_point = np.array([x0 + turn_deviation, (y0 + target_y)/2, (z0 + target_z)/2])
        control_points.append(mid_point)

        # 后半段飞行
        for j in range(mid_index + 1, num_control_points):
            t = (j - mid_index) / (num_control_points - mid_index - 1)
            pos = (1 - t) * mid_point + t * target_position
            control_points.append(pos)

        control_points = np.array(control_points).T

        # 样条曲线拟合
        tck, u = splprep(control_points, s=0, k=min(3, num_control_points - 1))
        u_fine = np.linspace(0, 1, num_points)
        x_fine, y_fine, z_fine = splev(u_fine, tck)
        trajectory = np.stack((x_fine, y_fine, z_fine), axis=-1)
        trajectories.append(trajectory)

    return trajectories

# 生成巡航导弹的轨迹
def generate_cruise_missile_trajectories(num_missiles,
                                         num_points=500,
                                         initial_positions=None,
                                         target_area_x=(0, 10000),
                                         target_area_y=(-20000, -10000),
                                         cruise_altitude=10000,
                                         cruise_altitude_variation=1000,
                                         dive_angle=np.deg2rad(45),
                                         level_flight_distance=15000,
                                         num_level_control_points=5,
                                         max_initial_deviation_angle=np.deg2rad(20)
):
    """
    生成巡航导弹的轨迹，转折主要发生在初始阶段
    :param num_missiles: 巡航导弹数量
    :param num_points: 每条轨迹上的总点数
    :param initial_positions: 初始位置数组，形状为 (n, 3)
    :param target_area_x: 目标区域的 x 坐标范围 (min, max)
    :param target_area_y: 目标区域的 y 坐标范围 (min, max)
    :param cruise_altitude: 巡航高度
    :param cruise_altitude_variation: 巡航高度变化范围（正负值）
    :param dive_angle: 俯冲角度
    return 巡航导弹轨迹列表
    """
    trajectories = []
    if initial_positions is None:
        raise ValueError("必须提供 initial_positions 参数")

    for i in range(num_missiles):
        initial_position = initial_positions[i]

        # 目标位置，位于指定的目标区域
        target_x = np.random.uniform(*target_area_x)
        target_y = np.random.uniform(*target_area_y)
        target_position = np.array([target_x, target_y, 0.0])

        # 总体方向
        overall_direction = target_position[:2] - initial_position[:2]
        overall_direction = overall_direction / np.linalg.norm(overall_direction)

        # 初始阶段方向偏移
        deviation_angle = np.random.uniform(-max_initial_deviation_angle, max_initial_deviation_angle)
        cos_theta = np.cos(deviation_angle)
        sin_theta = np.sin(deviation_angle)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta,  cos_theta]
        ])
        initial_direction = rotation_matrix @ overall_direction

        # 水平飞行结束位置，方向为 overall_direction
        level_end_position = initial_position + np.append(overall_direction * level_flight_distance, 0)
        # 不再固定高度，之后根据控制点计算

        # 计算俯冲阶段的水平距离和垂直距离
        dive_vertical_distance = cruise_altitude - target_position[2]
        dive_horizontal_distance = dive_vertical_distance / np.tan(dive_angle)

        # 调整目标位置
        adjusted_target_position_xy = level_end_position[:2] + overall_direction * dive_horizontal_distance
        adjusted_target_position = np.append(adjusted_target_position_xy, target_position[2])

        # 如果调整后的目标位置与原始目标位置有显著差异，则更新目标位置
        distance_to_original_target = np.linalg.norm(adjusted_target_position[:2] - target_position[:2])
        if distance_to_original_target > 1.0:
            target_position = adjusted_target_position

        # 生成水平飞行阶段的控制点
        level_control_points = [initial_position]
        # 初始阶段的控制点（有偏移）
        num_initial_control_points = num_level_control_points // 2
        segment_length = level_flight_distance / num_level_control_points

        current_position = initial_position.copy()
        current_direction = initial_direction.copy()

        for j in range(1, num_initial_control_points):
            delta_pos = np.append(current_direction * segment_length, 0)
            current_position += delta_pos
            # 引入高度变化
            altitude_variation = np.random.uniform(-cruise_altitude_variation, cruise_altitude_variation)
            current_position[2] = cruise_altitude + altitude_variation
            level_control_points.append(current_position.copy())

        # 剩余阶段的控制点（方向不变，为 overall_direction）
        current_direction = overall_direction
        for j in range(num_initial_control_points, num_level_control_points):
            delta_pos = np.append(current_direction * segment_length, 0)
            current_position += delta_pos
            # 引入高度变化
            altitude_variation = np.random.uniform(-cruise_altitude_variation, cruise_altitude_variation)
            current_position[2] = cruise_altitude + altitude_variation
            level_control_points.append(current_position.copy())

        level_control_points = np.array(level_control_points)

        # 更新水平飞行结束位置的高度
        level_end_position = level_control_points[-1]

        # 使用样条曲线拟合水平飞行阶段的轨迹
        tck_level, u_level = splprep([level_control_points[:, 0], level_control_points[:, 1], level_control_points[:, 2]], s=0)
        num_level_points = int(num_points * (level_flight_distance / (level_flight_distance + dive_horizontal_distance)))
        num_level_points = max(num_level_points, 2)  # 确保至少有两个点
        u_fine_level = np.linspace(0, 1, num_level_points)
        x_level, y_level, z_level = splev(u_fine_level, tck_level)
        level_trajectory = np.stack((x_level, y_level, z_level), axis=-1)

        # 生成俯冲阶段的轨迹，与水平飞行结束方向一致
        num_dive_points = num_points - num_level_points
        num_dive_points = max(num_dive_points, 2)  # 确保至少有两个点
        t_dive = np.linspace(0, 1, num_dive_points)
        # x 和 y 位置沿着 overall_direction 前进
        dive_positions_xy = level_end_position[:2] + overall_direction * (t_dive[:, np.newaxis] * dive_horizontal_distance)
        # z 位置从水平飞行结束高度下降到目标高度
        dive_positions_z = np.linspace(level_end_position[2], target_position[2], num_dive_points)
        dive_positions = np.hstack((dive_positions_xy, dive_positions_z[:, np.newaxis]))

        # 拼接水平飞行和俯冲阶段的轨迹
        trajectory = np.vstack((level_trajectory, dive_positions))
        trajectories.append(trajectory)

    return trajectories

# 雷达生成函数   
def generate_radars(num_radars=3,
                    radii=12000,
                    center_x=5000,
                    center_y=-10000
):
    """
    生成雷达信息，包含雷达数量、通道数、半径、位置
    :param num_radars: 雷达数量
    :param radii: 雷达半径
    :param center_x: 雷达x坐标
    :param center_y: 雷达y坐标
    :return: 雷达信息
    """
    # 如果 radii 是单个值，转换为数组
    if isinstance(radii, (int, float)):
        radii = np.full(num_radars, radii)
    else:
        radii = np.array(radii)
        assert len(radii) == num_radars, "radii 的长度必须等于 num_radars"

    # 雷达位置集中在指定的中心附近
    radar_positions = []
    radar_channal = []
    angle_step = 360 / num_radars
    for i in range(num_radars):
        angle = np.deg2rad(i * angle_step)
        x = center_x + 1000 * np.cos(angle)  # 半径为 1000 米的圆上
        y = center_y + 1000 * np.sin(angle)
        z = 0.0
        radar_positions.append([x, y, z])

    for i in range(num_radars):
        radar_channal.append(np.random.randint(5, 8))

    radar_positions = np.array(radar_positions)

    radar_channal = np.array(radar_channal)

    return radar_positions, radii, radar_channal

# 判断轨迹是否在雷达范围内
def get_trajectory_coverage(trajectory, 
                            radar_positions, 
                            radar_radii
):
    """
    判断轨迹是否在雷达范围内
    :param trajectory: 轨迹
    :param radar_positions: 雷达位置
    :param radar_radii: 雷达半径
    :return: 轨迹是否在雷达范围内
    """
    distances = np.linalg.norm(trajectory[:, np.newaxis, :] - radar_positions[np.newaxis, :, :], axis=2)
    in_range = np.any(distances <= radar_radii, axis=1)
    return in_range

# 分段轨迹
def split_trajectory_by_coverage(trajectory, coverage):
    """
    根据覆盖情况分段轨迹
    :param trajectory: 轨迹
    :param coverage: 覆盖情况
    :return: 分段轨迹
    """
    segments = []
    n_points = len(trajectory)
    if n_points == 0:
        return segments

    start_idx = 0
    current_state = coverage[0]

    for i in range(1, n_points):
        if coverage[i] != current_state:
            segment = {
                'points': trajectory[start_idx:i],
                'in_range': current_state
            }
            segments.append(segment)
            start_idx = i
            current_state = coverage[i] # 更新当前状态

    # 添加最后一个段
    segment = {
        'points': trajectory[start_idx:n_points],
        'in_range': current_state
    }
    segments.append(segment)

    return segments # 返回分段轨迹

# 保存轨迹到 CSV 文件的函数
def save_trajectories_to_csv(trajectories_dict, 
                             filename_prefix,
                             save_directory
):
    """
    将轨迹保存至CSV文件
    :param trajectories_dict: 轨迹字典
    :param filename_prefix: 文件名前缀
    :param save_path: 保存路径

    每种类型的轨迹将被保存为一个 CSV 文件，文件名为 "{save_directory}/{filename_prefix}_{key}.csv"
    CSV 文件包含列: id, x, y, z
    """
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    for key, trajectories in trajectories_dict.items():
        data = []
        for idx, trajectory in enumerate(trajectories):
            n_points = trajectory.shape[0]
            ids = np.full((n_points, 1), idx+1)  # 轨迹编号
            trajectory_with_id = np.hstack((ids, trajectory))
            data.append(trajectory_with_id)

        # 合并所有轨迹
        data = np.vstack(data)
        filename = os.path.join(save_directory, f"{filename_prefix}_{key}.csv")

        # 保存到 CSV 文件
        np.savetxt(filename, data, delimiter=',', header='id,x,y,z', comments='')
        print(f"轨迹已保存到 {filename}")

def save_radar_info_to_csv(radar_dict,
                           filename_prefix,
                           save_directory
):
    """
    将雷达信息保存到 CSV 文件
    :param radar_dict: 雷达信息字典
    :param filename_prefix: 文件名前缀
    :param save_directory: 保存路径
    """
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    radar_positions = radar_dict['radar_positions']
    radar_radii = radar_dict['radar_radii']
    radar_channal = radar_dict['radar_channal']

    data = np.column_stack((radar_positions, radar_radii, radar_channal))
    filename = os.path.join(save_directory, f"{filename_prefix}.csv")
    np.savetxt(filename, data, delimiter=',', header='x,y,z,radii,channel', comments='')
    print(f"雷达信息已保存到 {filename}")

if __name__ == '__main__':
    # 设置参数
    num_ballistic_missiles = 20
    num_airplanes = 80
    num_cruise_missiles = 100
    x_range = (0, 10000)
    y_range = (0, 10000)
    target_area_x = (0, 10000)
    target_area_y = (-20000, -10000)
    radar_center_x = 5000
    radar_center_y = -10000

    # 生成初始位置
    positions = generate_initial_positions(
        num_ballistic_missiles=num_ballistic_missiles,
        num_airplanes=num_airplanes,
        num_cruise_missiles=num_cruise_missiles,
        x_range=x_range,
        y_range=y_range,
        airplane_altitude=10000,
        ballistic_missile_altitude=30000,
        cruise_missile_altitude=10000
    )

    # 生成弹道导弹轨迹
    ballistic_missile_trajectories = generate_ballistic_missile_trajectories(
        initial_positions=positions['ballistic_missiles'],
        num_points=50, 
        target_area_x=target_area_x,
        target_area_y=target_area_y
    )

    # 生成飞机轨迹
    airplane_trajectories = generate_airplane_trajectories(
        num_planes=num_airplanes,
        num_points=500,
        initial_positions=positions['airplanes'],
        target_area_y=target_area_y,
        target_altitude_range=(8000, 12000)  # 目标高度范围
    )

    # 生成巡航导弹轨迹
    cruise_missile_trajectories = generate_cruise_missile_trajectories(
        num_missiles=num_cruise_missiles,
        num_points=1000,
        initial_positions=positions['cruise_missiles'],
        target_area_x=target_area_x,
        target_area_y=target_area_y,
        cruise_altitude=10000,
        cruise_altitude_variation=1000  # 巡航高度变化范围
    )

    # 生成雷达
    radar_positions, radar_radii, radar_channal = generate_radars(
        num_radars=20,
        radii=12000,
        center_x=radar_center_x,
        center_y=radar_center_y
    )

    # 绘制结果
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制弹道导弹轨迹
    for trajectory in ballistic_missile_trajectories:
        coverage = get_trajectory_coverage(trajectory, radar_positions, radar_radii)
        segments = split_trajectory_by_coverage(trajectory, coverage)
        # 绘制分段
        for segment in segments:
            points = segment['points']
            if len(points) < 2:
                continue
            linestyle = '-' if segment['in_range'] else '--'
            ax.plot(points[:, 0], points[:, 1], points[:, 2], color='red', linestyle=linestyle, alpha=0.7)
        # 起点用小正方形表示
        ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], color='red', marker='s')

    # 绘制飞机轨迹
    for trajectory in airplane_trajectories:
        coverage = get_trajectory_coverage(trajectory, radar_positions, radar_radii)
        segments = split_trajectory_by_coverage(trajectory, coverage)
        # 绘制分段
        for segment in segments:
            points = segment['points']
            if len(points) < 2:
                continue
            linestyle = '-' if segment['in_range'] else '--'
            ax.plot(points[:, 0], points[:, 1], points[:, 2], color='green', linestyle=linestyle, alpha=0.7)
        # 起点用小圆形表示
        ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], color='green', marker='o')

    # 绘制巡航导弹轨迹
    for trajectory in cruise_missile_trajectories:
        coverage = get_trajectory_coverage(trajectory, radar_positions, radar_radii)
        segments = split_trajectory_by_coverage(trajectory, coverage)
        # 绘制分段
        for segment in segments:
            points = segment['points']
            if len(points) < 2:
                continue
            linestyle = '-' if segment['in_range'] else '--'
            ax.plot(points[:, 0], points[:, 1], points[:, 2], color='blue', linestyle=linestyle, alpha=0.7)
        # 起点用小三角形表示
        ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], color='blue', marker='^')

    # 绘制雷达位置
    ax.scatter(radar_positions[:, 0], radar_positions[:, 1], radar_positions[:, 2], color='black', marker='x', s=100)

    # 绘制雷达的侦测范围（半球）
    for i in range(len(radar_positions)):
        x0, y0, z0 = radar_positions[i]
        r = radar_radii[i]
        # 生成上半球数据
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi / 2, 10)
        u, v = np.meshgrid(u, v)
        xs = x0 + r * np.cos(u) * np.sin(v)
        ys = y0 + r * np.sin(u) * np.sin(v)
        zs = z0 + r * np.cos(v)
        ax.plot_surface(xs, ys, zs, color='cyan', alpha=0.1)

    # 设置坐标轴标签
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')

    # 设置坐标轴范围
    ax.set_xlim(-25000, 25000)
    ax.set_ylim(-25000, 25000)
    ax.set_zlim(0, 40000)

    plt.title('Ballistic Missile, Plane, Cruise Missile Trajectories and Radar Coverage')
    plt.show()

    # 保存轨迹到 CSV 文件
    trajectories_dict = {
        'ballistic_missiles': ballistic_missile_trajectories,
        'airplanes': airplane_trajectories,
        'cruise_missiles': cruise_missile_trajectories
    }
    save_trajectories_to_csv(trajectories_dict, filename_prefix='trajectories', save_directory='./trajectories')
            
    radar_dict = {
        'radar_positions': radar_positions,
        'radar_radii': radar_radii,
        'radar_channal': radar_channal
    }
    save_radar_info_to_csv(radar_dict, filename_prefix='radar_info', save_directory='./trajectories')
        