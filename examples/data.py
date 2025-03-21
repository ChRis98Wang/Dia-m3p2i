# collect_data_example.py

import os
import math
import numpy as np
import csv
import torch

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

def main():
    # ==============================================================
    # 1. 初始化和参数
    # ==============================================================
    # 从命令行解析标准参数，例如：--headless
    args = gymutil.parse_arguments()

    # 创建 Isaac Gym 的 gym 对象
    gym = gymapi.acquire_gym()

    # 配置模拟参数
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z           # Z轴为竖直
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8) # 重力
    sim_params.dt = 1.0 / 60.0                      # 仿真时间步长
    sim_params.substeps = 2
    sim_params.physx.use_gpu = args.use_gpu         # 如果需要 GPU 物理
    sim_params.physx.num_threads = 4

    # 创建模拟器
    # 注意：如果你的 GPU 设备ID 或 Gpu Physics Device 不同，需要调整
    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
    if sim is None:
        raise Exception("Failed to create sim")

    # 创建一个 viewer (可视化窗口)，如果是 headless 模式就不用
    if not args.headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            raise Exception("Failed to create viewer")

    # ==============================================================
    # 2. 创建环境 (env) 并加载一个Actor (Franka或其它)
    # ==============================================================
    # 每个环境的尺寸（地面大小等）
    env_lower = gymapi.Vec3(-1.0, -1.0, 0.0)
    env_upper = gymapi.Vec3(1.0, 1.0, 1.0)

    # 创建单个环境
    num_envs = 1
    envs = []
    for i in range(num_envs):
        env = gym.create_env(sim, env_lower, env_upper, 1)
        envs.append(env)

    # 加载Franka(或你自己的URDF) 的 asset
    asset_root = os.path.join(os.path.dirname(__file__), "assets")  # 假设你的URDF放在这里
    franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True    # 机器人底座固定
    asset_options.armature = 0.01

    franka_asset = gym.load_asset(sim, asset_root, franka_asset_file, asset_options)

    # 在环境中创建Actor
    # 为了演示，我们只在第一个env创建1个Actor
    pose = gymapi.Transform()
    pose.p.x = 0.0
    pose.p.y = 0.0
    pose.p.z = 0.0

    franka_handles = []
    for i, env in enumerate(envs):
        franka_handle = gym.create_actor(env, franka_asset, pose, "franka", i, 1)
        franka_handles.append(franka_handle)

    # 获取Franka的关节数量
    franka_num_dofs = gym.get_asset_dof_count(franka_asset)

    # ==============================================================
    # 3. 准备读取张量 (GPU 或 CPU) 并做后续数据收集
    # ==============================================================
    # (1) 获取 root state tensor (每个 Actor 的根部位姿/速度)
    actor_root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
    # (2) 获取 DOF state tensor (每个关节的角度/速度)
    dof_state_tensor = gym.acquire_dof_state_tensor(sim)
    # (3) 把张量包装成 torch Tensor 方便操作
    #     device 可以是 "cpu" 或 "cuda"。如果要在GPU上操作，可以使用 "cuda"
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)

    root_states = gymtorch.wrap_tensor(actor_root_state_tensor)
    dof_states  = gymtorch.wrap_tensor(dof_state_tensor)

    # ==============================================================
    # 4. 仿真循环，收集数据
    # ==============================================================
    # 存储数据的列表
    # 每一步保存 root_states, dof_states, 以及你想要的其它信息(比如动作)
    data_records = []

    # 假设我们仿真 300 步
    max_steps = 300

    # 这里假设我们想让关节保持一个简单的目标位置(随机也行)
    # 需要先获取关节目标张量
    dof_props = gym.get_actor_dof_properties(envs[0], franka_handles[0])
    dof_targets = torch.zeros((franka_num_dofs,), dtype=torch.float32)

    for step in range(max_steps):
        # 刷新张量，将最新状态同步到 root_states / dof_states
        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)

        # 读取当前的根状态(位置,姿态,线速度,角速度等)
        # 注意 root_states 的大小是 [num_actors, 13]
        # 这里我们只有 1 个 env, 1 个 actor，那么 index=0
        current_root_state = root_states[0].cpu().numpy()  # 转到 numpy 方便存储
        # 读取当前关节状态(角度,角速度)
        # dof_states 的大小是 [num_total_dofs, 2]
        # franka_num_dofs = 7 (Franka一般7个关节)
        # 如果只有1个Actor，前7行就是它
        current_dof_state = dof_states[:franka_num_dofs].cpu().numpy()

        # 这里可以定义一些简单的动作:
        # 比如让第0个关节缓慢转动
        dof_targets[0] = 0.5 * math.sin(step * 0.1)

        # 设置目标 (position target)，让 Isaac Gym 进行PD控制
        # 先获取 dof_targ_tensor
        dof_targ_tensor = gym.acquire_dof_position_target_tensor(sim)
        targ_tensor = gymtorch.wrap_tensor(dof_targ_tensor)

        # 把本步的目标写进去
        targ_tensor[:franka_num_dofs] = dof_targets

        # 仿真一步
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # 如果使用 viewer，可视化更新
        if not args.headless:
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

        # 将当前状态保存到 data_records
        record = {
            "step": step,
            "root_state": current_root_state.tolist(),  # 转为python list
            "dof_state": current_dof_state.tolist(),    # 同上
            "action_targets": dof_targets.cpu().numpy().tolist()
        }
        data_records.append(record)

    # ==============================================================
    # 5. 仿真完成后，将数据导出
    # ==============================================================
    # 可以存CSV、JSON或Numpy等，这里以CSV为例
    csv_file_name = "simulation_data.csv"
    with open(csv_file_name, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["step", "root_state", "dof_state", "action_targets"])
        writer.writeheader()
        for row in data_records:
            writer.writerow(row)

    print(f"数据已保存到: {csv_file_name}")

    # 如果有 viewer，记得在结束前关闭
    if not args.headless:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
