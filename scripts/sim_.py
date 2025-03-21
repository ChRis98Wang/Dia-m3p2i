from isaacgym import gymtorch
import torch, hydra, zerorpc, time
from m3p2i_aip.config.config_store import ExampleConfig
import m3p2i_aip.utils.isaacgym_utils.isaacgym_wrapper as wrapper
from m3p2i_aip.utils.data_transfer import bytes_to_torch, torch_to_bytes
from m3p2i_aip.utils.skill_utils import check_and_apply_suction, time_tracking

torch.set_printoptions(precision=3, sci_mode=False, linewidth=160)

'''
Run in the command line:
    python3 sim.py
    python3 sim.py task=pull
    python3 sim.py task=push_pull
    python3 sim.py -cn config_panda
    python3 sim_.py -cn config_panda multi_modal=True cube_on_shelf=True
'''


@hydra.main(version_base=None, config_path="../src/m3p2i_aip/config", config_name="config_point")
def run_sim(cfg: ExampleConfig):
    sim = wrapper.IsaacGymWrapper(
        cfg.isaacgym,
        cfg.env_type,
        num_envs=1,
        viewer=True,
        device=cfg.mppi.device,
        cube_on_shelf=cfg.cube_on_shelf,
    )
    noise_paraA = 0.9
    noise_paraB = 0.8
    planner = zerorpc.Client(heartbeat=60, timeout=60)
    planner.connect("tcp://127.0.0.1:4242")
    print("Server found and wait for the viewer")
    for _ in range(150):
        sim.step()
    print("Start simulation!")
    total_episodes = 150
    success_count = 0
    failure_count = 0
    log_file = "episode_log.txt"
    with open(log_file, "a") as f:
        f.write("========== Simulation Log N=1 ==========\n")
    # t = time.time()
    for ep in range(total_episodes):
        failure_reason = ""  # 初始化为一个空字符串
        success_reason = ""
        print(f"\n=== Episode {ep + 1}/{total_episodes} ===")
        # 先检查状态是否正常
        if torch.isnan(sim._root_state).any() or torch.isnan(sim._dof_state).any():
            failure_reason = "Episode 开始时检测到 NaN"
            print(failure_reason)
            # 重建环境并重置规划器
            sim.full_reset()
            planner.full_reset_reactive()
        else:
            sim.reset()

        planner.reset_planner()
        # 重置模拟环境（假设 IsaacGymWrapper 提供了 reset 方法）
        # if torch.isnan(sim._root_state).any() or torch.isnan(sim._dof_state).any():
        # print("检测到仿真状态异常，立即重置仿真状态")
        # planner.full_reset_reactive()
        # sim.full_reset()

        # else:
        # 若没有 reset 方法，则需要自行重置状态变量，例如：
        # sim._dof_state.zero_()
        # sim._root_state.zero_()
        # 并重新设置初始状态到 Isaac Gym 中（具体代码根据你的 wrapper 实现）
        sim.reset()

        planner.reset_planner()
        print("******************************************")

        ep_start_time = time.time()
        episode_success = False
        step_counter = 0
        last_timestamp = time.time()
        stream = torch.cuda.Stream()
        while time.time() - ep_start_time < 240:
            current_time = time.time()
            dt = current_time - last_timestamp
            if dt > 0:
                current_freq = 1.0 / dt
                print(f"当前控制频率：{current_freq:.2f} Hz")

            # 更新 last_timestamp 为当前时间
            last_timestamp = current_time
            step_counter += 1
            sim.update_dyn_obs(step_counter)
            sim.play_with_cube()
            # print(noise_paraA)
            cubeA_id = sim._get_actor_index_by_name("cubeA")
            cubeA_pos = sim._root_state[:, cubeA_id, :3]
            cubeB_id = sim._get_actor_index_by_name("cubeB")
            cubeB_pos = sim._root_state[:, cubeB_id, :3]
            # print(cubeA_pos, cubeB_pos)
            dist = torch.norm(cubeA_pos - cubeB_pos, dim=-1)
            print(dist)

            action = bytes_to_torch(
                planner.run_tamp(
                    torch_to_bytes(sim._dof_state), torch_to_bytes(sim._root_state), 33)
            )

            sim.set_dof_velocity_target_tensor(action)

            cfg.suction_active = bytes_to_torch(
                planner.get_suction()
            )
            check_and_apply_suction(cfg, sim, action)

            sim.step()

            # sim.step()
            # if torch.isnan(sim._root_state).any() or torch.isnan(sim._dof_state).any():
            # failure_reason = "仿真过程中检测到 NaN，执行 full_reset"
            # print("检测到仿真状态中 NaN，立即重置！")
            # planner.full_reset_reactive()
            # sim.full_reset()
            # episode_success = False
            # break
            # 假设 "dyn-obs" 对应的 actor 在 sim.env_cfg 中保存了 handle

            dyn_obs_force = sim.get_actor_contact_forces_by_name("dyn-obs",
                                                                 "box") + sim.get_actor_contact_forces_by_name(
                "dyn-obs_", "sphere")
            force_norm = torch.linalg.norm(dyn_obs_force, dim=1)
            # static_obs_force = sim.get_actor_contact_forces_by_name("cubeC", "box")
            # static_force_norm = torch.linalg.norm(static_obs_force, dim=1)
            print("force_dynamic: ", force_norm)
            threshold = 7  # 根据实际情况调整这个阈值
            if torch.any(force_norm > threshold):
                failure_reason = f"检测到动态障碍碰撞，接触力: {force_norm}"
                print("检测到动态障碍碰撞，接触力:", force_norm, "宣布本回合失败！")
                episode_success = False
                break

            if torch.all(action == 0) and planner.task_success:
                print("本回合任务成功！")
                success_reason = "任务成功：动作全为0且 planner.task_success 为 True"
                print("*******************************************************")
                episode_success = True
                break
        ep_duration = time.time() - ep_start_time
        if episode_success:
            success_count += 1
        else:
            failure_count += 1
            # failure_reason = f"本回合超时"
            print("本回合超时，任务未成功。")
            print("*******************************************************")

        print(f"Episode {ep + 1}结束。当前成功次数: {success_count}，失败次数: {failure_count}")
        print("Allocated:", torch.cuda.memory_allocated())
        print("Reserved:", torch.cuda.memory_reserved())

        print("\n=== 模拟结束 ===")
        print(f"总成功次数: {success_count}")
        print(f"总失败次数: {failure_count}")

        with open(log_file, "a") as f:
            if episode_success:
                f.write(f"Episode {ep + 1}: Success. Reason: {success_reason} 任务完成时间: {ep_duration:.2f}秒\n")
            else:
                if failure_reason is None:
                    failure_reason = f"超时，任务未成功；回合时长: {ep_duration:.2f}秒"
                f.write(f"Episode {ep + 1}: Failure. Reason: {failure_reason}\n")
        # torch.cuda.synchronize()

        # sim.visualize_trajs(
        # bytes_to_torch(planner.get_trajs())
        # )

        # t = time_tracking(t, cfg)
    print("All 100 simulation runs finished!")
    with open(log_file, "a") as f:
        f.write(f"Episode {ep + 1}结束。当前成功次数: {success_count}，失败次数: {failure_count}")


if __name__ == "__main__":
    run_sim()