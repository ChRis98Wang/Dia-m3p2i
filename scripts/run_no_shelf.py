from isaacgym import gymtorch
import torch, hydra, zerorpc, time
from m3p2i_aip.config.config_store import ExampleConfig
import  m3p2i_aip.utils.isaacgym_utils.isaacgym_wrapper as wrapper
from m3p2i_aip.utils.data_transfer import bytes_to_torch, torch_to_bytes
from m3p2i_aip.utils.skill_utils import check_and_apply_suction, time_tracking
torch.set_printoptions(precision=3, sci_mode=False, linewidth=160)

'''
运行说明：
    这个脚本创建一个没有架子的环境，机械臂将从桌面上夹取物体
    运行命令：
    python3 run_no_shelf.py -cn config_panda
'''

@hydra.main(version_base=None, config_path="../src/m3p2i_aip/config", config_name="config_panda")
def run_sim(cfg: ExampleConfig):
    # 使用panda_env_no_shelf环境类型，这是我们创建的不含架子的环境
    sim = wrapper.IsaacGymWrapper(
        cfg.isaacgym,
        env_type="panda_env_no_shelf",  # 使用不含架子的环境
        num_envs=1,
        viewer=True,
        device=cfg.mppi.device,
        cube_on_shelf=False,  # 确保方块放置在桌面上
    )
    noise_paraA = 0.9
    noise_paraB = 0.8
    planner = zerorpc.Client(heartbeat=60, timeout=60)
    planner.connect("tcp://127.0.0.1:4242")
    print("服务器已连接，等待查看器初始化")
    for _ in range(150):
        sim.step()
    print("开始仿真！")
    total_episodes = 150
    success_count = 0
    failure_count = 0
    log_file = "no_shelf_log.txt"
    with open(log_file, "a") as f:
        f.write("========== 无架子仿真日志 ==========\n")
    
    try:
        while success_count + failure_count < total_episodes:
            # 获取当前状态
            dof_state = torch_to_bytes(sim._dof_state)
            root_state = torch_to_bytes(sim._root_state)
            
            # 调用规划器计算动作
            joint_action = planner.run_tamp(dof_state, root_state, noise_paraA)
            
            # 执行动作
            desired_joint_pos = bytes_to_torch(joint_action)
            sim.set_dof_velocity_target_tensor(desired_joint_pos)
            
            # 获取夹爪状态信息
            suction_active = bytes_to_torch(planner.get_suction())
            
            # 应用吸盘（如果启用）- 修复类型错误
            if isinstance(suction_active, (int, bool)) or (isinstance(suction_active, torch.Tensor) and suction_active.numel() == 1):
                # 如果是标量值，直接检查是否为True或非零
                if isinstance(suction_active, torch.Tensor):
                    suction_enabled = bool(suction_active.item())
                else:
                    suction_enabled = bool(suction_active)
                
                if suction_enabled:
                    check_and_apply_suction(sim, suction_active)
            else:
                # 如果是张量，使用.any()方法
                if suction_active.any():
                    check_and_apply_suction(sim, suction_active)
            
            # 每帧步进仿真
            sim.step()
            
            # 更新统计信息
            if planner.task_success:
                success_count += 1
                with open(log_file, "a") as f:
                    f.write(f"Episode {success_count + failure_count}: Success\n")
                planner.full_reset_reactive()
            
            # 添加交互操作 - 按ESC键退出
            for evt in sim._gym.query_viewer_action_events(sim.viewer):
                if evt.action == "exit" and evt.value > 0:
                    print("用户请求退出")
                    break
    
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        print(f"总成功次数: {success_count}")
        print(f"总失败次数: {failure_count}")
        with open(log_file, "a") as f:
            f.write(f"最终统计：成功 {success_count}，失败 {failure_count}\n")

if __name__ == "__main__":
    run_sim() 