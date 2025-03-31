import torch
from m3p2i_aip.utils import skill_utils
import  m3p2i_aip.utils.isaacgym_utils.isaacgym_wrapper as wrapper
import torch .jit
@torch.jit.script
def gripper_vertical_cost(
    quat: torch.Tensor,        # [N,4] EE 四元数 (x,y,z,w)
    z_world: torch.Tensor,     # [3]   世界 Z 轴  (0,0,1)
    w_far: float = 4.0,        # 远距离的权重
    w_near: float = 25.0,      # 近距离的权重
    r_switch: float = 0.15     # 切换半径 (m)
) -> torch.Tensor:
    """
    余弦损失：cost = w * (1 - |<z_local, z_world>|)
    - <·,·> 取绝对 ⇒ 无论爪尖向上/向下都 OK
    - w 随 EE-goal 距离 piece-wise 变化
    """
    # 局部 z 轴方向向量 (经典四元数旋转公式)
    x, y, z, w = quat[:,0], quat[:,1], quat[:,2], quat[:,3]
    z_local = torch.stack([
        2*(x*z + y*w),
        2*(y*z - x*w),
        1 - 2*(x*x + y*y)
    ], dim=1)                                 # [N,3]

    # 余弦 |cosθ|
    cos = torch.abs((z_local * z_world).sum(1))             # [N]

    # 动态权重
    # 这里直接在调用处给 dist ⇒ 用 torch.where
    return 1.0 - cos
@torch.jit.script
def _anisotropic_goal_cost(
        pos: torch.Tensor,
        goal: torch.Tensor,
        n_hat: torch.Tensor,
        w_para: float = 8.0,
        w_perp: float = 1.0,
) -> torch.Tensor:
    diff = goal - pos
    e_para = (diff @ n_hat)[:, None] * n_hat
    e_perp = diff - e_para
    return w_para * torch.linalg.norm(e_para, dim=1) + \
        w_perp * torch.linalg.norm(e_perp, dim=1)
@torch.jit.script
def _xy_clearance_cost(
    obj_xy: torch.Tensor,        # [N,2] cube 或 EE 的 xy
    obs_xy: torch.Tensor,        # [N,2] 障碍中心 xy
    safe_r: float = 0.25,        # 安全半径 (m)
    k: float = 200.0             # 权重
) -> torch.Tensor:
    """
    If distance in XY < safe_r:  cost = k * (safe_r - d)^2
    Else:                       cost = 0
    """
    d = torch.linalg.norm(obj_xy - obs_xy, dim=1)
    v = torch.clamp(safe_r - d, min=0.0)
    return k * v * v
def _piecewise_goal_cost(dist, r_switch=0.05,
                         alpha_far=30.0, alpha_near=5.0):
    far = dist > r_switch
    cost = torch.zeros_like(dist)
    cost[far]  = alpha_far * dist[far]
    cost[~far] = alpha_near * dist[~far]
    return cost

@torch.jit.script
def scripted_get_motion_cost(obs_force: torch.Tensor, threshold: float, k: float) -> torch.Tensor:
        # 计算 x,y 分量的接触力范数
    force_magnitude = torch.linalg.norm(obs_force[:, :2], dim=1)
        # 限制 force_magnitude 的最大值，防止数值异常
    force_magnitude = torch.clamp(force_magnitude, min=0.0, max=50.0)
    cost = torch.zeros_like(force_magnitude)
        # 当 force_magnitude 超过阈值时，成本按二次增长
    mask = force_magnitude > threshold
    cost[mask] = k * (force_magnitude[mask] - threshold) ** 2
    return cost
@torch.jit.script
def _verticality_cost(
    ee_quat: torch.Tensor,   # [N,4]
    w: float = 8.0           # 权重，可调
) -> torch.Tensor:
    """
    Encourage gripper's local Z‑axis to align with world Z.
    cost = w * (1 - |n_z|)^2
    """
    # 取局部 z 轴（常用四元数->方向向量公式）
    q = ee_quat
    nz = 2*(q[:,0]*q[:,2] + q[:,1]*q[:,3])          # 世界 z 分量
    nz = nz.clamp(-1.0, 1.0)                        # 数值安全
    return w * (1.0 - nz.abs()) ** 2

@torch.jit.script
def _height_piecewise_cost(
    z: torch.Tensor,           # [N]  物块当前 z 高度
    low_thresh: float = 1.2,   # 无惩罚上限
    high_thresh: float = 1.3,  # 二次段上限
    k: float = 1.0,            # 二次段权重
    hard_penalty: float = 1e3  # z > high_thresh 时的常数惩罚
) -> torch.Tensor:
    """
    Piece‑wise cost:
      region 1  (z <= low_thresh):        cost = 0
      region 2  (low_thresh < z <= high): cost = k * (z - low_thresh)^2
      region 3  (z > high_thresh):        cost = hard_penalty
    """
    cost = torch.zeros_like(z)

    # region 2: quadratic growth
    mask_quad = (z > low_thresh) & (z <= high_thresh)
    cost[mask_quad] = k * (z[mask_quad] - low_thresh) ** 2

    # region 3: constant large penalty
    cost[z > high_thresh] = hard_penalty

    return cost

class Objective(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.multi_modal = cfg.multi_modal
        self.num_samples = cfg.mppi.num_samples
        self.half_samples = int(cfg.mppi.num_samples/2)
        self.device = self.cfg.mppi.device
        self.pre_height_diff = cfg.pre_height_diff
        self.tilt_cos_theta = 0.5

    def update_objective(self, task, goal):
        self.task = task
        self.goal = goal if torch.is_tensor(goal) else torch.tensor(goal, device=self.device)

    def compute_cost(self, sim: wrapper):
        task_cost = 0
        if self.task == "navigation":
            task_cost = self.get_navigation_cost(sim)
        elif self.task == "push":
            return self.get_push_cost(sim, self.goal)
        elif self.task == "pull":
            return self.get_pull_cost(sim, self.goal)
        elif self.task == "push_pull":
            return torch.cat((self.get_push_cost(sim, self.goal)[:self.half_samples],
                              self.get_pull_cost(sim, self.goal)[self.half_samples:]), dim=0)
        elif self.task == "reach":
            task_cost= self.get_panda_reach_cost(sim, self.goal)
        elif self.task == "pick":
            task_cost = self.get_panda_pick_cost(sim, self.goal)
        elif self.task == "place":
            return self.get_panda_place_cost(sim)
        return task_cost + 10*self.get_motion_cost(sim)

    def get_height_piecewise_cost(self, sim):
        # 取红色物块 cubeA 的 z 高度
        cube_z = sim.get_actor_link_by_name("cubeA", "box")[:, 2]
        return _height_piecewise_cost(
            cube_z,
            low_thresh=1.15,   # <=1.2 m 免费
            high_thresh=1.25,  # >1.3 m 直接罚 hard_penalty
            k=5.0,            # 二次段权重 (可调)
            hard_penalty=1000 # 常数罚值 (可调)
        )



    def get_navigation_cost(self, sim: wrapper):
        return torch.linalg.norm(sim.robot_pos - self.goal, axis=1)


    def calculate_dist(self, sim: wrapper, block_goal: torch.tensor):
        self.block_pos = sim.get_actor_position_by_name("box")[:, :2]  # x, y position
        robot_to_block = sim.robot_pos - self.block_pos
        block_to_goal = block_goal - self.block_pos

        robot_to_block_dist = torch.linalg.norm(robot_to_block, axis = 1)
        block_to_goal_dist = torch.linalg.norm(block_to_goal, axis = 1)
        eps = 1e-6
        print("robot_to_block_dist:", robot_to_block_dist)
        print("block_to_goal_dist:", block_to_goal_dist)

        self.dist_cost = robot_to_block_dist + block_to_goal_dist * 10
        self.cos_theta = torch.sum(robot_to_block*block_to_goal, 1)/(torch.clamp(robot_to_block_dist * block_to_goal_dist, min=eps))
        print("cos_theta:", self.cos_theta)
    def get_push_cost(self, sim: wrapper, block_goal: torch.tensor):
        # Calculate dist cost
        self.calculate_dist(sim, block_goal)

        # Force the robot behind block and goal, align_cost is actually cos(theta)+1
        align_cost = torch.zeros(self.num_samples, device=self.device)
        align_cost[self.cos_theta>0] = self.cos_theta[self.cos_theta>0]

        return 3 * self.dist_cost + 1 * align_cost

    def get_pull_cost(self, sim: wrapper, block_goal: torch.tensor):
        self.calculate_dist(sim, block_goal)
        pos_dir = self.block_pos - sim.robot_pos
        robot_to_block_dist = torch.linalg.norm(pos_dir, axis = 1)

        # True means the velocity moves towards block, otherwise means pull direction
        flag_towards_block = torch.sum(sim.robot_vel*pos_dir, 1) > 0

        # simulate a suction to the box
        suction_force = skill_utils.calculate_suction(self.cfg, sim)
        # Set no suction force if robot moves towards the block
        suction_force[flag_towards_block] = 0
        if self.multi_modal:
            suction_force[:self.half_samples] = 0
        sim.apply_rigid_body_force_tensors(suction_force)

        self.calculate_dist(sim, block_goal)

        # Force the robot to be in the middle between block and goal, align_cost is actually 1-cos(theta)
        align_cost = torch.zeros(self.num_samples, device=self.device)
        align_cost[self.cos_theta<0] = -self.cos_theta[self.cos_theta<0]  # (1 - cos_theta)

        # Add the cost when the robot is close to the block and moves towards the block
        vel_cost = torch.zeros(self.num_samples, device=self.device)
        robot_block_close = robot_to_block_dist <= 0.5
        vel_cost[flag_towards_block*robot_block_close] = 0.6

        return 3 * self.dist_cost + 3 * vel_cost + 7 * align_cost

    """
    def get_panda_reach_cost(self, sim, pre_pick_goal: torch.Tensor) -> torch.Tensor:
        
       
        
        # 1. 获取末端执行器状态
        ee_l_state = sim.get_actor_link_by_name("panda", "panda_leftfinger")
        ee_r_state = sim.get_actor_link_by_name("panda", "panda_rightfinger")
        # 假设前 3 个元素为位置
        ee_state = (ee_l_state + ee_r_state) / 2  # shape: [num_envs, 7]，前3为位置

        # 2. 位置成本：鼓励 EE 走向目标
        pos_cost = torch.linalg.norm(ee_state[:, :3] - pre_pick_goal, dim=1)

        # 3. 障碍物避让成本：选用 "cubeC" 作为障碍物
        obs_pos = sim.get_actor_position_by_name("cubeC")  # shape: [num_envs, 3]
        obs_dist = torch.linalg.norm(ee_state[:, :3] - obs_pos, dim=1)
        safe_distance = 0.5  # 期望与障碍物保持的安全距离（单位：米）
        # 当距离小于安全距离时，采用二次函数惩罚
        obs_cost = torch.zeros_like(obs_dist)
        mask = obs_dist < safe_distance
        obs_cost[mask] = ((safe_distance - obs_dist[mask]) / safe_distance) ** 2

        # 4. 方向性成本：鼓励 EE 面向目标
        # 获取 panda 的姿态（四元数）
        panda_ori = sim.get_actor_orientation_by_name("panda")  # shape: [num_envs, 4]
        # 根据常用公式，将四元数转换为前向向量（这里假设前向为 panda 局部 x 轴）
        q = panda_ori
        forward = torch.stack([
            1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2),
            2 * (q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3]),
            2 * (q[:, 0] * q[:, 2] - q[:, 1] * q[:, 3])
        ], dim=1)
        # 计算 EE 到目标的单位方向向量
        to_target = pre_pick_goal - ee_state[:, :3]
        to_target = to_target / torch.clamp(torch.linalg.norm(to_target, dim=1, keepdim=True), min=1e-6)
        # 余弦相似度（越大说明朝向越正确）
        cos_angle = torch.sum(forward * to_target, dim=1)
        # 定义方向成本：1 - cos_angle
        direction_cost = 1 - cos_angle

        # 5. 速度平滑成本：鼓励 EE 运动平滑
        ee_velocity = sim.get_actor_velocity_by_name("panda")  # shape: [num_envs, 3]
        vel_cost = torch.linalg.norm(ee_velocity, dim=1)

        # 6. 综合成本，各项权重可以根据实际情况调整
        total_cost = (
                5 * pos_cost +  # 位置代价
                10 * obs_cost +  # 障碍避让代价（当靠近障碍时会迅速增加）
                3 * direction_cost +  # 方向性代价
                2 * vel_cost  # 速度平滑代价
        )
        return total_cost
    """
    def get_panda_reach_cost(self, sim, pre_pick_goal):
        ee_l_state = sim.get_actor_link_by_name("panda", "panda_leftfinger")
        ee_r_state = sim.get_actor_link_by_name("panda", "panda_rightfinger")
        ee_state = (ee_l_state + ee_r_state) / 2
        cube_state = sim.get_actor_link_by_name("cubeA", "box")
        ee_state = torch.nan_to_num(ee_state, nan=0.0, posinf=1e3, neginf=-1e3)
        cube_state = torch.nan_to_num(cube_state, nan=0.0, posinf=1e3, neginf=-1e3)
        if not self.multi_modal:
            pre_pick_goal = cube_state[0, :3].clone()
            pre_pick_goal[2] += self.pre_height_diff
            reach_cost = torch.linalg.norm(ee_state[:,:3] - pre_pick_goal, axis = 1)
        else:
            pre_pick_goal = cube_state[:, :3].clone()
            pre_pick_goal_1 = cube_state[0, :3].clone()
            pre_pick_goal_2 = cube_state[0, :3].clone()
            pre_pick_goal_1[2] += self.pre_height_diff
            pre_pick_goal_2[0] -= self.pre_height_diff * self.tilt_cos_theta
            pre_pick_goal_2[2] += self.pre_height_diff * (1 - self.tilt_cos_theta ** 2) ** 0.5
            pre_pick_goal[:self.half_samples, :] = pre_pick_goal_1
            pre_pick_goal[self.half_samples:, :] = pre_pick_goal_2
            reach_cost = torch.linalg.norm(ee_state[:,:3] - pre_pick_goal, axis = 1)
        static_obs_cost = self.get_static_obstacle_cost(sim)
        # Compute the tilt value between ee and cube
        tilt_cost = self.get_pick_tilt_cost(sim)
        total_cost = 10* reach_cost + tilt_cost+ static_obs_cost
        return torch.nan_to_num(total_cost, nan=1e6, posinf=1e6, neginf=0)

    def get_static_obstacle_cost(self, sim):
        """
        计算末端执行器与静态障碍之间的避碰成本。
        若末端与障碍物的距离小于 safe_distance，则二次惩罚，
        并按一定权重累加各个障碍的惩罚值。
        """
        # 定义静态障碍名称列表（根据实际环境修改）
        static_obstacles = ["cubeC", "cubeD", "cubeE", "shelf_stand"]
        # 获取末端执行器状态（这里采用左右手指的平均作为 EE 位置）
        ee_l_state = sim.get_actor_link_by_name("panda", "panda_leftfinger")
        ee_r_state = sim.get_actor_link_by_name("panda", "panda_rightfinger")
        ee_state = (ee_l_state + ee_r_state) / 2  # shape: [num_envs, 13]，前3个为位置

        safe_distance = 0.02  # 期望与障碍物保持的安全距离（单位：米）
        cost = torch.zeros(ee_state.shape[0], device=self.device)
        for obs_name in static_obstacles:
            obs_pos = sim.get_actor_position_by_name(obs_name)  # shape: [num_envs, 3]
            # 计算 EE 与障碍物的欧氏距离
            distance = torch.linalg.norm(ee_state[:, :3] - obs_pos, dim=1)
            # 当距离小于 safe_distance 时，采用二次惩罚（可根据实际情况调整惩罚系数）
            penalty = torch.clamp(safe_distance - distance, min=0) ** 2
            cost += penalty
        # 可按权重放大该成本（例如乘以 10）
        return cost * 10.0
    """"
    def get_panda_pick_cost(self, sim, pre_place_state):
        cube_state = sim.get_actor_link_by_name("cubeA", "box")

        # Move to pre-place location
        goal_cost = torch.linalg.norm(pre_place_state[:3] - cube_state[:,:3], axis = 1)
        cube_quaternion = cube_state[:, 3:7]
        goal_quatenion = pre_place_state[3:7].repeat(self.num_samples).view(self.num_samples, 4)
        ori_cost = skill_utils.get_general_ori_cube2goal(cube_quaternion, goal_quatenion) 

        return   goal_cost + ori_cost
    """
    def get_panda_place_cost(self, sim):
        # task planner will send discrete gripper commands instead of sampling

        # Just to make mppi running! Actually this is not useful!!
        ee_l_state = sim.get_actor_link_by_name("panda", "panda_leftfinger")
        ee_r_state = sim.get_actor_link_by_name("panda", "panda_rightfinger")
        gripper_dist = torch.linalg.norm(ee_l_state[:, :3] - ee_r_state[:, :3], axis=1)
        gripper_cost = 2 * (1 - gripper_dist)

        return gripper_cost
    def get_static_side_cost(self, sim):
        """
        只作用于——已夹块后（pick / place）——鼓励 cubeA 在 XY 上
        离静态障碍 > safe_r。你也可以换成 EE 中心。
        """
        cube_xy = sim.get_actor_link_by_name("cubeA", "box")[:, :2]

        # 障碍中心可硬编码或实时读位置
        obs_names = ["cubeC", "cubeD", "cubeE", "shelf_stand"]
        safe_r = 0.07  # 22 cm 环
        k_r = 250.0  # 权重

        cost = torch.zeros(self.num_samples, device=self.device)
        for name in obs_names:
            obs_xy = sim.get_actor_position_by_name(name)[:, :2]
            cost += _xy_clearance_cost(cube_xy, obs_xy, safe_r, k_r)
        return cost


    def get_panda_pick_cost(self, sim, pre_place_state):
        cube_state = sim.get_actor_link_by_name("cubeA", "box")

        # 分离出位置和方向
        pos_cube = cube_state[:, :3]
        pos_goal = pre_place_state[:3]
        obs_center = sim.get_actor_position_by_name("shelf_stand")[0, :3]  # 任选最碍眼的障碍
        diff = pos_goal - pos_cube
        n_hat = torch.nn.functional.normalize(pos_goal - obs_center, dim=0)
        e_para = (diff @ n_hat)[:, None] * n_hat
        e_perp = diff - e_para
        dist = torch.linalg.norm(diff, dim=1)

        # --- 1 动态各向异性 ---
        w_para = torch.where(dist > 0.2, 1.5, 9.5)
        goal_cost = (w_para * torch.linalg.norm(e_para, dim=1) + \
                    1.0 * torch.linalg.norm(e_perp, dim=1))
        ee_l = sim.get_actor_link_by_name("panda", "panda_leftfinger")
        ee_r = sim.get_actor_link_by_name("panda", "panda_rightfinger")
        ee_quat = 0.5 * (ee_l[:, 3:7] + ee_r[:, 3:7])  # [N,4]
        pos_cube = sim.get_actor_link_by_name("cubeA", "box")[:, :3]
        diff = pre_place_state[:3] - pos_cube  # [N,3]
        dist = torch.linalg.norm(diff, dim=1)  # [N]

        # (3) 垂直 cost + 权重
        raw_vert_cost = gripper_vertical_cost(
            ee_quat, torch.tensor([0., 0., 1.], device=self.device)
        )  # [N]

        w_vert = torch.where(dist > 0.15, 4.0, 25.0)  # [N]
        vert_cost = w_vert * raw_vert_cost


        # ---------- (A) 各向异性 goal cost ----------


        #weight = torch.tensor([5.0, 5.0, 1.0], device=pos_cube.device)
        #goal_cost = torch.linalg.norm((pos_goal - pos_cube) * weight, axis=1)

        print(f'goal cost is {goal_cost}')

        cube_quaternion = cube_state[:, 3:7]
        goal_quaternion = pre_place_state[3:7].repeat(self.num_samples).view(self.num_samples, 4)
        #ori_cost = skill_utils.get_general_ori_cube2goal(cube_quaternion, goal_quaternion)
        total_cost = 20*goal_cost  #+ ori_cost  # 或者加上其他 cost
        overcost=self.get_height_piecewise_cost(sim)
        #ee_l = sim.get_actor_link_by_name("panda", "panda_leftfinger")
        #total_cost += _verticality_cost(ee_l[:, 3:7], w=18.0)
        #static_obs_cost = self.get_static_obstacle_cost(sim)
        #xycost = self.get_static_side_cost(sim)

        #total_cost = torch.nan_to_num(total_cost, nan=1e6, posinf=1e6, neginf=0)
        return total_cost+overcost+ vert_cost#+xycost#+static_obs_cost

    def get_pick_tilt_cost(self, sim):
        # This measures the cost of the tilt angle between the end effector and the cube
        ee_l_state = sim.get_actor_link_by_name("panda", "panda_leftfinger")
        ee_quaternion = ee_l_state[:, 3:7]
        cubeA_ori = sim.get_actor_orientation_by_name("cubeA")
        ee_quaternion = torch.nan_to_num(ee_quaternion, nan=0.0)
        cubeA_ori = torch.nan_to_num(cubeA_ori, nan=0.0)
        # cube_quaternion = cube_state[:, 3:7]
        if not self.multi_modal:
            # To make the z-axis direction of end effector to be perpendicular to the cube surface
            ori_ee2cube = skill_utils.get_general_ori_ee2cube(ee_quaternion, cubeA_ori, tilt_value=0)
            # ori_ee2cube = skill_utils.get_ori_ee2cube(ee_quaternion, cubeA_ori)
        else:
            # To combine costs of different tilt angles
            cost_1 = skill_utils.get_general_ori_ee2cube(ee_quaternion[:self.half_samples],
                                                            cubeA_ori[:self.half_samples], tilt_value = 0)
            cost_2 = skill_utils.get_general_ori_ee2cube(ee_quaternion[self.half_samples:],
                                                            cubeA_ori[self.half_samples:], tilt_value = self.tilt_cos_theta)
            ori_ee2cube =  torch.cat((cost_1, cost_2), dim=0)
        ori_ee2cube = torch.nan_to_num(ori_ee2cube, nan=1e6, posinf=1e6, neginf=0)

        return 3 * ori_ee2cube

    '''
    def get_motion_cost(self, sim):
        if self.cfg.env_type == 'point_env':
            obs_force = sim.get_actor_contact_forces_by_name("dyn-obs", "box")  # [num_envs, 3]
        elif self.cfg.env_type == 'panda_env':
            # 静态障碍部分
            static_force = (sim.get_actor_contact_forces_by_name("table", "box") +
                            sim.get_actor_contact_forces_by_name("shelf_stand", "box") +
                            sim.get_actor_contact_forces_by_name("cubeB", "box") +
                            sim.get_actor_contact_forces_by_name("cubeC", "box"))
            # 动态障碍部分：放大权重以突出重要性
            dyn_force =( sim.get_actor_contact_forces_by_name("dyn-obs", "panda") + sim.get_actor_contact_forces_by_name("dyn-obs_", "panda"))
            obs_force = 2*static_force + 10 * dyn_force

        # 计算 x,y 分量的接触力大小（假设 z 方向的力对避碰不敏感）
        force_magnitude = torch.linalg.norm(obs_force[:, :2], dim=1)
        # 为防止异常值，限制 force_magnitude 的最大值（例如 50）
        force_magnitude = torch.clamp(force_magnitude, min=0.0, max=50.0)

        # 设定一个触发阈值，低于该阈值认为不产生碰撞风险
        threshold = 0.1
        # 这个值可以根据实际统计数据进行调节

        # 对于超过阈值的部分采用二次惩罚，平滑地增长成本
        # 即：cost = k * (force_magnitude - threshold)^2，当 force_magnitude > threshold，否则 cost = 0
        k = 5.0  # 惩罚系数，根据实际效果调整
        cost = torch.zeros_like(force_magnitude)
        mask = force_magnitude > threshold
        cost[mask] = k * (force_magnitude[mask] - threshold) ** 2

        # 如果需要，cost 还可以加上一个静态项（例如直接使用 force_magnitude 的一部分），
        # 但这里主要关注超过阈值部分的非线性惩罚
        return cost
    '''


    #sim.get_actor_contact_forces_by_name("shelf_stand", "box")+
    def get_motion_cost(self, sim):
        if self.cfg.env_type == 'point_env':
            obs_force = sim.get_actor_contact_forces_by_name("dyn-obs", "box")  # [num_envs, 3]
        elif self.cfg.env_type == 'panda_env'or self.cfg.env_type == 'panda_env_LiftedObstaclesShelf'or self.cfg.env_type=="panda_env_2dyn"or self.cfg.env_type=="panda_env_2dyx2"or self.cfg.env_type=="panda_env_lift_maze":
            static_force = (
                            sim.get_actor_contact_forces_by_name("cubeC", "box")+
                            sim.get_actor_contact_forces_by_name("cubeD", "box")+
                            sim.get_actor_contact_forces_by_name("cubeE", "box")+
                            sim.get_actor_contact_forces_by_name("shelf_stand", "box")+
                            sim.get_actor_contact_forces_by_name("cubeF", "box")
                            )
                            #sim.get_actor_contact_forces_by_name("cubeE", "box"))
            dyn_force = (sim.get_actor_contact_forces_by_name("dyn-obs", "panda")
                         #+ sim.get_actor_contact_forces_by_name("dyn-obs_", "panda")
                         )
            static_force = torch.nan_to_num(static_force, nan=0.0, posinf=1e5, neginf=-1e3)
            dyn_force = torch.nan_to_num(dyn_force, nan=0.0, posinf=1e3, neginf=-1e3)
            obs_force =  10*static_force + 25*dyn_force
            obs_force = torch.clamp(obs_force, min=-1e3, max=1e5)

        # 调用脚本化函数计算 cost
        return scripted_get_motion_cost(10*obs_force, 5, 20.0)

    """
    def get_motion_cost(self, sim):
        if self.cfg.env_type == 'point_env':
            obs_force = sim.get_actor_contact_forces_by_name("dyn-obs", "box") # [num_envs, 3]
        elif self.cfg.env_type == 'panda_env':
            obs_force = sim.get_actor_contact_forces_by_name("table", "box")
            obs_force +=   sim.get_actor_contact_forces_by_name("shelf_stand", "box")
            obs_force += sim.get_actor_contact_forces_by_name("cubeB", "box")
            obs_force += sim.get_actor_contact_forces_by_name("cubeC", "box")
            #obs_force += sim.get_actor_contact_forces_by_name("cubeD", "box")
            #print("dynamic obs cost",sim.get_actor_contact_forces_by_name("dyn-obs", "box"))
            obs_force += 100*sim.get_actor_contact_forces_by_name("dyn-obs", "panda")
            #obs_force += 10*sim.get_actor_contact_forces_by_name("dyn-obs_", "panda")
            #print("**************************************************************")
            #print(obs_force.shape)
        coll_cost = torch.sum(torch.abs(obs_force[:, :2]), dim=1) # [num_envs]
        # Binary check for collisions.
        coll_cost[coll_cost>0.1] = 1
        coll_cost[coll_cost<=0.1] = 0
       # print(coll_cost)

        return 10*coll_cost
"""

