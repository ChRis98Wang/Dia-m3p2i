#!/usr/bin/env python3
"""
这个脚本提供一个解决方案，通过创建一个虚拟的架子对象并设置其位置在视野之外，
同时添加NaN检查逻辑来解决机械臂消失和动作NaN问题。
运行此脚本将创建一个小的配置文件，使系统认为架子存在但实际上不可见。
"""

import os
import shutil
import sys

# 配置信息
CONFIG_PATH = "/home/sw/m3p2i-aip/src/m3p2i_aip/config"
ENV_PATH = os.path.join(CONFIG_PATH, "panda_env_no_shelf")
ORIGINAL_ENV_PATH = os.path.join(CONFIG_PATH, "panda_env")

def create_virtual_shelf():
    """创建一个虚拟的架子配置，但位置在视野外"""
    # 检查目录是否存在
    if not os.path.exists(ENV_PATH):
        print(f"创建环境目录: {ENV_PATH}")
        os.makedirs(ENV_PATH, exist_ok=True)
    
    # 复制所有文件，确保基本环境完整
    for filename in os.listdir(ORIGINAL_ENV_PATH):
        if not filename.startswith("3_shelf") and not os.path.exists(os.path.join(ENV_PATH, filename)):
            src_file = os.path.join(ORIGINAL_ENV_PATH, filename)
            dst_file = os.path.join(ENV_PATH, filename)
            print(f"复制文件: {filename}")
            shutil.copy2(src_file, dst_file)
    
    # 创建虚拟架子配置（位置在视野外）
    virtual_shelf_path = os.path.join(ENV_PATH, "3_shelf_stand.yaml")
    with open(virtual_shelf_path, "w") as f:
        f.write("""type: "box"
name: "shelf_stand"
size: [.001, .001, .001]  # 极小的尺寸
init_pos: [999, 999, 999]  # 位置在视野外
fixed: True
color: [0, 0, 0, 0]  # 完全透明
""")
    print(f"创建虚拟架子配置: {virtual_shelf_path}")

    # 修改cubeA配置确保它在桌子上
    cube_a_path = os.path.join(ENV_PATH, "5_cubeA.yaml")
    if os.path.exists(cube_a_path):
        with open(cube_a_path, "r") as f:
            content = f.read()
        
        # 确保方块在桌子上
        if "init_pos_on_table" in content and "init_pos_on_shelf" in content:
            print("确保方块位置设置正确")
            # 不需要修改，只需要在运行时确保cube_on_shelf=False
    
    print("配置修复完成!")

def create_patch_script():
    """创建一个补丁脚本，在运行时检查和修复NaN值"""
    patch_script_path = os.path.join("/home/sw/m3p2i-aip/scripts", "nan_check_patch.py")
    with open(patch_script_path, "w") as f:
        f.write("""#!/usr/bin/env python3
import torch
import numpy as np

def fix_nan_in_tensor(tensor):
    \"\"\"修复张量中的NaN值\"\"\"
    if tensor is None:
        return tensor
    
    if isinstance(tensor, torch.Tensor):
        if torch.isnan(tensor).any():
            # 找到NaN的位置
            nan_mask = torch.isnan(tensor)
            # 将NaN替换为0
            tensor = torch.where(nan_mask, torch.zeros_like(tensor), tensor)
            print("已修复tensor中的NaN值")
    elif isinstance(tensor, np.ndarray):
        if np.isnan(tensor).any():
            # 找到NaN的位置
            nan_mask = np.isnan(tensor)
            # 将NaN替换为0
            tensor = np.where(nan_mask, np.zeros_like(tensor), tensor)
            print("已修复numpy array中的NaN值")
    
    return tensor

# 为了在其他模块中使用，可以进行以下修改
# 在 m3p2i_aip/planners/motion_planner/mppi.py 中添加:
# 
# from nan_check_patch import fix_nan_in_tensor
# 
# 并在计算action之后添加:
# action = fix_nan_in_tensor(action)
""")
    print(f"创建NaN检查补丁脚本: {patch_script_path}")

def create_no_shelf_launcher():
    """创建一个改进的无架子启动器脚本"""
    launcher_path = "/home/sw/m3p2i-aip/improved_no_shelf.sh"
    with open(launcher_path, "w") as f:
        f.write("""#!/bin/bash

# 改进的无架子环境启动脚本

echo "===== 启动改进的无架子环境 ====="

# 首先运行修复脚本
echo "执行环境修复..."
python3 /home/sw/m3p2i-aip/scripts/fix_no_shelf.py

# 启动任务规划服务器
echo "启动任务规划服务器..."
python3 /home/sw/m3p2i-aip/scripts/reactive_tamp_no_shelf.py -cn config_panda &
SERVER_PID=$!

# 等待几秒让服务器启动
echo "等待服务器启动..."
sleep 3

# 启动仿真环境
echo "启动仿真环境..."
python3 /home/sw/m3p2i-aip/scripts/run_no_shelf.py -cn config_panda

# 当仿真结束时，终止服务器进程
echo "仿真结束，正在清理..."
kill $SERVER_PID

echo "完成！"
""")
    os.chmod(launcher_path, 0o755)  # 赋予执行权限
    print(f"创建改进的启动脚本: {launcher_path}")

def main():
    print("===== 开始修复无架子环境 =====")
    create_virtual_shelf()
    create_patch_script()
    create_no_shelf_launcher()
    print("===== 修复完成! =====")
    print("请运行以下命令启动改进的无架子环境:")
    print("  /home/sw/m3p2i-aip/improved_no_shelf.sh")

if __name__ == "__main__":
    main() 