#!/bin/bash

# 启动无架子环境的脚本

echo "启动无架子环境机械臂夹取任务..."

# 启动任务规划服务器
echo "启动任务规划服务器..."
python3 scripts/reactive_tamp_no_shelf.py -cn config_panda &
SERVER_PID=$!

# 等待几秒让服务器启动
echo "等待服务器启动..."
sleep 3

# 启动仿真环境
echo "启动仿真环境..."
python3 scripts/run_no_shelf.py -cn config_panda

# 当仿真结束时，终止服务器进程
echo "仿真结束，正在清理..."
kill $SERVER_PID

echo "完成！" 