#!/bin/bash

# 定义要遍历的 alpha 值（从极度偏重感知到极度偏重通信）
ALPHAS=(0.1 0.3 0.6 0.7 0.9)

echo "🚀 开始批量运行 GNN 训练，准备生成 Pareto 前沿数据..."

# 遍历数组中的每一个 alpha 值
for alpha in "${ALPHAS[@]}"; do
    echo "=================================================="
    echo "⏳ 正在启动训练: alpha = $alpha"
    echo "=================================================="
    
    # 执行你的训练命令
    pixi run python main.py --algo gnn_heuristic --alpha $alpha
    
    # 检查上一条命令是否成功执行（退出状态码为 0 表示成功）
    if [ $? -eq 0 ]; then
        echo "✅ alpha = $alpha 训练顺利完成！"
    else
        echo "❌ 警告: alpha = $alpha 训练异常中断！"
        # 如果你想在报错时停止整个脚本，可以取消下面这行的注释
        # exit 1 
    fi
    
    echo "" # 打印空行，为了输出美观
done

echo "🎉 所有 alpha 值的训练任务已全部执行完毕！快去查看你的面板吧！"
