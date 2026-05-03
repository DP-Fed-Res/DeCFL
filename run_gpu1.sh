#!/bin/bash

# ==========================================
# 联邦学习消融实验自动化脚本 (并行队列版 - 完美修复nohup)
# ==========================================

LOG_DIR="log"
mkdir -p ${LOG_DIR}

# 1. 设置最大并行任务数 (同时运行的实验组数)
MAX_JOBS=2
TASK_FILE="tasks.txt"

# 清空或创建任务文件
> ${TASK_FILE}

# 【修复核心 1】获取你当前终端所处环境(如 Conda)的 Python 绝对路径
# 这样 nohup 在后台运行时，绝对不会找不到 python 或跑错环境
PYTHON_EXEC=$(which python)
echo "======================================="
echo "  🛠️  锁定的 Python 环境: ${PYTHON_EXEC}"
echo "======================================="

# 全局固定参数
DATASET="text-distill"  # Cifar10
G1=4
G2=5
TC_1ST=150
TC=50
BR=0.02
LR=0.1
TG=20
CLIP=1.0
GPU=1

# 2. 定义消融变量数组
ALGS=("fedavg" "fedprox" "scaffold" "fed_pcdp" "flexcfl" "fesem" "fedrc" "decfl")
ALPHAS=("0.5")
EPSILONS=("5.0")

echo "  🚀 正在生成实验队列 (并发数: ${MAX_JOBS})  "

# 3. 嵌套循环生成所有需要执行的命令，写入 tasks.txt
for alg in "${ALGS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
        for eps in "${EPSILONS[@]}"; do

            log_file="${LOG_DIR}/${alg}_${DATASET}_alpha${alpha}_eps${eps}.log"

            # 【修复核心 2】使用绝对路径替代单纯的 python 命令
            cmd="${PYTHON_EXEC} main.py --alg ${alg} --dataset ${DATASET} --g1 ${G1} --g2 ${G2} --alpha ${alpha} --tc_1st ${TC_1ST} --tc ${TC} --br ${BR} --lr ${LR} --tg ${TG} --eps ${eps} --clip ${CLIP} --gpu ${GPU} > ${log_file} 2>&1"

            # 将命令追加到任务文本中
            echo "$cmd" >> ${TASK_FILE}

        done
    done
done

TOTAL_TASKS=$(wc -l < ${TASK_FILE})
echo "共生成 ${TOTAL_TASKS} 个实验任务，即将开始队列并行执行..."

# ==========================================
# 4. 核心：使用 xargs 维持严格的并发数量执行
# ==========================================
# 【修复核心 3】废弃 cat | xargs 管道，改用 -a 直接读取文件，-d '\n' 严格按行分割。
# 这能 100% 免疫 nohup 对 stdin 管道造成的破坏。
xargs -d '\n' -a ${TASK_FILE} -P ${MAX_JOBS} -I {} bash -c "{}"

echo "       🎉 所有实验已全部运行完成！ 🎉      "