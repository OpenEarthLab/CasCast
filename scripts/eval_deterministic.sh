#!/bin/bash

gpus=1
node_num=1
single_gpus=`expr $gpus / $node_num`

cpus=8

# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL


PORT=$((((RANDOM<<15)|RANDOM)%49152 + 10000))

echo $PORT

srun -p ai4earth --quotatype=reserved --ntasks-per-node=$single_gpus --cpus-per-task=$cpus --time=43200 -N $node_num --gres=gpu:$single_gpus python -u evaluation.py \
--init_method 'tcp://127.0.0.1:'$PORT \
--world_size $gpus \
--per_cpus $cpus \
--batch_size 8 \
--num_workers 8 \
--cfgdir /mnt/lustre/gongjunchao/release_code/cascast/experiments/EarthFormer/world_size1-ckpt \
--pred_len 12 \
--test_name test \
--metrics_type SEVIRSkillScore

sleep 2
rm -f batchscript-*