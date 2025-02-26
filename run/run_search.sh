#!/usr/bin/env bash

# Run this script from the project root dir.

function run_repeats {
    dataset=$1
    cfg_suffix=$2
    exp_indx=$4
    # The cmd line cfg overrides that will be passed to the main.py,
    # e.g. 'name_tag test01 gnn.layer_type gcnconv'
    cfg_overrides=$3

    cfg_file="${cfg_dir}/${dataset}/${dataset}-${cfg_suffix}.yaml"
    if [[ ! -f "$cfg_file" ]]; then
        echo "WARNING: Config does not exist: $cfg_file"
        echo "SKIPPING!"
        return 1
    fi

    main="python -m H2GB.main --cfg ${cfg_file} --gpu ${exp_indx}"
    out_dir="/nobackup/users/junhong/Logs/results/${dataset}"  # <-- Set the output dir.
    common_params="out_dir ${out_dir} ${cfg_overrides}"

    echo "Run program: ${main}"
    echo "  output dir: ${out_dir}"

    # Run once
    script="${main} --repeat 1 ${common_params} &"
    echo $script
    eval $script
    echo $script
}

cfg_dir="configs"
##### Parameter Grid Search ###
gpu_count=4
job_count=0
gpu_index=0

# DATASET="oag-chem"
# DATASET="oag-cs"
# DATASET="oag-eng"
# DATASET="ogbn-mag"
DATASET="mag-year"
# DATASET="RCDD"
# DATASET="IEEE-CIS"
# DATASET="Pokec"
# DATASET="PDNS"

# MLP
# num_layers_lst=(2 3)
# hidden_channels_lst=(32 64 128 256 512)
# for num_layers in "${num_layers_lst[@]}"; do
#     for hidden_channels in "${hidden_channels_lst[@]}"; do
#         run_repeats ${DATASET} MLP "name_tag MLP+${num_layers}+${hidden_channels} gnn.layers_post_mp ${num_layers} gnn.dim_inner ${hidden_channels}" $gpu_index
#         ((job_count++))
#         ((gpu_index++))
#         if [[ $job_count -eq $gpu_count ]]; then
#             wait  # Wait for all 4 processes to finish
#             job_count=0  # Reset the job counter
#             gpu_index=0  # Reset the GPU index
#         fi
#     done
# done
# wait

# # GCN
# hidden_channels_lst=(512) #(32 64 128 256 512)
# lr_lst=(0.1 0.01 0.001)
# for hidden_channels in "${hidden_channels_lst[@]}"; do
#     for lr in "${lr_lst[@]}"; do
#         run_repeats ${DATASET} GCN "name_tag GCN+${hidden_channels}+${lr} gnn.dim_inner ${hidden_channels} optim.base_lr ${lr}" $gpu_index
#         ((job_count++))
#         ((gpu_index++))
#         if [[ $job_count -eq $gpu_count ]]; then
#             wait  # Wait for all 4 processes to finish
#             job_count=0  # Reset the job counter
#             gpu_index=0  # Reset the GPU index
#         fi
#     done
# done
# wait

# # GraphSAGE
# hidden_channels_lst=(512) #(32 64 128 256 512)
# lr_lst=(0.1 0.01 0.001)
# for hidden_channels in "${hidden_channels_lst[@]}"; do
#     for lr in "${lr_lst[@]}"; do
#         run_repeats ${DATASET} GraphSAGE "name_tag GraphSAGE+${hidden_channels}+${lr} gnn.dim_inner ${hidden_channels} optim.base_lr ${lr}" $gpu_index
#         ((job_count++))
#         ((gpu_index++))
#         if [[ $job_count -eq $gpu_count ]]; then
#             wait  # Wait for all 4 processes to finish
#             job_count=0  # Reset the job counter
#             gpu_index=0  # Reset the GPU index
#         fi
#     done
# done
# wait

# # GAT
# hidden_channels_lst=(32 64 128 256 512)
# heads_lst=(2 4 8)
# lr_lst=(0.1 0.01 0.001)
# for hidden_channels in "${hidden_channels_lst[@]}"; do
#     for heads in "${heads_lst[@]}"; do
#         for lr in "${lr_lst[@]}"; do
#             run_repeats ${DATASET} GAT "name_tag GAT+${hidden_channels}+${heads}+${lr} gnn.dim_inner ${hidden_channels} gnn.attn_heads ${heads} optim.base_lr ${lr}" $gpu_index
#             ((job_count++))
#             ((gpu_index++))
#             if [[ $job_count -eq $gpu_count ]]; then
#                 wait  # Wait for all 4 processes to finish
#                 job_count=0  # Reset the job counter
#                 gpu_index=0  # Reset the GPU index
#             fi
#         done
#     done
# done
# wait

# # GIN
# num_layers_lst=(4) #(2 3 4)
# hidden_channels_lst=(64 128 256 512) #(32 64 128 256 512)
# lr_lst=(0.1 0.01 0.001) #(0.1 0.01 0.001)
# for num_layers in "${num_layers_lst[@]}"; do
#     for hidden_channels in "${hidden_channels_lst[@]}"; do
#         for lr in "${lr_lst[@]}"; do
#             run_repeats ${DATASET} GIN "name_tag GIN+${num_layers}+${hidden_channels}+${lr} gnn.dim_inner ${hidden_channels} optim.base_lr ${lr} gnn.layers_mp ${num_layers}" $gpu_index
#             ((job_count++))
#             ((gpu_index++))
#             if [[ $job_count -eq $gpu_count ]]; then
#                 wait  # Wait for all 4 processes to finish
#                 job_count=0  # Reset the job counter
#                 gpu_index=0  # Reset the GPU index
#             fi
#         done
#     done
# done
# wait

# # APPNP
# hidden_channels_lst=(64) #(32 64 128 256 512)
# lr_lst=(0.001) # (0.01 0.001)
# alpha_lst=(0.2) #(0.1 0.2 0.5 0.9)
# for hidden_channels in "${hidden_channels_lst[@]}"; do
#     for lr in "${lr_lst[@]}"; do
#         for alpha in "${alpha_lst[@]}"; do
#             run_repeats ${DATASET} APPNP "name_tag APPNP+${hidden_channels}+${lr}+${alpha} gnn.dim_inner ${hidden_channels} optim.base_lr ${lr} gnn.alpha ${alpha}" $gpu_index
#             ((job_count++))
#             ((gpu_index++))
#             if [[ $job_count -eq $gpu_count ]]; then
#                 wait  # Wait for all 4 processes to finish
#                 job_count=0  # Reset the job counter
#                 gpu_index=0  # Reset the GPU index
#             fi
#         done
#     done
# done
# wait

# # NAGphormer
# num_layers_lst=(2 3 4)
# hidden_channels_lst=(32 64 128 256 512)
# for num_layers in "${num_layers_lst[@]}"; do
#     for hidden_channels in "${hidden_channels_lst[@]}"; do
#         run_repeats ${DATASET} NAGphormer "name_tag NAGphormer+${num_layers}+${hidden_channels} gnn.dim_inner ${hidden_channels} gnn.layers_mp ${num_layers}" $gpu_index
#         ((job_count++))
#         ((gpu_index++))
#         if [[ $job_count -eq $gpu_count ]]; then
#             wait  # Wait for all 4 processes to finish
#             job_count=0  # Reset the job counter
#             gpu_index=0  # Reset the GPU index
#         fi
#     done
# done
# wait

# # RGCN
# hidden_channels_lst=(128) #(32 64 128 256 512)
# lr_lst=(0.001) #(0.1 0.01 0.001)
# for hidden_channels in "${hidden_channels_lst[@]}"; do
#     for lr in "${lr_lst[@]}"; do
#         run_repeats ${DATASET} RGCN "name_tag RGCN+${hidden_channels}+${lr} gnn.dim_inner ${hidden_channels} optim.base_lr ${lr}" $gpu_index
#         ((job_count++))
#         ((gpu_index++))
#         if [[ $job_count -eq $gpu_count ]]; then
#             wait  # Wait for all 4 processes to finish
#             job_count=0  # Reset the job counter
#             gpu_index=0  # Reset the GPU index
#         fi
#     done
# done
# wait

# # RGraphSAGE
# hidden_channels_lst=(512) #(32 64 128 256 512)
# lr_lst=(0.1 0.01 0.001)
# for hidden_channels in "${hidden_channels_lst[@]}"; do
#     for lr in "${lr_lst[@]}"; do
#         run_repeats ${DATASET} RGraphSAGE "name_tag RGraphSAGE+${hidden_channels}+${lr} gnn.dim_inner ${hidden_channels} optim.base_lr ${lr}" $gpu_index
#         ((job_count++))
#         ((gpu_index++))
#         if [[ $job_count -eq $gpu_count ]]; then
#             wait  # Wait for all 4 processes to finish
#             job_count=0  # Reset the job counter
#             gpu_index=0  # Reset the GPU index
#         fi
#     done
# done
# wait

# # MixHop
hidden_channels_lst=(512) #(32 64 128 256 512)
num_layers_lst=(2) #(2 3 4)
for hidden_channels in "${hidden_channels_lst[@]}"; do
    for num_layers in "${num_layers_lst[@]}"; do
        run_repeats ${DATASET} MixHop "name_tag MixHop+${hidden_channels}+${num_layers} gnn.dim_inner ${hidden_channels} gnn.layers_mp ${num_layers}" $gpu_index
        ((job_count++))
        ((gpu_index++))
        if [[ $job_count -eq $gpu_count ]]; then
            wait  # Wait for all 4 processes to finish
            job_count=0  # Reset the job counter
            gpu_index=0  # Reset the GPU index
        fi
    done
done
wait

# # FAGCN
# num_layers_lst=(2 3 4)
# hidden_channels_lst=(32 64 128 256 512)
# for num_layers in "${num_layers_lst[@]}"; do
#     for hidden_channels in "${hidden_channels_lst[@]}"; do
#         run_repeats ${DATASET} FAGCN "name_tag FAGCN+${num_layers}+${hidden_channels} gnn.dim_inner ${hidden_channels} gnn.layers_mp ${num_layers}" $gpu_index
#         ((job_count++))
#         ((gpu_index++))
#         if [[ $job_count -eq $gpu_count ]]; then
#             wait  # Wait for all 4 processes to finish
#             job_count=0  # Reset the job counter
#             gpu_index=0  # Reset the GPU index
#         fi
#     done
# done
# wait

# # ACM-GCN
# num_layers_lst=(3) #(2 3 4)
# hidden_channels_lst=(256 512) #(32 64 128 256 512)
# for num_layers in "${num_layers_lst[@]}"; do
#     for hidden_channels in "${hidden_channels_lst[@]}"; do
#         run_repeats ${DATASET} ACM-GCN "name_tag ACM-GCN+${num_layers}+${hidden_channels} gnn.dim_inner ${hidden_channels} gnn.layers_mp ${num_layers}" $gpu_index
#         ((job_count++))
#         ((gpu_index++))
#         if [[ $job_count -eq $gpu_count ]]; then
#             wait  # Wait for all 4 processes to finish
#             job_count=0  # Reset the job counter
#             gpu_index=0  # Reset the GPU index
#         fi
#     done
# done
# wait

# # LSGNN
# num_layers_lst=(3 4) #(2 3 4)
# hidden_channels_lst=(32 64 128 256 512)
# for num_layers in "${num_layers_lst[@]}"; do
#     for hidden_channels in "${hidden_channels_lst[@]}"; do
#         run_repeats ${DATASET} LSGNN "name_tag LSGNN+${num_layers}+${hidden_channels} gnn.dim_inner ${hidden_channels} gnn.layers_mp ${num_layers}" $gpu_index
#         ((job_count++))
#         ((gpu_index++))
#         if [[ $job_count -eq $gpu_count ]]; then
#             wait  # Wait for all 4 processes to finish
#             job_count=0  # Reset the job counter
#             gpu_index=0  # Reset the GPU index
#         fi
#     done
# done
# wait

# # HAN
# hidden_channels_lst=(512) #(32 64 128 256 512)
# num_layers_lst=(2 3) #(2 3 4)
# lr_lst=(0.01 0.001 0.0005)
# for hidden_channels in "${hidden_channels_lst[@]}"; do
#     for num_layers in "${num_layers_lst[@]}"; do
#         for lr in "${lr_lst[@]}"; do
#             run_repeats ${DATASET} HAN "name_tag HAN+${num_layers}+${hidden_channels}+${lr} gnn.dim_inner ${hidden_channels} gnn.layers_mp ${num_layers} optim.base_lr ${lr}" $gpu_index
#             ((job_count++))
#             ((gpu_index++))
#             if [[ $job_count -eq $gpu_count ]]; then
#                 wait  # Wait for all 4 processes to finish
#                 job_count=0  # Reset the job counter
#                 gpu_index=0  # Reset the GPU index
#             fi
#         done
#     done
# done
# wait

# # HGT
# hidden_channels_lst=(128 256 512) #(32 64 128 256 512)
# num_layers_lst=(2 3) #(2 3 4)
# lr_lst=(0.01 0.001 0.0005)
# for hidden_channels in "${hidden_channels_lst[@]}"; do
#     for num_layers in "${num_layers_lst[@]}"; do
#         for lr in "${lr_lst[@]}"; do
#             run_repeats ${DATASET} HGT "name_tag HGT+${num_layers}+${hidden_channels}+${lr} gnn.dim_inner ${hidden_channels} gnn.layers_mp ${num_layers} optim.base_lr ${lr}" $gpu_index
#             ((job_count++))
#             ((gpu_index++))
#             if [[ $job_count -eq $gpu_count ]]; then
#                 wait  # Wait for all 4 processes to finish
#                 job_count=0  # Reset the job counter
#                 gpu_index=0  # Reset the GPU index
#             fi
#         done
#     done
# done
# wait

# LINKX
# hidden_channels_lst=(512) #(16 32 64 128 256 512)
# num_layers_lst=(1 2 3)
# for num_layers in "${num_layers_lst[@]}"; do
#     for hidden_channels in "${hidden_channels_lst[@]}"; do
#         run_repeats ${DATASET} LINKX "name_tag LINKX+${num_layers}+${hidden_channels} gnn.dim_inner ${hidden_channels} gnn.layers_mp ${num_layers}" $gpu_index
#         ((job_count++))
#         ((gpu_index++))
#         if [[ $job_count -eq $gpu_count ]]; then
#             wait  # Wait for all 4 processes to finish
#             job_count=0  # Reset the job counter
#             gpu_index=0  # Reset the GPU index
#         fi
#     done
# done
# wait
