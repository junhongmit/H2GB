#!/bin/bash

function run_repeats {
    dataset=$1
    cfg_prefix=$2
    cfg_suffix=$3
    exp_indx=$4
    # The cmd line cfg overrides that will be passed to the main.py,
    # e.g. 'name_tag test01 gnn.layer_type gcnconv'
    cfg_overrides=$5

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

# Define an array of parameters
# cfg_dir="configs"
# DATASET="ogbn-arxiv"
# CONFIG='SparseEdgeGT'
# parameters=(
#     # Layer test
#     # "gt.layers 2 gt.dim_hidden 512 gt.layer_norm False gt.batch_norm False dataset.sample_width 128 dataset.sample_depth 6 train.iter_per_epoch 256 train.batch_size 16 optim.weight_decay 1e-5 optim.base_lr 0.0005"
#     # "gt.layers 3 gt.dim_hidden 512 gt.layer_norm False gt.batch_norm False dataset.sample_width 128 dataset.sample_depth 6 train.iter_per_epoch 256 train.batch_size 16 optim.weight_decay 1e-5 optim.base_lr 0.0005"
#     # "gt.layers 4 gt.dim_hidden 512 gt.layer_norm False gt.batch_norm False dataset.sample_width 128 dataset.sample_depth 6 train.iter_per_epoch 256 train.batch_size 16 optim.weight_decay 1e-5 optim.base_lr 0.0005"
#     # "gt.layers 5 gt.dim_hidden 512 gt.layer_norm False gt.batch_norm False dataset.sample_width 128 dataset.sample_depth 6 train.iter_per_epoch 256 train.batch_size 16 optim.weight_decay 1e-5 optim.base_lr 0.0005"
    
#     # Hidden dim test
#     # "gt.layers 3 gt.dim_hidden 128 gnn.dim_inner 128 gt.layer_norm False gt.batch_norm False dataset.sample_width 128 dataset.sample_depth 6 train.iter_per_epoch 256 train.batch_size 16 optim.weight_decay 1e-5 optim.base_lr 0.0005"
#     # "gt.layers 3 gt.dim_hidden 256 gnn.dim_inner 256 gt.layer_norm False gt.batch_norm False dataset.sample_width 128 dataset.sample_depth 6 train.iter_per_epoch 256 train.batch_size 16 optim.weight_decay 1e-5 optim.base_lr 0.0005"
#     # "gt.layers 3 gt.dim_hidden 512 gnn.dim_inner 512 gt.layer_norm False gt.batch_norm False dataset.sample_width 128 dataset.sample_depth 6 train.iter_per_epoch 256 train.batch_size 16 optim.weight_decay 1e-5 optim.base_lr 0.0005"
#     # "gt.layers 3 gt.dim_hidden 768 gnn.dim_inner 768 gt.layer_norm False gt.batch_norm False dataset.sample_width 128 dataset.sample_depth 6 train.iter_per_epoch 256 train.batch_size 16 optim.weight_decay 1e-5 optim.base_lr 0.0005"

#     # # Sampling test
#     # "gt.layers 3 gt.dim_hidden 256 gnn.dim_inner 256 gt.layer_norm False gt.batch_norm False gt.attn_dropout 0.2 gt.attn_mask Edge dataset.sample_width 192 dataset.sample_depth 6 train.iter_per_epoch 256 train.batch_size 16 optim.weight_decay 1e-5 optim.base_lr 0.001 optim.batch_accumulation 8"
#     # "gt.layers 3 gt.dim_hidden 256 gnn.dim_inner 256 gt.layer_norm False gt.batch_norm False gt.attn_dropout 0.2 gt.attn_mask Edge dataset.sample_width 128 dataset.sample_depth 6 train.iter_per_epoch 512 train.batch_size 10 optim.weight_decay 1e-5 optim.base_lr 0.0005 optim.batch_accumulation 16"
#     # "gt.layers 3 gt.dim_hidden 256 gnn.dim_inner 256 gt.layer_norm False gt.batch_norm False gt.attn_dropout 0.2 gt.attn_mask Edge dataset.sample_width 192 dataset.sample_depth 6 train.iter_per_epoch 512 train.batch_size 16 optim.weight_decay 1e-5 optim.base_lr 0.0005 optim.batch_accumulation 8"
#     # "gt.layers 3 gt.dim_hidden 256 gnn.dim_inner 256 gt.layer_norm False gt.batch_norm False gt.attn_dropout 0.2 gt.attn_mask Edge dataset.sample_width 144 dataset.sample_depth 6 train.iter_per_epoch 256 train.batch_size 12 optim.weight_decay 1e-5 optim.base_lr 0.001 optim.batch_accumulation 8"

#     # # Batch accumulation test
#     # "optim.batch_accumulation 24"
#     # "optim.batch_accumulation 16"
#     # "optim.batch_accumulation 32"
#     # "optim.batch_accumulation 64"

#     # Validation size test
#     "val.iter_per_epoch 128"
#     "val.iter_per_epoch 256"
#     "val.iter_per_epoch 512"
#     "val.iter_per_epoch 1024"

#     # Baseline test
#     # "gnn.layers_mp 3 gnn.dim_inner 250 gnn.layer_type GAT gnn.layer_norm False gnn.batch_norm False dataset.sample_width 32 dataset.sample_depth 6 train.sampler hgt        train.iter_per_epoch 1024 train.batch_size 4 optim.weight_decay 1e-5 optim.base_lr 0.001 optim.batch_accumulation 32"
#     # "gnn.layers_mp 3 gnn.dim_inner 250 gnn.layer_type GAT gnn.layer_norm True  gnn.batch_norm False dataset.sample_width 32 dataset.sample_depth 6 train.sampler hgt        train.iter_per_epoch 1024 train.batch_size 4 optim.weight_decay 1e-5 optim.base_lr 0.001 optim.batch_accumulation 32"
#     # "gnn.layers_mp 3 gnn.dim_inner 250 gnn.layer_type GAT gnn.layer_norm False gnn.batch_norm True  dataset.sample_width 32 dataset.sample_depth 6 train.sampler hgt        train.iter_per_epoch 1024 train.batch_size 4 optim.weight_decay 1e-5 optim.base_lr 0.001 optim.batch_accumulation 32"
#     # "gnn.layers_mp 3 gnn.dim_inner 250 gnn.layer_type GAT gnn.layer_norm False gnn.batch_norm False dataset.sample_width 32 dataset.sample_depth 6 train.sampler full_batch train.iter_per_epoch 1024 train.batch_size 4 optim.weight_decay 1e-5 optim.base_lr 0.001 optim.batch_accumulation 32"
# )
# tags=(
#     "val128"
#     "val256"
#     "val512"
#     "val1024"
# )

cfg_dir="configs"
DATASET="Pokec"
# "AML-Small-HI"
# "ogbn-products"
# "VOCSuperpixels"
# "COCOSuperpixels"
# "Heterophilic_arxiv-year"
# "Heterophilic_snap-patents"
CONFIG="PolyFormer"
parameters=(
    # Multi-seed experiments
    "seed 43 " # train.auto_resume True
    "seed 44 "
    "seed 45 "
    "seed 46 "
    # "seed 43 train.auto_resume True" # train.auto_resume True
    # "seed 44 train.auto_resume True"
    # "seed 45 train.auto_resume True"
    # "seed 46 train.auto_resume True"

    # # Walk length test
    # "train.neighbor_sizes \"[512, 512, 512, 512]\""
    # "train.neighbor_sizes \"[512, 512, 512, 512, 512, 512, 512, 512]\""
    # "train.neighbor_sizes \"[1024, 1024, 1024, 1024]\""
    # "train.neighbor_sizes \"[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024]\""
)
tags=(
    ""
    ""
    ""
    ""
)

# DATASET="mag"
# CONFIG='HGT'
# parameters=(
#     # # Sampling test
#     # "gt.layers 3 gt.dim_hidden 512 gnn.dim_inner 512 gt.layer_norm False gt.batch_norm False dataset.sample_width 64 dataset.sample_depth 6 train.iter_per_epoch 512 train.batch_size 32 optim.weight_decay 1e-5 optim.base_lr 0.0005 optim.batch_accumulation 32 optim.max_epoch 200"
#     # "gt.layers 3 gt.dim_hidden 512 gnn.dim_inner 512 gt.layer_norm False gt.batch_norm False dataset.sample_width 96 dataset.sample_depth 6 train.iter_per_epoch 512 train.batch_size 32 optim.weight_decay 1e-5 optim.base_lr 0.0005 optim.batch_accumulation 32 optim.max_epoch 200"
#     # "gt.layers 3 gt.dim_hidden 512 gnn.dim_inner 512 gt.layer_norm False gt.batch_norm False dataset.sample_width 128 dataset.sample_depth 6 train.iter_per_epoch 512 train.batch_size 32 optim.weight_decay 1e-5 optim.base_lr 0.0005 optim.batch_accumulation 32 optim.max_epoch 200"
#     # "gt.layers 3 gt.dim_hidden 512 gnn.dim_inner 512 gt.layer_norm False gt.batch_norm False dataset.sample_width 192 dataset.sample_depth 6 train.iter_per_epoch 512 train.batch_size 32 optim.weight_decay 1e-5 optim.base_lr 0.0005 optim.batch_accumulation 32 optim.max_epoch 200"

#     # Norm test
#     "gnn.layers_mp 3 gnn.dim_inner 512 gnn.layer_norm False gnn.batch_norm False gnn.dropout 0.5 dataset.sample_width 520 dataset.sample_depth 6 train.iter_per_epoch 256 train.batch_size 128 optim.weight_decay 1e-5 optim.base_lr 0.001 optim.batch_accumulation 32 optim.max_epoch 500"
#     "gnn.layers_mp 3 gnn.dim_inner 512 gnn.layer_norm True gnn.batch_norm False gnn.dropout 0.5 dataset.sample_width 520 dataset.sample_depth 6 train.iter_per_epoch 256 train.batch_size 128 optim.weight_decay 1e-5 optim.base_lr 0.001 optim.batch_accumulation 32 optim.max_epoch 500"
#     "gnn.layers_mp 3 gnn.dim_inner 512 gnn.layer_norm False gnn.batch_norm True gnn.dropout 0.5 dataset.sample_width 520 dataset.sample_depth 6 train.iter_per_epoch 256 train.batch_size 128 optim.weight_decay 1e-5 optim.base_lr 0.001 optim.batch_accumulation 32 optim.max_epoch 500"
#     "gnn.layers_mp 3 gnn.dim_inner 512 gnn.layer_norm True gnn.batch_norm False gnn.dropout 0.5 dataset.sample_width 256 dataset.sample_depth 6 train.iter_per_epoch 512 train.batch_size 64 optim.weight_decay 1e-5 optim.base_lr 0.001 optim.batch_accumulation 64 optim.max_epoch 500"
# )
# tags=(
#     "sw512+256iter+bs128+bc32"
#     "LN+sw512+256iter+bs128+bc32"
#     "BN+sw512+256iter+bs128+bc32"
#     "LN+sw256+512iter+bs64+bc64"
# )

# DATASET="mag"
# CONFIG='HGT'
# parameters=(
#     # # Sampling test
#     # "gt.layers 3 gt.dim_hidden 512 gnn.dim_inner 512 gt.layer_norm False gt.batch_norm False dataset.sample_width 64 dataset.sample_depth 6 train.iter_per_epoch 512 train.batch_size 32 optim.weight_decay 1e-5 optim.base_lr 0.0005 optim.batch_accumulation 32 optim.max_epoch 200"
#     # "gt.layers 3 gt.dim_hidden 512 gnn.dim_inner 512 gt.layer_norm False gt.batch_norm False dataset.sample_width 96 dataset.sample_depth 6 train.iter_per_epoch 512 train.batch_size 32 optim.weight_decay 1e-5 optim.base_lr 0.0005 optim.batch_accumulation 32 optim.max_epoch 200"
#     # "gt.layers 3 gt.dim_hidden 512 gnn.dim_inner 512 gt.layer_norm False gt.batch_norm False dataset.sample_width 128 dataset.sample_depth 6 train.iter_per_epoch 512 train.batch_size 32 optim.weight_decay 1e-5 optim.base_lr 0.0005 optim.batch_accumulation 32 optim.max_epoch 200"
#     # "gt.layers 3 gt.dim_hidden 512 gnn.dim_inner 512 gt.layer_norm False gt.batch_norm False dataset.sample_width 192 dataset.sample_depth 6 train.iter_per_epoch 512 train.batch_size 32 optim.weight_decay 1e-5 optim.base_lr 0.0005 optim.batch_accumulation 32 optim.max_epoch 200"

#     # Norm test
#     # "gt.layers 3 gt.dim_hidden 512 gnn.dim_inner 512 gt.layer_norm True gt.batch_norm False gt.dropout 0.2 gt.residual Learn dataset.sample_width 256 dataset.sample_depth 6 train.iter_per_epoch 512 train.batch_size 64 optim.weight_decay 1e-5 optim.base_lr 0.001 optim.batch_accumulation 64 optim.max_epoch 500"
#     # "gt.layers 3 gt.dim_hidden 512 gnn.dim_inner 512 gt.layer_norm True gt.batch_norm False gt.dropout 0.2 gt.residual Fixed dataset.sample_width 128 dataset.sample_depth 6 train.iter_per_epoch 1024 train.batch_size 32 optim.weight_decay 1e-5 optim.base_lr 0.001 optim.batch_accumulation 128 optim.max_epoch 500"
#     # "gt.layers 3 gt.dim_hidden 512 gnn.dim_inner 512 gt.layer_norm True gt.batch_norm False gt.dropout 0.2 dataset.sample_width 64 dataset.sample_depth 6 train.iter_per_epoch 2048 train.batch_size 16 optim.weight_decay 1e-5 optim.base_lr 0.001 optim.batch_accumulation 256 optim.max_epoch 500"
#     # "gt.layers 3 gt.dim_hidden 512 gnn.dim_inner 512 gt.layer_norm True gt.batch_norm False gt.dropout 0.2 dataset.sample_width 256 dataset.sample_depth 6 train.iter_per_epoch 512 train.batch_size 64 optim.weight_decay 1e-5 optim.base_lr 0.001 optim.batch_accumulation 64 optim.max_epoch 500"

#     "gnn.dropout 0.2 gt.dropout 0.2 gt.attn_dropout 0.2 dataset.sample_width 160 dataset.sample_depth 6 train.iter_per_epoch 256 train.batch_size 32 val.iter_per_epoch 512 optim.base_lr 0.0005 optim.batch_accumulation 32"
#     "gnn.dropout 0.2 gt.dropout 0.2 gt.attn_dropout 0.2 dataset.sample_width 160 dataset.sample_depth 6 train.iter_per_epoch 256 train.batch_size 32 val.iter_per_epoch 512 optim.base_lr 0.0005 optim.batch_accumulation 24"
#     "gnn.dropout 0.2 gt.dropout 0.2 gt.attn_dropout 0.2 dataset.sample_width 128 dataset.sample_depth 6 train.iter_per_epoch 256 train.batch_size 32 val.iter_per_epoch 512 optim.base_lr 0.0005 optim.batch_accumulation 48"
#     "gnn.dropout 0.2 gt.dropout 0.2 gt.attn_dropout 0.2 dataset.sample_width 256 dataset.sample_depth 6 train.iter_per_epoch 256 train.batch_size 32 val.iter_per_epoch 512 optim.base_lr 0.0005 optim.batch_accumulation 32"
# )
# tags=(
#     "sw160+bc32"
#     "sw160+bc24"
#     "sw128+bc48"
#     "sw256+bc32"
# )

# DATASET="ogbn-products"
# CONFIG='SparseEdgeGT'
# parameters=(
#     "gt.batch_norm False gt.layer_norm True gt.attn_heads 4"
#     "gt.batch_norm False gt.layer_norm True gt.input_dropout 0.1"
#     "gt.batch_norm False gt.layer_norm True gt.attn_dropout 0.2"
#     "gt.batch_norm False gt.layer_norm True gt.layers_post_gt 2"
# )
# tags=(
#     "+LN+attn_heads_4"
#     "+LN+intput_drop_0.1"
#     "+LN+attn_drop_0.2"
#     "+LN+layers_post_gt_2"
# )

# DATASET="Heterophilic_arxiv-year"
# CONFIG='GCN+SparseEdgeGT+2Hop+Metapath+LP'
# parameters=(
#     "gt.attn_heads 8"
#     "gt.act gelu"
#     "gt.residual Learn"
#     "gt.attn_dropout 0.0"
# )
# tags=(
#     "+attn_heads_8"
#     "+act_gelu"
#     "+residual_learn"
#     "+attn_dropout_0.0"
# )



# Total number of experiments
total_experiments=${#parameters[@]}

# Number of GPUs available
num_gpus=$(nvidia-smi -L | wc -l)

# Loop through the experiments in batches of num_gpus
for ((start_idx=0; start_idx<total_experiments; start_idx+=num_gpus)); do
    end_idx=$((start_idx + num_gpus - 1))
    
    # Limit the end index to total_experiments
    if ((end_idx >= total_experiments)); then
        end_idx=$((total_experiments - 1))
    fi

    # Run experiments in parallel
    for ((idx=start_idx; idx<=end_idx; idx++)); do
        params="${parameters[idx]}"
        tag="${tags[idx]}"
        echo "Running experiment $idx with parameters: $params"
        run_repeats ${DATASET} CUDA_VISIBLE_DEVICES=$((idx % num_gpus)) ${CONFIG} $((idx % num_gpus)) "name_tag ${CONFIG}$tag $params"
    done

    # Wait for all background processes to finish
    wait
done

echo "All experiments have finished."