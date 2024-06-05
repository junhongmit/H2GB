#!/usr/bin/env bash

# Run this script from the project root dir.

function run_repeats {
    dataset=$1
    cfg_suffix=$2
    # The cmd line cfg overrides that will be passed to the main.py,
    # e.g. 'name_tag test01 gnn.layer_type gcnconv'
    cfg_overrides=$3

    cfg_file="${cfg_dir}/${dataset}/${dataset}-${cfg_suffix}.yaml"
    if [[ ! -f "$cfg_file" ]]; then
        echo "WARNING: Config does not exist: $cfg_file"
        echo "SKIPPING!"
        return 1
    fi

    main="python -m H2GB.main --cfg ${cfg_file}"
    out_dir="/nobackup/users/junhong/Logs/results/${dataset}"  # <-- Set the output dir.
    common_params="out_dir ${out_dir} ${cfg_overrides}"

    echo "Run program: ${main}"
    echo "  output dir: ${out_dir}"

    # Run once
    script="${main} --repeat 1 ${common_params}"
    echo $script
    eval $script
    echo $script
}

cfg_dir="configs"

##### Node Classification Tasks ###########################################################################
DATASET="oag-chem"
# run_repeats ${DATASET} MLP                "name_tag MLP"
# run_repeats ${DATASET} MLP+Metapath       "name_tag MLP+Metapath"
# run_repeats ${DATASET} LP+1Hop            "name_tag LP+1Hop"
# run_repeats ${DATASET} LP+2Hop            "name_tag LP+2Hop"

# Homogeneous GNN Baselines
# run_repeats ${DATASET} GCN                "name_tag GCN"
# run_repeats ${DATASET} GraphSAGE          "name_tag GraphSAGE"
# run_repeats ${DATASET} GIN                "name_tag GIN"
# run_repeats ${DATASET} GAT                "name_tag GAT"
# run_repeats ${DATASET} APPNP                "name_tag APPNP"
# run_repeats ${DATASET} NAGphormer         "name_tag NAGphormer"
# run_repeats ${DATASET} GraphTrans         "name_tag GraphTrans"
# run_repeats ${DATASET} Gophormer          "name_tag Gophormer"
# run_repeats ${DATASET} GOAT               "name_tag GOAT"

# Heterogeneous GNN Baselines
# run_repeats ${DATASET} RGCN               "name_tag RGCN"
# run_repeats ${DATASET} HAN                "name_tag HAN"
# run_repeats ${DATASET} HGT                "name_tag HGT"

# Heterophilic GNN Baselines
# run_repeats ${DATASET} MixHop               "name_tag MixHop"
# run_repeats ${DATASET} LINKX               "name_tag LINKX"
# run_repeats ${DATASET} FAGCN               "name_tag FAGCN"
# run_repeats ${DATASET} LSGNN               "name_tag LSGNN"

# Proposed GT
# run_repeats ${DATASET} SparseEdgeGT         "name_tag SparseEdgeGT"
# run_repeats ${DATASET} FullEdgeGT         "name_tag FullEdgeGT wandb.use False"
# run_repeats ${DATASET} SparseEdgeGT                 "name_tag SparseEdgeGT"
# run_repeats ${DATASET} SparseEdgeGT+Metapath        "name_tag SparseEdgeGT+Metapath"
# run_repeats ${DATASET} SparseNodeGT                 "name_tag SparseNodeGT"
# run_repeats ${DATASET} SparseNodeGT_Test                 "name_tag SparseNodeGT_Test"
# run_repeats ${DATASET} SparseNodeGT+Metapath        "name_tag SparseNodeGT+Metapath"
# run_repeats ${DATASET} SparseNodeGT+LP        "name_tag SparseNodeGT+LP"
# run_repeats ${DATASET} GT                 "name_tag GT+TypeFFN"
# run_repeats ${DATASET} GT0                "name_tag GT+write_rel"
# run_repeats ${DATASET} GCN+GT             "name_tag GCN+GT+TypeFFN"
# run_repeats ${DATASET} SAGE+GT            "name_tag SAGE+GT+TypeFFN"
# run_repeats ${DATASET} SAGE+FullGT        "name_tag SAGE+FullGT+TypeFFN"
# run_repeats ${DATASET} SAGE-SAGE+GT       "name_tag SAGE-SAGE+GT+TypeFFN"

DATASET="oag-cs"
run_repeats ${DATASET} MLP                "name_tag MLP"
# run_repeats ${DATASET} MLP+Metapath       "name_tag MLP+Metapath"
# run_repeats ${DATASET} MLP+Node2Vec         "name_tag MLP+Node2Vec"
# run_repeats ${DATASET} LP+1Hop            "name_tag LP+1Hop"
# run_repeats ${DATASET} LP+2Hop            "name_tag LP+2Hop"

# Homogeneous GNN Baselines
# run_repeats ${DATASET} GCN                "name_tag GCN"
# run_repeats ${DATASET} GraphSAGE          "name_tag GraphSAGE"
# run_repeats ${DATASET} GIN                "name_tag GIN"
# run_repeats ${DATASET} GAT                "name_tag GAT"
# run_repeats ${DATASET} APPNP              "name_tag APPNP"
# run_repeats ${DATASET} NAGphormer         "name_tag NAGphormer"
# run_repeats ${DATASET} GraphTrans         "name_tag GraphTrans"
# run_repeats ${DATASET} Gophormer          "name_tag Gophormer"
# run_repeats ${DATASET} GOAT               "name_tag GOAT"

# Heterogeneous GNN Baselines
# run_repeats ${DATASET} RGCN               "name_tag RGCN"
# run_repeats ${DATASET} HAN                "name_tag HAN"
# run_repeats ${DATASET} HGT                "name_tag HGT"
# run_repeats ${DATASET} SHGN               "name_tag SHGN"

# Heterophilic GNN Baselines
# run_repeats ${DATASET} MixHop               "name_tag MixHop"
# run_repeats ${DATASET} LINKX               "name_tag LINKX"
# run_repeats ${DATASET} FAGCN               "name_tag FAGCN"
# run_repeats ${DATASET} LSGNN               "name_tag LSGNN"

# Proposed GT
# run_repeats ${DATASET} FullEdgeGT       "name_tag FullEdgeGT wandb.use False"
# run_repeats ${DATASET} FullEdgeGT+Metapath       "name_tag FullEdgeGT+Metapath"
# run_repeats ${DATASET} FullNodeGT       "name_tag FullNodeGT"
# run_repeats ${DATASET} GCN-FullEdgeGT       "name_tag GCN-FullEdgeGT"
# run_repeats ${DATASET} GCN+FullEdgeGT             "name_tag GCN+FullEdgeGT"
# run_repeats ${DATASET} EdgeGT       "name_tag EdgeGT"
# run_repeats ${DATASET} SparseEdgeGT                 "name_tag SparseEdgeGT train.auto_resume True"
# run_repeats ${DATASET} SparseEdgeGT+Metapath        "name_tag SparseEdgeGT+Metapath"
# run_repeats ${DATASET} SparseNodeGT                 "name_tag SparseNodeGT"
# run_repeats ${DATASET} SparseNodeGT                 "name_tag SparseNodeGT+PAP_relation"
# run_repeats ${DATASET} SparseNodeGT_edgeType       "name_tag SparseNodeGT_edgeType"
# run_repeats ${DATASET} SparseNodeGT+2Hop                 "name_tag SparseNodeGT+2Hop"
# run_repeats ${DATASET} SparseNodeGT+GlobalNodes                 "name_tag SparseNodeGT+GlobalNodes"
# run_repeats ${DATASET} SparseNodeGT+2Hop+GlobalNodes                 "name_tag SparseNodeGT+2Hop+GlobalNodes"
# run_repeats ${DATASET} SparseNodeGT+2Hop+LP                 "name_tag SparseNodeGT+2Hop+LP"
# run_repeats ${DATASET} SparseNodeGT_Test                 "name_tag SparseNodeGT_Test"
# run_repeats ${DATASET} SparseNodeGT+Metapath                 "name_tag SparseNodeGT+Metapath"
# run_repeats ${DATASET} SparseNodeGT+LP                 "name_tag SparseNodeGT+LP"
# run_repeats ${DATASET} SparseNodeGT+Node2Vec+LP                 "name_tag SparseNodeGT+Node2Vec+LP"
# run_repeats ${DATASET} GCN+SparseEdgeGT             "name_tag GCN+SparseEdgeGT"
# run_repeats ${DATASET} GCN+SparseEdgeGT+Metapath             "name_tag GCN+SparseEdgeGT+Metapath"
# run_repeats ${DATASET} SAGE+SparseNodeGT                 "name_tag SAGE+SparseNodeGT"
# run_repeats ${DATASET} SAGE+FullGT        "name_tag SAGE+FullGT+TypeFFN"
# run_repeats ${DATASET} SAGE-SAGE+GT       "name_tag SAGE-SAGE+GT+TypeFFN"

# run_repeats ${DATASET} MLP+Metapath       "name_tag MLP+Metapath"
# run_repeats ${DATASET} MLP+TransE         "name_tag MLP+TransE"
# run_repeats ${DATASET} MLP+ComplEx        "name_tag MLP+ComplEx"
# run_repeats ${DATASET} MLP+DistMult       "name_tag MLP+DistMult"

# run_repeats ${DATASET} HGT+Metapath       "name_tag HGT+Metapath"
# run_repeats ${DATASET} HGT+TransE         "name_tag HGT+TransE"

# run_repeats ${DATASET} Hetero_GT          "name_tag Hetero_GT"
# run_repeats ${DATASET} GT128              "name_tag GT128"
# run_repeats ${DATASET} GT+Metapath        "name_tag GT+Metapath"
# run_repeats ${DATASET} GT+TransE          "name_tag GT+TransE"
# run_repeats ${DATASET} GT+attnRes         "name_tag GT+attnRes"
# run_repeats ${DATASET} GT0                "name_tag GT+TypeFFN"
# run_repeats ${DATASET} NodeGT             "name_tag NodeGT+TypeFFN"
# run_repeats ${DATASET} NodeGT0            "name_tag NodeGT_Debug"
# run_repeats ${DATASET} HierarchyNodeGT    "name_tag HierarchyNodeGT"
# run_repeats ${DATASET} NodetypeGT         "name_tag NodetypeGT"
# run_repeats ${DATASET} EdgetypeGT         "name_tag EdgetypeGT"
# run_repeats ${DATASET} khopGT             "name_tag khopAug+GT"
# run_repeats ${DATASET} FullGT             "name_tag FullGT"
# run_repeats ${DATASET} GT0                "name_tag GT+TypeFFN"
# run_repeats ${DATASET} GCN+GT             "name_tag GCN+GT+TypeFFN"
# run_repeats ${DATASET} SAGE+GT            "name_tag SAGE+GT+TypeFFN"
# run_repeats ${DATASET} SAGE+FullGT        "name_tag SAGE+FullGT+TypeFFN+Dr"
# run_repeats ${DATASET} SAGE+GT            "name_tag SAGE+GT+TypeFFN+Dr"
# run_repeats ${DATASET} SAGE-FullGT        "name_tag SAGE-FullGT+TypeFFN"
# run_repeats ${DATASET} SAGE-GT            "name_tag SAGE-GT+TypeFFN"
# run_repeats ${DATASET} SAGE-SAGE+GT       "name_tag SAGE-SAGE+GT+TypeFFN"
# run_repeats ${DATASET} FullGT--SAGE       "name_tag FullGT--SAGE+TypeFFN"
# run_repeats ${DATASET} GT--SAGE           "name_tag GT--SAGE+TypeFFN"
# run_repeats ${DATASET} SAGE-NodeGT+TypeFFN0       "name_tag SAGE-NodeGT+TypeFFN_r42 seed 42 wandb.use False"

# run_repeats ${DATASET} FullGT+Hetero_RWSE "name_tag FullGT+Hetero_RWSE"
# run_repeats ${DATASET} FullGT+Hetero_SDPE "name_tag FullGT+Hetero_SDPE"


DATASET="oag-eng"
# run_repeats ${DATASET} MLP                "name_tag MLP"
# run_repeats ${DATASET} MLP+Metapath                "name_tag MLP+Metapath"
# run_repeats ${DATASET} LP+1Hop                "name_tag LP+1Hop"
# run_repeats ${DATASET} LP+2Hop                "name_tag LP+2Hop"

# Homogeneous GNN Baselines
# run_repeats ${DATASET} GCN                "name_tag GCN"
# run_repeats ${DATASET} GraphSAGE          "name_tag GraphSAGE"
# run_repeats ${DATASET} GIN                "name_tag GIN"
# run_repeats ${DATASET} GAT                "name_tag GAT"
# run_repeats ${DATASET} NAGphormer         "name_tag NAGphormer"
# run_repeats ${DATASET} GraphTrans         "name_tag GraphTrans"
# run_repeats ${DATASET} Gophormer          "name_tag Gophormer"
# run_repeats ${DATASET} GOAT               "name_tag GOAT"

# Heterogeneous GNN Baselines
# run_repeats ${DATASET} RGCN               "name_tag RGCN"
# run_repeats ${DATASET} RGraphSAGE         "name_tag RGraphSAGE"
# run_repeats ${DATASET} RGAT               "name_tag RGAT"
# run_repeats ${DATASET} HAN                "name_tag HAN"
# run_repeats ${DATASET} HGT                "name_tag HGT"
# run_repeats ${DATASET} HINormer           "name_tag HINormer wandb.use False"

# Heterophilic GNN Baselines
# run_repeats ${DATASET} MixHop               "name_tag MixHop"
# run_repeats ${DATASET} LINKX               "name_tag LINKX"
# run_repeats ${DATASET} FAGCN               "name_tag FAGCN"
# run_repeats ${DATASET} LSGNN               "name_tag LSGNN"

# Proposed GT
# run_repeats ${DATASET} FullEdgeGT       "name_tag FullEdgeGT wandb.use False"
# run_repeats ${DATASET} FullEdgeGT+Metapath       "name_tag FullEdgeGT+Metapath"
# run_repeats ${DATASET} FullNodeGT       "name_tag FullNodeGT"
# run_repeats ${DATASET} GCN-FullEdgeGT       "name_tag GCN-FullEdgeGT"
# run_repeats ${DATASET} GCN+FullEdgeGT             "name_tag GCN+FullEdgeGT"
# run_repeats ${DATASET} EdgeGT       "name_tag EdgeGT"
# run_repeats ${DATASET} SparseEdgeGT                 "name_tag SparseEdgeGT"
# run_repeats ${DATASET} SparseEdgeGT+Metapath        "name_tag SparseEdgeGT+Metapath"
# run_repeats ${DATASET} SparseNodeGT                 "name_tag SparseNodeGT"
# run_repeats ${DATASET} SparseNodeGT+Metapath                 "name_tag SparseNodeGT+Metapath"
# run_repeats ${DATASET} GCN+SparseEdgeGT             "name_tag GCN+SparseEdgeGT"
# run_repeats ${DATASET} GCN+SparseEdgeGT+Metapath             "name_tag GCN+SparseEdgeGT+Metapath"
# run_repeats ${DATASET} GCN+GT             "name_tag GCN+GT+TypeFFN"
# run_repeats ${DATASET} SAGE+GT            "name_tag SAGE+GT+TypeFFN"
# run_repeats ${DATASET} SAGE+FullGT        "name_tag SAGE+FullGT+TypeFFN"
# run_repeats ${DATASET} SAGE-SAGE+GT       "name_tag SAGE-SAGE+GT+TypeFFN"


DATASET="ogbn-mag"
# run_repeats ${DATASET} MLP                "name_tag MLP"
# run_repeats ${DATASET} MLP+Node2Vec       "name_tag MLP+Node2Vec"
# run_repeats ${DATASET} MLP+Metapath       "name_tag MLP+Metapath"
# run_repeats ${DATASET} MLP+TransE         "name_tag MLP+TransE"
# run_repeats ${DATASET} LP+1Hop            "name_tag LP+1Hop"
# run_repeats ${DATASET} LP+2Hop            "name_tag LP+2Hop"
# run_repeats ${DATASET} SGC+1Hop           "name_tag SGC+1Hop"
# run_repeats ${DATASET} SGC+2Hop           "name_tag SGC+2Hop"

# Homogeneous GNN Baselines
# run_repeats ${DATASET} GCN                "name_tag GCN"
# run_repeats ${DATASET} GCN+MS             "name_tag GCN+MS"
# run_repeats ${DATASET} GraphSAGE          "name_tag GraphSAGE seed 45"
# run_repeats ${DATASET} GIN                "name_tag GIN"
# run_repeats ${DATASET} GAT                "name_tag GAT"
# run_repeats ${DATASET} APPNP                "name_tag APPNP"
# run_repeats ${DATASET} NAGphormer         "name_tag NAGphormer"
# run_repeats ${DATASET} GraphTrans         "name_tag GraphTrans"
# run_repeats ${DATASET} Gophormer          "name_tag Gophormer"
# run_repeats ${DATASET} GOAT               "name_tag GOAT"

# Heterogeneous GNN Baselines
# run_repeats ${DATASET} RGCN               "name_tag RGCN"
# run_repeats ${DATASET} RGAT               "name_tag RGAT"
# run_repeats ${DATASET} HAN                "name_tag HAN"
# run_repeats ${DATASET} HGT                "name_tag HGT"
# run_repeats ${DATASET} HGT+CL             "name_tag HGT+CL"
# run_repeats ${DATASET} HGT256             "name_tag HGT256"
# run_repeats ${DATASET} HGT_eval           "name_tag HGT_eval"
# run_repeats ${DATASET} HINormer           "name_tag HINormer wandb.use False" # wandb.use False
# run_repeats ${DATASET} SHGN               "name_tag SHGN"

# Heterophilic GNN Baselines
# run_repeats ${DATASET} MixHop               "name_tag MixHop"
# run_repeats ${DATASET} LINKX               "name_tag LINKX"
# run_repeats ${DATASET} FAGCN               "name_tag FAGCN"
# run_repeats ${DATASET} ACM-GCN             "name_tag ACM-GCN" # wandb.use False
# run_repeats ${DATASET} LSGNN               "name_tag LSGNN"

# Proposed GT
# run_repeats ${DATASET} FullEdgeGT         "name_tag FullEdgeGT seed 43"
# run_repeats ${DATASET} FullEdgeGT+Metapath         "name_tag FullEdgeGT+Metapath seed 43"
# run_repeats ${DATASET} GCN-FullEdgeGT       "name_tag GCN-FullEdgeGT seed 43"
# run_repeats ${DATASET} GCN+FullEdgeGT       "name_tag GCN+FullEdgeGT seed 43"
# run_repeats ${DATASET} GCN+FullEdgeGT+Metapath "name_tag GCN+FullEdgeGT+Metapath seed 43"
# run_repeats ${DATASET} EdgeGT             "name_tag EdgeGT"
# run_repeats ${DATASET} EdgeGT_eval        "name_tag EdgeGT_eval"
# run_repeats ${DATASET} EdgeGT+JK          "name_tag EdgeGT+JK"
# run_repeats ${DATASET} FullNodeGT         "name_tag FullNodeGT seed 43 train.auto_resume True"
# run_repeats ${DATASET} FullNodeGT+Metapath         "name_tag FullNodeGT+Metapath seed 43"
# run_repeats ${DATASET} NodeGT             "name_tag NodeGT"
# run_repeats ${DATASET} NodeGT+1Hop        "name_tag NodeGT+1Hop"
# run_repeats ${DATASET} NodeGT+2Hop        "name_tag NodeGT+2Hop"
# run_repeats ${DATASET} SparseEdgeGT       "name_tag SparseEdgeGT"
# run_repeats ${DATASET} SparseEdgeGT+MS       "name_tag SparseEdgeGT+MS"
# run_repeats ${DATASET} SparseEdgeGT+LP       "name_tag SparseEdgeGT+LP" # wandb.use False
# run_repeats ${DATASET} SparseEdgeGT_Test       "name_tag SparseEdgeGT_Test"
# run_repeats ${DATASET} SparseEdgeGT+GlobalNodes       "name_tag SparseEdgeGT+GlobalNodes"
# run_repeats ${DATASET} SparseEdgeGT+Metapath          "name_tag SparseEdgeGT+Metapath"
# run_repeats ${DATASET} SparseEdgeGT+Metapath+MS          "name_tag SparseEdgeGT+Metapath+MS"
# run_repeats ${DATASET} SparseEdgeGT+Metapath+LP          "name_tag SparseEdgeGT+Metapath+LP"
# run_repeats ${DATASET} SparseEdgeGT+Metapath+LP+MS          "name_tag SparseEdgeGT+Metapath+LP+MS"
# run_repeats ${DATASET} SparseEdgeGT+TransE          "name_tag SparseEdgeGT+TransE"
# run_repeats ${DATASET} SparseEdgeGT+Metapath+GlobalNodes       "name_tag SparseEdgeGT+Metapath+GlobalNodes"
# run_repeats ${DATASET} SparseEdgeGT+Metapath_eval       "name_tag SparseEdgeGT+Metapath_eval"
# run_repeats ${DATASET} SparseNodeGT       "name_tag SparseNodeGT+layer3+512*3 train.auto_resume True"
# run_repeats ${DATASET} SparseNodeGT_fast       "name_tag SparseNodeGT_fast"
# run_repeats ${DATASET} SparseNodeGT_edgeType_fast       "name_tag SparseNodeGT_edgeType_fast"
# run_repeats ${DATASET} SparseNodeGT+2Hop       "name_tag SparseNodeGT+2Hop"
# run_repeats ${DATASET} SparseNodeGT+Metapath       "name_tag SparseNodeGT+Metapath seed 43"
# run_repeats ${DATASET} SparseNodeGT+Metapath+LP       "name_tag SparseNodeGT+Metapath+LP"
# run_repeats ${DATASET} GCN+SparseEdgeGT       "name_tag GCN+SparseEdgeGT seed 43"
# run_repeats ${DATASET} GCN+SparseEdgeGT+Metapath       "name_tag GCN+SparseEdgeGT+Metapath seed 43"
# run_repeats ${DATASET} GCN+SparseEdgeGT+Metapath+LP       "name_tag GCN+SparseEdgeGT+Metapath+LP"
# run_repeats ${DATASET} SAGE+SparseEdgeGT+Metapath       "name_tag SAGE+SparseEdgeGT+Metapath"
# run_repeats ${DATASET} GCN-SparseEdgeGT       "name_tag GCN-SparseEdgeGT seed 43" # wandb.use False
# run_repeats ${DATASET} GIN-SparseEdgeGT       "name_tag GIN-SparseEdgeGT" # wandb.use False
# run_repeats ${DATASET} GCN-GCN+SparseEdgeGT       "name_tag GCN-GCN+SparseEdgeGT seed 43"
# run_repeats ${DATASET} GCN-GCN+FullEdgeGT       "name_tag GCN-GCN+FullEdgeGT seed 43" # wandb.use False
# run_repeats ${DATASET} SAGE-SAGE+SparseEdgeGT+Metapath       "name_tag SAGE-SAGE+SparseEdgeGT+Metapath"


# run_repeats ${DATASET} SAGE+Hetero_RWSE   "name_tag SAGE+Hetero_RWSE"
# run_repeats ${DATASET} GT                   "name_tag GT+sw256+512iter+bs64+bc64 gt.layers 3 gt.dim_hidden 512 gnn.dim_inner 512 gt.layer_norm True gt.batch_norm False gt.dropout 0.2 gt.residual Learn dataset.sample_width 256 dataset.sample_depth 6 train.iter_per_epoch 512 train.batch_size 64 optim.weight_decay 1e-5 optim.base_lr 0.001 optim.batch_accumulation 64 optim.max_epoch 500"
# run_repeats ${DATASET} GT                   "name_tag GT"
# run_repeats ${DATASET} NodeGT                   "name_tag FullNodeGT"
# run_repeats ${DATASET} khopNodeGT                   "name_tag khopNodeGT"
# run_repeats ${DATASET} HierarchyNodeGT                   "name_tag HierarchyNodeGT"
# run_repeats ${DATASET} FullGT                   "name_tag FullGT"
# run_repeats ${DATASET} GT+TypeFFN         "name_tag GT+128+6+32+512+lr0.001"
# run_repeats ${DATASET} GT+AllFFN            "name_tag FullGT+FFN+520+6+128+lr0.001"

# run_repeats ${DATASET} SAGE-GT              "name_tag SAGE-EdgeGT"
# run_repeats ${DATASET} SAGE+GT            "name_tag SAGE+EdgeGT+sw512+256iter+bs128+bc32 gnn.dropout 0.2 gt.dropout 0.2 gt.attn_dropout 0.2 dataset.sample_width 512 dataset.sample_depth 6 train.iter_per_epoch 256 train.batch_size 128 val.iter_per_epoch 128 optim.base_lr 0.0005 optim.batch_accumulation 32"
# run_repeats ${DATASET} SAGE+khopNodeGT            "name_tag SAGE+khopNodeGT"
# run_repeats ${DATASET} SAGE-SAGE+GT            "name_tag SAGE-SAGE+EdgeGT"
# run_repeats ${DATASET} GT--SAGE            "name_tag FullGT--SAGE"
# run_repeats ${DATASET} SAGE+GT+Hetero_RWSE    "name_tag GPSwLapPE.GatedGCN+Trf.10run"
# run_repeats ${DATASET} NodeGT+Hetero_SDAB    "name_tag NodeGT+Hetero_SDAB"
# run_repeats ${DATASET} GT+Hetero_RWSE     "name_tag FullGT+Hetero_RWSE"
# run_repeats ${DATASET} GT+Hetero_SDPE              "name_tag FullGT+SDPE"

DATASET="mag-year"
# run_repeats ${DATASET} MLP                "name_tag MLP"
# run_repeats ${DATASET} MLP+Metapath       "name_tag MLP+Metapath"
# run_repeats ${DATASET} MLP+TransE         "name_tag MLP+TransE"
# run_repeats ${DATASET} LP+1Hop            "name_tag LP+1Hop"
# run_repeats ${DATASET} LP+2Hop            "name_tag LP+2Hop"
# run_repeats ${DATASET} SGC+1Hop           "name_tag SGC+1Hop"
# run_repeats ${DATASET} SGC+2Hop           "name_tag SGC+2Hop"

# Homogeneous GNN Baselines
# run_repeats ${DATASET} GCN                "name_tag GCN"
# run_repeats ${DATASET} GCN+LP                "name_tag GCN+LP"
# run_repeats ${DATASET} GCN+MS                "name_tag GCN+MS"
# run_repeats ${DATASET} GraphSAGE          "name_tag GraphSAGE"
# run_repeats ${DATASET} GIN                "name_tag GIN"
# run_repeats ${DATASET} GAT                "name_tag GAT"
# run_repeats ${DATASET} APPNP              "name_tag APPNP"
# run_repeats ${DATASET} NAGphormer         "name_tag NAGphormer"
# run_repeats ${DATASET} GraphTrans         "name_tag GraphTrans train.auto_resume True"
# run_repeats ${DATASET} Gophormer          "name_tag Gophormer"
# run_repeats ${DATASET} GOAT               "name_tag GOAT"

# Heterogeneous GNN Baselines
# run_repeats ${DATASET} RGCN               "name_tag RGCN"
# run_repeats ${DATASET} RGCN+LP            "name_tag RGCN+LP"
# run_repeats ${DATASET} RGAT               "name_tag RGAT"
# run_repeats ${DATASET} RGraphSAGE         "name_tag RGraphSAGE"
# run_repeats ${DATASET} RGraphSAGE+LP      "name_tag RGraphSAGE+LP"
# run_repeats ${DATASET} HAN                "name_tag HAN"
# run_repeats ${DATASET} HGT                "name_tag HGT"
# run_repeats ${DATASET} HGT+CL             "name_tag HGT+CL"
# run_repeats ${DATASET} HGT256             "name_tag HGT256"
# run_repeats ${DATASET} HGT_eval           "name_tag HGT_eval"
# run_repeats ${DATASET} HINormer           "name_tag HINormer wandb.use False"
# run_repeats ${DATASET} SHGN               "name_tag SHGN seed 44 train.auto_resume True"

# Heterophilic GNN Baselines
# run_repeats ${DATASET} MixHop               "name_tag MixHop"
# run_repeats ${DATASET} LINKX               "name_tag LINKX"
# run_repeats ${DATASET} FAGCN               "name_tag FAGCN"
# run_repeats ${DATASET} LSGNN               "name_tag LSGNN"

# Proposed GT
# run_repeats ${DATASET} SparseNodeGT       "name_tag SparseNodeGT"
# run_repeats ${DATASET} SparseNodeGT_Test       "name_tag SparseNodeGT_Test"
# run_repeats ${DATASET} SparseNodeGT+2Hop       "name_tag SparseNodeGT+2Hop"
# run_repeats ${DATASET} SparseNodeGT+Metapath+LP       "name_tag SparseNodeGT+Metapath+LP"
# run_repeats ${DATASET} GCN+SparseNodeGT+LP         "name_tag GCN+SparseNodeGT+LP"
# run_repeats ${DATASET} GCN+SparseNodeGT+Metapath+LP         "name_tag GCN+SparseNodeGT+Metapath+LP"
# run_repeats ${DATASET} GCN+SparseNodeGT_Test+Metapath+LP         "name_tag GCN+SparseNodeGT_Test+Metapath+LP"
# run_repeats ${DATASET} SparseEdgeGT       "name_tag SparseEdgeGT"
# run_repeats ${DATASET} SparseEdgeGT_Test       "name_tag SparseEdgeGT_Test"
# run_repeats ${DATASET} SparseEdgeGT+Metapath       "name_tag SparseEdgeGT+Metapath"
# run_repeats ${DATASET} GCN+SparseEdgeGT        "name_tag GCN+SparseEdgeGT"
# run_repeats ${DATASET} GCN+SparseEdgeGT+LP         "name_tag GCN+SparseEdgeGT+LP"
# run_repeats ${DATASET} GCN+SparseEdgeGT+Metapath+LP         "name_tag GCN+SparseEdgeGT+Metapath+LP"


DATASET="RCDD"
# run_repeats ${DATASET} MLP                "name_tag MLP"
# run_repeats ${DATASET} MLP+Metapath       "name_tag MLP+Metapath"
# run_repeats ${DATASET} MLP+Node2Vec         "name_tag MLP+Node2Vec"
# run_repeats ${DATASET} MLP+TransE         "name_tag MLP+TransE"
# run_repeats ${DATASET} LP+1Hop            "name_tag LP+1Hop"
# run_repeats ${DATASET} LP+2Hop            "name_tag LP+2Hop"
# run_repeats ${DATASET} SGC+1Hop           "name_tag SGC+1Hop"
# run_repeats ${DATASET} SGC+2Hop           "name_tag SGC+2Hop seed 46"

# Homogeneous GNN Baselines
# run_repeats ${DATASET} GCN                "name_tag GCN"
# run_repeats ${DATASET} GCN+MS                "name_tag GCN+MS"
# run_repeats ${DATASET} GraphSAGE          "name_tag GraphSAGE"
# run_repeats ${DATASET} GIN                "name_tag GIN"
# run_repeats ${DATASET} GAT                "name_tag GAT"
# run_repeats ${DATASET} APPNP              "name_tag APPNP"
# run_repeats ${DATASET} NAGphormer         "name_tag NAGphormer"
# run_repeats ${DATASET} GraphTrans         "name_tag GraphTrans"
# run_repeats ${DATASET} Gophormer          "name_tag Gophormer"
# run_repeats ${DATASET} GOAT               "name_tag GOAT"

# Heterogeneous GNN Baselines
# run_repeats ${DATASET} RGCN               "name_tag RGCN"
# run_repeats ${DATASET} RGAT               "name_tag RGAT"
# run_repeats ${DATASET} RGraphSAGE         "name_tag RGraphSAGE"
# run_repeats ${DATASET} HAN                "name_tag HAN"
# run_repeats ${DATASET} HGT                "name_tag HGT"
# run_repeats ${DATASET} HGT+CL             "name_tag HGT+CL"
# run_repeats ${DATASET} HGT256             "name_tag HGT256"
# run_repeats ${DATASET} HGT_eval           "name_tag HGT_eval"
# run_repeats ${DATASET} HINormer           "name_tag HINormer wandb.use False"
# run_repeats ${DATASET} SHGN               "name_tag SHGN"

# Heterophilic GNN Baselines
# run_repeats ${DATASET} MixHop               "name_tag MixHop"
# run_repeats ${DATASET} LINKX               "name_tag LINKX"
# run_repeats ${DATASET} FAGCN               "name_tag FAGCN"
# run_repeats ${DATASET} ACM-GCN             "name_tag ACM-GCN"
# run_repeats ${DATASET} LSGNN               "name_tag LSGNN"

# Proposed GT
# run_repeats ${DATASET} SparseNodeGT       "name_tag SparseNodeGT"
# run_repeats ${DATASET} SparseNodeGT+LP       "name_tag SparseNodeGT+LP"
# run_repeats ${DATASET} SparseNodeGT+Metapath+LP       "name_tag SparseNodeGT+Metapath+LP"
# run_repeats ${DATASET} GCN+SparseNodeGT        "name_tag GCN+SparseNodeGT"
# run_repeats ${DATASET} GCN+SparseNodeGT+LP         "name_tag GCN+SparseNodeGT+LP"
# run_repeats ${DATASET} SparseEdgeGT       "name_tag SparseEdgeGT"
# run_repeats ${DATASET} GCN+SparseEdgeGT        "name_tag GCN+SparseEdgeGT"
# run_repeats ${DATASET} GCN+SparseEdgeGT+LP         "name_tag GCN+SparseEdgeGT+LP"

DATASET="CCF"
# run_repeats ${DATASET} MLP                "name_tag MLP"
# run_repeats ${DATASET} MLP+Metapath       "name_tag MLP+Metapath"
# run_repeats ${DATASET} MLP+TransE         "name_tag MLP+TransE"

# Homogeneous GNN Baselines
# run_repeats ${DATASET} GCN                "name_tag GCN"
# run_repeats ${DATASET} GCN+MS                "name_tag GCN+MS"
# run_repeats ${DATASET} GraphSAGE          "name_tag GraphSAGE"
# run_repeats ${DATASET} GIN                "name_tag GIN"
# run_repeats ${DATASET} GAT                "name_tag GAT"
# run_repeats ${DATASET} NAGphormer         "name_tag NAGphormer"

# Heterogeneous GNN Baselines
# run_repeats ${DATASET} RGCN               "name_tag RGCN"
# run_repeats ${DATASET} RGCN_fast               "name_tag RGCN_fast"
# run_repeats ${DATASET} RGCN_weighted               "name_tag RGCN_weighted"
# run_repeats ${DATASET} RGCN+fullbatch               "name_tag RGCN+fullbatch"
# run_repeats ${DATASET} HAN                "name_tag HAN"
# run_repeats ${DATASET} HGT                "name_tag HGT"
# run_repeats ${DATASET} HGT_eval           "name_tag HGT_eval"


DATASET="IEEE-CIS"
# run_repeats ${DATASET} MLP                "name_tag MLP"
# run_repeats ${DATASET} MLP+Metapath       "name_tag MLP+Metapath"
# run_repeats ${DATASET} MLP+TransE         "name_tag MLP+TransE"
# run_repeats ${DATASET} LP+1Hop            "name_tag LP+1Hop"
# run_repeats ${DATASET} LP+2Hop            "name_tag LP+2Hop"
# run_repeats ${DATASET} SGC+1Hop           "name_tag SGC+1Hop"
# run_repeats ${DATASET} SGC+2Hop           "name_tag SGC+2Hop"

# Homogeneous GNN Baselines
# run_repeats ${DATASET} GCN                "name_tag GCN"
# run_repeats ${DATASET} GCN+MS             "name_tag GCN+MS"
# run_repeats ${DATASET} GraphSAGE          "name_tag GraphSAGE"
# run_repeats ${DATASET} GIN                "name_tag GIN"
# run_repeats ${DATASET} GAT                "name_tag GAT"
# run_repeats ${DATASET} APPNP              "name_tag APPNP"
# run_repeats ${DATASET} NAGphormer         "name_tag NAGphormer"
# run_repeats ${DATASET} GraphTrans         "name_tag GraphTrans"
# run_repeats ${DATASET} Gophormer          "name_tag Gophormer"
# run_repeats ${DATASET} GOAT               "name_tag GOAT"

# Heterogeneous GNN Baselines
# run_repeats ${DATASET} RGCN               "name_tag RGCN"
# run_repeats ${DATASET} RGCN_fast               "name_tag RGCN_fast"
# run_repeats ${DATASET} RGCN_weighted               "name_tag RGCN_weighted"
# run_repeats ${DATASET} RGCN+fullbatch               "name_tag RGCN+fullbatch"
# run_repeats ${DATASET} HAN                "name_tag HAN"
# run_repeats ${DATASET} HGT                "name_tag HGT"
# run_repeats ${DATASET} HGT_eval           "name_tag HGT_eval"

# Heterophilic GNN Baselines
# run_repeats ${DATASET} MixHop              "name_tag MixHop"
# run_repeats ${DATASET} LINKX               "name_tag LINKX"
# run_repeats ${DATASET} FAGCN               "name_tag FAGCN"
# run_repeats ${DATASET} LSGNN               "name_tag LSGNN"

# Proposed GT
# run_repeats ${DATASET} SparseNodeGT               "name_tag SparseNodeGT"
# run_repeats ${DATASET} SparseNodeGT+LP             "name_tag SparseNodeGT+LP"
# run_repeats ${DATASET} SparseNodeGT+Metapath+LP             "name_tag SparseNodeGT+Metapath+LP"
# run_repeats ${DATASET} SparseNodeGT+Node2Vec+LP     "name_tag SparseNodeGT+Node2Vec+LP"
# run_repeats ${DATASET} SparseEdgeGT               "name_tag SparseEdgeGT"
# run_repeats ${DATASET} SparseEdgeGT_Test               "name_tag SparseEdgeGT_Test"
# run_repeats ${DATASET} SparseEdgeGT+2Hop      "name_tag SparseEdgeGT+2Hop"
# run_repeats ${DATASET} GCN+SparseEdgeGT           "name_tag GCN+SparseEdgeGT"
# run_repeats ${DATASET} SparseEdgeGT+LP               "name_tag SparseEdgeGT+LP"
# run_repeats ${DATASET} GCN+SparseEdgeGT+LP           "name_tag GCN+SparseEdgeGT+LP"


DATASET="Pokec"
# run_repeats ${DATASET} MLP                "name_tag MLP"
# run_repeats ${DATASET} MLP+Metapath       "name_tag MLP+Metapath"
# run_repeats ${DATASET} LP+1Hop            "name_tag LP+1Hop"
# run_repeats ${DATASET} LP+2Hop            "name_tag LP+2Hop"
# run_repeats ${DATASET} SGC+1Hop           "name_tag SGC+1Hop"
# run_repeats ${DATASET} SGC+2Hop           "name_tag SGC+2Hop"

# Homogeneous GNN Baselines
# run_repeats ${DATASET} GCN                "name_tag GCN"
# run_repeats ${DATASET} GraphSAGE          "name_tag GraphSAGE"
# run_repeats ${DATASET} GAT                "name_tag GAT"
# run_repeats ${DATASET} GIN                "name_tag GIN"
# run_repeats ${DATASET} APPNP              "name_tag APPNP"
# run_repeats ${DATASET} NAGphormer         "name_tag NAGphormer"
# run_repeats ${DATASET} GraphTrans         "name_tag GraphTrans train.auto_resume True"
# run_repeats ${DATASET} Gophormer          "name_tag Gophormer"
# run_repeats ${DATASET} GOAT               "name_tag GOAT"

# Heterogeneous GNN Baselines
# run_repeats ${DATASET} RGCN               "name_tag RGCN_nonuniform_sampling"
# run_repeats ${DATASET} RGraphSAGE         "name_tag RGraphSAGE_nonuniform_sampling"

# Heterophilic GNN Baselines
# run_repeats ${DATASET} MixHop               "name_tag MixHop"
# run_repeats ${DATASET} LINKX               "name_tag LINKX"
# run_repeats ${DATASET} FAGCN               "name_tag FAGCN"

# Proposed GT
# run_repeats ${DATASET} SparseNodeGT               "name_tag SparseNodeGT_Hetero"
# run_repeats ${DATASET} SparseNodeGT+LP             "name_tag SparseNodeGT_Hetero+LP_nonuniform_sampling"
# run_repeats ${DATASET} SparseNodeGT+Node2Vec+LP             "name_tag SparseNodeGT+Node2Vec+LP"
# run_repeats ${DATASET} SparseNodeGT+Metapath+LP             "name_tag SparseNodeGT+Metapath+LP"


DATASET="PDNS"
# run_repeats ${DATASET} MLP                "name_tag MLP"
# run_repeats ${DATASET} MLP+Metapath       "name_tag MLP+Metapath"
# run_repeats ${DATASET} LP+1Hop            "name_tag LP+1Hop"
# run_repeats ${DATASET} LP+2Hop            "name_tag LP+2Hop"
# run_repeats ${DATASET} SGC+1Hop           "name_tag SGC+1Hop"
# run_repeats ${DATASET} SGC+2Hop           "name_tag SGC+2Hop"

# Homogeneous GNN Baselines
# run_repeats ${DATASET} GCN                "name_tag GCN"
# run_repeats ${DATASET} GraphSAGE          "name_tag GraphSAGE"
# run_repeats ${DATASET} GAT                "name_tag GAT"
# run_repeats ${DATASET} GIN                "name_tag GIN"
# run_repeats ${DATASET} APPNP              "name_tag APPNP"
# run_repeats ${DATASET} NAGphormer         "name_tag NAGphormer"
# run_repeats ${DATASET} GraphTrans         "name_tag GraphTrans"
# run_repeats ${DATASET} Gophormer          "name_tag Gophormer"
# run_repeats ${DATASET} GOAT               "name_tag GOAT"

# Heterogeneous GNN Baselines
# run_repeats ${DATASET} RGCN               "name_tag RGCN"
# run_repeats ${DATASET} RGraphSAGE         "name_tag RGraphSAGE"
# run_repeats ${DATASET} RGAT               "name_tag RGAT"
# run_repeats ${DATASET} HAN                "name_tag HAN"
# run_repeats ${DATASET} HGT                "name_tag HGT"
# run_repeats ${DATASET} HINormer           "name_tag HINormer wandb.use False"
# run_repeats ${DATASET} SHGN               "name_tag SHGN"

# Heterophilic GNN Baselines
# run_repeats ${DATASET} MixHop               "name_tag MixHop"
# run_repeats ${DATASET} LINKX               "name_tag LINKX"
# run_repeats ${DATASET} FAGCN               "name_tag FAGCN"
# run_repeats ${DATASET} ACM-GCN             "name_tag ACM-GCN"
# run_repeats ${DATASET} LSGNN               "name_tag LSGNN"

# Proposed GT
# run_repeats ${DATASET} SparseNodeGT               "name_tag SparseNodeGT"
# run_repeats ${DATASET} SparseNodeGT+LP             "name_tag SparseNodeGT+LP"
# run_repeats ${DATASET} SparseNodeGT+Metapath+LP             "name_tag SparseNodeGT+Metapath+LP"