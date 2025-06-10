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
    common_params="${cfg_overrides}"

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
# run_repeats ${DATASET} PolyFormer         "name_tag PolyFormer"

# Proposed GT
# run_repeats ${DATASET} H2Gformer          "name_tag H2Gformer"

DATASET="oag-cs"
# run_repeats ${DATASET} MLP                "name_tag MLP"
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
# run_repeats ${DATASET} PolyFormer         "name_tag PolyFormer wandb.use False"

# Proposed GT
# run_repeats ${DATASET} H2Gformer          "name_tag H2Gformer"


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
# run_repeats ${DATASET} HINormer           "name_tag HINormer"

# Heterophilic GNN Baselines
# run_repeats ${DATASET} MixHop               "name_tag MixHop"
# run_repeats ${DATASET} LINKX               "name_tag LINKX"
# run_repeats ${DATASET} FAGCN               "name_tag FAGCN"
# run_repeats ${DATASET} LSGNN               "name_tag LSGNN"
# run_repeats ${DATASET} PolyFormer         "name_tag PolyFormer"

# Proposed GT
# run_repeats ${DATASET} H2Gformer          "name_tag H2Gformer"


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
# run_repeats ${DATASET} PolyFormer         "name_tag PolyFormer"

# Proposed GT
# run_repeats ${DATASET} H2Gformer          "name_tag H2Gformer"


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
# run_repeats ${DATASET} PolyFormer         "name_tag PolyFormer"

# Proposed GT
# run_repeats ${DATASET} H2Gformer          "name_tag H2Gformer"


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
# run_repeats ${DATASET} PolyFormer         "name_tag PolyFormer"

# Proposed GT
# run_repeats ${DATASET} H2Gformer          "name_tag H2Gformer"


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
# run_repeats ${DATASET} RGCN_fast          "name_tag RGCN_fast"
# run_repeats ${DATASET} RGCN_weighted      "name_tag RGCN_weighted"
# run_repeats ${DATASET} RGCN+fullbatch     "name_tag RGCN+fullbatch"
# run_repeats ${DATASET} HAN                "name_tag HAN"
# run_repeats ${DATASET} HGT                "name_tag HGT"
# run_repeats ${DATASET} HGT_eval           "name_tag HGT_eval"

# Heterophilic GNN Baselines
# run_repeats ${DATASET} MixHop             "name_tag MixHop"
# run_repeats ${DATASET} LINKX              "name_tag LINKX"
# run_repeats ${DATASET} FAGCN              "name_tag FAGCN"
# run_repeats ${DATASET} LSGNN              "name_tag LSGNN"
# run_repeats ${DATASET} PolyFormer         "name_tag PolyFormer"

# Proposed GT
# run_repeats ${DATASET} H2Gformer          "name_tag H2Gformer"



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
# run_repeats ${DATASET} PolyFormer         "name_tag PolyFormer"

# Proposed GT
# run_repeats ${DATASET} H2Gformer          "name_tag H2Gformer"


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
# run_repeats ${DATASET} MixHop             "name_tag MixHop"
# run_repeats ${DATASET} LINKX              "name_tag LINKX"
# run_repeats ${DATASET} FAGCN              "name_tag FAGCN"
# run_repeats ${DATASET} ACM-GCN            "name_tag ACM-GCN"
# run_repeats ${DATASET} LSGNN              "name_tag LSGNN"
# run_repeats ${DATASET} PolyFormer         "name_tag PolyFormer"

# Proposed GT
# run_repeats ${DATASET} H2Gformer          "name_tag H2Gformer"