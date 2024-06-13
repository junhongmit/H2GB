#!/usr/bin/env bash

# Run this script from the project root dir.

function calc_homophily {
    dataset=$1
    cfg_suffix=$2
    # The cmd line cfg overrides that will be passed to the main.py,
    # e.g. 'name_tag test01 gnn.layer_type gcnconv'

    cfg_file="${cfg_dir}/${dataset}/${dataset}-${cfg_suffix}.yaml"
    if [[ ! -f "$cfg_file" ]]; then
        echo "WARNING: Config does not exist: $cfg_file"
        echo "SKIPPING!"
        return 1
    fi

    main="python -m H2GB.calcHomophily --cfg ${cfg_file}"
    out_dir="/nobackup/users/junhong/Logs/results/${dataset}/homohomophily.log"  # <-- Set the output dir.

    echo "Run program: ${main} > ${out_dir}"

    # Run once
    script="${main} > ${out_dir}"
    echo $script
    eval $script
    echo $script
}

cfg_dir="configs"

# Uncomment the dataset that you want to calculate homophily

# calc_homophily  ogbn-mag  MLP
# calc_homophily  mag-year  MLP
# calc_homophily  oag-cs  MLP
# calc_homophily  oag-eng  MLP
# calc_homophily  oag-chem  MLP
# calc_homophily  RCDD  MLP
# calc_homophily  IEEE-CIS  MLP
# calc_homophily  Pokec  MLP
# calc_homophily  PDNS  MLP