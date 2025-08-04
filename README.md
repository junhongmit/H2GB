<div align="center">
  <img src="https://raw.githubusercontent.com/junhongmit/H2GB/main/imgs/logo_bg.png" width="30%" height="auto"/>
</div>

<div align="center">

# [When Heterophily Meets Heterogeneity:<br>Challenges and a New Large-Scale Graph Benchmark](https://dl.acm.org/doi/10.1145/3711896.3737421)
# 🏆 Best Paper Award, KDD 2025 Datasets & Benchmarks Track
[![ACM](https://img.shields.io/badge/ACM%20DL-KDD%202025-blue?logo=acm&logoColor=white)](https://dl.acm.org/doi/10.1145/3711896.3737421)
[![preprint](https://img.shields.io/static/v1?label=arXiv&message=2407.10916&color=B31B1B&logo=arXiv)](https://arxiv.org/pdf/2407.10916)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-FFD21E)](https://huggingface.co/datasets/junhongmit/H2GB)
[![PyPI](https://img.shields.io/static/v1?label=PyPI&message=H2GB&color=brightgreen&logo=pypi)](https://pypi.org/project/H2GB/)
[![ReadTheDocs](https://img.shields.io/static/v1?label=latest&message=ReadTheDocs&color=orange&logo=readthedocs)](https://junhongmit.github.io/H2GB/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/junhongmit/H2GB/blob/main/LICENSE)

<strong>Junhong Lin¹</strong>, <strong>Xiaojie Guo²</strong>, <strong>Shuaicheng Zhang³</strong>, <strong>Yada Zhu²</strong>, <strong>Dawei Zhou³</strong>, <strong>Julian Shun¹</strong><br>
  ¹ MIT CSAIL, ² IBM Research, ³ Virginia Tech

</div>


--------------------------------------------------------------------------------

## 📌 Overview
The Heterophilic and Heterogeneous Graph Benchmark (ℋ²GB) is a collection of graph benchmark datasets, data loaders, modular graph transformer framework (UnifiedGT) and evaluators for graph learning.
The ℋ²GB encompasses 9 diverse real-world datasets across 5 domains. Its data loaders are fully compatible with popular graph deep learning framework [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/). They provide automatic dataset downloading, standardized dataset splits, and unified performance evaluation.

<p align='center'>
  <img src="https://raw.githubusercontent.com/junhongmit/H2GB/main/imgs/flowchart_v4.png" width="80%" height="auto"/>
</p>


## ⚙️ Environment Setup
You can create a conda environment to easily run the code. For example, we can create a virtual environment named `H2GB`:
```
conda create -n H2GB python=3.9 -y
conda activate H2GB
```
Install the required packages using the following commands:
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg -c pyg
pip install -r requirements.txt
```

> 💡 **We have also pack everything for you on the PyPI, and the easiest way to use the library is simply running:**
> ```
> pip install H2GB
> ```


## 🚀 Run the UnifiedGT
To summarize and systematically compare the performance of existing GNNs on `H2GB`, we designed `UnifiedGT`. `UnifiedGT` is a modular graph transformer framework that  designed to encompass many existing GTs and GNNs by leveraging unified components: (1) graph sampling, (2) graph encoding, (3) graph attention, (4) attention masking, (5) heterogeneous GNN, and (6) feedforward networks (FFN). It is implemented as a Python library and is user-friendly. It includes a unified data loader and evaluator, making it easy for researchers to access datasets, evaluate methods, and compare performance.

<p align='center'>
  <img src="https://raw.githubusercontent.com/junhongmit/H2GB/main/imgs/framework_v4.png" width="80%" height="auto"/>
</p> 

We implement 9 existing GT baselines and 19 GNN models based on `UnifiedGT` and provide comprehensive experiment configurations available in `./configs`. To run `UnifiedGT`, you will need to firstly specify the dataset and log location by editing the config file provided under `./configs/{dataset_name}/`. An example configuration is
```
......
out_dir: ./results/{Model Name} # Put your log output path here
dataset:
  dir: ./data # Put your input data path here
......
```
Dataset download will be automatically initiated if dataset is not found under the specified location.

For convenience, a script file is created to run the experiment with specified configuration. For instance, you can edit and run the `interactive_run.sh` to start the experiment.
```
# Assuming you are located in the H2GB repo
chmox +x ./run/interactive_run.sh
./run/interactive_run.sh
```

You can also directly enter this command into your terminal:
```
python -m H2GB.main --cfg {Path to Your Configs} name_tag {Custom Name Tag}
```
For example, the following command is to run `MLP` model experiment for `oag-cs` dataset.
```
python -m H2GB.main --cfg configs/oag-cs/oag-cs-MLP.yaml name_tag MLP
```

## 📊 Caclulate the class-adjusted heterogeneous heterophily index (ℋ² Index)
We provide a extended heterophily measurement from homogeneous grpah into the heterogeneous setting, which is called metapath-induced heterophily measrement. The calcualtion function is available in `./H2GB/calcHomophily.py`. You can simply import it by using `from H2GB.calcHomophily import calcHomophily` and measure the heterophily of your data. For convenience, we also provide a script to reproduce the heterophily measurement on our developed datasets. Note that $$\text{Heterophily} = 1 - \text{Homophily}$$ So just do a simple transformation to obtain the heterophily.
```
chmox +x ./run/calcHomo.sh
./run/calcHomo.sh
```

## 📚 Citation
If you find this dataset or benchmark useful, please consider citing the following paper:

```
@inproceedings{lin2025h2gb,
  author    = {Lin, Junhong and Guo, Xiaojie and Zhang, Shuaicheng and Zhu, Yada and Shun, Julian},
  title     = {When Heterophily Meets Heterogeneity: Challenges and a New Large-Scale Graph Benchmark},
  booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
  year      = {2025},
  pages     = {5607--5618},
  doi       = {10.1145/3711896.3737421},
  url       = {https://doi.org/10.1145/3711896.3737421}
}
```

## 📝 Additional Notes
### 🧩 Encoders
The Hetero_Raw encoder are supposed to be used for heterogeneous GNN or graph dataset that has  different node encoding dimensions for different node type. Therefore, each node type can be transformed separately. To reproduce results of homogeneous GNN, consider using the Raw encoder, which apply the same transformation for each node type. Otherwise, using Hetero_Raw for homogeneous GNN will misleadingly increase the task performance.
