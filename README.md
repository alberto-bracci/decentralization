# The decentralized evolution of decentralization

## Description

## Installation

## Collect the data
For this project, we use one of the releases of the Semantic Scholar Corpus.

Download corpus from
https://api.semanticscholar.org/corpus/download/

Release used in the paper: 2022-01-01 release

In order to download the selected release, from the root folder, run the following commands:

```
mkdir corpus
cd corpus
wget https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/2022-01-01/manifest.txt
wget -B https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/2022-01-01/ -i manifest.txt
```

After this, selected the keywords you want to use to create your dataset, run the following notebook:
```
./notebooks/dataset_creation.ipynb
```

## Run hierarchical Stochastic Block Model (hSBM)


## TODO list
1. add docstrings to functions in hsbm_creation.py
1. go through all functions in hsbm_fit.py in detail
1. go through all functions in hsbm_partitions.py in detail
1. clean edited_texts part in hsbm.py
1. go through all analysis part in hsbm.py
1. go through notebook semschol_graph_tool_analysis for plots/analysis