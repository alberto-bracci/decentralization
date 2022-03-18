# The decentralized evolution of decentralization

## Description

## Installation
1. Clone this directory to your computer in a new folder, using the following command:
```
git clone https://github.com/alberto-bracci/decentralization
```
2. In order to ensure that everything is working as intended, create a dedicated environment using the specified requirements file, using: 
```conda env create -f decentralization.yml```
ACHTUNG: If you want to specify a specific install path rather than the default for your system, just use the -p flag followed by the required path, e.g.: 
```conda env create -f decentralization.yml -p /home/user/anaconda3/envs/decentralization```

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

After this, selected the keywords you want to use to create your dataset, open and run the following Jupyter notebook:
```
./notebooks/dataset_creation.ipynb
```

## Run hierarchical Stochastic Block Model (hSBM)


## TODO list
1. Go through notebooks/paper_analysis.ipynb, cut all unnecessary cells and move functions in new utils
1. Reorder clusters in figures
1. Add requirements file decentralization.yml
1. Update hsbm_equilibrate.sh and hsbm_consensus.sh
1. Add disc and running_time-memory requirements