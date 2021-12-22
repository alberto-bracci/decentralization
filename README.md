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