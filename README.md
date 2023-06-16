# The decentralized evolution of decentralization across fields: from Governance to Blockchain (by Gabriele Di Bona*, Alberto Bracci*, Nicola Perra, Vito Latora, and Andrea Baronchelli)
\*[gabriele-di-bona](https://github.com/gabriele-di-bona) and [alberto-bracci](https://github.com/alberto-bracci/) contributed equally to this work

## Description

This Github repository provides the pipeline published with the article at https://arxiv.org/abs/2207.14260.
With this pipeline, one can analyse the time evolution of the scholarly debate around any concept (in the article, "decentralization" has been studied). It combines both semantic and bibliographic information using a multilayer hierarchical stochastic block model and knowledge flows. 

### Reference
If you want to use any part of this code, you are more than welcome to do so, but you need to cite our paper as
```
Di Bona, G., Bracci, A., Perra, N., Latora, V., & Baronchelli, A. (2022). The decentralized evolution of decentralization across fields: from Governance to Blockchain. arXiv preprint arXiv:2207.14260.
```
or using the following bibtex entry.
```
@article{dibona2022decentralization,
    title = {The decentralized evolution of decentralization across fields: from Governance to Blockchain},
    abstract = {Decentralization is a widespread concept across disciplines such as Economics, Political Science and Computer Science, which use it in distinct but overlapping ways. 
Here, we investigate the scholarly history of the term by analysing 425k academic publications mentioning (de)centralization. We find that the fraction of papers on the topic has been exponentially increasing since the 1950s, with 1 author in 154 mentioning (de)centralization in the title or abstract of an article in 2021. 
We then cluster papers using both semantic and citation information and show that the topic has independently emerged in different fields, while  cross-disciplinary contamination started only more recently.
Finally, we show how Blockchain has become the most influential field about 10 years ago, while Governance dominated before the 1990s, and we characterize their interactions with other fields.
Our findings help quantify the evolution of a key yet elusive concept. Furthermore, our general framework-whose code is publicly released alongside this paper-may be used to run similar analyses on virtually any other concept in the academic literature.},
    author = {Di Bona, Gabriele and Bracci, Alberto and Perra, Nicola and Latora, Vito and Baronchelli, Andrea},
    journal = {arXiv preprint arXiv:2207.14260},
    archivePrefix = {arXiv},
    arxivId = {2207.14260},
}
```

## Installation
1. Clone this directory to your computer in a new folder, using the following command:
```
git clone https://github.com/alberto-bracci/decentralization
```
2. All computations in this repository is done in Python 3.9.5, apart from ```./notebooks/r_plots.ipynb``` which is run in R. In order to ensure that everything is working as intended, create a dedicated environment using the specified requirements file, using: 
```conda env create -f decentralization.yml```
ACHTUNG: If you want to specify a specific install path rather than the default for your system, just use the -p flag followed by the required path, e.g.: 
```conda env create -f decentralization.yml -p /home/user/anaconda3/envs/decentralization```

## Download S2AG dataset

**WARNING**: this guide has been written in early January 2022. Since then, the Semantic Scholar website has changed, requesting to be authenticated to use these APIs. This section might hence not be fully updated. As of early August 2022, it seems like the present guide still works for the 2022-01-01 release.

For this project, we use one of the releases of the Semantic Scholar Academic Graph dataset (S2AG).

Download corpus from
https://api.semanticscholar.org/corpus/download/.

Release used in the paper: 2022-01-01 release.

In order to download the selected release, from the root folder, run the following commands:

```
mkdir corpus
cd corpus
wget https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/2022-01-01/manifest.txt
wget -B https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/2022-01-01/ -i manifest.txt
```

**WARNING**: The downloaded corpus (made of 6000 zip files) weighs about 192GB.

## Create Dataset

After downloading a local version of the S2AG dataset, you can create your own dataset including the information about all the papers containing a certain keyword root in title and/or abstract using the following Jupyter notebook:
```
./notebooks/dataset_creation.ipynb
```

The downloaded data is stored in subfolders of the created folder `./data`

**WARNING**: For decentralization, the downloaded dataset weighs around 600MB.


## Run multilayer hierarchical Stochastic Block Model (hSBM)
In order to run the multilayer hSBM as described in the article, one needs to run a series of bash scripts, that launch python instances. These are:
1. ```./bash_scripts/hsbm_prep.sh```
1. ```./bash_scripts/hsbm_fit.sh```
1. ```./bash_scripts/hsbm_equilibrate.sh```
1. ```./bash_scripts/hsbm_consensus.sh```

These bash scripts are written for the QMUL apocrita HPC cluster, where all computations have been run. Possibly, you can use a similar script for your cluster. If you instead need to run this on your local machine, then you can use these bash scripts as a reference for what you need to do (just look at the line `python ...`).

**WARNING**: The scripts are meant to be executed SEQUENTIALLY, i.e., one after the other. This means that first you need to run `hsbm_prep.sh`, then, after the previous is finished, you can launch `hsbm_fit.sh`, etc. 

**WARNING**: Notice that `hsbm_fit.sh` and `hsbm_equilibrate` launch an array of 100 different instances, which can be executed separately or at the same time, i.e., they do not interfere with one another. They are then brought together when launching `hsbm_consensus.sh`.

**WARNING**: For decentralization, this process creates subfolders under `./data` where all the necessary files are dumped and stored, for a total of about 20GB. Once the whole process is finished, one could delete the various instances folders, keeping only the consensus one, which is about 3 to 4 GB.

## Analysis
### Annotation of keywords
In order to complete the analysis started in the previous section when running the hSBM, one now needs to do first annotate the keywords. In order to do, you can launch the Jupyter notebook 
```
./notebooks/cluster_labelling.ipynb
```

In this notebook, all clusters at the chosen level are analysed at length. This analysis focuses on the normalized mixture proportion of the identified topics (which are recognized through ranked words), the most important paper titles according to different centrality measures, as well word frequencies.
Keywords are hence annotated within the notebook.

### Main analysis
Almost all other figures shown in the notebook are generated using the Jupyter notebook 
```
./notebooks/figures.ipynb
```

Notice that Figure 1 is partially created in the notebook, since it needs to be unified with a image-editing program, like Inkscape.

Figure 4 instead is generated in R. In order to create it, one first needs to create the proper data in the related section in `./notebooks/figures.ipynb`. After this, one runs `./notebooks/r_plots.ipynb` with the files created in the python notebook. The generated figures are then visually modified in Inkscape to obtain the figures in the manuscript.

Further notice that the maximum overlap and NMI between the single partitions of the 100 runs of the hSBM and the consensus partitions are computed separately using the script `./bash_scripts/overlap.sh`, using the python script in `./scripts/overlap.py`.


## Acknowledgements
Thank you for the interest and for using this repository. If you need to contact us for any reason, please do not hesitate to send an email to g.dibona@qmul.ac.uk.

The authors
