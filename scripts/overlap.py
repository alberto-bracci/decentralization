#!/usr/bin/env python
# coding: utf-8


# IMPORTS

root_dir = '../'

import gzip
import pickle
import platform 
from random import choice
import scipy.stats
import gi
from gi.repository import Gtk, Gdk
import graph_tool.all as gt
import graph_tool as graph_tool
import pandas as pd
import numpy as np
import os
import copy
import time
from sklearn.feature_extraction import text
from nltk.stem import  WordNetLemmatizer
import re
from tqdm.notebook import tqdm
from datetime import datetime
import json
from scipy.stats import linregress
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import random
import seaborn as sn
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
from IPython.display import display

import ast # to get list comprehension from a string
import functools, builtins # to impose flush=True on every print
builtins.print = functools.partial(print, flush=True)

# move to repo root folder
os.chdir(root_dir)
import sys
sys.path.insert(0, os.path.join(os.getcwd(),"utils"))
# from utils import sbmmultilayer 
# # from hsbm.utils.nmi import * 
# from utils.doc_clustering import construct_nmi_matrix # *
# from hsbm_creation import *
# # from hsbm_fit import *
# from hsbm_partitions import *
# from hsbm_knowledge_flow import *
# from hsbm_analysis_topics import *

import sbmmultilayer 
# from hsbm.utils.nmi import * 
from doc_clustering import construct_consensus_nmi_matrix # *
from doc_clustering import _nmis # *

# from doc_clustering import *
from hsbm_creation import *
# from hsbm_fit import *
from hsbm_partitions import *
from hsbm_knowledge_flow import *
from hsbm_analysis_topics import *


import distinctipy

# DEFAULT PARAMETERS IN THE FIGURES TO BE ADJUSTED!!!!

plt.style.use("default")

height_fig = 8
width_fig = 21

params_default = {
    # no upper and right axes
    'axes.spines.right' : False,
    'axes.spines.top' : False,
    # no frame around the legend
    "legend.frameon" : False,

    # dimensions of figures and labels
    # we will play with these once we see how they are rendered in the latex
    'figure.figsize' : (width_fig, height_fig),

    'axes.labelsize' : 22,
    'axes.titlesize' : 25,
    'xtick.labelsize' : 18,
    'ytick.labelsize' : 18,
    'legend.fontsize' : 16, 
    
    # no grids (?)
    'axes.grid' : False,

    # the default color(s) for lines in the plots: in order if multiple lines. We can change them or add colors if needed
#     'axes.prop_cycle' : mpl.cycler(color=["#00008B", "#BF0000", "#006400"]), 

    # default quality of the plot. Not too high but neither too low
    "savefig.dpi" : 100,
    "savefig.bbox" : 'tight', 
    
    # To use standard latex font
#     'mathtext.fontset': 'stix',
#     'font.family': 'STIXGeneral',
    # to use sans serif arial, i.e., standard figure font
    'font.family' : 'sans-serif',
    'font.sans-serif' : 'Arial', # 'Tahoma', # 'Comic Sans MS'
    'mathtext.fontset' : 'custom',
    'mathtext.it' : 'Arial:italic',
    'mathtext.rm' : 'Arial',
}

plt.rcParams.update(params_default)

# To make colored prints!
class print_color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def split_words(string):
    '''
        Get a string and divide into list of words and other characters.
        
        Paramaters
        ----------
            string: string to split (str)
        
        Returns
        ----------
            words: list of words and characters (list of str)
    '''
    words = []
    word_tmp = ""
    for letter in string:
        if letter in "QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm":
            word_tmp += letter
        else:
            words.append(word_tmp)
            words.append(letter)
            word_tmp = ""
    if letter in "QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm":
        words.append(word_tmp)
    return words




# ARGUMENTS FROM SHELL

import argparse
parser = argparse.ArgumentParser(description='Filtering the documents and creating the GT network.')

parser.add_argument('-i', '--ID', type=int,
    help='Array-ID of multiple hSBM, useful only if do_analysis=0. Used also as seed for the mcmc iteration [default 1]',
    default=1)

arguments = parser.parse_args()

ID = arguments.ID - 1


print('\nStarting')
start = datetime.now()


# FOLDERS

corpus_version = '2022-01-01'
all_dataset_path = f"data/{corpus_version}/"
dataset_path = f"data/{corpus_version}/decentralization/"
paper_figures_folder = f"figures/"

filter_label = '' # if you want a specific label to do some testing
min_inCitations = 0
min_word_occurences = 5
use_titles = True

results_folder = os.path.join(dataset_path,f'{min_inCitations}_min_inCitations_{min_word_occurences}_min_word_occurrences/')

chosen_text_attribute = 'paperAbstract'
if use_titles:
    # USE TITLES TEXT INSTEAD OF ABSTRACTS
    print('Using titles text instead of abstracts!', flush=True)
    chosen_text_attribute = 'title'
    results_folder = results_folder[:-1] + '_titles/'
prep_results_folder = results_folder

# results_folder += 'ID_1_no_iterMC_5000/'
results_folder += 'consensus_all_100_iter/'

number_iterations_MC_equilibrate = 5000
ID_iteration_list = list(range(1,101))
dir_list = [os.path.join(results_folder, f'ID_{ID_iteration}_no_iterMC_{number_iterations_MC_equilibrate}/') for ID_iteration in ID_iteration_list]

overlap_results_folder = results_folder + 'overlap/'
os.makedirs(overlap_results_folder, exist_ok = True)

nmi_results_folder = results_folder + 'nmi/'
os.makedirs(nmi_results_folder, exist_ok = True)


# PREP

with gzip.open(os.path.join(results_folder,f'results_fit_greedy_partitions_docs_all.pkl.gz'),'rb') as fp:
    hyperlink_text_hsbm_partitions_by_level,duration = pickle.load(fp)
print(f'Loaded doc partitions from {results_folder}', flush=True)    

with gzip.open(os.path.join(results_folder,f'results_fit_greedy_partitions_words_all.pkl.gz'),'rb') as fp:
    H_T_word_hsbm_partitions_by_level, H_T_word_hsbm_num_groups_by_level = pickle.load(fp)
print(f'Loaded word partitions from {results_folder}', flush=True)

with gzip.open(f'{results_folder}results_fit_greedy_partitions_consensus_all{filter_label}.pkl.gz','rb') as fp:
    h_t_doc_consensus_by_level, h_t_word_consensus_by_level, h_t_consensus_summary_by_level = pickle.load(fp)
print(f'Loaded consensus partitions from {results_folder}', flush=True)    


end = datetime.now()
print('Time duration load',end-start,flush=True)



def compute_partition_overlap(partition_i, partition_j):
    """
    Compute the maximum partition overlap between the two partitions.
    """
    return gt.partition_overlap(partition_i, partition_j)


def _max_overlap_partition(partitions):
    """
    Helper function for calculating the partition overlap. Take in list of partitions from 2 different models and
    computes the partition overlap between the two models' partitions.

    partitions: list of list
    The partitions will be two lists of lists of size N_ITER where each list corresponds to partitions of a model 
    where we retrieve N_ITER partitions each time.

    Return: list of partition overlap between partitions.
    """
    n = len(partitions) # 10 * 10, depends on number of iterations to retrieve partitions
    overlap_partition_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            overlap_partition_matrix[i,j] = compute_partition_overlap(partitions[i], partitions[j])
    return list(overlap_partition_matrix[np.triu_indices(n,1)]) # return upper triangle for array.

def construct_maximum_overlap_partition_matrix_consensus(partitions, consensus_partitions):
    """
    Compute maximum overlap partition matrix for all partitions generated by models against true partition.

    partitions: list of list of partitions for each model (iteration). 
    true_partition: single list of true partitions.

    Remark: For example, we may generate 20 different partitions for the hSBM and compare it to the true partitions.
    First column is for the true partition.
    """
    num_models = len(partitions) # number of different models we are testing.
    num_consensus_models = len(consensus_partitions) # number of different models we are testing.
    # Store the average and standard deviation of partition overlap between partitions in a n x n matrix
    max_partition_overlap_avg = np.zeros((num_models, num_consensus_models))
    max_partition_overlap_std = np.zeros((num_models, num_consensus_models))

    # Iterate through NMI matrix and compute partition overlap between models excluding the ground truth.
    # We do not iterate through the first column.
    for i in tqdm(range(num_models)):
        for j in tqdm(range(num_consensus_models)):
            partition_overlaps = _max_overlap_partition(partitions[i] + consensus_partitions[j]) # retrieve list of partitions for model i-1 and model j-1
            # Store mean and std of partition overlap.
            max_partition_overlap_avg[i,j] = np.average(partition_overlaps)
            max_partition_overlap_std[i,j] = np.std(partition_overlaps)

#     max_partition_overlap_avg[0,0], max_partition_overlap_std[0,0] = 1, 0 # true partition should have NMI of 1 with itself.
#     # Compute the NMI for each model against ground truth. Corresponds to 1st column of NMI matrix.
#     for i in range(num_models):
#         # Compute NMI of model's partition with ground truth labels.
#         partition_overlap_with_true = [compute_partition_overlap(p, true_partition) for p in partitions[i]]
#         max_partition_overlap_avg[0, i+1] = np.average(partition_overlap_with_true)
#         max_partition_overlap_std[0, i+1] = np.std(partition_overlap_with_true)
    return (max_partition_overlap_avg.T, max_partition_overlap_std.T)            



print('\nStarting overlap doc')
start = datetime.now()

num_level_text_partitions = len(hyperlink_text_hsbm_partitions_by_level)
num_level_word_partitions = len(H_T_word_hsbm_partitions_by_level)
print(num_level_text_partitions, num_level_word_partitions, flush=True)

num_level_text_consensus_partitions = len(h_t_doc_consensus_by_level)
num_level_word_consensus_partitions = len(h_t_word_consensus_by_level)
print(num_level_text_consensus_partitions, num_level_word_consensus_partitions, flush=True)

num_calc_IDs = num_level_text_partitions * num_level_text_consensus_partitions
lev_partition = ID % num_level_text_partitions
lev_consensus_partition = int(ID / num_level_text_partitions)
print(ID, lev_partition, lev_consensus_partition, flush=True)

# Compute the average partition overlap and standard deviation
max_partition_overlap_avg, max_partition_overlap_std = construct_maximum_overlap_partition_matrix_consensus([hyperlink_text_hsbm_partitions_by_level[lev_partition]], [h_t_doc_consensus_by_level[lev_consensus_partition]])
print(max_partition_overlap_avg, max_partition_overlap_std, flush=True)
with open(os.path.join(overlap_results_folder, f'doc_{lev_partition}_{lev_consensus_partition}.pkl'), 'wb') as fp:
    pickle.dump((max_partition_overlap_avg, max_partition_overlap_std), fp)
    
end = datetime.now()
print('Overlap doc Time duration',end-start,flush=True)


print('\nStarting overlap word')
start = datetime.now()

# Compute the average partition overlap and standard deviation
max_partition_overlap_avg, max_partition_overlap_std = construct_maximum_overlap_partition_matrix_consensus([H_T_word_hsbm_partitions_by_level[lev_partition]], [h_t_word_consensus_by_level[lev_consensus_partition]])
print(max_partition_overlap_avg, max_partition_overlap_std, flush=True)
with open(os.path.join(overlap_results_folder, f'word_{lev_partition}_{lev_consensus_partition}.pkl'), 'wb') as fp:
    pickle.dump((max_partition_overlap_avg, max_partition_overlap_std), fp)
    
end = datetime.now()
print('Overlap doc Time duration',end-start,flush=True)



print('\nStarting nmi doc')
start = datetime.now()

# Compute the average partition overlap and standard deviation
nmi_avg, nmi_std = construct_consensus_nmi_matrix([hyperlink_text_hsbm_partitions_by_level[lev_partition]], [h_t_doc_consensus_by_level[lev_consensus_partition]])
print(nmi_avg, nmi_std, flush=True)
with open(os.path.join(nmi_results_folder, f'doc_{lev_partition}_{lev_consensus_partition}.pkl'), 'wb') as fp:
    pickle.dump((nmi_avg, nmi_std), fp)
    
end = datetime.now()
print('NMI doc Time duration',end-start,flush=True)


print('\nStarting nmi word')
start = datetime.now()

# Compute the average partition overlap and standard deviation
nmi_avg, nmi_std = construct_consensus_nmi_matrix([H_T_word_hsbm_partitions_by_level[lev_partition]], [h_t_word_consensus_by_level[lev_consensus_partition]])
print(nmi_avg, nmi_std, flush=True)
with open(os.path.join(nmi_results_folder, f'word_{lev_partition}_{lev_consensus_partition}.pkl'), 'wb') as fp:
    pickle.dump((nmi_avg, nmi_std), fp)
    
end = datetime.now()
print('NMI word Time duration',end-start,flush=True)


print('FINISHED')