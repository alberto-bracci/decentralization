#!/usr/bin/env python
# coding: utf-8

# IMPORTS

import gzip
import pickle
import pandas as pd
import numpy as np
import os
# move to repo root directory
os.chdir('../')
from datetime import datetime
import sys
sys.path.insert(0, os.path.join(os.getcwd(),'utils'))
from hsbm import sbmmultilayer 
from hsbm.utils.nmi import *
from hsbm.utils.doc_clustering import *
from hsbm_creation import *
from hsbm_fit import *
from hsbm_partitions import *
from hsbm_analysis_topics import *
from hsbm_knowledge_flow import *
from gi.repository import Gtk, Gdk
import graph_tool.all as gt
import graph_tool as graph_tool
import ast # to get list comprehension from a string
import functools, builtins # to impose flush=True on every print
builtins.print = functools.partial(print, flush=True)


# ARGUMENTS FROM SHELL

import argparse
parser = argparse.ArgumentParser(description='Filtering the documents and creating the GT network.')

parser.add_argument('-data', '--dataset_path', type=str,
    help='Path to the dataset with respect to the repo root folder, e.g. "data/2021-09-01/decentralization/" (do NOT start with "/" and DO end with "/") [REQUIRED]',
    required=True)
parser.add_argument('-analysis_results_subfolder', '--analysis_results_subfolder', type=str,
    help='If not changed, if do_analysis==0 it is set automatically to the iteration subfolder, while if do_analysis==1 it is the same as dataset_path.\
            If instead it is provided, it creates a subfolder inside results_folder in which to save the results (specific iterations loaded from dataset_path)',
    default='')
parser.add_argument('-i', '--ID', type=int,
    help='Array-ID of multiple hSBM, useful only if do_analysis=0. Used also as seed for the mcmc iteration [default 1]',
    default=1)
parser.add_argument('-NoIterMC', '--number_iterations_MC_equilibrate', type=int,
    help='Number of iterations in the MC equilibration step of the hSBM algorithm. [default 5000]',
    default=5000)
parser.add_argument('-occ', '--min_word_occurences', type=int,
    help='Minimum number of articles a word needs to present in. [default 5]',
    default=5)
parser.add_argument('-titles', '--use_titles', type=int,
    help='If equal to 1 use titles instead of abstracts, otherwise use abstracts [default 1]',
    default=1)
parser.add_argument('-analysis', '--do_analysis', type=int,
    help='Do analysis after gathering all iterations, instead of just the one provided by the ID. [default 0]',
    default=0)
parser.add_argument('-id_list', '--id_list_for_analysis', type=str,
    help='list of ids for which we do the analysis, list written as string (with brackets, see default), and then converted to list in the script. By default set to the list of the provided id. [default "[1]"]',
    default='[1]')
parser.add_argument('-stop', '--stop_at_fit', type=int,
    help='If stop_at_fit is 1, it does only the fit and saves it in a temporary file, otherwise it does also the equilibrate. [default 0]',
    default=0)
parser.add_argument('-prep', '--do_only_prep', type=int,
    help='If do_only_prep is 1, it does only the preparation of all files needed to do the hsbm and the following analysis. [default 0]',
    default=0)
parser.add_argument('-lev_kf', '--first_level_knowledge_flow', type=int,
    help='Calculate the knowledge flows starting from this first level. [default 2]',
    default=2)

arguments = parser.parse_args()

dataset_path = arguments.dataset_path
analysis_results_subfolder = arguments.analysis_results_subfolder
if len(analysis_results_subfolder) > 0 and analysis_results_subfolder[-1] != '/':
    analysis_results_subfolder += '/'
ID = arguments.ID
number_iterations_MC_equilibrate = arguments.number_iterations_MC_equilibrate
min_word_occurences = arguments.min_word_occurences
use_titles = arguments.use_titles == 1
do_analysis = arguments.do_analysis
_id_list_string = arguments.id_list_for_analysis
_id_list = ast.literal_eval(_id_list_string)
if do_analysis == 0:
    _id_list = [ID]
stop_at_fit = arguments.stop_at_fit == 1
do_only_prep = arguments.do_only_prep == 1
first_level_knowledge_flow = arguments.first_level_knowledge_flow


# Parameters needed in the process.
# ACHTUNG: Be aware to change this!
min_inCitations = 0 # Minimum number of citations filter
N_iter = 1 # Number of iterations of the hSBM algorithm. Now it's not parallelized, so be aware of computational time
n_cores = 1 # If you do plots or other graph_tool stuff, this needs to be properly set with the number of available cores
gt.openmp_set_num_threads(n_cores)


# FILES DIRECTORY

filter_label = '' # if you want a specific label to do some testing, to use at the end of the names to be saved

results_folder = os.path.join(dataset_path,f'{min_inCitations}_min_inCitations_{min_word_occurences}_min_word_occurrences/')
chosen_text_attribute = 'paperAbstract'
if use_titles:
    # USE TITLES TEXT INSTEAD OF ABSTRACTS
    print('Using titles text instead of abstracts!', flush=True)
    chosen_text_attribute = 'title'
    results_folder = results_folder[:-1] + '_titles/'
results_folder_iteration = os.path.join(results_folder, f'ID_{ID}_no_iterMC_{number_iterations_MC_equilibrate}/')
print(results_folder_iteration,flush=True)
print(f'Filtering with at least {min_inCitations} citations and {N_iter} iterations and {number_iterations_MC_equilibrate} swaps in MC-equilibrate.',flush=True)
os.makedirs(results_folder, exist_ok = True)
if len(analysis_results_subfolder) > 0:
    os.makedirs(results_folder+analysis_results_subfolder, exist_ok = True)

    
# LOADING DATA

print('\nLoading data set')
with gzip.open(f'{dataset_path}papers_dict.pkl.gz', 'rb') as fp:
    all_docs_dict = pickle.load(fp)

# Check that every doc has the chosen attribute
all_docs_dict = {x:all_docs_dict[x] for x in all_docs_dict if chosen_text_attribute in all_docs_dict[x] and all_docs_dict[x][chosen_text_attribute] is not None and len(all_docs_dict[x][chosen_text_attribute])>0}
print('total number of docs: %d'%(len(all_docs_dict)))

# tokenized texts
tokenized_texts_dict = load_tokenized_texts_dict(all_docs_dict, results_folder, chosen_text_attribute=chosen_text_attribute, file_name = 'tokenized_texts_dict_all.pkl.gz')

# article category (fields of study)
article_category = load_article_category(all_docs_dict, results_folder, file_name = 'article_category_all.pkl.gz')

# citations edgelist (hyperlinks)
sorted_paper_ids_with_texts = list(tokenized_texts_dict.keys())
citations_df = load_citations_edgelist(all_docs_dict, sorted_paper_ids_with_texts, results_folder, file_name = 'citations_edgelist_all.csv')


# FILTER NETWORK

ordered_papers_with_cits, new_filtered_words, tokenized_texts_dict, results_folder, filter_label, IDs, texts, edited_text = filter_dataset(
    all_docs_dict,
    tokenized_texts_dict,
    min_inCitations,
    min_word_occurences,
    results_folder,
    filter_label,
)


# CREATE GT OBJECT

hyperlink_g, hyperlinks = create_hyperlink_g(
    article_category,
    results_folder,
    filter_label
)


# ACHTUNG: h_t_doc_consensus is (for some reason) not ordered like edited_text, 
# but with the same order as hyperlink_g.vp['name']... 
ordered_paper_ids = list(hyperlink_g.vp['name'])
ordered_edited_texts = [edited_text[IDs.index(paper_id)] for paper_id in ordered_paper_ids]
    

# COMPUTE CENTRALITIES
centralities = load_centralities(
    all_docs_dict,
    citations_df,
    ordered_paper_ids,
    hyperlink_g, 
    results_folder,
    filter_label,
)

print('\nFinished all preparation.\n')
# EXIT HERE IF YOU WANT ONLY THE PREPARATION, WITHOUT ANY HSBM CALCULATION OR CONSENSUS PARTITIONS        
if do_only_prep:
    print('Exiting after preparation finished.')
    exit()

    
    
# ACHTUNG seed is what before whas the job array id
SEED_NUM = ID
print(f'seed is {SEED_NUM}',flush=True)
print('gt version:', gt.__version__)



if not do_analysis:
    # Execute multiple runs of fitting multilayer SBM using greedy moves.
    try:
        ################ ACHTUNG CHANGING FOLDER TO SUBFOLDER HERE!!!!!!!! ################
        results_folder = results_folder_iteration
        os.makedirs(results_folder, exist_ok=True)
        with gzip.open(f'{results_folder}results_fit_greedy{filter_label}.pkl.gz','rb') as fp:
            hyperlink_text_hsbm_states,time_duration = pickle.load(fp)
        print(f'Fit and calibration already done (in {time_duration}), loaded.',flush=True)
        
    except:
        print('Fit and calibration files not found, starting fit.',flush=True)
        start = datetime.now()
        
        hyperlink_text_hsbm_states =  fit_hyperlink_text_hsbm(edited_text, 
                                                                      IDs, 
                                                                      hyperlinks, 
                                                                      N_iter, 
                                                                      results_folder, 
                                                                      stop_at_fit = stop_at_fit, 
                                                                      filename_fit = f'results_fit_greedy{filter_label}_tmp.pkl.gz', 
                                                                      SEED_NUM=SEED_NUM, 
                                                                      number_iterations_MC_equilibrate = number_iterations_MC_equilibrate)
        if stop_at_fit == True:
            exit()
            
        end = datetime.now()
        time_duration = end - start
        
        with gzip.open(f'{results_folder}results_fit_greedy{filter_label}.pkl.gz','wb') as fp:
            pickle.dump((hyperlink_text_hsbm_states,end-start),fp)

        print('Time duration algorithm',time_duration,flush=True)

else:
    # ACHTUNG: if you change the list of states (or the states are modified), 
    #          remove all the files created in the following code in results_folder
    #          otherwise they will not do again the analysis and use old data, giving errors
    results_folder = results_folder # useless, but just to remember we save analysis results to results_folder, as they're _id independent
    try:
        start = datetime.now()
        print('Doing analysis, not fit',flush = True)
        print('Loading list of iterations', _id_list_string, flush=True)
        with gzip.open(f'{results_folder+analysis_results_subfolder}results_fit_greedy{filter_label}.pkl.gz','rb') as fp:
            hyperlink_text_hsbm_states,time_duration = pickle.load(fp)
        print('Average time duration algorithm',time_duration,flush=True)
        end = datetime.now()
    except:
        print('Loading states 1 by 1')
        start = datetime.now()
        time_duration_list = []
        results_folder_iteration = os.path.join(results_folder, f'ID_{_id_list[0]}_no_iterMC_{number_iterations_MC_equilibrate}/')
        
        with gzip.open(f'{results_folder_iteration}results_fit_greedy{filter_label}.pkl.gz','rb') as fp:
            hyperlink_text_hsbm_states,time_duration = pickle.load(fp)
        time_duration_list.append(time_duration)
#         print('Loaded %d'%_id_list[0],flush = True)
#         for _id in _id_list[1:]:
#             results_folder_iteration = os.path.join(results_folder, f'ID_{_id}_no_iterMC_{number_iterations_MC_equilibrate}/')
#             with gzip.open(f'{results_folder_iteration}results_fit_greedy{filter_label}.pkl.gz','rb') as fp:
#     #                 hyperlink_text_hsbm_states.append(pickle.load(fp)[0])
#                 hyperlink_text_hsbm_state,time_duration = pickle.load(fp)
#                 hyperlink_text_hsbm_states += hyperlink_text_hsbm_state
#             time_duration_list.append(time_duration)
#             print('Loaded %d'%_id,flush = True)
#         time_duration = np.mean(time_duration_list)
#         os.makedirs(results_folder+analysis_results_subfolder, exist_ok = True)
#         with gzip.open(f'{results_folder+analysis_results_subfolder}results_fit_greedy{filter_label}.pkl.gz','wb') as fp:
#             pickle.dump((hyperlink_text_hsbm_states,time_duration),fp)
        end = datetime.now()
#         print('Average time duration algorithm',time_duration,flush=True)
    print('Time loading states',end-start,flush=True)
        

# RETRIEVE PARTITIONS
# Retrieve the partitions assigned to the document nodes by examining the highest non-trivial level of the hierarchical degree-corrected SBM.
print('\nRetrieve doc partitions',flush=True)
print('\nHighest level',flush=True)


# Retrieve partitions assigned to documents in each run. Also save index of highest non-trivial level.
start = datetime.now()
if do_analysis == 0:
    dir_list = [results_folder]
else:
    dir_list = [os.path.join(results_folder, f'ID_{_id}_no_iterMC_{number_iterations_MC_equilibrate}/') for _id in _id_list]
hyperlink_text_hsbm_partitions, levels = get_highest_level_hsbm_partitions_from_iterations(hyperlink_g, dir_list, results_folder+analysis_results_subfolder)
end = datetime.now()
print('Time duration',end - start,flush=True)


# Retrieve partitions assigned to documents in each run for every level up to the highest level among all iterations
print('\nAll levels',flush=True)
hyperlink_text_hsbm_partitions_by_level, time_duration = get_hsbm_partitions_from_iterations(hyperlink_g,
                                        dir_list, 
                                        levels,
                                        results_folder+analysis_results_subfolder,
                                       )
print('Time duration',time_duration, flush=True)


print('\nRetrieve word partitions',flush=True)
# We now show how this framework tackles the problem of topic modelling simultaneously.
# Retrieve the topics associated to the consensus partition.
H_T_word_hsbm_partitions_by_level, H_T_word_hsbm_num_groups_by_level = get_hsbm_word_partitions_from_iterations(hyperlink_g,
                                        dir_list, 
                                        levels,
                                        results_folder + analysis_results_subfolder,
                                        IDs
                                       )

# CONSENSUS PARTITION
print('\nConsensus partition by level', flush=True)
start = datetime.now()

h_t_doc_consensus_by_level, h_t_word_consensus_by_level, h_t_consensus_summary_by_level = get_consensus(
    hyperlink_text_hsbm_states = hyperlink_text_hsbm_states,
    hyperlink_text_hsbm_partitions_by_level = hyperlink_text_hsbm_partitions_by_level,
    H_T_word_hsbm_partitions_by_level = H_T_word_hsbm_partitions_by_level,
    ordered_paper_ids = ordered_paper_ids,
    results_folder = results_folder+analysis_results_subfolder, 
    filter_label = filter_label,
    )

highest_non_trivial_level = max(list(h_t_doc_consensus_by_level.keys()))
end = datetime.now()
print('Time duration',end-start,flush=True)


print('\nGet all topics by level',flush=True)

g_words, dict_groups_by_level, topics_df_by_level = get_topics(
    hyperlink_text_hsbm_states,
    h_t_consensus_summary_by_level,
    h_t_doc_consensus_by_level,
)


# Topic frequency in clusters
print('\nMixture proportion',flush=True)
start = datetime.now()
mixture_proportion_by_level, normalized_mixture_proportion_by_level, avg_topic_frequency_by_level = get_mixture_proportion(
    h_t_doc_consensus_by_level, 
    dict_groups_by_level, 
    ordered_edited_texts,
    topics_df_by_level,
    results_folder = results_folder + analysis_results_subfolder,
    filter_label = ''
)
end = datetime.now()
print('Time duration',end-start,flush=True)


# RECOVER HIERARCHY
start = datetime.now()
hierarchy_docs,hierarchy_words = get_hierarchy(
    highest_non_trivial_level=highest_non_trivial_level,
    h_t_doc_consensus_by_level=h_t_doc_consensus_by_level,
    h_t_word_consensus_by_level=h_t_word_consensus_by_level,
    results_folder=results_folder+analysis_results_subfolder,
    filter_label = filter_label,
)

try:
    print('hierarchy words at non trivial level:', hierarchy_words[highest_non_trivial_level],flush=True)
except:
    print('There is only one level, so hierarchy_words is an empty dictionary.')
    
try:
    print('hierarchy docs at non trivial level:', hierarchy_docs[highest_non_trivial_level],flush=True)
except:
    print('There is only one level, so hierarchy_docs is an empty dictionary.')
end = datetime.now()
print('Time duration',end-start,flush=True)

    
    
    
# Normalized mixture proportion by different level partition and topic

print('\nNormalized mixture proportion between different levels')
start = datetime.now()
mixture_proportion_by_level_partition_by_level_topics, normalized_mixture_proportion_by_level_partition_by_level_topics, avg_topic_frequency_by_level_partition_by_level_topics = get_mixture_proportion_by_level(
    h_t_doc_consensus_by_level, 
    dict_groups_by_level, 
    ordered_edited_texts,
    topics_df_by_level,
    highest_non_trivial_level,
    results_folder = results_folder + analysis_results_subfolder,
    filter_label = ''
)
end = datetime.now()
print('Time duration',end-start,flush=True)

        
        
        

# KNOWLEDGE FLOW
print('\nStarting knowledge flow calculations')
start = datetime.now()

run_knowledge_flow_analysis(
    all_docs_dict,
    h_t_doc_consensus_by_level,
    hyperlink_g,
    ordered_paper_ids,
    first_level_knowledge_flow,
    highest_non_trivial_level,
    dataset_path,
    results_folder,
    last_year = 2021,
    first_year = 1962,
)

end = datetime.now()
print('Time duration',end-start,flush=True)



print('FINISHED')