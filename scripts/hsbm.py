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
sys.path.insert(0, os.path.join(os.getcwd(),"utils"))
from hsbm import sbmmultilayer 
from hsbm.utils.nmi import *
from hsbm.utils.doc_clustering import *
from hsbm_creation import *
from hsbm_fit import *
from hsbm_partitions import *

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
    default="")

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
    help='Do analysis gathering all iterations, instead of just doing the fit. [default 0]',
    default=0)

parser.add_argument('-id_list', '--id_list_for_analysis', type=str,
    help='list of ids for which we do the analysis, list written as string (with brackets, see default), and then converted to list in the script. Needs to contain at least two elements. [default "[1,2]"]',
    default='[1,2]')

parser.add_argument('-stop', '--stop_at_fit', type=int,
    help='If stop_at_fit is 1, it does only the fit and saves it in a temporary file, otherwise it does also the equilibrate. [default 0]',
    default=0)

parser.add_argument('-prep', '--do_only_prep', type=int,
    help='If do_only_prep is 1, it does only the preparation of all files needed to do the hsbm and the following analysis. [default 0]',
    default=0)



arguments = parser.parse_args()

dataset_path = arguments.dataset_path
analysis_results_subfolder = arguments.analysis_results_subfolder
if len(analysis_results_subfolder) > 0 and analysis_results_subfolder[-1] != "/":
        analysis_results_subfolder += "/"
ID = arguments.ID
number_iterations_MC_equilibrate = arguments.number_iterations_MC_equilibrate
min_word_occurences = arguments.min_word_occurences
use_titles = arguments.use_titles == 1
do_analysis = arguments.do_analysis
_id_list_string = arguments.id_list_for_analysis
_id_list = ast.literal_eval(_id_list_string)
stop_at_fit = arguments.stop_at_fit == 1
do_only_prep = arguments.do_only_prep == 1



# FILTERING OF THE DATASET
# 1. keep only articles in the citations layer (citing or cited)
# 2. filter out words appearing in at least min_word_occurences article






# Parameters needed in the process.
# ACHTUNG: Be aware to change this!

# Minimum number of citations filter
min_inCitations = 0
# Number of iterations of the hSBM algorithm. Now it's not parallelized, so be aware of computational time
N_iter = 1
# If you do plots or other graph_tool stuff, this needs to be properly set with the number of available cores
n_cores = 1
gt.openmp_set_num_threads(n_cores)

# DELETE!!!!
edge_divider = 1


# FILES DIRECTORY


# filter_label = f'_{min_inCitations}_min_inCitations' # to use at the end of the names to be saved
filter_label = '' # if you want a specific label to do some testing


print('Loading data')


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

## Pruning of papers
print('Pruning of papers',flush=True)
# eliminating duplicated papers

with gzip.open(f'{dataset_path}papers_dict.pkl.gz', 'rb') as fp:
    all_docs_dict = pickle.load(fp)
        
all_docs_dict = {x:all_docs_dict[x] for x in all_docs_dict if chosen_text_attribute in all_docs_dict[x] and all_docs_dict[x][chosen_text_attribute] is not None and len(all_docs_dict[x][chosen_text_attribute])>0}


print('total number of docs: %d'%(len(all_docs_dict)))




## tokenized texts

tokenized_texts_dict = load_tokenized_texts_dict(all_docs_dict, results_folder, chosen_text_attribute=chosen_text_attribute, file_name = "tokenized_texts_dict_all.pkl.gz")

# article category (fields of study)

article_category = load_article_category(all_docs_dict, results_folder, file_name = "article_category_all.pkl.gz")

## citations edgelist (hyperlinks)
sorted_paper_ids_with_texts = list(tokenized_texts_dict.keys())
citations_df = load_citations_edgelist(all_docs_dict, sorted_paper_ids_with_texts, results_folder, file_name = 'citations_edgelist_all.csv')



# graph-tool analysis
# The following ordered lists are required
# - texts: list of tokenized texts
# - IDs: list of unique paperIds

IDs = []
texts = []
for paper_id in sorted_paper_ids_with_texts:
    IDs.append(paper_id)
    texts.append(tokenized_texts_dict[paper_id])




## Filter the network
citations_df, ordered_papers_with_cits = load_filtered_papers_with_cits(all_docs_dict, tokenized_texts_dict, results_folder, min_inCitations=min_inCitations)
print(f"number of document-document links: {len(citations_df)}",flush=True)
# words appearing in at least two of these articles

to_count_words_in_doc_df = pd.DataFrame(data = {'paperId': ordered_papers_with_cits, 'word':[tokenized_texts_dict[x] for x in ordered_papers_with_cits]})

to_count_words_in_doc_df = to_count_words_in_doc_df.explode('word')
x = to_count_words_in_doc_df.groupby('word').paperId.nunique()

new_filtered_words = set(x[x>=min_word_occurences].index.values)
print('Number of new filtered words', len(new_filtered_words),flush=True)


    
    
    
# graph-tool analysis
# The following ordered lists are required
# - texts: list of tokenized texts
# - IDs: list of unique paperIds

IDs = []
texts = []
for paper_id in ordered_papers_with_cits:
    IDs.append(paper_id)
    texts.append(tokenized_texts_dict[paper_id])



# Remove stop words in text data
try:
    with gzip.open(f'{results_folder}IDs_texts_and_edited_text_papers_with_abstract{filter_label}.pkl.gz', 'rb') as fp:
        IDs,texts,edited_text = pickle.load(fp)
except:
    edited_text = []
    from nltk.corpus import stopwords
    import nltk
    nltk.download('stopwords')

    stop_words = stopwords.words('english') + stopwords.words('italian') + stopwords.words('german') + stopwords.words('french') + stopwords.words('spanish')
    
    # Recall texts is list of lists of words in each document.
    for doc in texts:
        temp_doc = []
        for word in doc:
            if word not in stop_words and word in new_filtered_words:
                temp_doc.append(word)
        edited_text.append(temp_doc)

    with gzip.open(f'{results_folder}edited_text_papers_with_abstract{filter_label}.pkl.gz', 'wb') as fp:
            pickle.dump(edited_text,fp)

    with gzip.open(f'{results_folder}IDs_texts_and_edited_text_papers_with_abstract{filter_label}.pkl.gz', 'wb') as fp:
            pickle.dump((IDs,texts,edited_text),fp)
    print('Dumped edited texts')

print(f"number of word-document links: {np.sum([len(set(x)) for x in edited_text])}",flush=True)

## Create gt object
if os.path.exists(f'{results_folder}gt_network{filter_label}.gt'):
    hyperlink_g = gt.load_graph(f'{results_folder}gt_network{filter_label}.gt')
    num_vertices = hyperlink_g.num_vertices()
    num_edges = hyperlink_g.num_edges()
    label = hyperlink_g.vp['label']
    name = hyperlink_g.vp['name']
    for v in hyperlink_g.vertices():
        category_of_article = article_category[name[v]]
    # Retrieve true partition of graph
    true_partition = list(hyperlink_g.vp.label)    
    # Retrieve ordering of articles
    article_names = list(hyperlink_g.vp.name)
    filename = f'{results_folder}citations_edgelist{filter_label}.csv'
    x = pd.read_csv(filename)
    hyperlinks = [(row[0],row[1]) for source, row in x.iterrows()]  
    unique_hyperlinks = hyperlinks.copy()
else:
    print('Creating gt object...')
    hyperlink_edgelist = f'{results_folder}citations_edgelist{filter_label}.csv'
    hyperlink_g = gt.load_graph_from_csv(hyperlink_edgelist,
                              skip_first=True,
                              directed=True,
                              csv_options={'delimiter': ','},)
    num_vertices = hyperlink_g.num_vertices()
    num_edges = hyperlink_g.num_edges()

    # Create hyperlinks list
    filename = f'{results_folder}citations_edgelist{filter_label}.csv'
    x = pd.read_csv(filename)
    hyperlinks = [(row[0],row[1]) for source, row in x.iterrows()]  


    label = hyperlink_g.vp['label'] = hyperlink_g.new_vp('string')
    name = hyperlink_g.vp['name'] # every vertex has a name already associated to it!


    # We now assign category article to each Wikipedia article
    for v in hyperlink_g.vertices():
        category_of_article = article_category[name[v]]
        label[v] = category_of_article # assign wikipedia category to article

    # Retrieve true partition of graph
    true_partition = list(hyperlink_g.vp.label)    
    # Retrieve ordering of articles
    article_names = list(hyperlink_g.vp.name)

    # Remove parallel edges in hyperlink graph # Deprecated
    # gt.remove_parallel_edges(hyperlink_g)
    # unique_hyperlinks = []
    # for e in hyperlinks:
    #     if e not in unique_hyperlinks:
    #         unique_hyperlinks.append(e)

    unique_hyperlinks = hyperlinks.copy()

    hyperlink_g.save(f'{results_folder}gt_network{filter_label}.gt')


    
    
# ACHTUNG: h_t_doc_consensus is (for some reason) not ordered like edited_text, 
# but with the same order as hyperlink_g.vp['name']... 
ordered_paper_ids = list(hyperlink_g.vp['name'])
ordered_edited_texts = [tokenized_texts_dict[paper_id] for paper_id in ordered_paper_ids]
    
    
# COMPUTE CENTRALITIES
try:
    with gzip.open(f'{results_folder}hyperlink_g_centralities{filter_label}.pkl.gz','rb') as fp:
        centralities = pickle.load(fp)
except:
    print("Calculating centralities...")
    id2NoCits = {x: len(all_docs_dict[x]['inCitations']) for x in all_docs_dict.keys()}

    centralities = {}
    centralities['citations_overall'] = np.vectorize(id2NoCits.get)(ordered_paper_ids)
    print("Done citations_overall centrality", flush=True)
    paper_with_cits = citations_df['from'].value_counts().index.values
    paper_without_cits = set(ordered_paper_ids).difference(set(paper_with_cits))
    centralities['in_degree'] = citations_df['from'].value_counts().append(pd.Series(data = np.zeros(len(paper_without_cits)),index=paper_without_cits)).loc[ordered_paper_ids].values
    print("Done in_degree centrality", flush=True)
    if len(ordered_paper_ids) > 1000:
        centralities['eigenvector'] = graph_tool.centrality.eigenvector(hyperlink_g)[1]._get_data()
        print("Done eigenvector centrality", flush=True)
    centralities['betweenness'] = graph_tool.centrality.betweenness(hyperlink_g)[0]._get_data()
    print("Done betweenness centrality", flush=True)
    centralities['closeness'] = graph_tool.centrality.closeness(hyperlink_g)._get_data()
    print("Done closeness centrality", flush=True)
    centralities['pagerank'] = graph_tool.centrality.pagerank(hyperlink_g)._get_data()
    print("Done pagerank centrality", flush=True)
    centralities['katz'] = graph_tool.centrality.katz(hyperlink_g)._get_data()
    print("Done katz centrality", flush=True)

    with gzip.open(f'{results_folder}hyperlink_g_centralities{filter_label}.pkl.gz','wb') as fp:
        pickle.dump(centralities,fp)

    
    
    
if do_only_prep:
    exit()

## algorithm run
## ACHTUNG seed is what before whas the job array id
SEED_NUM = ID
print(f'seed is {SEED_NUM}',flush=True)
print(gt.__version__)




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
                                                                      filename_fit = f'results_fit_greedy{filter_label}_tmp.gz', 
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
        print('Time loading states',end-start,flush=True)
    except:
        start = datetime.now()
        time_duration_list = []
        results_folder_iteration = os.path.join(results_folder, f'ID_{_id_list[0]}_no_iterMC_{number_iterations_MC_equilibrate}/')
        
        with gzip.open(f'{results_folder_iteration}results_fit_greedy{filter_label}.pkl.gz','rb') as fp:
                    hyperlink_text_hsbm_states,time_duration = pickle.load(fp)
        time_duration_list.append(time_duration)
        print('Loaded %d'%_id_list[0],flush = True)

        for _id in _id_list[1:]:
            results_folder_iteration = os.path.join(results_folder, f'ID_{_id}_no_iterMC_{number_iterations_MC_equilibrate}/')
            with gzip.open(f'{results_folder_iteration}results_fit_greedy{filter_label}.pkl.gz','rb') as fp:
    #                 hyperlink_text_hsbm_states.append(pickle.load(fp)[0])
                hyperlink_text_hsbm_state,time_duration = pickle.load(fp)
                hyperlink_text_hsbm_states += hyperlink_text_hsbm_state
            time_duration_list.append(time_duration)
            print('Loaded %d'%_id,flush = True)
        time_duration = np.mean(time_duration_list)
        with gzip.open(f'{results_folder+analysis_results_subfolder}results_fit_greedy{filter_label}.pkl.gz','wb') as fp:
            pickle.dump((hyperlink_text_hsbm_states,time_duration),fp)
        end = datetime.now()
        print('Average time duration algorithm',time_duration,flush=True)
        print('Time loading states',end-start,flush=True)
        

# ## Retrieve partitions
# 
# Retrieve the partitions assigned to the document nodes by examining the highest non-trivial level of the hierarchical degree-corrected SBM.
print('\nRetrieve partitions',flush=True)



# Retrieve partitions assigned to documents in each run. Also save index of highest non-trivial level.
try:
    with gzip.open(f'{results_folder+analysis_results_subfolder}results_fit_greedy_partitions{filter_label}.pkl.gz','rb') as fp:
        hyperlink_text_hsbm_partitions, levels = pickle.load(fp)
    print('Loaded', flush=True)
except:
    start = datetime.now()
    hyperlink_text_hsbm_partitions, levels = get_hsbm_partitions(hyperlink_g, hyperlink_text_hsbm_states)
    end = datetime.now()
    print('Time duration retrieving partitions',end - start,flush=True)

#     print('number of partitions',len(set(hyperlink_text_hsbm_partitions[0])),flush=True)

    with gzip.open(f'{results_folder+analysis_results_subfolder}results_fit_greedy_partitions{filter_label}.pkl.gz','wb') as fp:
        pickle.dump((hyperlink_text_hsbm_partitions, levels),fp)


# ## Consensus Partition
# 
# Compute the consensus partition assignment to document nodes over all the iterations.

print('\nConsensus Partition',flush=True)
try:
    with gzip.open(f'{results_folder+analysis_results_subfolder}results_fit_greedy_partitions_docs_all{filter_label}.pkl.gz','rb') as fp:
        hyperlink_text_hsbm_partitions_by_level,duration = pickle.load(fp)
    print('Loaded', flush=True)
except:
    start = datetime.now()
    print(start)

    clustering_info = doc_clustering('./', hyperlink_g)
#     no_levels = hyperlink_text_hsbm_states[0].n_levels

    clustering_info.seeds = [0]

    hyperlink_text_hsbm_partitions_by_level = {}
    hyperlink_text_hsbm_partitions_by_level_info = {}

    for l in range(max(levels)+1):
        hyperlink_text_hsbm_partitions_by_level[l] = []
        hyperlink_text_hsbm_partitions_by_level_info[l] = []

    for iteration,curr_hsbm in enumerate(hyperlink_text_hsbm_states):
        print('Iteration %d'%iteration, flush=True)
#         for l in range(levels[iteration]+1):
        for l in range(max(levels)+1):
            print('\tlevel',l)
            tmp = clustering_info.collect_info2('1-layer-doc', curr_hsbm.g, [l], curr_hsbm.state)
            hyperlink_text_hsbm_partitions_by_level_info[l].append(tmp)
            hyperlink_text_hsbm_partitions_by_level[l].append(tmp[0][0])

    end = datetime.now()
    print(end - start)

    with gzip.open(f'{results_folder+analysis_results_subfolder}results_fit_greedy_partitions_docs_all{filter_label}.pkl.gz','wb') as fp:
        pickle.dump((hyperlink_text_hsbm_partitions_by_level,end-start),fp)

    with gzip.open(f'{results_folder+analysis_results_subfolder}results_fit_greedy_partitions_docs_all_info{filter_label}.pkl.gz','wb') as fp:
        pickle.dump((hyperlink_text_hsbm_partitions_by_level_info,end-start),fp)

# # THIS IS USELESS HERE
# start = datetime.now()
# print(start)
# hyperlink_text_consensus_partitions_by_level, hyperlink_text_consensus_partitions_sd_by_level = {}, {}
# for l in hyperlink_text_hsbm_partitions_by_level.keys():
#     print('Consensus level',l,flush=True)
#     tmp = [x[:] for x in hyperlink_text_hsbm_partitions_by_level[l]] # [:3] when using collect_info2 else [:]
#     hyperlink_text_consensus_partitions_by_level[l], hyperlink_text_consensus_partitions_sd_by_level[l] = gt.partition_overlap_center(tmp)
# end = datetime.now()
# print(end - start)

# # Topic Modelling
print('\nTopic Modelling',flush=True)
# 
# We now show how this framework tackles the problem of topic modelling simultaneously.

# We now retrieve the topics associated to the consensus partition.

def get_word_type_blocks(h_t_state, h_t_graph, level):
    '''
    Retrieve the block assignment of WORD types for the model.
    '''
    partitions = []
    num_of_groups = []
    entropies = []
    block_SBM_partitions = {} # store dictionary to map nodes to partition.
    b = h_t_state.project_level(level).get_blocks()
    # Need to specify to retrieve partitions for WORD type nodes.
    for node in h_t_graph.vertices():
        if h_t_graph.vp['kind'][node] == 1:
            block_SBM_partitions[h_t_graph.vp.name[node]] = b[node]                    

    # Retrieve the partition from the SBM and store as parameter.    
    partition = h_t_graph.vp['partition'] = h_t_graph.new_vp('int')
    # Assign partition label to node properties.
    for v in h_t_graph.vertices():
        if h_t_graph.vp['kind'][v] == 1:
            partition[v] = block_SBM_partitions[h_t_graph.vp.name[v]]
    # IGNORE FIRST 120 NODES (there are document nodes)
    partitions.append(list(h_t_graph.vp.partition)[len(IDs):])
    num_of_groups.append(len(set(partitions[0])))
    entropies.append(h_t_state.entropy())
    return (partitions, num_of_groups, entropies)

# THIS IS ONLY FOR MAXIMUM LEVEL, USELESS
# # for each iteration get word groups for the maximum non trivial level
# H_T_word_hsbm_partitions = []
# H_T_word_hsbm_num_groups = []
# start = datetime.now()
# for i in range(len(levels)):
#     print('\tIteration',i,flush=True)
#     word_partitions, num_word_groups, en = get_word_type_blocks(hyperlink_text_hsbm_states[i].state, hyperlink_text_hsbm_states[i].g, levels[i])
#     H_T_word_hsbm_partitions.append(word_partitions[0])
#     H_T_word_hsbm_num_groups.append(num_word_groups)
# end = datetime.now()
# print(end-start,flush=True)

# for each iteration get word groups for all non-trivial level
try:
    with gzip.open(f'{results_folder+analysis_results_subfolder}results_fit_greedy_partitions_words_all{filter_label}.pkl.gz','rb') as fp:
        H_T_word_hsbm_partitions_by_level, H_T_word_hsbm_num_groups_by_level = pickle.load(fp)
    print('Loaded', flush=True)
except:
    start = datetime.now()
    H_T_word_hsbm_partitions_by_level, H_T_word_hsbm_num_groups_by_level = {},{}

    for l in hyperlink_text_hsbm_partitions_by_level.keys():
        print('level',l)
        H_T_word_hsbm_partitions_by_level[l] = []
        H_T_word_hsbm_num_groups_by_level[l] = []
        for i in range(len(levels)):
            print('\titeration',i,flush=True)
            word_partitions, num_word_groups, en = get_word_type_blocks(hyperlink_text_hsbm_states[i].state, hyperlink_text_hsbm_states[i].g, l)
            H_T_word_hsbm_partitions_by_level[l].append(word_partitions[0])
            H_T_word_hsbm_num_groups_by_level[l].append(num_word_groups)

    with gzip.open(f'{results_folder+analysis_results_subfolder}results_fit_greedy_partitions_words_all{filter_label}.pkl.gz','wb') as fp:
            pickle.dump((H_T_word_hsbm_partitions_by_level, H_T_word_hsbm_num_groups_by_level),fp)

    end = datetime.now()
    print('Time duration',end-start,flush=True)





# We now retrieve the consensus partitions for the document and word type nodes respectively. We now 'count' the number of edges between document clusters and word type groups (i.e. topics) in order to compute the distributions required.
# start = datetime.now()
# THIS IS ONLY FOR MAXIMUM LEVEL, USELESS
# h_t_doc_consensus = gt.partition_overlap_center(hyperlink_text_hsbm_partitions)[0]


# THIS IS ONLY FOR MAXIMUM LEVEL, USELESS
# D = len((h_t_doc_consensus)) # number of document nodes

# h_t_word_consensus = gt.partition_overlap_center(H_T_word_hsbm_partitions)[0]
# h_t_word_consensus += len(set(h_t_doc_consensus)) # to get cluster number to not start from 0

# V = len((h_t_word_consensus)) # number of word-type nodes
# # number of word-tokens (edges excluding hyperlinks)
# N = int(np.sum([hyperlink_text_hsbm_states[0].g.ep.edgeCount[e] for e in hyperlink_text_hsbm_states[0].g.edges() if hyperlink_text_hsbm_states[0].g.ep['edgeType'][e]== 0 ])) 

# # Number of blocks
# B = len(set(h_t_word_consensus)) + len(set(h_t_doc_consensus))

# # Count labeled half-edges, total sum is # of edges
# # Number of half-edges incident on word-node w and labeled as word-group tw
# n_wb = np.zeros((V,B)) # will be reduced to (V, B_w)

# # Number of half-edges incident on document-node d and labeled as document-group td
# n_db = np.zeros((D,B)) # will be reduced to (D, B_d)

# # Number of half-edges incident on document-node d and labeled as word-group tw
# n_dbw = np.zeros((D,B))  # will be reduced to (D, B_w)



# # All graphs created the same for each iteration
# for e in hyperlink_text_hsbm_states[0].g.edges():
#     # Each edge will be between a document node and word-type node
#     if hyperlink_text_hsbm_states[0].g.ep.edgeType[e] == 0:        
#         # v1 ranges from 0, 1, 2, ..., D - 1
#         # v2 ranges from D, ..., (D + V) - 1 (V # of word types)
#         v1 = int(e.source()) # document node index
#         v2 = int(e.target()) # word type node index
#         # z1 will have values from 1, 2, ..., B_d; document-group i.e document block that doc node is in 
#         # z2 will have values from B_d + 1, B_d + 2,  ..., B_d + B_w; word-group i.e word block that word type node is in
#         # Recall that h_t_word_consensus starts at 0 so need to -120
#         z1, z2 = h_t_doc_consensus[v1], h_t_word_consensus[v2-D]
#         n_wb[v2-D,z2] += 1 # word type v2 is in topic z2
#         n_db[v1,z1] += 1 # document v1 is in doc cluster z1
#         n_dbw[v1,z2] += 1 # document v1 has a word in topic z2


# n_db = n_db[:, np.any(n_db, axis=0)] # (D, B_d)
# n_wb = n_wb[:, np.any(n_wb, axis=0)] # (V, B_w)
# n_dbw = n_dbw[:, np.any(n_dbw, axis=0)] # (D, B_d)

# B_d = n_db.shape[1]  # number of document groups
# B_w = n_wb.shape[1] # number of word groups (topics)



# # Group membership of each word-type node in topic, matrix of ones or zeros, shape B_w x V
# # This tells us the probability of topic over word type
# p_tw_w = (n_wb / np.sum(n_wb, axis=1)[:, np.newaxis]).T

# # Group membership of each doc-node, matrix of ones of zeros, shape B_d x D
# p_td_d = (n_db / np.sum(n_db, axis=1)[:, np.newaxis]).T

# # Mixture of word-groups into documents P(t_w | d), shape B_d x D
# p_tw_d = (n_dbw / np.sum(n_dbw, axis=1)[:, np.newaxis]).T

# # Per-topic word distribution, shape V x B_w
# p_w_tw = n_wb / np.sum(n_wb, axis=0)[np.newaxis, :]


# h_t_consensus_summary = {}
# h_t_consensus_summary['Bd'] = B_d # Number of document groups
# h_t_consensus_summary['Bw'] = B_w # Number of word groups
# h_t_consensus_summary['p_tw_w'] = p_tw_w # Group membership of word nodes
# h_t_consensus_summary['p_td_d'] = p_td_d # Group membership of document nodes
# h_t_consensus_summary['p_tw_d'] = p_tw_d # Topic proportions over documents
# h_t_consensus_summary['p_w_tw'] = p_w_tw # Topic distribution over words
# end = datetime.now()
# print('Time duration',end-start,flush=True)


# and now do it for all levels
print('\nConsensus summary by level', flush=True)
start = datetime.now()

print('Calculating nested consensus partition among docs', flush=True)
h_t_doc_consensus_by_level, hyperlink_docs_hsbm_partitions_by_level = get_consensus_nested_partition(hyperlink_text_hsbm_partitions_by_level)

print('Calculating nested consensus partition among words', flush=True)
h_t_word_consensus_by_level, hyperlink_words_hsbm_partitions_by_level = get_consensus_nested_partition(H_T_word_hsbm_partitions_by_level)

    
for l in list(hyperlink_text_hsbm_partitions_by_level.keys()):
    if len(set(h_t_word_consensus_by_level[l])) == 1 and len(set(h_t_doc_consensus_by_level[l])) == 1:
        print('Removing level %d because it has only 1 cluster of docs and 1 of words'%l, flush=True)
        del h_t_word_consensus_by_level[l]
        del h_t_doc_consensus_by_level[l]
        del hyperlink_words_hsbm_partitions_by_level[l]
        del hyperlink_text_hsbm_partitions_by_level[l]
        
        
print('Calculating summary', flush=True)

highest_non_trivial_level = max(list(h_t_doc_consensus_by_level.keys()))

# h_t_word_consensus_by_level = {}
h_t_consensus_summary_by_level = {}

for l in h_t_doc_consensus_by_level.keys():
    print('level %d'%l, flush=True)
    D = len((h_t_doc_consensus_by_level[l])) # number of document nodes
    print('\tdocs blocks are %d, end with index %d'%(len(set(h_t_doc_consensus_by_level[l])),max(h_t_doc_consensus_by_level[l])))
    if len(set(h_t_doc_consensus_by_level[l])) != max(h_t_doc_consensus_by_level[l]) + 1:
        print('Reordering list of doc blocks', flush=True)
        old_blocks = sorted(set(h_t_doc_consensus_by_level[l]))
        new_blocks = list(range(len(set(h_t_doc_consensus_by_level[l]))))
        remap_blocks_dict = {old_blocks[i]:new_blocks[i] for i in range(len(set(h_t_doc_consensus_by_level[l])))}
        for i in range(D):
            h_t_doc_consensus_by_level[l][i] = remap_blocks_dict[h_t_doc_consensus_by_level[l][i]]
            
#     h_t_word_consensus_by_level[l] = gt.partition_overlap_center(H_T_word_hsbm_partitions_by_level[l])[0]
    print('\twords blocks are %d, end with index %d'%(len(set(h_t_word_consensus_by_level[l])),max(h_t_word_consensus_by_level[l])))
    V = len((h_t_word_consensus_by_level[l])) # number of word-type nodes
    if len(set(h_t_word_consensus_by_level[l])) != max(h_t_word_consensus_by_level[l]) + 1:
        print('Reordering list of word blocks', flush=True)
        old_blocks = sorted(set(h_t_word_consensus_by_level[l]))
        new_blocks = list(range(len(set(h_t_word_consensus_by_level[l]))))
        remap_blocks_dict = {old_blocks[i]:new_blocks[i] for i in range(len(set(h_t_word_consensus_by_level[l])))}
        for i in range(V):
            h_t_word_consensus_by_level[l][i] = remap_blocks_dict[h_t_word_consensus_by_level[l][i]]
    
    h_t_word_consensus_by_level[l] += len(set(h_t_doc_consensus_by_level[l])) # to get cluster number to not start from 0
#     h_t_word_consensus_by_level[l] += max(h_t_doc_consensus_by_level[l])+1 # to get cluster number to not start from 0
    
    # number of word-tokens (edges excluding hyperlinks)
    N = int(np.sum([hyperlink_text_hsbm_states[0].g.ep.edgeCount[e] for e in hyperlink_text_hsbm_states[0].g.edges() if hyperlink_text_hsbm_states[0].g.ep['edgeType'][e]== 0 ])) 

    # Number of blocks
    B = len(set(h_t_word_consensus_by_level[l])) + len(set(h_t_doc_consensus_by_level[l]))
#     B = max(h_t_word_consensus_by_level[l])+1

    # Count labeled half-edges, total sum is # of edges
    # Number of half-edges incident on word-node w and labeled as word-group tw
    n_wb = np.zeros((V,B)) # will be reduced to (V, B_w)

    # Number of half-edges incident on document-node d and labeled as document-group td
    n_db = np.zeros((D,B)) # will be reduced to (D, B_d)

    # Number of half-edges incident on document-node d and labeled as word-group tw
    n_dbw = np.zeros((D,B))  # will be reduced to (D, B_w)


    # All graphs created the same for each H+T model
    for e in hyperlink_text_hsbm_states[0].g.edges():
        # Each edge will be between a document node and word-type node
        if hyperlink_text_hsbm_states[0].g.ep.edgeType[e] == 0:        
            # v1 ranges from 0, 1, 2, ..., D - 1
            # v2 ranges from D, ..., (D + V) - 1 (V # of word types)
            
            v1 = int(e.source()) # document node index
            v2 = int(e.target()) # word type node index # THIS MAKES AN IndexError!
#             v1, v2 = e
#             v1 = int(v1)
#             v2 = int(v2)
            
            # z1 will have values from 1, 2, ..., B_d; document-group i.e document block that doc node is in 
            # z2 will have values from B_d + 1, B_d + 2,  ..., B_d + B_w; word-group i.e word block that word type node is in
            # Recall that h_t_word_consensus starts at 0 so need to -120
            z1, z2 = h_t_doc_consensus_by_level[l][v1], h_t_word_consensus_by_level[l][v2-D]
            n_wb[v2-D,z2] += 1 # word type v2 is in topic z2
            n_db[v1,z1] += 1 # document v1 is in doc cluster z1
            n_dbw[v1,z2] += 1 # document v1 has a word in topic z2

    n_db = n_db[:, np.any(n_db, axis=0)] # (D, B_d)
    n_wb = n_wb[:, np.any(n_wb, axis=0)] # (V, B_w)
    n_dbw = n_dbw[:, np.any(n_dbw, axis=0)] # (D, B_d)

    B_d = n_db.shape[1]  # number of document groups
    B_w = n_wb.shape[1] # number of word groups (topics)

    # Group membership of each word-type node in topic, matrix of ones or zeros, shape B_w x V
    # This tells us the probability of topic over word type
    p_tw_w = (n_wb / np.sum(n_wb, axis=1)[:, np.newaxis]).T

    # Group membership of each doc-node, matrix of ones of zeros, shape B_d x D
    p_td_d = (n_db / np.sum(n_db, axis=1)[:, np.newaxis]).T

    # Mixture of word-groups into documents P(t_w | d), shape B_d x D
    p_tw_d = (n_dbw / np.sum(n_dbw, axis=1)[:, np.newaxis]).T

    # Per-topic word distribution, shape V x B_w
    p_w_tw = n_wb / np.sum(n_wb, axis=0)[np.newaxis, :]


    h_t_consensus_summary_by_level[l] = {}
    h_t_consensus_summary_by_level[l]['Bd'] = B_d # Number of document groups
    h_t_consensus_summary_by_level[l]['Bw'] = B_w # Number of word groups
    h_t_consensus_summary_by_level[l]['p_tw_w'] = p_tw_w # Group membership of word nodes
    h_t_consensus_summary_by_level[l]['p_td_d'] = p_td_d # Group membership of document nodes
    h_t_consensus_summary_by_level[l]['p_tw_d'] = p_tw_d # Topic proportions over documents
    h_t_consensus_summary_by_level[l]['p_w_tw'] = p_w_tw # Topic distribution over words

with gzip.open(f'{results_folder+analysis_results_subfolder}results_fit_greedy_partitions_consensus_all{filter_label}.pkl.gz','wb') as fp:
    pickle.dump((h_t_doc_consensus_by_level, h_t_word_consensus_by_level, h_t_consensus_summary_by_level),fp)
    
end = datetime.now()
print('Time duration',end-start,flush=True)




g_words = [ hyperlink_text_hsbm_states[0].g.vp['name'][v] for v in  hyperlink_text_hsbm_states[0].g.vertices() if hyperlink_text_hsbm_states[0].g.vp['kind'][v]==1   ]
dict_groups_by_level = {l:get_topics_h_t_consensus_model(h_t_consensus_summary_by_level[l], g_words, n=1000000) for l in h_t_doc_consensus_by_level.keys()}



# THIS IS ONLY FOR MAXIMUM LEVEL, USELESS
# # Write out topics as dataframe
# topic_csv_dict = {}
# for key in dict_groups.keys():
#     topic_csv_dict[key] = [entry[0] for entry in dict_groups[key]]

# keys = topic_csv_dict.keys()
# topics_df = pd.DataFrame()

# for key in dict_groups.keys():
#     temp_df = pd.DataFrame(topic_csv_dict[key], columns=[key])
#     topics_df = pd.concat([topics_df, temp_df], ignore_index=True, axis=1)


topics_df_by_level = {}

for l in h_t_doc_consensus_by_level.keys():
    # Write out topics as dataframe
    topic_csv_dict = {}
    for key in dict_groups_by_level[l].keys():
        topic_csv_dict[key] = [entry[0] for entry in dict_groups_by_level[l][key]]

    keys = topic_csv_dict.keys()
    topics_df_by_level[l] = pd.DataFrame()

    for key in dict_groups_by_level[l].keys():
        temp_df = pd.DataFrame(topic_csv_dict[key], columns=[key])
        topics_df_by_level[l] = pd.concat([topics_df_by_level[l], temp_df], ignore_index=True, axis=1)


# We can now retrieve the top 10 words associated to topics associated to consensus partition.topics_df_by_level
# print(topics_df_by_level[2].iloc[:5],flush=True)#[range(0,15)]

# ## Topic frequency in clusters
print('Mixture proportion...',flush=True)


# THIS IS ONLY FOR MAXIMUM LEVEL, USELESS
# # mixture_proportion, normalized_mixture_proportion, avg_topic_frequency = topic_mixture_proportion(dict_groups,edited_text,h_t_doc_consensus)
# mixture_proportion, normalized_mixture_proportion, avg_topic_frequency = topic_mixture_proportion(dict_groups,ordered_edited_texts,h_t_doc_consensus)


mixture_proportion_by_level, normalized_mixture_proportion_by_level, avg_topic_frequency_by_level = {}, {}, {}
for l in h_t_doc_consensus_by_level.keys():
#     mixture_proportion_by_level[l], normalized_mixture_proportion_by_level[l], avg_topic_frequency_by_level[l] = topic_mixture_proportion(dict_groups_by_level[l],edited_text,h_t_doc_consensus_by_level[l])
    mixture_proportion_by_level[l], normalized_mixture_proportion_by_level[l], avg_topic_frequency_by_level[l] = topic_mixture_proportion(dict_groups_by_level[l],ordered_edited_texts,h_t_doc_consensus_by_level[l])


with gzip.open(f'{results_folder+analysis_results_subfolder}results_fit_greedy_topic_frequency_all{filter_label}.pkl.gz','wb') as fp:
        pickle.dump((topics_df_by_level,mixture_proportion_by_level, normalized_mixture_proportion_by_level, avg_topic_frequency_by_level),fp)


# Recover Hierarchy

hierarchy_docs, hierarchy_words = {}, {}


for l in range(highest_non_trivial_level,0,-1):

    tmp1_docs, tmp2_docs = h_t_doc_consensus_by_level[l], h_t_doc_consensus_by_level[l-1]
    tmp1_words, tmp2_words = h_t_word_consensus_by_level[l], h_t_word_consensus_by_level[l-1]

    hierarchy_docs[l] = {d: set() for d in np.unique(tmp1_docs)}
    hierarchy_words[l] = {w: set() for w in np.unique(tmp1_words)}

    for i in range(len(tmp1_docs)):
        hierarchy_docs[l][tmp1_docs[i]].add(tmp2_docs[i])

    for i in range(len(tmp1_words)):
        hierarchy_words[l][tmp1_words[i]].add(tmp2_words[i])

# Add higher layer of hierarchy words so that we have a unique root
hierarchy_words[highest_non_trivial_level+1] = {0:set(list(hierarchy_words[highest_non_trivial_level].keys()))}

# Add higher layer of hierarchy docs so that we have a unique root
hierarchy_docs[highest_non_trivial_level+1] = {0:set(list(hierarchy_docs[highest_non_trivial_level].keys()))}

with gzip.open(f'{results_folder+analysis_results_subfolder}results_fit_greedy_topic_hierarchy_all{filter_label}.pkl.gz','wb') as fp:
        pickle.dump((hierarchy_docs,hierarchy_words),fp)

try:
    print('hierarchy words at non trivial level:', hierarchy_words[highest_non_trivial_level],flush=True)
except:
    print('There is only one level, so hierarchy_words is an empty dictionary.')
    
try:
    print('hierarchy docs at non trivial level:', hierarchy_docs[highest_non_trivial_level],flush=True)
except:
    print('There is only one level, so hierarchy_docs is an empty dictionary.')

    
    
    
    
    
    
    
# Normalized mixture proportion by different level partition and topic

try:
    with gzip.open(os.path.join(results_folder+analysis_results_subfolder, f"results_fit_greedy_topic_frequency_all_by_level_partition_by_level_topics{filter_label}_all.pkl.gz"),"rb") as fp:
        topics_df_by_level,mixture_proportion_by_level_partition_by_level_topics, normalized_mixture_proportion_by_level_partition_by_level_topics, avg_topic_frequency_by_level_partition_by_level_topics = pickle.load(fp)
except FileNotFoundError:
    mixture_proportion_by_level_partition_by_level_topics, normalized_mixture_proportion_by_level_partition_by_level_topics, avg_topic_frequency_by_level_partition_by_level_topics = {}, {}, {}
    for level_partition in range(highest_non_trivial_level + 1):
        mixture_proportion_by_level_partition_by_level_topics[level_partition], normalized_mixture_proportion_by_level_partition_by_level_topics[level_partition], avg_topic_frequency_by_level_partition_by_level_topics[level_partition] = {}, {}, {}
        for l in range(highest_non_trivial_level + 1):
            mixture_proportion_by_level_partition_by_level_topics[level_partition][l], normalized_mixture_proportion_by_level_partition_by_level_topics[level_partition][l], avg_topic_frequency_by_level_partition_by_level_topics[level_partition][l] = \
                topic_mixture_proportion(dict_groups_by_level[l],ordered_edited_texts,h_t_doc_consensus_by_level[level_partition])

    with gzip.open(os.path.join(results_folder+analysis_results_subfolder, f"results_fit_greedy_topic_frequency_all_by_level_partition_by_level_topics{filter_label}_all.pkl.gz"),"wb") as fp:
        pickle.dump((topics_df_by_level,mixture_proportion_by_level_partition_by_level_topics, normalized_mixture_proportion_by_level_partition_by_level_topics, avg_topic_frequency_by_level_partition_by_level_topics),fp)

        
        
        
        
        
        
        
        
        
# KNOWLEDGE FLOW
all_docs = list(all_docs_dict.values())

paper2field = {x['id']:x['fieldsOfStudy'] for x in all_docs if 'id' in x and 'fieldsOfStudy' in x}

# all_docs_dict[paper]['ye_partition'] is a list of all the fields of the paper!!

# ACHTUNG PUT TO TRUE ONLY ONE!!
partition_first_field = False # Deprecated
partition_exploded = False
gt_partition = True

for gt_partition_level in range(highest_non_trivial_level + 1):

    if partition_first_field:
        partition_used = "original_partition_first_field"
        # ACHTUNG here we assign the partition, can be changed
        # THIS IS NOW DEPRECATED
        for paper in all_docs_dict:
            if 'fieldsOfStudy' in all_docs_dict[paper] and all_docs_dict[paper]['fieldsOfStudy'] is not None and len(all_docs_dict[paper]['fieldsOfStudy']) > 0:
                all_docs_dict[paper]['ye_partition'] = all_docs_dict[paper]['fieldsOfStudy'][0]
            elif 'fieldsOfStudy' in all_docs_dict[paper]: 
                all_docs_dict[paper]['ye_partition'] = None

        all_partitions = set([x['ye_partition'] for x in all_docs_dict.values()])
        ye_partition = {field:set() for field in all_partitions}

        for paper in all_docs_dict:
            if 'ye_partition' in all_docs_dict[paper]:
                ye_partition[all_docs_dict[paper]['ye_partition']].add(paper)


    if partition_exploded:
        partition_used = "original_partition_exploded"
        # ACHTUNG: papers may be assigned to multiple fields

        for paper in all_docs_dict:
            if 'fieldsOfStudy' in all_docs_dict[paper] and all_docs_dict[paper]['fieldsOfStudy'] is not None and len(all_docs_dict[paper]['fieldsOfStudy']) > 0:
                all_docs_dict[paper]['ye_partition'] = all_docs_dict[paper]['fieldsOfStudy']
            elif 'fieldsOfStudy' in all_docs_dict[paper]: 
                all_docs_dict[paper]['ye_partition'] = []


        all_partitions = set([x for y in paper2field.values() for x in y])
        ye_partition = {field:set() for field in all_partitions}

        for paper in all_docs:
            for field in paper['fieldsOfStudy']:
                ye_partition[field].add(paper['id'])


    if gt_partition:
        partition_used = "gt_partition_lev_%d"%(gt_partition_level)
        hyperlink_text_consensus_partitions_by_level = {}
        for l in h_t_doc_consensus_by_level.keys():
            print(l,flush=True)
            hyperlink_text_consensus_partitions_by_level[l] = h_t_doc_consensus_by_level[l]


        name2partition = {}
        for i,name in enumerate(hyperlink_g.vp["name"]):
            name2partition[name] = hyperlink_text_consensus_partitions_by_level[gt_partition_level][i]
        paper_ids_with_partition = set(name2partition.keys())

        doc_partition_remapping = {}
        doc_partition_remapping_inverse = {}
        lista1 = []
        for paper in ordered_paper_ids:
            lista1.append(name2partition[paper])
        lista2 = hyperlink_text_consensus_partitions_by_level[gt_partition_level]
        for part1, part2 in set(list(zip(lista1,lista2))):
            if part1 in doc_partition_remapping:
                print("THERE ARE MULTIPLE INSTANCES... ERROR")
                break
            else:
                doc_partition_remapping[part1] = part2  
                doc_partition_remapping_inverse[part2] = part1

        labelling_partition_remapping_by_level = {gt_partition_level:{x:str(x) for x in doc_partition_remapping.values()}}
        labelling_partition_remapping_by_level_inverse = {level:{y:x for x,y in labelling_partition_remapping_by_level[level].items()} for level in labelling_partition_remapping_by_level}


        print("Assigning partition field in docs")

        for paper_id in all_docs_dict:
            if paper_id in paper_ids_with_partition:
                all_docs_dict[paper_id]['ye_partition'] = [labelling_partition_remapping_by_level[gt_partition_level][doc_partition_remapping[name2partition[paper_id]]]]
            else: 
                all_docs_dict[paper_id]['ye_partition'] = []


        all_partitions = set([x for x in labelling_partition_remapping_by_level[gt_partition_level].values()])
        all_partitions = list(all_partitions)
        all_partitions.sort()
    #     all_partitions = set(all_partitions)
        ye_partition = {field:set() for field in all_partitions}

        for paper in all_docs:
            for field in paper['ye_partition']:
                ye_partition[field].add(paper['id'])


    # count citations between fields and between different times

    years_with_none = set([x['year'] for x in all_docs_dict.values()])

    # citation_count_per_field_in_time = {partition: {partition: {year: {year: 0 for year in years_with_none} for year in years_with_none} for partition in all_partitions} for partition in all_partitions}
    with gzip.open(dataset_path + "../" +"no_papers_in_fields_by_year.pkl.gz","rb") as fp:
        papers_per_year_per_field = pickle.load(fp)
    years = sorted([x for x in papers_per_year_per_field.keys() if x is not None and x > 1500])

    knowledge_units_count_per_field_in_time = {partition: {partition: {year: {year: 0 for year in years} for year in years} for partition in all_partitions} for partition in all_partitions}


    all_papers_ids = set([x["id"] for x in all_docs])

    # paper2 cites paper1 
    # so citation_count_per_field_in_time is ordered like partition_from - partition_to - year_from - year_to

    # ordered like knowledge_units_count_per_field_in_time[cited_field][citing_field][cited_year][citing_year]

    N_partitions = len(ye_partition.keys())

    for paper1 in all_docs:
        citing_papers = paper1['inCitations']
        for paper2_id in set(citing_papers).intersection(all_papers_ids):
            paper2 = all_docs_dict[paper2_id]
            partitions1 = paper1['ye_partition']
            partitions2 = paper2['ye_partition']
            if len(partitions1)>0 and len(partitions2)>0:
                for partition1 in partitions1:
                    for partition2 in partitions2:
                        year1 = paper1['year']
                        year2 = paper2['year']
                        if year1 is not None and year2 is not None:
                            knowledge_units_count_per_field_in_time[partition1][partition2][year1][year2] += (1/len(partitions1))*(1/len(partitions2))

    with gzip.open(os.path.join(results_folder+analysis_results_subfolder, f"knowledge_units_count_per_field_in_time_{partition_used}.pkl.gz"),"wb") as fp:
        pickle.dump(knowledge_units_count_per_field_in_time,fp)

    field_units_per_year = {year:{field:0 for field in all_partitions} for year in years}

    for paper in all_docs:
        if 'year' in paper and paper['year'] is not None and len(paper['ye_partition']) > 0:
            for field in paper['ye_partition']:
                field_units_per_year[paper['year']][field] += 1/len(paper['ye_partition'])


                
    def normalize_citations_count(citing_field,cited_field,citing_year,cited_year):
        # normalize citation_count_per_field using YE's prescription (null model)

        # fraction of knowledge units received by citing field in citing_year from cited field in cited years over all knowledge units received by citing field in citing years
        numerator = knowledge_units_count_per_field_in_time[cited_field][citing_field][cited_year][citing_year]/np.sum([knowledge_units_count_per_field_in_time[x][citing_field][cited_year][citing_year] for x in knowledge_units_count_per_field_in_time.keys()])

        # fraction of papers produced by cited field in cited year over all papers produced in cited years
        denominator = field_units_per_year[cited_year][cited_field]/np.sum([field_units_per_year[cited_year][field]for field in all_partitions])

        return numerator/denominator if denominator>0 and pd.notna(numerator) else 0


    knowledge_flow_normalized_per_field_in_time = {partition: {partition: {year: {year: 0 for year in years} for year in years} for partition in all_partitions} for partition in all_partitions}

    # NOTE: we iteratote over years_with_none because it only has years where a decentralization papers has been published, all other years are already initialized with 0
    for cited_field in all_partitions:
        for citing_field in all_partitions:
            for cited_year in years_with_none - {None}:
                for citing_year in years_with_none - {None}:
                    if citing_year < cited_year:
                        continue
                    knowledge_flow_normalized_per_field_in_time[cited_field][citing_field][cited_year][citing_year] = normalize_citations_count(citing_field,cited_field,citing_year,cited_year)

    with gzip.open(os.path.join(results_folder+analysis_results_subfolder, f"knowledge_flow_normalized_per_field_in_time_{partition_used}.pkl.gz"),"wb") as fp:
        pickle.dump(knowledge_flow_normalized_per_field_in_time,fp)

    # TODO: average over time window of field knowledge flow
    # it's the simple mean of the knowledge flow over the considered time window for the cited years given a citing year (and then you average this over a window of citing years)

    def time_window_average_knowledge_flow(citing_field,cited_field,citing_years,cited_years):

        tmp = []
        for cited_year in cited_years:
            for citing_year in citing_years:
                if citing_year>cited_year:
                    tmp.append(knowledge_flow_normalized_per_field_in_time[cited_field][citing_field][cited_year][citing_year])
        return np.nanmean(tmp)

    time_window_size_range = [5,10]
    last_year = 2021
    first_year = 1962

    knowledge_flow_normalized_per_field_per_time_window = {}
    for time_window_size in time_window_size_range:
        tmp = knowledge_flow_normalized_per_field_per_time_window[time_window_size] = {}
        for final_year in range(last_year,first_year,-time_window_size):
            tmp2 = tmp[(final_year-time_window_size+1,final_year)] = {}
    #         for final_year2 in range(final_year,last_year,time_window_size):
            for final_year2 in range(final_year,last_year+1,time_window_size):
                tmp3 = tmp2[(final_year2+1-time_window_size,final_year2)] = {}
                for partition in all_partitions:
                    tmp3[partition] = {partition: 0 for partition in all_partitions} 

    for time_window_size in knowledge_flow_normalized_per_field_per_time_window.keys():
        for cited_time_window in knowledge_flow_normalized_per_field_per_time_window[time_window_size].keys():
            for citing_time_window,tmp_dict in knowledge_flow_normalized_per_field_per_time_window[time_window_size][cited_time_window].items():
                for cited_field in all_partitions:
                    for citing_field in all_partitions:
                        tmp_dict[cited_field][citing_field] = time_window_average_knowledge_flow(citing_field,cited_field,range(citing_time_window[0],citing_time_window[1]+1),range(cited_time_window[0],cited_time_window[1]+1))

    with gzip.open(os.path.join(results_folder+analysis_results_subfolder, f"knowledge_flow_normalized_per_field_per_time_window_{partition_used}.pkl.gz"),"wb") as fp:
        pickle.dump(knowledge_flow_normalized_per_field_per_time_window,fp)



    # TODO: average over time window of field knowledge flow
    # it's the simple mean of the knowledge flow over the considered time window for the cited years given a citing year (and then you average this over a window of citing years)

    def time_window_average_knowledge_flow_to_future(citing_field,cited_field,cited_years):

        tmp = []
        for cited_year in cited_years:
            for citing_year in range(cited_year+1, 2022):
                if citing_year>cited_year:
                    tmp.append(knowledge_flow_normalized_per_field_in_time[cited_field][citing_field][cited_year][citing_year])
        return np.nanmean(tmp)

    time_window_size_range = [5,10]
    last_year = 2021
    first_year = 1962

    knowledge_flow_normalized_per_field_per_time_window_to_future = {time_window_size: {(final_year-time_window_size+1,final_year): {"future": {partition: {partition: 0 for partition in all_partitions} for partition in all_partitions}} for final_year in range(last_year,first_year,-time_window_size)} for time_window_size in time_window_size_range}

    for time_window_size in knowledge_flow_normalized_per_field_per_time_window_to_future.keys():
        for cited_time_window,tmp_dict in knowledge_flow_normalized_per_field_per_time_window_to_future[time_window_size].items():
            for cited_field in all_partitions:
                for citing_field in all_partitions:
                    tmp_dict["future"][cited_field][citing_field] = time_window_average_knowledge_flow_to_future(citing_field,cited_field,range(cited_time_window[0],cited_time_window[1]+1))

    with gzip.open(os.path.join(results_folder+analysis_results_subfolder, f"knowledge_flow_normalized_per_field_per_time_window_to_future_{partition_used}.pkl.gz"),"wb") as fp:
        pickle.dump(knowledge_flow_normalized_per_field_per_time_window_to_future,fp)

        

    knowledge_flow_normalized_per_field_per_time_window_to_future_by_level = {}
    knowledge_flow_normalized_per_field_in_time_by_level = {}
    knowledge_flow_normalized_per_field_per_time_window_by_level = {}
    
    lev = gt_partition_level
    
    with gzip.open(os.path.join(results_folder+analysis_results_subfolder, f"knowledge_flow_normalized_per_field_per_time_window_to_future_gt_partition_lev_{lev}.pkl.gz"),"rb") as fp:
        knowledge_flow_normalized_per_field_per_time_window_to_future_by_level[lev] = pickle.load(fp)
    with gzip.open(os.path.join(results_folder+analysis_results_subfolder, f"knowledge_flow_normalized_per_field_in_time_gt_partition_lev_{lev}.pkl.gz"),"rb") as fp:
        knowledge_flow_normalized_per_field_in_time_by_level[lev] = pickle.load(fp)
    with gzip.open(os.path.join(results_folder+analysis_results_subfolder, f"knowledge_flow_normalized_per_field_per_time_window_gt_partition_lev_{lev}.pkl.gz"),"rb") as fp:
        knowledge_flow_normalized_per_field_per_time_window_by_level[lev] = pickle.load(fp)

    def time_average_knowledge_flow_to_future(lev,citing_field,cited_field,cited_year):

        tmp = []

        for citing_year in range(cited_year+1, 2022):
            if citing_year>cited_year:
                tmp.append(knowledge_flow_normalized_per_field_in_time_by_level[lev][cited_field][citing_field][cited_year][citing_year])
        return np.nanmean(tmp)

    knowledge_flow_normalized_per_field_in_time_to_future_by_level = {}
    knowledge_flow_normalized_per_field_in_time_to_future_by_level[lev] = {}
    for cluster_from in knowledge_flow_normalized_per_field_in_time_by_level[lev].keys():
        knowledge_flow_normalized_per_field_in_time_to_future_by_level[lev][cluster_from] = {}
        for cluster_to in knowledge_flow_normalized_per_field_in_time_by_level[lev][cluster_from].keys():
            knowledge_flow_normalized_per_field_in_time_to_future_by_level[lev][cluster_from][cluster_to] = {}
            for year_from in knowledge_flow_normalized_per_field_in_time_by_level[lev][cluster_from][cluster_to].keys():
                knowledge_flow_normalized_per_field_in_time_to_future_by_level[lev][cluster_from][cluster_to][year_from] = time_average_knowledge_flow_to_future(lev,cluster_to,cluster_from,year_from)

    knowledge_flow_normalized_per_field_in_time_by_level_df = {}

    _t_from = []
    _t_to = []
    _from = []
    _to = []
    k = []

    for cluster_from in knowledge_flow_normalized_per_field_in_time_by_level[lev].keys():
        for cluster_to in knowledge_flow_normalized_per_field_in_time_by_level[lev][cluster_from].keys():
            for year_from in knowledge_flow_normalized_per_field_in_time_by_level[lev][cluster_from][cluster_to].keys():
                for year_to in knowledge_flow_normalized_per_field_in_time_by_level[lev][cluster_from][cluster_to][year_from].keys():
                    if year_to >= year_from:
                        _t_from.append(year_from)
                        _t_to.append(year_to)
                        _from.append(cluster_from)
                        _to.append(cluster_to)
                        k.append(knowledge_flow_normalized_per_field_in_time_by_level[lev][cluster_from][cluster_to][year_from][year_to])

    knowledge_flow_normalized_per_field_in_time_by_level_df[lev] = pd.DataFrame({'year_from':_t_from,
                                                                                 'year_to':_t_to,
                                                                                 'cluster_from':_from,
                                                                                 'cluster_to':_to,
                                                                                 'knowledge_flow':k
                                                                                })

    knowledge_flow_normalized_per_field_in_time_by_level_df[lev].cluster_from = knowledge_flow_normalized_per_field_in_time_by_level_df[lev].cluster_from.astype(int)
    knowledge_flow_normalized_per_field_in_time_by_level_df[lev].cluster_to = knowledge_flow_normalized_per_field_in_time_by_level_df[lev].cluster_to.astype(int)
    
    
    knowledge_flow_normalized_per_field_in_time_by_level_df[lev].to_csv(os.path.join(results_folder+analysis_results_subfolder,f"knowledge_flow_normalized_per_field_in_time_df_gt_partition_lev_{lev}.csv"), index=False)

print("FINISHED")