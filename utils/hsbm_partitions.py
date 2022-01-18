# IMPORTS

import gzip
import pickle
import pandas as pd
import numpy as np
import os
# move to repo root directory
from datetime import datetime
import sys
sys.path.insert(0, os.path.join(os.getcwd(),"utils"))
from hsbm import sbmmultilayer 
from hsbm.utils.nmi import *
from hsbm.utils.doc_clustering import *

# import gi
from gi.repository import Gtk, Gdk
import graph_tool.all as gt
import ast # to get list comprehension from a string


import functools, builtins # to impose flush=True on every print
builtins.print = functools.partial(print, flush=True)


def find_highest_non_trivial_group(hyperlink_g, 
                                   num_levels, 
                                   curr_hsbm, 
                                  ):
    '''
    Find the highest group that is not just 1 group.
    '''
    top_levels = curr_hsbm.n_levels
    temp_l = 0
    highest_l = [temp_l]
    final_l = 0

    # Compute values of interest.
    clustering_info = doc_clustering('./', hyperlink_g)
    clustering_info.seeds = [0] # we only want 1 iteration.

    for i in range(top_levels):
        # Iterate until no longer high enough level.
        highest_l = [temp_l]
        doc_partitions, doc_num_groups, doc_entropies = clustering_info.collect_info('1-layer-doc', curr_hsbm.g, highest_l, curr_hsbm.state)
        if doc_num_groups == [1]:
            # We have found highest level we can go.
            final_l = i-1
            break
        # Still on non-trivial level, so still continue
        temp_l += 1
    highest_l = [final_l]
    doc_partitions, doc_num_groups, doc_entropies = clustering_info.collect_info('1-layer-doc', curr_hsbm.g, highest_l, curr_hsbm.state)
    print(f'\tWe chose level {final_l} out of levels {num_levels}')
    print('\tNumber of groups', doc_num_groups)
#     print('\n')
    return doc_partitions, final_l



def get_hsbm_partitions(hyperlink_g,
                        model_states, 
                       ):
    '''
    For each iteration, retrieve the partitions.
    Return:
        - model_partitions: list of lists of length the number of iterations used in the algorithm. Each sublist contains the positions of each node in the partition with highest non-trivial level
        - levels: list of the highest non-trivial level for each iteration of the algorithm.
    '''
    model_partitions = []
    levels = [] # store highest non-trivial level
    # Retrieve partitions
    for i in range(len(model_states)):
        print('Iteration %d'%(i), flush = True)
        curr_hsbm = model_states[i] #retrieve hSBM model       
        # Figure out top layer
        top_levels = curr_hsbm.n_levels
        print('\tNumber of levels', top_levels)
        model_partition, highest_level = find_highest_non_trivial_group(hyperlink_g,
                                                                        top_levels, curr_hsbm)
        model_partitions.append(model_partition[0])
        levels.append(highest_level)
    return model_partitions, levels


def get_highest_level_hsbm_partitions_from_iterations(hyperlink_g,
                                        dir_list, 
                                        results_folder,
                                       ):
    '''
    For each iteration, retrieve the partitions.
    Return:
        - model_partitions: list of lists of length the number of iterations used in the algorithm. Each sublist contains the positions of each node in the partition with highest non-trivial level
        - levels: list of the highest non-trivial level for each iteration of the algorithm.
    '''
    # Retrieve partitions assigned to documents in each run. Also save index of highest non-trivial level.
    try:
        with gzip.open(os.path.join(results_folder, f'results_fit_greedy_partitions.pkl.gz'),'rb') as fp:
            hyperlink_text_hsbm_partitions, levels = pickle.load(fp)
        print(f'Loaded {len(hyperlink_text_hsbm_partitions)} partitions from {results_folder}', flush=True)
    except FileNotFoundError:
        print(f'Retrieving all iterations from provided list of directories', flush=True)
        hyperlink_text_hsbm_partitions = []
        count = 0
        levels = []
        for dir_ in dir_list:
            try:
                with gzip.open(os.path.join(dir_,f'results_fit_greedy_partitions.pkl.gz'),'rb') as fp:
                    tmp_hyperlink_text_hsbm_partitions, tmp_levels = pickle.load(fp)
                print(f'Loaded partitions from {dir_}', flush=True)
            except FileNotFoundError:
                try:
                    with gzip.open(os.path.join(dir_,f'results_fit_greedy.pkl.gz'),'rb') as fp:
                        tmp_hyperlink_text_hsbm_states,time_duration = pickle.load(fp)
                    print(f'Loaded states from {dir_}', flush=True)
                    tmp_hyperlink_text_hsbm_partitions, tmp_levels = get_hsbm_partitions(hyperlink_g, tmp_hyperlink_text_hsbm_states)
                    print(f'Computed partitions from {dir_}', flush=True)
                except FileNotFoundError:
                    print(f'PARTITIONS OR STATES NOT FOUND IN {dir_}', flush=True)
                    continue
            for i in range(len(tmp_hyperlink_text_hsbm_partitions)):
                hyperlink_text_hsbm_partitions.append(tmp_hyperlink_text_hsbm_partitions[i])
                levels.append(tmp_levels[i])
                count += 1
                print(f'\tAdded iteration {count} from {dir_}', flush=True)

        print('Number of iterations retrieved:',len(hyperlink_text_hsbm_partitions),flush=True)

        with gzip.open(os.path.join(results_folder,f'results_fit_greedy_partitions.pkl.gz'),'wb') as fp:
            pickle.dump((hyperlink_text_hsbm_partitions, levels),fp)

    return hyperlink_text_hsbm_partitions, levels



# ## Consensus Partition
# 
# Compute the consensus partition assignment to document nodes over all the iterations.
def get_hsbm_partitions_from_iterations(hyperlink_g,
                                        dir_list, 
                                        levels,
                                        results_folder,
                                       ):
    start = datetime.now()
    try:
        with gzip.open(os.path.join(results_folder,f'results_fit_greedy_partitions_docs_all.pkl.gz'),'rb') as fp:
            hyperlink_text_hsbm_partitions_by_level,duration = pickle.load(fp)
        print(f'Loaded partitions from {results_folder}', flush=True)
    except FileNotFoundError:
        clustering_info = doc_clustering('./', hyperlink_g)
        clustering_info.seeds = [0]

        hyperlink_text_hsbm_partitions_by_level = {}
        hyperlink_text_hsbm_partitions_by_level_info = {}

        for l in range(max(levels)+1):
            hyperlink_text_hsbm_partitions_by_level[l] = []
            hyperlink_text_hsbm_partitions_by_level_info[l] = []
        
        print(f'Retrieving all iterations from provided list of directories', flush=True)
            
        count = 0
        for dir_ in dir_list:
            try:
                with gzip.open(os.path.join(dir_,f'results_fit_greedy_partitions_docs_all.pkl.gz'),'rb') as fp:
                    tmp_hyperlink_text_hsbm_partitions_by_level, tmp_duration = pickle.load(fp)
                with gzip.open(os.path.join(dir_,f'results_fit_greedy_partitions_docs_all_info.pkl.gz'),'rb') as fp:
                    tmp_hyperlink_text_hsbm_partitions_by_level_info, tmp_duration = pickle.load(fp)
                print(f'Loaded partitions from {dir_}', flush=True)
                for iteration in range(len(tmp_hyperlink_text_hsbm_partitions_by_level[list(tmp_hyperlink_text_hsbm_partitions_by_level.keys())[0]])):
                    count += 1
                    print('Iteration %d'%count, flush=True)
                    for l in range(max(levels)+1):
                        hyperlink_text_hsbm_partitions_by_level_info[l].append(tmp_hyperlink_text_hsbm_partitions_by_level_info[l][iteration])
                        hyperlink_text_hsbm_partitions_by_level[l].append(tmp_hyperlink_text_hsbm_partitions_by_level[l][iteration])
            except FileNotFoundError:
                try:
                    with gzip.open(os.path.join(dir_,f'results_fit_greedy.pkl.gz'),'rb') as fp:
                        tmp_hyperlink_text_hsbm_states,time_duration = pickle.load(fp)
                    print(f'Loaded states from {dir_}', flush=True)
                    for iteration,curr_hsbm in enumerate(tmp_hyperlink_text_hsbm_states):
                        count += 1
                        print('Iteration %d'%count, flush=True)
                        for l in range(max(levels)+1):
                            tmp = clustering_info.collect_info2('1-layer-doc', curr_hsbm.g, [l], curr_hsbm.state)
                            hyperlink_text_hsbm_partitions_by_level_info[l].append(tmp)
                            hyperlink_text_hsbm_partitions_by_level[l].append(tmp[0][0])                    
                    print(f'Computed partitions from {dir_}', flush=True)
                except FileNotFoundError:
                    print(f'PARTITIONS OR STATES NOT FOUND IN {dir_}', flush=True)
                    continue

        print('Number of iterations retrieved:',count,flush=True)
        end = datetime.now()
        with gzip.open(os.path.join(results_folder, f'results_fit_greedy_partitions_docs_all.pkl.gz'),'wb') as fp:
            pickle.dump((hyperlink_text_hsbm_partitions_by_level,end-start),fp)

        with gzip.open(os.path.join(results_folder, f'results_fit_greedy_partitions_docs_all_info.pkl.gz'),'wb') as fp:
            pickle.dump((hyperlink_text_hsbm_partitions_by_level_info,end-start),fp)#
        return hyperlink_text_hsbm_partitions_by_level, end-start
        
        

def get_word_type_blocks(h_t_state, h_t_graph, level, IDs):
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

        
        
def get_hsbm_word_partitions_from_iterations(hyperlink_g,
                                        dir_list, 
                                        levels,
                                        results_folder,
                                        IDs,
                                       ):
    try:
        with gzip.open(os.path.join(results_folder,f'results_fit_greedy_partitions_words_all.pkl.gz'),'rb') as fp:
            H_T_word_hsbm_partitions_by_level, H_T_word_hsbm_num_groups_by_level = pickle.load(fp)
        print(f'Loaded word partitions from {results_folder}', flush=True)
    except FileNotFoundError:
        start = datetime.now()
        H_T_word_hsbm_partitions_by_level, H_T_word_hsbm_num_groups_by_level = {},{}

        for l in range(max(levels)+1):
            H_T_word_hsbm_partitions_by_level[l] = []
            H_T_word_hsbm_num_groups_by_level[l] = []
            
        count = 0
        for dir_ in dir_list:
            try:
                with gzip.open(os.path.join(dir_,f'results_fit_greedy_partitions_words_all.pkl.gz'),'rb') as fp:
                    tmp_H_T_word_hsbm_partitions_by_level, tmp_H_T_word_hsbm_num_groups_by_level = pickle.load(fp)
                
                for iteration in range(len(tmp_H_T_word_hsbm_partitions_by_level[list(tmp_H_T_word_hsbm_partitions_by_level.keys())[0]])):
                    count += 1
                    print('\titeration',count,flush=True)
                    for l in range(max(levels)+1):
                        H_T_word_hsbm_partitions_by_level[l].append(tmp_H_T_word_hsbm_partitions_by_level[l][iteration])
                        H_T_word_hsbm_num_groups_by_level[l].append(tmp_H_T_word_hsbm_num_groups_by_level[l][iteration])
                    
            except FileNotFoundError:
                try:
                    with gzip.open(os.path.join(dir_,f'results_fit_greedy.pkl.gz'),'rb') as fp:
                        tmp_hyperlink_text_hsbm_states,time_duration = pickle.load(fp)
                    print(f'Loaded states from {dir_}', flush=True)
                    for iteration,curr_hsbm in enumerate(tmp_hyperlink_text_hsbm_states):
                        count += 1
                        print('\titeration',count,flush=True)
                        for l in range(max(levels)+1):
                            word_partitions, num_word_groups, en = get_word_type_blocks(curr_hsbm.state, curr_hsbm.g, l, IDs)
                            H_T_word_hsbm_partitions_by_level[l].append(word_partitions[0])
                            H_T_word_hsbm_num_groups_by_level[l].append(num_word_groups)
                    print(f'Computed word partitions from {dir_}', flush=True)
                except FileNotFoundError:
                    print(f'PARTITIONS OR STATES NOT FOUND IN {dir_}', flush=True)
                    continue

        with gzip.open(os.path.join(results_folder, f'results_fit_greedy_partitions_words_all.pkl.gz'),'wb') as fp:
                pickle.dump((H_T_word_hsbm_partitions_by_level, H_T_word_hsbm_num_groups_by_level),fp)

        end = datetime.now()
        print('Time duration',end-start,flush=True)
    
    return H_T_word_hsbm_partitions_by_level, H_T_word_hsbm_num_groups_by_level
    




def get_consensus_nested_partition(H_T_word_hsbm_partitions_by_level, 
                                  ):
    '''
    As parameter it takes a dictionary {level: [[partition[paper] for paper in papers] for i in iterations]}.
    It calculates the nested consensus partition (reordered correctly)
    '''
    hierarchy_words_partitions = {}
    for l in range(max(H_T_word_hsbm_partitions_by_level),0,-1):
        hierarchy_words_partitions[l] = []
        for iteration in range(len(H_T_word_hsbm_partitions_by_level[l])):
            tmp1_words, tmp2_words = H_T_word_hsbm_partitions_by_level[l][iteration], H_T_word_hsbm_partitions_by_level[l-1][iteration]

            old_blocks1 = sorted(set(tmp1_words))
            new_blocks1 = list(range(len(set(tmp1_words))))
            remap_blocks_dict1 = {old_blocks1[i]:new_blocks1[i] for i in range(len(old_blocks1))}
            for i in range(len(tmp1_words)):
                tmp1_words[i] = remap_blocks_dict1[tmp1_words[i]]

            old_blocks2 = sorted(set(tmp2_words))
            new_blocks2 = list(range(len(set(tmp2_words))))
            remap_blocks_dict2 = {old_blocks2[i]:new_blocks2[i] for i in range(len(old_blocks2))}
            for i in range(len(tmp2_words)):
                tmp2_words[i] = remap_blocks_dict2[tmp2_words[i]]

            hierarchy_words_partitions[l].append({w: set() for w in np.unique(tmp1_words)})

            for i in range(len(tmp1_words)):
                hierarchy_words_partitions[l][iteration][tmp1_words[i]].add(tmp2_words[i])

    hyperlink_words_hsbm_partitions_by_level = {0:H_T_word_hsbm_partitions_by_level[0]}

    for l in range(1,len(H_T_word_hsbm_partitions_by_level)):
        hyperlink_words_hsbm_partitions_by_level[l] = []
        for iteration in range(len(hierarchy_words_partitions[l])):
            tmp_list = -np.ones(len(set(hyperlink_words_hsbm_partitions_by_level[l-1][iteration])))
            for i,group in hierarchy_words_partitions[l][iteration].items():
                for j in group:
                    tmp_list[j] = i
            hyperlink_words_hsbm_partitions_by_level[l].append(tmp_list)

    bs_w = [[hyperlink_words_hsbm_partitions_by_level[l][i] for l in hyperlink_words_hsbm_partitions_by_level.keys()] for i in range(len(hyperlink_words_hsbm_partitions_by_level[0]))]
    c_w,r_w = gt.nested_partition_overlap_center(bs_w)    

    h_t_word_consensus_by_level = {0:np.array([c_w[0][word] for word in range(len(hyperlink_words_hsbm_partitions_by_level[0][0]))])}
    for l in range(1,len(c_w)):
        h_t_word_consensus_by_level[l] = np.array([c_w[l][h_t_word_consensus_by_level[l-1][word]] for word in range(len(hyperlink_words_hsbm_partitions_by_level[0][0]))])
    
    return h_t_word_consensus_by_level, hyperlink_words_hsbm_partitions_by_level


def get_topics_h_t_consensus_model(groups, 
                                   words, 
                                   n=10, 
                                  ):
    '''
    Retrieve topics in consensus partition for H+T model.
    '''
    dict_groups = groups
    Bw = dict_groups['Bw'] # number of word-groups
    p_w_tw = dict_groups['p_w_tw'] # topic proportions over words
    words = words
    # Loop over all word-groups
    dict_group_words = {}
    for tw in range(Bw):
        p_w_ = p_w_tw[:, tw]
        ind_w_ = np.argsort(p_w_)[::-1]
        list_words_tw = []
        for i in ind_w_[:n]:
            if p_w_[i] > 0:
                list_words_tw+=[(words[i],p_w_[i])]
            else:
                break
        dict_group_words[tw] = list_words_tw
    return dict_group_words    

def topic_mixture_proportion(dict_groups,edited_text, document_partitions):

    topics = dict_groups.keys()
    partitions = np.unique(document_partitions)

    avg_topic_frequency = {}
    mixture_proportion = {}
    normalized_mixture_proportion = {}

    doc_texts = np.array(edited_text, dtype=object)

    n_i_t = {}

    topic_doc_group_words = {}

    for doc_group in partitions:
        topic_doc_group_words[doc_group] = set()
        for i,doc_group_membership in enumerate(document_partitions):
            if doc_group_membership != doc_group:
                continue
            topic_doc_group_words[doc_group] = topic_doc_group_words[doc_group].union(set(edited_text[i]))

    for topic in topics:
        topic_words = set([x[0] for x in dict_groups[topic]])
        n_i_t[topic] = {}

        for doc_group in partitions:
            n_i_t[topic][doc_group] = len(topic_words.intersection(topic_doc_group_words[doc_group]))

    for doc_group in partitions:
        mixture_proportion[f'doc_group {doc_group}'] = {}
        for topic in topics:
            mixture_proportion[f'doc_group {doc_group}'][f'topic {topic}'] = n_i_t[topic][doc_group] / np.sum([n_i_t[topic_j][doc_group] for topic_j in topics])

    S = np.sum([n_i_t[topic_j][doc_group] for doc_group in partitions for topic_j in topics])
    for topic in topics:
        avg_topic_frequency[f'topic {topic}'] = np.sum([n_i_t[topic][doc_group] for doc_group in partitions]) / S

    for doc_group in partitions:
        normalized_mixture_proportion[f'doc_group {doc_group}'] = {}
        for topic in topics:
            normalized_mixture_proportion[f'doc_group {doc_group}'][f'topic {topic}'] = ( mixture_proportion[f'doc_group {doc_group}'][f'topic {topic}'] - avg_topic_frequency[f'topic {topic}'] ) / avg_topic_frequency[f'topic {topic}']

    return mixture_proportion, normalized_mixture_proportion, avg_topic_frequency


