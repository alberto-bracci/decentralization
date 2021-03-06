# IMPORTS

import gzip
import pickle
import pandas as pd
import numpy as np
import os
from datetime import datetime
import sys
sys.path.insert(0, os.path.join(os.getcwd(),"utils"))
import sbmmultilayer 
from nmi import *
from doc_clustering import *
from gi.repository import Gtk, Gdk
import graph_tool.all as gt
import scipy


def find_highest_non_trivial_group(
    hyperlink_g, 
    num_levels, 
    curr_hsbm, 
):
    '''
        Find the highest level with more than just 1 group.
        
        Paramaters
        ----------
            hyperlink_g: gt network containing all papers and their links, i.e., citations or hyperlinks (gt.Graph)
            num_levels: number of levels in the curr_hsbm (int)
            curr_hsbm: hierarchical stochastick block model to analyse (sbmmultilayer.sbmmultilayer)
        
        Returns
        ----------
            doc_partitions: list with only one list made of the corresponding cluster (int) for each doc at the highest non trivial level (list of list)
            final_l: highest level with more than just 1 group (int)
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
    print(f'\tWe chose level {final_l} out of {num_levels} levels')
    print('\tNumber of doc-groups:', doc_num_groups)
    return doc_partitions, final_l


def get_hsbm_highest_level_partitions(
    hyperlink_g,
    model_states, 
):
    '''
        For each iteration provided, retrieve the partitions at their highest non trivial level.

        Paramaters
        ----------
            hyperlink_g: gt network containing all papers and their links, i.e., citations or hyperlinks (gt.Graph)
            model_states: list of all the states, loaded from dir_list, of all iterations (list of hsbm.state)

        Returns
        ----------
            model_partitions: list of lists of length the number of iterations used in the algorithm. Each sublist contains the positions of each node in the partition with highest non-trivial level (list of lists)
            levels: list of the highest non-trivial level for each iteration of the algorithm (list of ints)
    '''
    model_partitions = []
    levels = [] # store highest non-trivial level
    # Retrieve partitions
    for i,curr_hsbm in enumerate(model_states):
        print('Iteration %d'%(i), flush = True)
        # Figure out top layer
        top_levels = curr_hsbm.n_levels
        print('\tNumber of levels', top_levels)
        model_partition, highest_level = find_highest_non_trivial_group(
            hyperlink_g,
            top_levels,
            curr_hsbm
        )
        model_partitions.append(model_partition[0])
        levels.append(highest_level)
    return model_partitions, levels


def get_highest_level_hsbm_partitions_from_iterations(
    hyperlink_g,
    dir_list, 
    results_folder,
):
    '''
        For each iteration, retrieve the partitions at their highest non trivial level.
        
        Paramaters
        ----------
            hyperlink_g: gt network containing all papers and their links, i.e., citations or hyperlinks (gt.Graph)
            dir_list: list of all the paths to the directories where all iterations results have been dumped (list of str, valid paths)
            results_folder: path to directory of the results_folder where to save tokenized_texts_dict (str, valid path)
        
        Returns
        ----------
            model_partitions: list of lists of length the number of iterations used in the algorithm. Each sublist contains the positions of each node in the partition with highest non-trivial level (list of lists)
            levels: list of the highest non-trivial level for each iteration of the algorithm (list of ints)
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
                    tmp_hyperlink_text_hsbm_partitions, tmp_levels = get_hsbm_highest_level_partitions(
                        hyperlink_g, 
                        tmp_hyperlink_text_hsbm_states
                    )
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


def get_hsbm_partitions_from_iterations(
    hyperlink_g,
    dir_list, 
    levels,
    results_folder,
):
    '''
        Compute the consensus partition assigned to document nodes over all the iterations.
        
        Paramaters
        ----------
            hyperlink_g: gt network containing all papers and their links, i.e., citations or hyperlinks (gt.Graph)
            dir_list: list of all the paths to the directories where all iterations results have been dumped (list of str, valid paths)
            levels: list of the highest non-trivial level for each iteration of the algorithm (list of ints)
            results_folder: path to directory of the results_folder where to save tokenized_texts_dict (str, valid path)
            
        Returns
        ---------- 
            hyperlink_text_hsbm_partitions_by_level: dict of level:list (one for each iteration with that level) of lists, containing the group assignment of each text (dict)
            time_duration: datetime difference from start to finish of the function (datetime object)
    '''
    start = datetime.now()
    try:
        with gzip.open(os.path.join(results_folder,f'results_fit_greedy_partitions_docs_all.pkl.gz'),'rb') as fp:
            hyperlink_text_hsbm_partitions_by_level,duration = pickle.load(fp)
        print(f'Loaded partitions from {results_folder}', flush=True)
    except FileNotFoundError:
        clustering_info = doc_clustering('./', hyperlink_g)
        clustering_info.seeds = [0]

        hyperlink_text_hsbm_partitions_by_level = {}

        for l in range(max(levels)+1):
            hyperlink_text_hsbm_partitions_by_level[l] = []
        
        print(f'Retrieving all iterations from provided list of directories', flush=True)
            
        count = 0
        for dir_ in dir_list:
            try:
                with gzip.open(os.path.join(dir_,f'results_fit_greedy_partitions_docs_all.pkl.gz'),'rb') as fp:
                    tmp_hyperlink_text_hsbm_partitions_by_level, tmp_duration = pickle.load(fp)
                print(f'Loaded partitions from {dir_}', flush=True)
                for iteration in range(len(tmp_hyperlink_text_hsbm_partitions_by_level[list(tmp_hyperlink_text_hsbm_partitions_by_level.keys())[0]])):
                    count += 1
                    print('Iteration %d'%count, flush=True)
                    for l in range(max(levels)+1):
                        try:
                            hyperlink_text_hsbm_partitions_by_level[l].append(tmp_hyperlink_text_hsbm_partitions_by_level[l][iteration])
                        except KeyError as e:
                            print(f'count is {count}, level is {l}, got KeyError:\n\t{e}',flush=True)
                            hyperlink_text_hsbm_partitions_by_level[l].append(list(np.zeros(len(tmp_hyperlink_text_hsbm_partitions_by_level[0][iteration]))))
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
                            hyperlink_text_hsbm_partitions_by_level[l].append(tmp[0][0])                    
                    print(f'Computed partitions from {dir_}', flush=True)
                except FileNotFoundError:
                    print(f'PARTITIONS OR STATES NOT FOUND IN {dir_}', flush=True)
                    continue

        print('Number of iterations retrieved:',count,flush=True)
        end = datetime.now()
        with gzip.open(os.path.join(results_folder, f'results_fit_greedy_partitions_docs_all.pkl.gz'),'wb') as fp:
            pickle.dump((hyperlink_text_hsbm_partitions_by_level,end-start),fp)
    end = datetime.now()
    time_duration = end-start
    return hyperlink_text_hsbm_partitions_by_level, time_duration
        
        

def get_word_type_blocks(
    h_t_state, 
    h_t_graph, 
    level, 
    IDs
):
    '''
        Retrieves the block assignment of word types for the model. 
        Results are lists because are intended to collect results from multiple iterations.
        
        Paramaters
        ----------
            h_t_state: state of the hsbm (hsbm.state)
            h_t_graph: graph of the hsbm (hsbm.g)
            level: level chosen to get the blocks (int)
            IDs: list of the unique paper IDs kept after filtering, used to create the hsbm (list of str)
        
        Returns
        ----------
            partitions: list of list containing the positions of each node in the partition at the selected level of the state (list of list of ints)
            num_of_groups: list with the number of groups in the partition of the state at the selected level (list of int)
            entropies: list of entropy of the selected hsbm, representing the minimum description length of inferred state (list of float)
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

        
        
def get_hsbm_word_partitions_from_iterations(
    hyperlink_g,
    dir_list, 
    levels,
    results_folder,
    IDs,
):
    '''
        Get all word group assignments at all levels and for all hsbm iterations in dir_list.
        
        Paramaters
        ----------
            hyperlink_g: gt network containing all papers and their links, i.e., citations or hyperlinks (gt.Graph)
            dir_list: list of all the paths to the directories where all iterations results have been dumped (list of str, valid paths)
            levels: list of the highest non-trivial level for each iteration of the algorithm (list of ints)
            results_folder: path to directory of the results_folder where to save tokenized_texts_dict (str, valid path)
            IDs: list of the unique paper IDs kept after filtering, used to create the hsbm (list of str)
        
        Returns
        ----------
            H_T_word_hsbm_partitions_by_level: dict {level: [[partition[paper] for paper in papers] for i in iterations]} containing for each level the partition list of each iteration (dict)
            H_T_word_hsbm_num_groups_by_level: dict {level: [num_groups for iteration in iterations]} containing the number of groups in the partition of the iterations for each level (dict)
    '''
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
                        try:
                            H_T_word_hsbm_partitions_by_level[l].append(tmp_H_T_word_hsbm_partitions_by_level[l][iteration])
                            H_T_word_hsbm_num_groups_by_level[l].append(tmp_H_T_word_hsbm_num_groups_by_level[l][iteration])
                        except KeyError as e:
                            print(f'count is {count}, level is {l}, got KeyError:\n\t{e}',flush=True)
                            H_T_word_hsbm_partitions_by_level[l].append(list(np.zeros(len(tmp_H_T_word_hsbm_partitions_by_level[0][iteration]))))
                            H_T_word_hsbm_num_groups_by_level[l].append([1])
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
    


def get_consensus_nested_partition(
    H_T_word_hsbm_partitions_by_level, 
):
    '''
        It calculates the nested consensus partition (reordered correctly). 
        If words are provided, the partition is of words (topics).
        If docs are provided, the partition is of docs (clusters).
        
        Paramaters
        ----------
            H_T_word_hsbm_partitions_by_level: dict {level: [[partition[paper] for paper in papers] for i in iterations]} containing for each level the partition list of each iteration (dict)
        
        Returns
        ----------
            h_t_word_consensus_by_level: dict of level as key and an array of same length as H_T_word_hsbm_partitions_by_level with value the cluster of that element at that level on the computed consensus partition (dict, level:np.array)
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
            
            for i in range(len(tmp2_words)):
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
    
    return h_t_word_consensus_by_level

def get_consensus(
    dir_list,
    hyperlink_text_hsbm_partitions_by_level,
    H_T_word_hsbm_partitions_by_level,
    ordered_paper_ids,
    results_folder,
    filter_label = ''
):
    '''
        It calculates the nested consensus partition (reordered correctly), computes some statistics and summary. 
        
        Paramaters
        ----------
            dir_list: list of all the paths to the directories where all iterations results have been dumped (list of str, valid paths)
            hyperlink_text_hsbm_partitions_by_level: 
            H_T_word_hsbm_partitions_by_level: dict {level: [[partition[paper] for paper in papers] for i in iterations]} containing for each level the partition list of each iteration (dict)
            ordered_paper_ids: ordered list of paper_ids in hyperlink_g, with the same ordering as hyperlink_g.vp['name'] (list of str, i.e., list(hyperlink_g.vp['name']))
            results_folder: path to directory of the results_folder where to save tokenized_texts_dict (str, valid path)
            filter_label: possible label to use if a special filtering is applied (str)
        
        Returns
        ----------
            h_t_doc_consensus_by_level: dict of level as key and an array of length the number of docs in the hsbm with value the cluster at that level (dict, level:np.array)
            h_t_word_consensus_by_level: dict of level as key and an array of length the number of words in the hsbm with value the cluster at that level (dict, level:np.array)
            h_t_consensus_summary_by_level: dict in which for each level there are statistics on group membership of nodes from the inferred state. These keys are:
                - B_d, int, number of doc-groups
                - B_w, int, number of word-groups
                - p_td_d, scipy.sparse.dok_matrix (B_d, D);
                          doc-group membership:
                          # group membership of each doc-node, matrix of ones and zeros, shape B_d x D
                          prob that doc-node d belongs to doc-group td: P(td | d)
                - p_tw_w, scipy.sparse.dok_matrix (B_w, V);
                          word-group membership:
                          # group membership of each word-node, matrix of ones or zeros, shape B_w x V
                          prob that word-node w belongs to word-group tw: P(tw | w)
                - p_tw_d, scipy.sparse.dok_matrix (B_w, D);
                          doc-topic mixtures:
                          # Mixture of word-groups into documents P(t_w | d), shape B_w x D
                          prob of word-group tw in doc d P(tw | d)
                - p_w_tw, scipy.sparse.dok_matrix (V, B_w);
                          per-topic word distribution, shape V x B_w
                          prob of word w given topic tw P(w | tw)
    '''
    try:
        with gzip.open(f'{results_folder}results_fit_greedy_partitions_consensus_all{filter_label}.pkl.gz','rb') as fp:
            h_t_doc_consensus_by_level, h_t_word_consensus_by_level, h_t_consensus_summary_by_level = pickle.load(fp)
        print('Loaded consensus from file', flush=True)
    except FileNotFoundError:
        print('Calculating nested consensus partition among docs', flush=True)
        h_t_doc_consensus_by_level = get_consensus_nested_partition(hyperlink_text_hsbm_partitions_by_level)
        print('Calculating nested consensus partition among words', flush=True)
        h_t_word_consensus_by_level = get_consensus_nested_partition(H_T_word_hsbm_partitions_by_level)
        print('Loading a state', flush=True)
        for dir_ in dir_list:
            try:
                with gzip.open(os.path.join(dir_,f'results_fit_greedy.pkl.gz'),'rb') as fp:
                    hyperlink_text_hsbm_states,time_duration = pickle.load(fp)
                print(f'Loaded state from dir {dir_}', flush=True)
                break
            except FileNotFoundError:
                continue 
        
        for l in list(hyperlink_text_hsbm_partitions_by_level.keys()):
            if len(set(h_t_word_consensus_by_level[l])) == 1 and len(set(h_t_doc_consensus_by_level[l])) == 1:
                print('Removing level %d because it has only 1 cluster of docs and 1 of words'%l, flush=True)
                del h_t_word_consensus_by_level[l]
                del h_t_doc_consensus_by_level[l]
                del hyperlink_text_hsbm_partitions_by_level[l]

        print('\nCalculating summary', flush=True)

        h_t_consensus_summary_by_level = {}

        for l in h_t_doc_consensus_by_level.keys():
            print('level %d'%l, flush=True)
            D = len((h_t_doc_consensus_by_level[l])) # number of document nodes
            print('\tdocs blocks are %d, end with index %d'%(len(set(h_t_doc_consensus_by_level[l])),max(h_t_doc_consensus_by_level[l])))
            if len(set(h_t_doc_consensus_by_level[l])) != max(h_t_doc_consensus_by_level[l]) + 1:
                print('\tReordering list of doc blocks', flush=True)
                old_blocks = sorted(set(h_t_doc_consensus_by_level[l]))
                new_blocks = list(range(len(set(h_t_doc_consensus_by_level[l]))))
                remap_blocks_dict = {old_blocks[i]:new_blocks[i] for i in range(len(set(h_t_doc_consensus_by_level[l])))}
                for i in range(D):
                    h_t_doc_consensus_by_level[l][i] = remap_blocks_dict[h_t_doc_consensus_by_level[l][i]]

            print('\twords blocks are %d, end with index %d'%(len(set(h_t_word_consensus_by_level[l])),max(h_t_word_consensus_by_level[l])))
            V = len((h_t_word_consensus_by_level[l])) # number of word-type nodes
            if len(set(h_t_word_consensus_by_level[l])) != max(h_t_word_consensus_by_level[l]) + 1:
                print('\tReordering list of word blocks', flush=True)
                old_blocks = sorted(set(h_t_word_consensus_by_level[l]))
                new_blocks = list(range(len(set(h_t_word_consensus_by_level[l]))))
                remap_blocks_dict = {old_blocks[i]:new_blocks[i] for i in range(len(set(h_t_word_consensus_by_level[l])))}
                for i in range(V):
                    h_t_word_consensus_by_level[l][i] = remap_blocks_dict[h_t_word_consensus_by_level[l][i]]

            h_t_word_consensus_by_level[l] += len(set(h_t_doc_consensus_by_level[l])) # to get cluster number to not start from 0

            # number of word-tokens (edges excluding hyperlinks)
            N = int(np.sum([hyperlink_text_hsbm_states[0].g.ep.edgeCount[e] for e in hyperlink_text_hsbm_states[0].g.edges() if hyperlink_text_hsbm_states[0].g.ep['edgeType'][e]== 0 ])) 

            # Number of blocks
            B = len(set(h_t_word_consensus_by_level[l])) + len(set(h_t_doc_consensus_by_level[l]))

            # Count number of blocks of both types
            remap_B_d = {}
            B_d = 0
            remap_B_w = {}
            B_w = 0
            for e in hyperlink_text_hsbm_states[0].g.edges():
                # We only care about edges in text network
                if hyperlink_text_hsbm_states[0].g.ep.edgeType[e] == 0:        
                    # v1 ranges from 0, 1, 2, ..., D - 1
                    # v2 ranges from D, ..., (D + V) - 1 (V # of word types)
                    v1 = int(e.source()) # document node index
                    v2 = int(e.target()) # word type node index # THIS MAKES AN IndexError!
                    v1_name = hyperlink_text_hsbm_states[0].g.vp['name'][v1]
                    v2_name = hyperlink_text_hsbm_states[0].g.vp['name'][v2]

                    # z1 will have values from 1, 2, ..., B_d; document-group i.e document block that doc node is in 
                    # z2 will have values from B_d + 1, B_d + 2,  ..., B_d + B_w; word-group i.e word block that word type node is in
                    # Recall that h_t_word_consensus starts at 0 so need to -120

                    z1, z2 = h_t_doc_consensus_by_level[l][ordered_paper_ids.index(v1_name)], h_t_word_consensus_by_level[l][v2-D]
                    if z1 not in remap_B_d:
                        remap_B_d[z1] = B_d
                        B_d += 1
                    if z2 not in remap_B_w:
                        remap_B_w[z2] = B_w
                        B_w += 1
            print(f"B {B}, B_d {B_d}, B_w {B_w}")

            # Count labeled half-edges, total sum is # of edges
            # Number of half-edges incident on word-node w and labeled as word-group tw
            n_wb = scipy.sparse.dok_matrix((V,B_w))

            # Number of half-edges incident on document-node d and labeled as document-group td
            n_db = scipy.sparse.dok_matrix((D,B_d))

            # Number of half-edges incident on document-node d and labeled as word-group tw
            n_dbw = scipy.sparse.dok_matrix((D,B_w))

            # Count labeled half-edges, total sum is # of edges
            for e in hyperlink_text_hsbm_states[0].g.edges():
                # We only care about edges in text network
                if hyperlink_text_hsbm_states[0].g.ep.edgeType[e] == 0:        
                    # v1 ranges from 0, 1, 2, ..., D - 1
                    # v2 ranges from D, ..., (D + V) - 1 (V # of word types)
                    v1 = int(e.source()) # document node index
                    v2 = int(e.target()) # word type node index # THIS MAKES AN IndexError!
                    v1_name = hyperlink_text_hsbm_states[0].g.vp['name'][v1]
                    v2_name = hyperlink_text_hsbm_states[0].g.vp['name'][v2]

                    # z1 will have values from 1, 2, ..., B_d; document-group i.e document block that doc node is in 
                    # z2 will have values from B_d + 1, B_d + 2,  ..., B_d + B_w; word-group i.e word block that word type node is in
                    # Recall that h_t_word_consensus starts at 0 so need to -120

                    z1, z2 = h_t_doc_consensus_by_level[l][ordered_paper_ids.index(v1_name)], h_t_word_consensus_by_level[l][v2-D]

                    v1 = int(e.source()) # document node index
                    v2 = int(e.target()) # word type node index
                    n_wb[v2-D,remap_B_w[z2]] += 1 # word type v2 is in topic z2
                    n_db[v1,remap_B_d[z1]] += 1 # document v1 is in doc cluster z1
                    n_dbw[v1,remap_B_w[z2]] += 1 # document v1 has a word in topic z2                    

            # Group membership of each word-type node in topic, matrix of ones or zeros, shape B_w x V
            # This tells us the probability of topic over word type
            p_tw_w = n_wb.copy()
            tmp_sum = p_tw_w.sum(axis=1)
            for row,col in p_tw_w.keys():
                sum_row = tmp_sum[row,0]
                p_tw_w[row,col] = p_tw_w[row,col] / sum_row
            p_tw_w = p_tw_w.T

            # Group membership of each doc-node, matrix of ones of zeros, shape B_d x D
            tmp_sum = n_db.sum(axis=1)
            for row,col in n_db.keys():
                sum_row = tmp_sum[row,0]
                n_db[row,col] = n_db[row,col] / sum_row
            p_td_d = n_db.T

            # Mixture of word-groups into documents P(t_w | d), shape B_d x D
            tmp_sum = n_dbw.sum(axis=1)
            for row,col in n_dbw.keys():
                sum_row = tmp_sum[row,0]
                n_dbw[row,col] = n_dbw[row,col] / sum_row
            p_tw_d = n_db.T

            # Per-topic word distribution, shape V x B_w
            tmp_sum = n_wb.sum(axis=0)
            for row,col in n_wb.keys():
                sum_col = tmp_sum[0,col]
                n_dbw[row,col] = n_wb[row,col] / sum_col
            p_w_tw = n_wb.T
            
            # Collect statistics in dict
            h_t_consensus_summary_by_level[l] = {}
            h_t_consensus_summary_by_level[l]['Bd'] = B_d # Number of document groups
            h_t_consensus_summary_by_level[l]['Bw'] = B_w # Number of word groups
            h_t_consensus_summary_by_level[l]['p_tw_w'] = p_tw_w # Group membership of word nodes
            h_t_consensus_summary_by_level[l]['p_td_d'] = p_td_d # Group membership of document nodes
            h_t_consensus_summary_by_level[l]['p_tw_d'] = p_tw_d # Topic proportions over documents
            h_t_consensus_summary_by_level[l]['p_w_tw'] = p_w_tw # Topic distribution over words

        with gzip.open(f'{results_folder}results_fit_greedy_partitions_consensus_all{filter_label}.pkl.gz','wb') as fp:
            pickle.dump((h_t_doc_consensus_by_level, h_t_word_consensus_by_level, h_t_consensus_summary_by_level),fp)
    
    return h_t_doc_consensus_by_level, h_t_word_consensus_by_level, h_t_consensus_summary_by_level



def get_hierarchy(
    highest_non_trivial_level,
    h_t_doc_consensus_by_level,
    h_t_word_consensus_by_level,
    results_folder,
    filter_label = '',
):
    '''
        Recovers the hierarchy of clusters, both docs and words, from the consensus partition at each level.
        
        Paramaters
        ----------
            highest_non_trivial_level: highest level of the hsbm consensus partition for which there are more than 1 groups (int)
            h_t_doc_consensus_by_level: dict of level as key and an array of length the number of docs in the hsbm with value the cluster at that level (dict, level:np.array)
            h_t_word_consensus_by_level: dict of level as key and an array of length the number of words in the hsbm with value the cluster at that level (dict, level:np.array)
            results_folder: path to directory of the results_folder where to save tokenized_texts_dict (str, valid path)
            filter_label: possible label to use if a special filtering is applied (str)
        
        Returns
        ----------
            hierarchy_docs: dict of dicts (by level), in which, each key is a doc-cluster in the current level and values is the set of doc-clusters under the key doc-cluster at one lower level (dict of dicts)
            hierarchy_words: dict of dicts (by level), in which, each key is a word-cluster in the current level and values is the set of word-clusters under the key word-cluster at one lower level (dict of dicts)
    '''
    # Recover Hierarchy
    try:
        # Try to load if already done
        # ACHTUNG: if you want to calculate it again, you have to remove it (if already done)
        with gzip.open(f'{results_folder}results_fit_greedy_topic_hierarchy_all{filter_label}.pkl.gz','rb') as fp:
            hierarchy_docs,hierarchy_words = pickle.load(fp)
        print('Loaded hierarchy from file', flush=True)
    except FileNotFoundError:
        print('Recovering hierarchy')
        hierarchy_docs, hierarchy_words = {}, {}
        # Compute hierarchy
        for l in range(highest_non_trivial_level,0,-1):
            tmp1_docs, tmp2_docs = h_t_doc_consensus_by_level[l], h_t_doc_consensus_by_level[l-1]
            tmp1_words, tmp2_words = h_t_word_consensus_by_level[l], h_t_word_consensus_by_level[l-1]
            hierarchy_docs[l] = {d: set() for d in np.unique(tmp1_docs)}
            hierarchy_words[l] = {w: set() for w in np.unique(tmp1_words)}
            for i in range(len(tmp1_docs)):
                hierarchy_docs[l][tmp1_docs[i]].add(tmp2_docs[i])
            for i in range(len(tmp1_words)):
                hierarchy_words[l][tmp1_words[i]].add(tmp2_words[i])
        
        try:
            # Add higher layer of hierarchy words so that we have a unique root
            hierarchy_words[highest_non_trivial_level+1] = {0:set(list(hierarchy_words[highest_non_trivial_level].keys()))}
            # Add higher layer of hierarchy docs so that we have a unique root
            hierarchy_docs[highest_non_trivial_level+1] = {0:set(list(hierarchy_docs[highest_non_trivial_level].keys()))}
        except KeyError:
            print(f'ACHTUNG: highest_non_trivial_level is {highest_non_trivial_level}')
            # AHTUNG: We get KeyError if highest_non_trivial_level = 0, so the previous for cycle is skipped!
            tmp1_docs, tmp2_docs = list(np.zeros(len(h_t_doc_consensus_by_level[0]))), h_t_doc_consensus_by_level[0]
            tmp1_words, tmp2_words = list(np.zeros(len(h_t_word_consensus_by_level[0]))), h_t_word_consensus_by_level[0]
            hierarchy_docs[1] = {d: set() for d in np.unique(tmp1_docs)}
            hierarchy_words[1] = {w: set() for w in np.unique(tmp1_words)}
            for i in range(len(tmp1_docs)):
                hierarchy_docs[1][tmp1_docs[i]].add(tmp2_docs[i])
            for i in range(len(tmp1_words)):
                hierarchy_words[1][tmp1_words[i]].add(tmp2_words[i])

        with gzip.open(f'{results_folder}results_fit_greedy_topic_hierarchy_all{filter_label}.pkl.gz','wb') as fp:
            pickle.dump((hierarchy_docs,hierarchy_words),fp)
    
    return (hierarchy_docs,hierarchy_words)