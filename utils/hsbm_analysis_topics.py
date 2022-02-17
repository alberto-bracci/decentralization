# IMPORTS

import gzip
import pickle
import pandas as pd
import numpy as np
import os
# move to repo root directory
# from datetime import datetime
# import sys
# sys.path.insert(0, os.path.join(os.getcwd(),"utils"))
# from hsbm import sbmmultilayer 
# from hsbm.utils.nmi import *
# from hsbm.utils.doc_clustering import *

# import gi
# from gi.repository import Gtk, Gdk
# import graph_tool.all as gt
# import ast # to get list comprehension from a string
# import scipy


# import functools, builtins # to impose flush=True on every print
# builtins.print = functools.partial(print, flush=True)



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

def topic_mixture_proportion(dict_groups, edited_text, document_partitions):

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

    return (mixture_proportion, normalized_mixture_proportion, avg_topic_frequency)


def get_topics(
    hyperlink_text_hsbm_states,
    h_t_consensus_summary_by_level,
    h_t_doc_consensus_by_level,
):
    g_words = [hyperlink_text_hsbm_states[0].g.vp['name'][v] for v in hyperlink_text_hsbm_states[0].g.vertices() if hyperlink_text_hsbm_states[0].g.vp['kind'][v]==1   ]
    dict_groups_by_level = {l:get_topics_h_t_consensus_model(h_t_consensus_summary_by_level[l], g_words, n=1000000) for l in h_t_doc_consensus_by_level.keys()}

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
    
    return (g_words, dict_groups_by_level, topics_df_by_level)


def get_mixture_proportion(
    h_t_doc_consensus_by_level, 
    dict_groups_by_level, 
    ordered_edited_texts,
    topics_df_by_level,
    results_folder,
    filter_label = ''
):
    try:
        with gzip.open(f'{results_folder}results_fit_greedy_topic_frequency_all{filter_label}.pkl.gz','rb') as fp:
            topics_df_by_level,mixture_proportion_by_level, normalized_mixture_proportion_by_level, avg_topic_frequency_by_level = pickle.load(fp)
    except FileNotFoundError:
        mixture_proportion_by_level, normalized_mixture_proportion_by_level, avg_topic_frequency_by_level = {}, {}, {}
        for l in h_t_doc_consensus_by_level.keys():
            mixture_proportion_by_level[l], normalized_mixture_proportion_by_level[l], avg_topic_frequency_by_level[l] = \
                topic_mixture_proportion(dict_groups_by_level[l],ordered_edited_texts,h_t_doc_consensus_by_level[l])
        with gzip.open(f'{results_folder}results_fit_greedy_topic_frequency_all{filter_label}.pkl.gz','wb') as fp:
            pickle.dump((topics_df_by_level,mixture_proportion_by_level, normalized_mixture_proportion_by_level, avg_topic_frequency_by_level),fp)
    
    return (mixture_proportion_by_level, normalized_mixture_proportion_by_level, avg_topic_frequency_by_level)



def get_mixture_proportion_by_level(
    h_t_doc_consensus_by_level, 
    dict_groups_by_level, 
    ordered_edited_texts,
    topics_df_by_level,
    highest_non_trivial_level,
    results_folder,
    filter_label = ''
):
    try:
        with gzip.open(os.path.join(results_folder, f'results_fit_greedy_topic_frequency_all_by_level_partition_by_level_topics{filter_label}_all.pkl.gz'),'rb') as fp:
            topics_df_by_level,mixture_proportion_by_level_partition_by_level_topics, normalized_mixture_proportion_by_level_partition_by_level_topics, avg_topic_frequency_by_level_partition_by_level_topics = pickle.load(fp)
    except FileNotFoundError:
        mixture_proportion_by_level_partition_by_level_topics, normalized_mixture_proportion_by_level_partition_by_level_topics, avg_topic_frequency_by_level_partition_by_level_topics = {}, {}, {}
        for level_partition in range(highest_non_trivial_level + 1):
            mixture_proportion_by_level_partition_by_level_topics[level_partition], normalized_mixture_proportion_by_level_partition_by_level_topics[level_partition], avg_topic_frequency_by_level_partition_by_level_topics[level_partition] = {}, {}, {}
            for l in range(highest_non_trivial_level + 1):
                mixture_proportion_by_level_partition_by_level_topics[level_partition][l], normalized_mixture_proportion_by_level_partition_by_level_topics[level_partition][l], avg_topic_frequency_by_level_partition_by_level_topics[level_partition][l] = \
                    topic_mixture_proportion(dict_groups_by_level[l],ordered_edited_texts,h_t_doc_consensus_by_level[level_partition])

        with gzip.open(os.path.join(results_folder, f'results_fit_greedy_topic_frequency_all_by_level_partition_by_level_topics{filter_label}_all.pkl.gz'),'wb') as fp:
            pickle.dump((topics_df_by_level,mixture_proportion_by_level_partition_by_level_topics, normalized_mixture_proportion_by_level_partition_by_level_topics, avg_topic_frequency_by_level_partition_by_level_topics),fp)
    
    return (mixture_proportion_by_level_partition_by_level_topics, normalized_mixture_proportion_by_level_partition_by_level_topics, avg_topic_frequency_by_level_partition_by_level_topics)
