# IMPORTS

import gzip
import pickle
import pandas as pd
import numpy as np
import os


def get_topics_consensus_model(
    consensus_summary, 
    words, 
    n=10, 
):
    '''
        Retrieve topics in consensus partition for hsbm model.
        
        Args:
            consensus_summary: summary of the consensus partition at a certain level, with info about the blocks and the topic proportions, see hsbm_partitions.get_consensus() (dict)
            words: ordered list of words in the hyperlink network (list)
            n: number of words for each topic
        
        Returns:
            dict_group_words: dict of topic: list of n most important words according to the topic proportion in consensus_summary (dict {int:[(word,p_w)]})
    '''
    Bw = consensus_summary['Bw'] # number of word-groups
    p_w_tw = consensus_summary['p_w_tw'] # topic proportions over words
    # Loop over all word-groups
    dict_group_words = {}
    for tw in range(Bw):
        p_w_ = p_w_tw[:, tw] # get word importance proportion in the selected topic
        ind_w_ = np.argsort(p_w_)[::-1] # order words in the topic
        list_words_tw = [] # ordered list of important words
        for i in ind_w_[:n]:
            if p_w_[i] > 0:
                list_words_tw+=[(words[i],p_w_[i])] # add the word
            else:
                break
        dict_group_words[tw] = list_words_tw
    return dict_group_words    


def topic_mixture_proportion(
    dict_groups, 
    edited_texts, 
    document_partitions
):
    '''
        For each pair of doc group (cluster) and text group (topic), computes the mixture proportion representing how much the topic is present in the cluster with value between 0 and 1.
        Calculates also the normalized mixture proportion, looking at the ixture proportion of the same topic in the other clusters, 
        to see how much more (or less) that topic is present in the cluster with respect to the others.
        Also calculates the average topic frequency among all clusters.
        
        
        Args:
            dict_groups: dict of topic: list of n most important words according to the topic proportion in consensus_summary (dict {int:[(word,p_w)]})
            edited_texts: list of the filtered texts, ordered with the same ordering of the doc network (list of lists)
            document_partitions: 1D array long as the number of docs, where for each doc the value is the cluster the doc belongs to (array)
        
        Returns:
            mixture_proportion: nested dict where for each cluster of docs and topic the mixture proportion is stored (nested dict {str:{str:float}})
            normalized_mixture_proportion: nested dict where for each cluster of docs and topic the normalized mixture proportion is stored (nested dict {str:{str:float}})
            avg_topic_frequency: dict where for each topic the average frequency among all clusters is stored (dict {str:float})
    '''
    # Initialization
    topics = dict_groups.keys()
    partitions = np.unique(document_partitions)

    avg_topic_frequency = {}
    mixture_proportion = {}
    normalized_mixture_proportion = {}

    doc_texts = np.array(edited_texts, dtype=object)

    n_i_t = {} # store the number of words of each topic in each doc cluster

    topic_doc_group_words = {}
    
    for doc_group in partitions:
        # get all words in this doc_group
        topic_doc_group_words[doc_group] = set()
        for i,doc_group_membership in enumerate(document_partitions):
            if doc_group_membership != doc_group:
                continue
            topic_doc_group_words[doc_group] = topic_doc_group_words[doc_group].union(set(edited_texts[i]))

    for topic in topics:
        topic_words = set([x[0] for x in dict_groups[topic]]) # get all the important words in the topic
        n_i_t[topic] = {}
        for doc_group in partitions:
            # get the number of words in the topic present in the cluster
            n_i_t[topic][doc_group] = len(topic_words.intersection(topic_doc_group_words[doc_group]))

    for doc_group in partitions:
        mixture_proportion[f'doc_group {doc_group}'] = {}
        for topic in topics:
            # Calculate mixture proportion
            mixture_proportion[f'doc_group {doc_group}'][f'topic {topic}'] = n_i_t[topic][doc_group] / np.sum([n_i_t[topic_j][doc_group] for topic_j in topics])

    # calculate total frequency of words in the clusters
    S = np.sum([n_i_t[topic_j][doc_group] for doc_group in partitions for topic_j in topics])
    for topic in topics:
        # Calculate the average topic frequency
        avg_topic_frequency[f'topic {topic}'] = np.sum([n_i_t[topic][doc_group] for doc_group in partitions]) / S

    for doc_group in partitions:
        normalized_mixture_proportion[f'doc_group {doc_group}'] = {}
        for topic in topics:
            # Calculate normalized mixture proportion
            normalized_mixture_proportion[f'doc_group {doc_group}'][f'topic {topic}'] = ( mixture_proportion[f'doc_group {doc_group}'][f'topic {topic}'] - avg_topic_frequency[f'topic {topic}'] ) / avg_topic_frequency[f'topic {topic}']

    return (mixture_proportion, normalized_mixture_proportion, avg_topic_frequency)


def get_topics(
    dir_list,
    h_t_consensus_summary_by_level,
    h_t_doc_consensus_by_level,
):
    '''
        Get the words, important words, and a DataFrame representation of each topic and its important words, in each level.
        
        Args:
            dir_list: list of all the paths to the directories where all iterations results have been dumped (list of str, valid paths)
            h_t_consensus_summary_by_level: summary of the consensus partition at a certain level, with info about the blocks and the topic proportions, see hsbm_partitions.get_consensus() (dict)
            h_t_doc_consensus_by_level: dict of level as key and an array of length the number of docs in the hsbm with value the cluster at that level (dict, level:np.array)
        
        Returns:
            g_words: ordered list of words in the hyperlink network (list)
            dict_groups_by_level: dict of dicts (for each level) topic: list of n most important words according to the topic proportion in consensus_summary (dict {int:{int:[(word,p_w)]}})
            topics_df_by_level: dict with level:DataFrame with the important words for each topic (dict {int:pd.DataFrame})
    '''
    print('Loading a state', flush=True)
    for dir_ in dir_list:
        try:
            with gzip.open(os.path.join(dir_,f'results_fit_greedy.pkl.gz'),'rb') as fp:
                hyperlink_text_hsbm_states,time_duration = pickle.load(fp)
            print(f'Loaded state from dir {dir_}', flush=True)
            break
        except FileNotFoundError as e:
            continue
    
    # Get words in the state
    g_words = [hyperlink_text_hsbm_states[0].g.vp['name'][v] for v in hyperlink_text_hsbm_states[0].g.vertices() if hyperlink_text_hsbm_states[0].g.vp['kind'][v]==1]
    # Get most important words in each topic at each level
    # n=1000000 to take into consideration all words in the topic
    dict_groups_by_level = {l:get_topics_consensus_model(h_t_consensus_summary_by_level[l], g_words, n=1000000) for l in h_t_doc_consensus_by_level.keys()}
    
    # Create dataframe for each level, where for each topic the important words are shown
    topics_df_by_level = {}
    for l in h_t_doc_consensus_by_level.keys():
        # Write out topics as dataframe
        topic_csv_dict = {}
        for topic in dict_groups_by_level[l].keys():
            # Get important words in the topic
            topic_csv_dict[topic] = [entry[0] for entry in dict_groups_by_level[l][topic]]

        keys = topic_csv_dict.keys()
        topics_df_by_level[l] = pd.DataFrame()

        for key in dict_groups_by_level[l].keys():
            # Build the dataframe
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
    '''
        Load or compute mixture proportion measures between topics and clusters at the same level on the hsbm.
        
        Args:
            h_t_doc_consensus_by_level: dict of level as key and an array of length the number of docs in the hsbm with value the cluster at that level (dict, level:np.array)
            dict_groups_by_level: dict of dicts (for each level) topic: list of n most important words according to the topic proportion in consensus_summary (dict {int:{int:[(word,p_w)]}})
            ordered_edited_texts: list of the filtered texts, ordered with the same ordering of the doc network (list of lists)
            topics_df_by_level: dict with level:DataFrame with the important words for each topic (dict {int:pd.DataFrame})
            results_folder: path to directory of the results_folder where to save tokenized_texts_dict (str, valid path)
            filter_label: possible label to use if a special filtering is applied (str)
        
        Returns:
            mixture_proportion_by_level: nested dict where, at each level, for each cluster of docs and topic the mixture proportion is stored (nested dict {int:{str:{str:float}}})
            normalized_mixture_proportion_by_level: nested dict where, at each level, for each cluster of docs and topic the normalized mixture proportion is stored (nested dict {int:{str:{str:float}}})
            avg_topic_frequency_by_level: dict where, at each level, for each topic the average frequency among all clusters is stored (dict {int:{str:float}})
    '''
    try:
        # Try to load
        with gzip.open(f'{results_folder}results_fit_greedy_topic_frequency_all{filter_label}.pkl.gz','rb') as fp:
            topics_df_by_level,mixture_proportion_by_level, normalized_mixture_proportion_by_level, avg_topic_frequency_by_level = pickle.load(fp)
    except FileNotFoundError:
        # Calculate it
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
    '''
        Load or compute mixture proportion measures between topics and clusters across different levels between clusters of docs and topics of words.
        
        Args:
            h_t_doc_consensus_by_level: dict of level as key and an array of length the number of docs in the hsbm with value the cluster at that level (dict, level:np.array)
            dict_groups_by_level: dict of dicts (for each level) topic: list of n most important words according to the topic proportion in consensus_summary (dict {int:{int:[(word,p_w)]}})
            ordered_edited_texts: list of the filtered texts, ordered with the same ordering of the doc network (list of lists)
            highest_non_trivial_level: highest level of the hsbm consensus partition for which there are more than 1 groups (int)
            topics_df_by_level: dict with level:DataFrame with the important words for each topic (dict {int:pd.DataFrame})
            results_folder: path to directory of the results_folder where to save tokenized_texts_dict (str, valid path)
            filter_label: possible label to use if a special filtering is applied (str)
        
        Returns:
            mixture_proportion_by_level_partition_by_level_topics: nested dict where, for each pair of level of cluster and level of topic, 
                for each cluster of docs and topic the mixture proportion is stored (nested dict {int:{int{str:{str:float}}}})
            normalized_mixture_proportion_by_level_partition_by_level_topics: nested dict where, for each pair of level of cluster and level of topic, 
                for each cluster of docs and topic the normalized mixture proportion is stored (nested dict {int:{int{str:{str:float}}}})
            avg_topic_frequency_by_level_partition_by_level_topics: dict where, for each pair of level of cluster and level of topic, 
                for each topic the average frequency among all clusters is stored (dict {int:{int{str:float}}})
    '''
    try:
        # Try to load it
        with gzip.open(os.path.join(results_folder, f'results_fit_greedy_topic_frequency_all_by_level_partition_by_level_topics{filter_label}_all.pkl.gz'),'rb') as fp:
            topics_df_by_level,mixture_proportion_by_level_partition_by_level_topics, normalized_mixture_proportion_by_level_partition_by_level_topics, avg_topic_frequency_by_level_partition_by_level_topics = pickle.load(fp)
    except FileNotFoundError:
        # Calculate it
        mixture_proportion_by_level_partition_by_level_topics, normalized_mixture_proportion_by_level_partition_by_level_topics, avg_topic_frequency_by_level_partition_by_level_topics = {}, {}, {}
        for level_partition in range(highest_non_trivial_level + 1):
            # at each level for the docs
            mixture_proportion_by_level_partition_by_level_topics[level_partition], normalized_mixture_proportion_by_level_partition_by_level_topics[level_partition], avg_topic_frequency_by_level_partition_by_level_topics[level_partition] = {}, {}, {}
            for l in range(highest_non_trivial_level + 1):
                # at each level for the words
                mixture_proportion_by_level_partition_by_level_topics[level_partition][l], normalized_mixture_proportion_by_level_partition_by_level_topics[level_partition][l], avg_topic_frequency_by_level_partition_by_level_topics[level_partition][l] = \
                    topic_mixture_proportion(dict_groups_by_level[l],ordered_edited_texts,h_t_doc_consensus_by_level[level_partition])
        # Dump it
        with gzip.open(os.path.join(results_folder, f'results_fit_greedy_topic_frequency_all_by_level_partition_by_level_topics{filter_label}_all.pkl.gz'),'wb') as fp:
            pickle.dump((topics_df_by_level,mixture_proportion_by_level_partition_by_level_topics, normalized_mixture_proportion_by_level_partition_by_level_topics, avg_topic_frequency_by_level_partition_by_level_topics),fp)
    
    return (mixture_proportion_by_level_partition_by_level_topics, normalized_mixture_proportion_by_level_partition_by_level_topics, avg_topic_frequency_by_level_partition_by_level_topics)
