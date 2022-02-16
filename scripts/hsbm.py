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


# TODO: wrappare in funzione che restituisce ...  fino a mixture proportion
# Fare nuova utils, tipo analysis_hsbm
print('\nGet all topics by level',flush=True)
g_words = [ hyperlink_text_hsbm_states[0].g.vp['name'][v] for v in hyperlink_text_hsbm_states[0].g.vertices() if hyperlink_text_hsbm_states[0].g.vp['kind'][v]==1   ]
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


# ## Topic frequency in clusters
print('\nMixture proportion',flush=True)
start = datetime.now()
mixture_proportion_by_level, normalized_mixture_proportion_by_level, avg_topic_frequency_by_level = {}, {}, {}
for l in h_t_doc_consensus_by_level.keys():
    mixture_proportion_by_level[l], normalized_mixture_proportion_by_level[l], avg_topic_frequency_by_level[l] = topic_mixture_proportion(dict_groups_by_level[l],ordered_edited_texts,h_t_doc_consensus_by_level[l])
with gzip.open(f'{results_folder+analysis_results_subfolder}results_fit_greedy_topic_frequency_all{filter_label}.pkl.gz','wb') as fp:
    pickle.dump((topics_df_by_level,mixture_proportion_by_level, normalized_mixture_proportion_by_level, avg_topic_frequency_by_level),fp)
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

    
    
    
    
# TODO: wrappare in funzione che computa la Normalized mixture proportion between different levels
# Sistemare i nomi (attento ai files gia' creati)
    
    
# Normalized mixture proportion by different level partition and topic

print('\nNormalized mixture proportion between different levels')
start = datetime.now()
try:
    with gzip.open(os.path.join(results_folder+analysis_results_subfolder, f'results_fit_greedy_topic_frequency_all_by_level_partition_by_level_topics{filter_label}_all.pkl.gz'),'rb') as fp:
        topics_df_by_level,mixture_proportion_by_level_partition_by_level_topics, normalized_mixture_proportion_by_level_partition_by_level_topics, avg_topic_frequency_by_level_partition_by_level_topics = pickle.load(fp)
except FileNotFoundError:
    mixture_proportion_by_level_partition_by_level_topics, normalized_mixture_proportion_by_level_partition_by_level_topics, avg_topic_frequency_by_level_partition_by_level_topics = {}, {}, {}
    for level_partition in range(highest_non_trivial_level + 1):
        mixture_proportion_by_level_partition_by_level_topics[level_partition], normalized_mixture_proportion_by_level_partition_by_level_topics[level_partition], avg_topic_frequency_by_level_partition_by_level_topics[level_partition] = {}, {}, {}
        for l in range(highest_non_trivial_level + 1):
            mixture_proportion_by_level_partition_by_level_topics[level_partition][l], normalized_mixture_proportion_by_level_partition_by_level_topics[level_partition][l], avg_topic_frequency_by_level_partition_by_level_topics[level_partition][l] = \
                topic_mixture_proportion(dict_groups_by_level[l],ordered_edited_texts,h_t_doc_consensus_by_level[level_partition])

    with gzip.open(os.path.join(results_folder+analysis_results_subfolder, f'results_fit_greedy_topic_frequency_all_by_level_partition_by_level_topics{filter_label}_all.pkl.gz'),'wb') as fp:
        pickle.dump((topics_df_by_level,mixture_proportion_by_level_partition_by_level_topics, normalized_mixture_proportion_by_level_partition_by_level_topics, avg_topic_frequency_by_level_partition_by_level_topics),fp)
end = datetime.now()
print('Time duration',end-start,flush=True)

        
        
        
        
        
        
        
        
        
# KNOWLEDGE FLOW
print('\nStarting knowledge flow calculations')
start = datetime.now()
all_docs = list(all_docs_dict.values())

paper2field = {x['id']:x['fieldsOfStudy'] for x in all_docs if 'id' in x and 'fieldsOfStudy' in x}

# all_docs_dict[paper]['ye_partition'] is a list of all the fields of the paper!!

for gt_partition_level in range(first_level_knowledge_flow,highest_non_trivial_level + 1):
    
    # Load the correct partitions in the key 'ye_partition' for each paper
    
    partition_used = 'gt_partition_lev_%d'%(gt_partition_level)
    hyperlink_text_consensus_partitions_by_level = {}
    for l in h_t_doc_consensus_by_level.keys():
        print(l,flush=True)
        hyperlink_text_consensus_partitions_by_level[l] = h_t_doc_consensus_by_level[l]


    name2partition = {}
    for i,name in enumerate(hyperlink_g.vp['name']):
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
            print('THERE ARE MULTIPLE INSTANCES... ERROR')
            break
        else:
            doc_partition_remapping[part1] = part2  
            doc_partition_remapping_inverse[part2] = part1

    labelling_partition_remapping_by_level = {gt_partition_level:{x:str(x) for x in doc_partition_remapping.values()}}
    labelling_partition_remapping_by_level_inverse = {level:{y:x for x,y in labelling_partition_remapping_by_level[level].items()} for level in labelling_partition_remapping_by_level}


    print('Assigning partition field in docs')

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
    with gzip.open(dataset_path + '../' +'no_papers_in_fields_by_year.pkl.gz','rb') as fp:
        papers_per_year_per_field = pickle.load(fp)
    years = sorted([x for x in papers_per_year_per_field.keys() if x is not None and x > 1500])

    knowledge_units_count_per_field_in_time = {partition: {partition: {year: {year: 0 for year in years} for year in years} for partition in all_partitions} for partition in all_partitions}


    all_papers_ids = set([x['id'] for x in all_docs])

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

    with gzip.open(os.path.join(results_folder+analysis_results_subfolder, f'knowledge_units_count_per_field_in_time_{partition_used}.pkl.gz'),'wb') as fp:
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

    with gzip.open(os.path.join(results_folder+analysis_results_subfolder, f'knowledge_flow_normalized_per_field_in_time_{partition_used}.pkl.gz'),'wb') as fp:
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

    with gzip.open(os.path.join(results_folder+analysis_results_subfolder, f'knowledge_flow_normalized_per_field_per_time_window_{partition_used}.pkl.gz'),'wb') as fp:
        pickle.dump(knowledge_flow_normalized_per_field_per_time_window,fp)



    # TODO: average over time window of field knowledge flow
    # it's the simple mean of the knowledge flow over the considered time window for the cited years given a citing year (and then you average this over a window of citing years)

    def time_window_average_knowledge_flow_to_future(citing_field,cited_field,cited_years):

        tmp = []
        for cited_year in cited_years:
            for citing_year in knowledge_flow_normalized_per_field_in_time[cited_field][citing_field][cited_year].keys(): # range(cited_year+1, 2022):
                if citing_year>cited_year:
                    tmp.append(knowledge_flow_normalized_per_field_in_time[cited_field][citing_field][cited_year][citing_year])
        return np.nanmean(tmp)

    time_window_size_range = [5,10]
    last_year = 2021
    first_year = 1962

    knowledge_flow_normalized_per_field_per_time_window_to_future = {time_window_size: {(final_year-time_window_size+1,final_year): {'future': {partition: {partition: 0 for partition in all_partitions} for partition in all_partitions}} for final_year in range(last_year,first_year,-time_window_size)} for time_window_size in time_window_size_range}

    for time_window_size in knowledge_flow_normalized_per_field_per_time_window_to_future.keys():
        for cited_time_window,tmp_dict in knowledge_flow_normalized_per_field_per_time_window_to_future[time_window_size].items():
            for cited_field in all_partitions:
                for citing_field in all_partitions:
                    tmp_dict['future'][cited_field][citing_field] = time_window_average_knowledge_flow_to_future(citing_field,cited_field,range(cited_time_window[0],cited_time_window[1]+1))

    with gzip.open(os.path.join(results_folder+analysis_results_subfolder, f'knowledge_flow_normalized_per_field_per_time_window_to_future_{partition_used}.pkl.gz'),'wb') as fp:
        pickle.dump(knowledge_flow_normalized_per_field_per_time_window_to_future,fp)

        
    print(f'level {gt_partition_level} Knowledge flow to be put in a dict by lev', flush=True)

    knowledge_flow_normalized_per_field_per_time_window_to_future_by_level = {}
    knowledge_flow_normalized_per_field_in_time_by_level = {}
    knowledge_flow_normalized_per_field_per_time_window_by_level = {}
    
    lev = gt_partition_level
    
    with gzip.open(os.path.join(results_folder+analysis_results_subfolder, f'knowledge_flow_normalized_per_field_per_time_window_to_future_{partition_used}.pkl.gz'),'rb') as fp:
        knowledge_flow_normalized_per_field_per_time_window_to_future_by_level[lev] = pickle.load(fp)
    with gzip.open(os.path.join(results_folder+analysis_results_subfolder, f'knowledge_flow_normalized_per_field_in_time_{partition_used}.pkl.gz'),'rb') as fp:
        knowledge_flow_normalized_per_field_in_time_by_level[lev] = pickle.load(fp)
    with gzip.open(os.path.join(results_folder+analysis_results_subfolder, f'knowledge_flow_normalized_per_field_per_time_window_{partition_used}.pkl.gz'),'rb') as fp:
        knowledge_flow_normalized_per_field_per_time_window_by_level[lev] = pickle.load(fp)

    def time_average_knowledge_flow_to_future(lev,citing_field,cited_field,cited_year):

        tmp = []

        for citing_year in knowledge_flow_normalized_per_field_in_time_by_level[lev][cited_field][citing_field][cited_year].keys(): # range(cited_year+1, 2022):
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
    
    
    knowledge_flow_normalized_per_field_in_time_by_level_df[lev].to_csv(os.path.join(results_folder+analysis_results_subfolder,f'knowledge_flow_normalized_per_field_in_time_df_gt_partition_lev_{lev}.csv'), index=False)

end = datetime.now()
print('Time duration',end-start,flush=True)

print('FINISHED')