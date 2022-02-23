# IMPORTS

import gzip
import pickle
import pandas as pd
import numpy as np
import os


def assign_partition(
    h_t_doc_consensus_by_level,
    hyperlink_g,
    ordered_paper_ids,
    gt_partition_level,
    all_docs_dict,
):
    '''
        
        
        Args:
            h_t_doc_consensus_by_level: 
            hyperlink_g: 
            ordered_paper_ids: 
            gt_partition_level: 
            all_docs_dict: 
        
        Returns:
            assigned_partition: 
            all_partitions: 
    '''
    hyperlink_text_consensus_partitions_by_level = {}
    for l in h_t_doc_consensus_by_level.keys():
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
            all_docs_dict[paper_id]['assigned_partition'] = [labelling_partition_remapping_by_level[gt_partition_level][doc_partition_remapping[name2partition[paper_id]]]]
        else: 
            all_docs_dict[paper_id]['assigned_partition'] = []


    all_partitions = set([x for x in labelling_partition_remapping_by_level[gt_partition_level].values()])
    all_partitions = list(all_partitions)
    all_partitions.sort()
#     all_partitions = set(all_partitions)
    assigned_partition = {field:set() for field in all_partitions}

    for paper in all_docs_dict.values():
        for field in paper['assigned_partition']:
            assigned_partition[field].add(paper['id'])
    
    return (assigned_partition, all_partitions)
 

def normalize_citations_count(
    all_partitions,
    citing_field,
    cited_field,
    citing_year,
    cited_year,
    knowledge_units_count_per_field_in_time,
    field_units_per_year,
):
    '''
        Normalize citation_count_per_field using YE's prescription (null model)
        
        Args:
            all_partitions: 
            citing_field: 
            cited_field: 
            citing_year: 
            cited_year: 
            knowledge_units_count_per_field_in_time: 
            field_units_per_year: 
        
        Returns:
            float :
    '''

    # fraction of knowledge units received by citing field in citing_year from cited field in cited years over all knowledge units received by citing field in citing years
    numerator = knowledge_units_count_per_field_in_time[cited_field][citing_field][cited_year][citing_year]/np.sum([knowledge_units_count_per_field_in_time[x][citing_field][cited_year][citing_year] for x in knowledge_units_count_per_field_in_time.keys()])

    # fraction of papers produced by cited field in cited year over all papers produced in cited years
    denominator = field_units_per_year[cited_year][cited_field]/np.sum([field_units_per_year[cited_year][field]for field in all_partitions])

    return numerator/denominator if denominator>0 and pd.notna(numerator) else 0


def time_window_average_knowledge_flow(
    citing_field,
    cited_field,
    citing_years,
    cited_years,
    knowledge_flow_normalized_per_field_in_time
):
    '''
        Computes the average knowledge flow of 
        
        Args:
            citing_field: 
            cited_field: 
            citing_years: 
            cited_years: 
            knowledge_flow_normalized_per_field_in_time: 
        
        Returns:
            float: 
    '''
    # TODO: average over time window of field knowledge flow
    # it's the simple mean of the knowledge flow over the considered time window for the cited years given a citing year (and then you average this over a window of citing years)
    tmp = []
    for cited_year in cited_years:
        for citing_year in citing_years:
            if citing_year>cited_year:
                tmp.append(knowledge_flow_normalized_per_field_in_time[cited_field][citing_field][cited_year][citing_year])
    return np.nanmean(tmp)

        
def time_window_average_knowledge_flow_to_future(
    citing_field,
    cited_field,
    cited_years,
    knowledge_flow_normalized_per_field_in_time
):
    '''
        
        
        Args:
            citing_field: 
            cited_field: 
            cited_years: 
            knowledge_flow_normalized_per_field_in_time: 
        
        Returns:
            float:
    '''
    # TODO: average over time window of field knowledge flow
    # it's the simple mean of the knowledge flow over the considered time window for the cited years given a citing year (and then you average this over a window of citing years)
    tmp = []
    for cited_year in cited_years:
        for citing_year in knowledge_flow_normalized_per_field_in_time[cited_field][citing_field][cited_year].keys(): # range(cited_year+1, 2022):
            if citing_year>cited_year:
                tmp.append(knowledge_flow_normalized_per_field_in_time[cited_field][citing_field][cited_year][citing_year])
    return np.nanmean(tmp)  
        

def time_average_knowledge_flow_to_future(
    lev,
    citing_field,
    cited_field,
    cited_year,
    knowledge_flow_normalized_per_field_in_time
):
    '''
        
        Args:
            lev: 
            citing_field: 
            cited_field: 
            cited_year: 
            knowledge_flow_normalized_per_field_in_time: 
        
        Returns:
            float: 
    '''
    tmp = []
    for citing_year in knowledge_flow_normalized_per_field_in_time[cited_field][citing_field][cited_year].keys(): # range(cited_year+1, 2022):
        if citing_year>cited_year:
            tmp.append(knowledge_flow_normalized_per_field_in_time[cited_field][citing_field][cited_year][citing_year])
    return np.nanmean(tmp)
    
    
   
    
def count_knowledge_units_per_field_in_time(
    all_partitions,
    assigned_partition,
    years,
    all_docs_dict,
    all_papers_ids,
    results_folder,
    partition_used,
):
    '''
        
        
        Args:
            all_partitions: 
            assigned_partition: 
            years: 
            all_docs_dict: 
            all_papers_ids: 
            results_folder: 
            partition_used: 
        
        Returns:
            knowledge_units_count_per_field_in_time: 
            field_units_per_year: 
    '''
    knowledge_units_count_per_field_in_time = {partition: {partition: {year: {year: 0 for year in years} for year in years} for partition in all_partitions} for partition in all_partitions}

    # paper2 cites paper1 
    # so citation_count_per_field_in_time is ordered like partition_from - partition_to - year_from - year_to

    # ordered like knowledge_units_count_per_field_in_time[cited_field][citing_field][cited_year][citing_year]

    N_partitions = len(assigned_partition.keys())

    for paper1 in all_docs_dict.values():
        citing_papers = paper1['inCitations']
        for paper2_id in set(citing_papers).intersection(all_papers_ids):
            paper2 = all_docs_dict[paper2_id]
            partitions1 = paper1['assigned_partition']
            partitions2 = paper2['assigned_partition']
            if len(partitions1)>0 and len(partitions2)>0:
                for partition1 in partitions1:
                    for partition2 in partitions2:
                        year1 = paper1['year']
                        year2 = paper2['year']
                        if year1 is not None and year2 is not None:
                            knowledge_units_count_per_field_in_time[partition1][partition2][year1][year2] += (1/len(partitions1))*(1/len(partitions2))

    with gzip.open(os.path.join(results_folder, f'knowledge_units_count_per_field_in_time_{partition_used}.pkl.gz'),'wb') as fp:
        pickle.dump(knowledge_units_count_per_field_in_time,fp)

    field_units_per_year = {year:{field:0 for field in all_partitions} for year in years}

    for paper in all_docs_dict.values():
        if 'year' in paper and paper['year'] is not None and len(paper['assigned_partition']) > 0:
            for field in paper['assigned_partition']:
                field_units_per_year[paper['year']][field] += 1/len(paper['assigned_partition'])
                
    return (knowledge_units_count_per_field_in_time, field_units_per_year)

        
    
    
def compute_knowledge_flow_normalized_per_field_in_time(
    all_partitions,
    years,
    years_with_none,
    knowledge_units_count_per_field_in_time,
    field_units_per_year,
    results_folder,
    partition_used
):
    '''
        
        
        Args:
            all_partitions: 
            years: 
            years_with_none: 
            knowledge_units_count_per_field_in_time: 
            field_units_per_year: 
            results_folder: 
            partition_used: 
        
        Returns:
            knowledge_flow_normalized_per_field_in_time: 
    '''
    knowledge_flow_normalized_per_field_in_time = {partition: {partition: {year: {year: 0 for year in years} for year in years} for partition in all_partitions} for partition in all_partitions}

    # NOTE: we iteratote over years_with_none because it only has years where a decentralization papers has been published, all other years are already initialized with 0
    for cited_field in all_partitions:
        for citing_field in all_partitions:
            for cited_year in years_with_none - {None}:
                for citing_year in years_with_none - {None}:
                    if citing_year < cited_year:
                        continue
                    knowledge_flow_normalized_per_field_in_time[cited_field][citing_field][cited_year][citing_year] = \
                        normalize_citations_count(
                            all_partitions,
                            citing_field,
                            cited_field,
                            citing_year,
                            cited_year,
                            knowledge_units_count_per_field_in_time,
                            field_units_per_year
                        )

    with gzip.open(os.path.join(results_folder, f'knowledge_flow_normalized_per_field_in_time_{partition_used}.pkl.gz'),'wb') as fp:
        pickle.dump(knowledge_flow_normalized_per_field_in_time,fp)
    return knowledge_flow_normalized_per_field_in_time


def compute_knowledge_flow_normalized_per_field_per_time_window(
    all_partitions,
    results_folder,
    partition_used,
    knowledge_flow_normalized_per_field_in_time,
    last_year = 2021,
    first_year = 1962,
):
    '''
        
        
        Args:
            all_partitions: 
            results_folder: 
            partition_used: 
            knowledge_flow_normalized_per_field_in_time: 
            last_year: 
            first_year: 
        
        Returns:
            knowledge_flow_normalized_per_field_per_time_window: 
    '''
    time_window_size_range = [5,10]
    
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
                        tmp_dict[cited_field][citing_field] = \
                            time_window_average_knowledge_flow(
                                citing_field,
                                cited_field,
                                range(citing_time_window[0],citing_time_window[1]+1),
                                range(cited_time_window[0],cited_time_window[1]+1),
                                knowledge_flow_normalized_per_field_in_time
                            )

    with gzip.open(os.path.join(results_folder, f'knowledge_flow_normalized_per_field_per_time_window_{partition_used}.pkl.gz'),'wb') as fp:
        pickle.dump(knowledge_flow_normalized_per_field_per_time_window,fp)
    
    return knowledge_flow_normalized_per_field_per_time_window
        
    
def compute_knowledge_flow_normalized_per_field_per_time_window_to_future(
    all_partitions,
    results_folder,
    partition_used,
    knowledge_flow_normalized_per_field_in_time,
    last_year = 2021,
    first_year = 1962,
):
    '''
        
        
        Args:
            all_partitions: 
            results_folder: 
            partition_used: 
            knowledge_flow_normalized_per_field_in_time: 
            last_year: 
            first_year: 
        
        Returns:
            knowledge_flow_normalized_per_field_per_time_window_to_future
    '''
    time_window_size_range = [5,10]

    knowledge_flow_normalized_per_field_per_time_window_to_future = {time_window_size: {(final_year-time_window_size+1,final_year): {'future': {partition: {partition: 0 for partition in all_partitions} for partition in all_partitions}} for final_year in range(last_year,first_year,-time_window_size)} for time_window_size in time_window_size_range}

    for time_window_size in knowledge_flow_normalized_per_field_per_time_window_to_future.keys():
        for cited_time_window,tmp_dict in knowledge_flow_normalized_per_field_per_time_window_to_future[time_window_size].items():
            for cited_field in all_partitions:
                for citing_field in all_partitions:
                    tmp_dict['future'][cited_field][citing_field] = \
                        time_window_average_knowledge_flow_to_future(
                            citing_field,
                            cited_field,
                            range(cited_time_window[0],cited_time_window[1]+1),
                            knowledge_flow_normalized_per_field_in_time
                        )

    with gzip.open(os.path.join(results_folder, f'knowledge_flow_normalized_per_field_per_time_window_to_future_{partition_used}.pkl.gz'),'wb') as fp:
        pickle.dump(knowledge_flow_normalized_per_field_per_time_window_to_future,fp)

    return knowledge_flow_normalized_per_field_per_time_window_to_future




def compute_knowledge_flow_normalized_per_field_in_time_df(
    lev,
    knowledge_flow_normalized_per_field_in_time,
    results_folder,
    partition_used
):
    '''
        
        Args:
            lev
            knowledge_flow_normalized_per_field_in_time
            results_folder
            partition_used
        
        Returns:
            knowledge_flow_normalized_per_field_in_time_df: 
    '''
    knowledge_flow_normalized_per_field_in_time_to_future = {}
    for cluster_from in knowledge_flow_normalized_per_field_in_time.keys():
        knowledge_flow_normalized_per_field_in_time_to_future[cluster_from] = {}
        for cluster_to in knowledge_flow_normalized_per_field_in_time[cluster_from].keys():
            knowledge_flow_normalized_per_field_in_time_to_future[cluster_from][cluster_to] = {}
            for year_from in knowledge_flow_normalized_per_field_in_time[cluster_from][cluster_to].keys():
                knowledge_flow_normalized_per_field_in_time_to_future[cluster_from][cluster_to][year_from] = \
                    time_average_knowledge_flow_to_future(
                        lev,
                        cluster_to,
                        cluster_from,
                        year_from,
                        knowledge_flow_normalized_per_field_in_time
                    )

    knowledge_flow_normalized_per_field_in_time_df = {}

    _t_from = []
    _t_to = []
    _from = []
    _to = []
    k = []

    for cluster_from in knowledge_flow_normalized_per_field_in_time.keys():
        for cluster_to in knowledge_flow_normalized_per_field_in_time[cluster_from].keys():
            for year_from in knowledge_flow_normalized_per_field_in_time[cluster_from][cluster_to].keys():
                for year_to in knowledge_flow_normalized_per_field_in_time[cluster_from][cluster_to][year_from].keys():
                    if year_to >= year_from:
                        _t_from.append(year_from)
                        _t_to.append(year_to)
                        _from.append(cluster_from)
                        _to.append(cluster_to)
                        k.append(knowledge_flow_normalized_per_field_in_time[cluster_from][cluster_to][year_from][year_to])

    knowledge_flow_normalized_per_field_in_time_df = pd.DataFrame({
        'year_from':_t_from,
        'year_to':_t_to,
        'cluster_from':_from,
        'cluster_to':_to,
        'knowledge_flow':k
    })

    knowledge_flow_normalized_per_field_in_time_df.cluster_from = knowledge_flow_normalized_per_field_in_time_df.cluster_from.astype(int)
    knowledge_flow_normalized_per_field_in_time_df.cluster_to = knowledge_flow_normalized_per_field_in_time_df.cluster_to.astype(int)
    
    
    knowledge_flow_normalized_per_field_in_time_df.to_csv(os.path.join(results_folder,f'knowledge_flow_normalized_per_field_in_time_df_{partition_used}.csv'), index=False)

    return knowledge_flow_normalized_per_field_in_time_df


def run_knowledge_flow_analysis(
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
):
    '''
        
        
        Args:
            all_docs_dict
            h_t_doc_consensus_by_level
            hyperlink_g
            ordered_paper_ids
            first_level_knowledge_flow
            highest_non_trivial_level
            dataset_path
            results_folder
            last_year
            first_year
        
        Returns:
            None
    '''
    paper2field = {x['id']:x['fieldsOfStudy'] for x in all_docs_dict.values() if 'id' in x and 'fieldsOfStudy' in x}

    for gt_partition_level in range(first_level_knowledge_flow,highest_non_trivial_level + 1):
        print(f'level {gt_partition_level}')
        
        # Load the correct partitions in the key 'assigned_partition' for each paper
        partition_used = 'gt_partition_lev_%d'%(gt_partition_level)
        assigned_partition, all_partitions = \
            assign_partition(
                h_t_doc_consensus_by_level,
                hyperlink_g,
                ordered_paper_ids,
                gt_partition_level,
                all_docs_dict,
            )


        # count citations between fields and between different times
        # citation_count_per_field_in_time = {partition: {partition: {year: {year: 0 for year in years_with_none} for year in years_with_none} for partition in all_partitions} for partition in all_partitions}
        with gzip.open(dataset_path + '../' +'no_papers_in_fields_by_year.pkl.gz','rb') as fp:
            papers_per_year_per_field = pickle.load(fp)

        years = sorted([x for x in papers_per_year_per_field.keys() if x is not None and x > 1500])
        years_with_none = set([x['year'] for x in all_docs_dict.values()])
        all_papers_ids = set([x['id'] for x in all_docs_dict.values()])


        knowledge_units_count_per_field_in_time, field_units_per_year = \
            count_knowledge_units_per_field_in_time(
                all_partitions,
                assigned_partition,
                years,
                all_docs_dict,
                all_papers_ids,
                results_folder,
                partition_used,
            )

        knowledge_flow_normalized_per_field_in_time = \
            compute_knowledge_flow_normalized_per_field_in_time(
                all_partitions,
                years,
                years_with_none,
                knowledge_units_count_per_field_in_time,
                field_units_per_year,
                results_folder,
                partition_used
            )

        knowledge_flow_normalized_per_field_per_time_window = \
            compute_knowledge_flow_normalized_per_field_per_time_window(
                all_partitions,
                results_folder,
                partition_used,
                knowledge_flow_normalized_per_field_in_time,
                last_year = last_year,
                first_year = first_year,
            )

        knowledge_flow_normalized_per_field_per_time_window_to_future = \
            compute_knowledge_flow_normalized_per_field_per_time_window_to_future(
                all_partitions,
                results_folder,
                partition_used,
                knowledge_flow_normalized_per_field_in_time,
                last_year = last_year,
                first_year = first_year
            )

        knowledge_flow_normalized_per_field_in_time_df = \
            compute_knowledge_flow_normalized_per_field_in_time_df(
                gt_partition_level,
                knowledge_flow_normalized_per_field_in_time,
                results_folder,
                partition_used
            )

    return None