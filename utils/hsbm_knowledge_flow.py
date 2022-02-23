# IMPORTS

import gzip
import pickle
import pandas as pd
import numpy as np
import os

def assign_partition(
    h_t_doc_consensus_by_level,
    hyperlink_g,
    gt_partition_level,
    all_docs_dict,
):
    '''
        Consider the partition at level gt_partition_level created with the hsbm, and assign the correct partition to the papers in all_docs_dict.
        Papers in all_docs_dict without a partition are assigned to an empty list.

        Args:
            h_t_doc_consensus_by_level: dict of level as key and an array of length the number of docs in the hsbm with value the cluster at that level (dict, level:np.array)
            hyperlink_g: gt network containing all papers and their links, i.e., citations or hyperlinks (gt.Graph)
            ordered_paper_ids: ordered list of paper_ids in hyperlink_g, with the same ordering as hyperlink_g.vp['name'] (list of str, i.e., list(hyperlink_g.vp['name']))
            gt_partition_level: level of hsbm at which to take the partition (int)
            all_docs_dict: dictionary with id of paper as key and a dict with all the info as value (dict of dicts)
        
        Returns:
            assigned_cluster_dict: dict of all clusters as keys and the set of all papers of the cluster as value (dict of cluster:set(papers in clusters))
            all_clusters: sorted list of all the clusters in the considered partition (list of int)
    '''
    # Create name2partition, where name is taken from hyperlink_g and the partition is taken from the correct level of the hsbm doc_consensus
    name2partition = {}
    for i,name in enumerate(hyperlink_g.vp['name']):
        name2partition[name] = h_t_doc_consensus_by_level[gt_partition_level][i]
    paper_ids_with_partition = set(name2partition.keys())
    
    print('Assigning partition in all_docs_docs')

    for paper_id in all_docs_dict:
        if paper_id in paper_ids_with_partition:
            all_docs_dict[paper_id]['assigned_cluster_list'] = [name2partition[paper_id]]
        else: 
            all_docs_dict[paper_id]['assigned_cluster_list'] = []


    all_clusters = set([x for x in name2partition.values()])
    all_clusters = list(all_clusters)
    all_clusters.sort()
    assigned_cluster_dict = {cluster:set() for cluster in all_clusters}

    for paper in all_docs_dict.values():
        for cluster in paper['assigned_cluster_list']:
            assigned_cluster_dict[cluster].add(paper['id'])
    
    return (assigned_cluster_dict, all_clusters)


def create_papers_existence_set(
    all_docs_dict
):
    '''
        Creates a set of tuples (partition, year) for each partition and year that has at least one paper in the dataset.
        Only valid assigned partitions are considered
        
        Args:
            all_docs_dict: dictionary with id of paper as key and a dict with all the info as value (dict of dicts)

        Returns:
            papers_existence_set: set of tuples (partition, year) for which there is at least a paper (set of tuples)
    '''
    papers_existence_set = set()
    
    for paper_id in all_docs_dict:
        paper = all_docs_dict[paper_id]
        partition = paper['assigned_cluster_list']
        year = paper['year']
        if (len(partition) > 0) and (year is not None):
            for p in partition:
                papers_existence_set.add((p, year))
    
    return papers_existence_set
    

def check_papers_existence(
    cluster,
    year,
    papers_existence_set
):
    '''
        Check if the tuple (partition, year) has at least one paper in the dataset.

        Args: 
            cluster: cluster in the particion of which to check existence of paper (int TODO check if str?)
            year: year in which to check existence of paper (int)
            papers_existence_set: set of tuples created by create_papers_existence_set (set of tuples (partition, year))
        
        Returns:
            bool: confirming existence/not existence of papers in that cluster in that year
    '''
    return (cluster,year) in papers_existence_set

  
def count_knowledge_units_per_cluster_in_time(
    all_clusters,
    years,
    all_docs_dict,
    all_papers_ids,
    results_folder,
    partition_used,
):
    '''
        Counts knowledge units between clusters and between years.
        Each citation of a paper in citing_cluster and citing_year to a paper in cited_cluster and cited_year
        creates a knowledge flow from the cited_cluster and cited_year to the citing_cluster and citing_year of 1 unit.
        If the paper belongs to more than one cluster, than the knowledge unit from each of the clusters is a fraction.
        In this function, all knowledge flow units coming from a cluster and a year to all other clusters and years are computed.
        
        Args:
            all_clusters: sorted list of all the clusters names in the considered partition (list of str)
            years: sorted list of years in the dataset (list of int)
            all_docs_dict: dictionary with id of paper as key and a dict with all the info as value (dict of dicts)
            all_papers_ids: set of paper_ids in all_docs_dict (set of str)
            results_folder: path to directory of the results_folder where to save tokenized_texts_dict (str, valid path)
            partition_used: str to recognize what kind of partition has been used (str)
        
        Returns:
            knowledge_units_count_per_cluster_in_time_dict: nested dict (ordered with keys [cited_cluster][citing_cluster][cited_year][citing_year]) 
                counting for each tuple of clusters and years the knowledge units between them due to citations (nested dict)
            cluster_units_per_year_dict: nested dict with weighted number of papers of each cluster in each year (ordered with keys [year][cluster]) (dict of dicts)
    '''
    papers_existence_set = create_papers_existence_set(all_docs_dict)
    # Create the nested dict, with years only if the corresponding tuple (cluster,year) has at least a paper, using check_papers_existence()
    knowledge_units_count_per_cluster_in_time_dict = {cited_cluster: {citing_cluster: {cited_year: {citing_year: 0 for citing_year in years if check_papers_existence(citing_cluster,citing_year,papers_existence_set)} for cited_year in years if check_papers_existence(cited_cluster,cited_year,papers_existence_set)} for citing_cluster in all_clusters} for cited_cluster in all_clusters}
    # Go through all papers to get all knowledge units
    for paper1 in all_docs_dict.values():
        citing_papers = paper1['inCitations']
        for paper2_id in set(citing_papers).intersection(all_papers_ids):
            paper2 = all_docs_dict[paper2_id]
            clusters1 = paper1['assigned_cluster_list']
            clusters2 = paper2['assigned_cluster_list']
            if len(clusters1)>0 and len(clusters2)>0:
                for partition1 in clusters1:
                    for partition2 in clusters2:
                        year1 = paper1['year']
                        year2 = paper2['year']
                        if year1 is not None and year2 is not None:
                            # paper2 cites paper1 
                            # so citation_count_per_cluster_in_time is ordered like partition_from - partition_to - year_from - year_to
                            # ordered like knowledge_units_count_per_cluster_in_time_dict[cited_cluster][citing_cluster][cited_year][citing_year]
                            knowledge_units_count_per_cluster_in_time_dict[partition1][partition2][year1][year2] += (1/len(clusters1))*(1/len(clusters2))
    with gzip.open(os.path.join(results_folder, f'knowledge_units_count_per_cluster_in_time_dict_{partition_used}.pkl.gz'),'wb') as fp:
        pickle.dump(knowledge_units_count_per_cluster_in_time_dict,fp)
    # Now calculate cluster_units coming from a cluster and a year to all other clusters and years
    cluster_units_per_year_dict = {year:{cluster:0 for cluster in all_clusters if check_papers_existence(cluster,year,papers_existence_set)} for year in years}
    # Go through all papers to get all knowledge units
    for paper in all_docs_dict.values():
        if 'year' in paper and paper['year'] is not None and len(paper['assigned_cluster_list']) > 0:
            for cluster in paper['assigned_cluster_list']:
                cluster_units_per_year_dict[paper['year']][cluster] += 1/len(paper['assigned_cluster_list'])
                
    return (knowledge_units_count_per_cluster_in_time_dict, cluster_units_per_year_dict)


def normalize_citations_count(
    all_clusters,
    citing_cluster,
    cited_cluster,
    citing_year,
    cited_year,
    knowledge_units_count_per_cluster_in_time_dict,
    cluster_units_per_year_dict,
):
    '''
        Normalize citation_count_per_cluster using Ye's prescription (null model), see the paper:
        Ye Sun and Vito Latora. The evolution of knowledge within and across fields in modern physics. Scientific Reports, 10(1):12097, December 2020.
        
        Args:
            all_clusters: sorted list of all the clusters names in the considered partition (list of str)
            citing_cluster: ID of the citing cluster (int)
            cited_cluster: ID of the cited cluster (int)
            citing_year: citing year (int)
            cited_year: cited year (int)
            knowledge_units_count_per_cluster_in_time_dict: nested dict (ordered with keys [cited_cluster][citing_cluster][cited_year][citing_year]) 
                counting for each tuple of clusters and years the knowledge units between them due to citations (nested dict)
            cluster_units_per_year_dict: nested dict with weighted number of papers of each cluster in each year (ordered with keys [year][cluster]) (dict of dicts)
        
        Returns:
            float:
    '''
    # fraction of knowledge units received by citing cluster in citing_year from cited cluster in cited years over all knowledge units received by citing cluster in citing years
    numerator = knowledge_units_count_per_cluster_in_time_dict[cited_cluster][citing_cluster][cited_year][citing_year]/np.sum([knowledge_units_count_per_cluster_in_time_dict[x][citing_cluster][cited_year][citing_year] for x in knowledge_units_count_per_cluster_in_time_dict.keys()])
    # fraction of papers produced by cited cluster in cited year over all papers produced in cited years
    denominator = cluster_units_per_year_dict[cited_year][cited_cluster]/np.sum([cluster_units_per_year_dict[cited_year][cluster]for cluster in all_clusters])
    return numerator/denominator if denominator>0 and pd.notna(numerator) else 0


def compute_knowledge_flow_normalized_per_cluster_in_time_dict(
    all_clusters,
    all_docs_dict,
    years,
    years_with_none,
    knowledge_units_count_per_cluster_in_time_dict,
    cluster_units_per_year_dict,
    results_folder,
    partition_used
):
    '''
        Returns a dictionary similar to knowledge_units_count_per_cluster_in_time_dict but normalized according to a null model, see the paper:
        Ye Sun and Vito Latora. The evolution of knowledge within and across fields in modern physics. Scientific Reports, 10(1):12097, December 2020.
        
        Args:
            all_clusters: sorted list of all the clusters in the considered partition (list of int)
            all_docs_dict: dictionary with id of paper as key and a dict with all the info as value (dict of dicts)
            years: sorted list of years in the whole dataset (list of int)
            years_with_none: sorted list of years in the sample used, including also None (list)
            knowledge_units_count_per_cluster_in_time_dict: nested dict (ordered with keys [cited_cluster][citing_cluster][cited_year][citing_year]) 
                counting for each tuple of clusters and years the knowledge units between them due to citations (nested dict)
            cluster_units_per_year_dict: nested dict with weighted number of papers of each cluster in each year (ordered with keys [year][cluster]) (dict of dicts)
            results_folder: path to directory of the results_folder where to save tokenized_texts_dict (str, valid path)
            partition_used: str to recognize what kind of partition has been used (str)
        
        Returns:
            knowledge_flow_normalized_per_cluster_in_time_dict: 
    '''
    papers_existence_set = create_papers_existence_set(all_docs_dict)
    knowledge_flow_normalized_per_cluster_in_time_dict = {cited_cluster: {citing_cluster: {cited_year: {citing_year: 0 for citing_year in years if check_papers_existence(citing_cluster,citing_year,papers_existence_set)} for cited_year in years if check_papers_existence(cited_cluster,cited_year,papers_existence_set)} for citing_cluster in all_clusters} for cited_cluster in all_clusters}

    # NOTE: we iterate over years_with_none because it only has years where a decentralization papers has been published, all other years are already initialized with 0
    for cited_cluster in all_clusters:
        for citing_cluster in all_clusters:
            for cited_year in years_with_none - {None}:
                if not check_papers_existence(cited_cluster,cited_year,papers_existence_set):
                    continue
                for citing_year in years_with_none - {None}:
                    if citing_year < cited_year and not check_papers_existence(citing_cluster,citing_year,papers_existence_set):
                        continue
                    knowledge_flow_normalized_per_cluster_in_time_dict[cited_cluster][citing_cluster][cited_year][citing_year] = \
                        normalize_citations_count(
                            all_clusters,
                            citing_cluster,
                            cited_cluster,
                            citing_year,
                            cited_year,
                            knowledge_units_count_per_cluster_in_time_dict,
                            cluster_units_per_year_dict
                        )

    with gzip.open(os.path.join(results_folder, f'knowledge_flow_normalized_per_cluster_in_time_dict_{partition_used}.pkl.gz'),'wb') as fp:
        pickle.dump(knowledge_flow_normalized_per_cluster_in_time_dict,fp)
    return knowledge_flow_normalized_per_cluster_in_time_dict


def compute_knowledge_flow_normalized_per_cluster_in_time_df(
    all_clusters,
    all_docs_dict,
    years,
    years_with_none,
    knowledge_units_count_per_cluster_in_time_dict,
    cluster_units_per_year_dict,
    results_folder,
    partition_used
):
    '''
        Uses knowledge_flow_normalized_per_cluster_in_time_dict to create a pd.DataFrame with columns 'year_from','year_to':_t_t
        'cluster_from':_from,
        'cluster_to':_to,
        'knowledge_flow':k

        Returns a dataframe similar to knowledge_units_count_per_cluster_in_time_dict but normalized according to a null model, see the paper:
        Ye Sun and Vito Latora. The evolution of knowledge within and across fields in modern physics. Scientific Reports, 10(1):12097, December 2020.
        
        Args:
            all_clusters: sorted list of all the clusters in the considered partition (list of int)
            all_docs_dict: dictionary with id of paper as key and a dict with all the info as value (dict of dicts)
            years: sorted list of years in the whole dataset (list of int)
            years_with_none: sorted list of years in the sample used, including also None (list)
            knowledge_units_count_per_cluster_in_time_dict: nested dict (ordered with keys [cited_cluster][citing_cluster][cited_year][citing_year]) 
                counting for each tuple of clusters and years the knowledge units between them due to citations (nested dict)
            cluster_units_per_year_dict: nested dict with weighted number of papers of each cluster in each year (ordered with keys [year][cluster]) (dict of dicts)
            results_folder: path to directory of the results_folder where to save tokenized_texts_dict (str, valid path)
            partition_used: str to recognize what kind of partition has been used (str)
        
        Returns:
            knowledge_flow_normalized_per_cluster_in_time_df: 
    '''

    knowledge_flow_normalized_per_cluster_in_time_dict = compute_knowledge_flow_normalized_per_cluster_in_time_dict(
        all_clusters,
        all_docs_dict,
        years,
        years_with_none,
        knowledge_units_count_per_cluster_in_time_dict,
        cluster_units_per_year_dict,
        results_folder,
        partition_used
    )

    _t_from = []
    _t_to = []
    _from = []
    _to = []
    k = []

    for cluster_from in knowledge_flow_normalized_per_cluster_in_time_dict.keys():
        for cluster_to in knowledge_flow_normalized_per_cluster_in_time_dict[cluster_from].keys():
            for year_from in knowledge_flow_normalized_per_cluster_in_time_dict[cluster_from][cluster_to].keys():
                for year_to in knowledge_flow_normalized_per_cluster_in_time_dict[cluster_from][cluster_to][year_from].keys():
                    if year_to >= year_from:
                        _t_from.append(year_from)
                        _t_to.append(year_to)
                        _from.append(cluster_from)
                        _to.append(cluster_to)
                        k.append(knowledge_flow_normalized_per_cluster_in_time_dict[cluster_from][cluster_to][year_from][year_to])

    knowledge_flow_normalized_per_cluster_in_time_df = pd.DataFrame({
        'year_from':_t_from,
        'year_to':_t_to,
        'cluster_from':_from,
        'cluster_to':_to,
        'knowledge_flow':k
    })

    knowledge_flow_normalized_per_cluster_in_time_df.cluster_from = knowledge_flow_normalized_per_cluster_in_time_df.cluster_from.astype(int)
    knowledge_flow_normalized_per_cluster_in_time_df.cluster_to = knowledge_flow_normalized_per_cluster_in_time_df.cluster_to.astype(int)
    
    knowledge_flow_normalized_per_cluster_in_time_df.to_csv(os.path.join(results_folder,f'knowledge_flow_normalized_per_cluster_in_time_df_{partition_used}.csv'), index=False)

    return knowledge_flow_normalized_per_cluster_in_time_df


def compute_knowledge_flow_normalized_per_cluster_per_time_window(
    all_clusters,
    results_folder,
    partition_used,
    knowledge_flow_normalized_per_cluster_in_time_df,
    last_year = 2022,
    first_year = 1950,
    time_window_size_range = [5,10],
    significance_threshold = 1
):
    '''
        
        
        Args:
            all_clusters: sorted list of all the clusters in the considered partition (list of int)
            results_folder: path to directory of the results_folder where to save tokenized_texts_dict (str, valid path)
            partition_used: str to recognize what kind of partition has been used (str)
            knowledge_flow_normalized_per_cluster_in_time_df: 
            last_year: most recent year to consider in the analysis
            first_year: least recent year to consider in the analysis
            time_window_size: list of integers representing the time window sizes to make the computation for
            significance_threshold: threshold to consider a knowledge flow value significant

        
        Returns:
            knowledge_flow_normalized_per_cluster_per_time_window: 
    '''
    time_window_size_range = [5,10]
    
    knowledge_flow_normalized_per_cluster_per_time_window = {}
    for time_window_size in time_window_size_range:

        knowledge_flow_normalized_per_cluster_in_time['significant_kf'] = knowledge_flow_normalized_per_cluster_in_time_df.knowledge_flow > significance_threshold

        bins = range(last_year,first_year,-time_window_size)
        knowledge_flow_normalized_per_cluster_per_time[f'time_window_{time_window}_years_from'] = pd.cut(knowledge_flow_normalized_per_cluster_in_time.year_from],bins = bins, labels = bins[:-1]) 
        knowledge_flow_normalized_per_cluster_per_time[f'time_window_{time_window}_years_to'] = pd.cut(knowledge_flow_normalized_per_cluster_in_time.year_to],bins = bins, labels = bins[:-1]) 

        knowledge_flow_normalized_per_cluster_per_time_window[time_window_size] = knowledge_flow_normalized_per_cluster_per_time.groupby(['cluster_from','cluster_to',f'time_window_{time_window}_years_from',f'time_window_{time_window}_years_to'],as_index=False).significant_kf.mean()

    with gzip.open(os.path.join(results_folder, f'knowledge_flow_normalized_per_cluster_per_time_window_{partition_used}.pkl.gz'),'wb') as fp:
        pickle.dump(knowledge_flow_normalized_per_cluster_per_time_window,fp)
    
    return knowledge_flow_normalized_per_cluster_per_time_window

def compute_knowledge_flow_normalized_per_cluster_per_time_window_to_future(
    all_clusters,
    results_folder,
    partition_used,
    knowledge_flow_normalized_per_cluster_in_time_df,
    last_year = 2022,
    first_year = 1950,
    time_window_size_range = [5,10],
    significance_threshold = 1
):
    '''
        
        
        Args:
            all_clusters: sorted list of all the clusters in the considered partition (list of int)
            results_folder: path to directory of the results_folder where to save tokenized_texts_dict (str, valid path)
            partition_used: str to recognize what kind of partition has been used (str)
            knowledge_flow_normalized_per_cluster_in_time_df: 
            last_year: most recent year to consider in the analysis
            first_year: least recent year to consider in the analysis
            time_window_size: list of integers representing the time window sizes to make the computation for
            significance_threshold: threshold to consider a knowledge flow value significant
        
        Returns:
            knowledge_flow_normalized_per_cluster_per_time_window: 
    '''
    time_window_size_range = [5,10]
    
    knowledge_flow_normalized_per_cluster_per_time_window_to_future = {}
    for time_window_size in time_window_size_range:

        knowledge_flow_normalized_per_cluster_in_time['significant_kf'] = knowledge_flow_normalized_per_cluster_in_time_df.knowledge_flow > significance_threshold

        bins = range(last_year,first_year,-time_window_size)
        knowledge_flow_normalized_per_cluster_per_time[f'time_window_{time_window}_years_from'] = pd.cut(knowledge_flow_normalized_per_cluster_in_time.year_from],bins = bins, labels = bins[:-1]) 

        knowledge_flow_normalized_per_cluster_per_time_window_to_future[time_window_size] = knowledge_flow_normalized_per_cluster_per_time.groupby(['cluster_from','cluster_to',f'time_window_{time_window}_years_from'],as_index=False).significant_kf.mean()
    
    with gzip.open(os.path.join(results_folder, f'knowledge_flow_normalized_per_cluster_per_time_window_to_future_{partition_used}.pkl.gz'),'wb') as fp:
        pickle.dump(knowledge_flow_normalized_per_cluster_per_time_window_to_future,fp)
    
    return knowledge_flow_normalized_per_cluster_per_time_window_to_future


def compute_knowledge_flow_normalized_per_cluster_in_time_to_future(
    all_clusters,
    results_folder,
    partition_used,
    knowledge_flow_normalized_per_cluster_in_time_df,
    significance_threshold = 1
):
    '''
        Args:
            all_clusters: sorted list of all the clusters in the considered partition (list of int)
            results_folder: path to directory of the results_folder where to save tokenized_texts_dict (str, valid path)
            partition_used: str to recognize what kind of partition has been used (str)
            knowledge_flow_normalized_per_cluster_in_time_df: 
            significance_threshold: threshold to consider a knowledge flow value significant
        
        Returns:
            knowledge_flow_normalized_per_cluster_per_time_window: 
    '''
    time_window_size_range = [5,10]
    
    knowledge_flow_normalized_per_cluster_in_time_to_future = {}
    for time_window_size in time_window_size_range:

        knowledge_flow_normalized_per_cluster_in_time['significant_kf'] = knowledge_flow_normalized_per_cluster_in_time_df.knowledge_flow > significance_threshold

        knowledge_flow_normalized_per_cluster_in_time_to_future[time_window_size] = knowledge_flow_normalized_per_cluster_per_time.groupby(['cluster_from','cluster_to','year_from'],as_index=False).significant_kf.mean()
    
    with gzip.open(os.path.join(results_folder, f'knowledge_flow_normalized_per_cluster_in_time_to_future_{partition_used}.pkl.gz'),'wb') as fp:
        pickle.dump(knowledge_flow_normalized_per_cluster_in_time_to_future,fp)
    
    return knowledge_flow_normalized_per_cluster_in_time_to_future


def run_knowledge_flow_analysis(
    all_docs_dict,
    h_t_doc_consensus_by_level,
    hyperlink_g,
    ordered_paper_ids,
    lowest_level_knowledge_flow,
    highest_non_trivial_level,
    dataset_path,
    results_folder,
    last_year = 2022,
    first_year = 1950,
    time_window_size_range = [5,10],
    significance_threshold = 1,
):
    '''
        
        
        Args:
            all_docs_dict: dictionary with id of paper as key and a dict with all the info as value (dict of dicts)
            h_t_doc_consensus_by_level: dict of level as key and an array of length the number of docs in the hsbm with value the cluster at that level (dict, level:np.array)
            hyperlink_g: gt network containing all papers and their links, i.e., citations or hyperlinks (gt.Graph)
            ordered_paper_ids: ordered list of paper_ids in hyperlink_g, with the same ordering as hyperlink_g.vp['name'] (list of str, i.e., list(hyperlink_g.vp['name']))
            lowest_level_knowledge_flow: lowest level of the hsbm consensus partition on which to calculate the knowledge flows (int)
            highest_non_trivial_level: highest level of the hsbm consensus partition for which there are more than 1 groups (int)
            dataset_path: path to the whole dataset with respect to the repo root folder (str, valid path)
            results_folder: path to directory of the results_folder where to save tokenized_texts_dict (str, valid path)
            last_year: most recent year to consider in the analysis
            first_year: least recent year to consider in the analysis
            time_window_size: list of integers representing the time window sizes to make the computation for
            significance_threshold: threshold to consider a knowledge flow value significant
        
        Returns:
            None
    '''
    paper2cluster = {x['id']:x['clustersOfStudy'] for x in all_docs_dict.values() if 'id' in x and 'clustersOfStudy' in x}

    for gt_partition_level in range(lowest_level_knowledge_flow,highest_non_trivial_level + 1):
        print(f'level {gt_partition_level}')
        
        # Load the correct partitions in the key 'assigned_cluster_list' for each paper
        partition_used = 'gt_partition_lev_%d'%(gt_partition_level)
        assigned_cluster_dict, all_clusters = \
            assign_partition(
                h_t_doc_consensus_by_level,
                hyperlink_g,
                 gt_partition_level,
                all_docs_dict,
            )


        # count citations between clusters and between different times
        # citation_count_per_cluster_in_time = {partition: {partition: {year: {year: 0 for year in years_with_none} for year in years_with_none} for partition in all_clusters} for partition in all_clusters}
        with gzip.open(dataset_path + '../' +'no_papers_in_clusters_by_year.pkl.gz','rb') as fp:
            papers_per_year_per_cluster = pickle.load(fp)

        years = sorted([x for x in papers_per_year_per_cluster.keys() if x is not None and x > 1500])
        years_with_none = set([x['year'] for x in all_docs_dict.values()])
        all_papers_ids = set([x['id'] for x in all_docs_dict.values()])


        knowledge_units_count_per_cluster_in_time_dict, cluster_units_per_year_dict = \
            count_knowledge_units_per_cluster_in_time(
                all_clusters,
                years,
                all_docs_dict,
                all_papers_ids,
                results_folder,
                partition_used,
            )

        knowledge_flow_normalized_per_cluster_in_time_df = \
            compute_knowledge_flow_normalized_per_cluster_in_time_df(
                all_clusters,
                all_docs_dict,
                years,
                years_with_none,
                knowledge_units_count_per_cluster_in_time_dict,
                cluster_units_per_year_dict,
                results_folder,
                partition_used
            )

        knowledge_flow_normalized_per_cluster_per_time_window = \
            compute_knowledge_flow_normalized_per_cluster_per_time_window(
                all_clusters,
                results_folder,
                partition_used,
                knowledge_flow_normalized_per_cluster_in_time_df,
                last_year = last_year,
                first_year = first_year,
                time_window_size_range = time_window_size_range,
                significance_threshold = significance_threshold
            )

        knowledge_flow_normalized_per_cluster_per_time_window_to_future = \
            compute_knowledge_flow_normalized_per_cluster_per_time_window_to_future(
                all_clusters,
                results_folder,
                partition_used,
                knowledge_flow_normalized_per_cluster_in_time_df,
                last_year = last_year,
                first_year = first_year,
                time_window_size_range = time_window_size_range,
                significance_threshold = significance_threshold
            )

        knowledge_flow_normalized_per_cluster_in_time_to_future = \
            compute_knowledge_flow_normalized_per_cluster_per_time_window_to_future(
                all_clusters,
                results_folder,
                partition_used,
                knowledge_flow_normalized_per_cluster_in_time_df,
                significance_threshold = significance_threshold
            )

    return None