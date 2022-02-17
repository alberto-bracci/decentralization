#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Compute partitions, number of groups, entropies of SBM model. Also compute NMI comparison between models.

Author: Alex Tao and Charles Hyland
'''

import os
import re
import numpy as np
import graph_tool.all as gt
from nmi import *
import pickle
import matplotlib.pyplot as plt
import matplotlib
cmap = matplotlib.colors.ListedColormap(
    np.vstack(plt.cm.Set3.colors[:6]))

class doc_clustering:
    def __init__(self, output_path, hyperlink_graph, seeds=[1213, 28, 520, 39, 47, 111111, 9, 180, 823, 324]):
        """
        Initialise clustering object where hyperlink is the Wikipedia article hyperlink graph.

        Parameters:
        output_path: File location to store the output from our other methods.
        hyperlink: The Wikipedia hyperlink graph.
        seeds: 10 Random seed numbers.
        """
        self.output_path = output_path
        self.hyperlink_graph = hyperlink_graph
        self.seeds = seeds


    def collect_info(self, nType, sbm_graph, highest_l, state, draw=False):
        """
        Retrieve the partitions, number of groups, and entropies for the hyperlink layer 
        of the network.

        Parameters:
        nType: String for type of network (hyperlink, 2-layer-hyperlink-word, 3-layer-hyperlink-word-tag)
        """        
        partitions = []
        num_of_groups = []
        entropies = []
        block_SBM_partitions = {} # store dictionary to map nodes to partition.
        
        # Due to NP-hardness of retrieving optimal partitions, repeat process numerous times.
        for i in range(len(self.seeds)):
            try:
                b = state.project_level(highest_l[i]).get_blocks()
                # Store dictionary of partitions
                if nType == "hSBM":
                    # Only contains hyperlink SBM.
                    for node in sbm_graph.vertices():
                        block_SBM_partitions[sbm_graph.vp.name[node]] = b[node]
                else:
                    # Need to specify to retrieve partitions for document nodes.
                    for node in sbm_graph.vertices():
                        if sbm_graph.vp['kind'][node] == 0:
                            block_SBM_partitions[sbm_graph.vp.name[node]] = b[node]                    
                # Retrieve the partition from the SBM and store as parameter.
                partition = self.hyperlink_graph.vp["partition"] = self.hyperlink_graph.new_vp("int")
                # Assign partition label to node properties.
                for v in self.hyperlink_graph.vertices():
                    partition[v] = block_SBM_partitions[self.hyperlink_graph.vp.name[v]]

                # Print output of graph
                if draw:
                    # Visualise graph
                    print(f"{i}: {state}")
                    gt.graph_draw(self.hyperlink_graph, pos=self.hyperlink_graph.vp.pos, vertex_fill_color=self.hyperlink_graph.vp.partition)
                
                # Add in results.
                partitions.append(list(self.hyperlink_graph.vp.partition))
                num_of_groups.append(len(set(block_SBM_partitions.values())))
                entropies.append(state.entropy())
            except (FileNotFoundError) as e:
                pass

        return (partitions, num_of_groups, entropies)
    
    def collect_info2(self, nType, sbm_graph, highest_l, state, draw=False):
        """
        Retrieve the partitions, number of groups, and entropies for the hyperlink layer 
        of the network.

        Parameters:
        nType: String for type of network (hyperlink, 2-layer-hyperlink-word, 3-layer-hyperlink-word-tag)
        """        
        partitions = []
        num_of_groups = []
        entropies = []
        block_SBM_partitions = {} # store dictionary to map nodes to partition.
        
        # Due to NP-hardness of retrieving optimal partitions, repeat process numerous times.
        for i in range(len(self.seeds)):
            try:
                b = state.project_level(highest_l[i]).get_blocks()
                # Store dictionary of partitions
                if nType == "hSBM":
                    # Only contains hyperlink SBM.
                    for node in sbm_graph.vertices():
                        block_SBM_partitions[sbm_graph.vp.name[node]] = b[node]
                else:
                    # Need to specify to retrieve partitions for document nodes.
                    for node in sbm_graph.vertices():
                        if sbm_graph.vp['kind'][node] == 0:
                            block_SBM_partitions[sbm_graph.vp.name[node]] = b[node]                    
                # Retrieve the partition from the SBM and store as parameter.
                partition = self.hyperlink_graph.vp["partition"] = self.hyperlink_graph.new_vp("int")
                # Assign partition label to node properties.
                for v in self.hyperlink_graph.vertices():
                    partition[v] = block_SBM_partitions[self.hyperlink_graph.vp.name[v]]

                # Print output of graph
                if draw:
                    # Visualise graph
                    print(f"{i}: {state}")
                    gt.graph_draw(self.hyperlink_graph, pos=self.hyperlink_graph.vp.pos, vertex_fill_color=self.hyperlink_graph.vp.partition)
                
                # Add in results.
                partitions.append(list(self.hyperlink_graph.vp.partition))
                num_of_groups.append(len(set(block_SBM_partitions.values())))
                entropies.append(state.entropy())
            except (FileNotFoundError) as e:
                pass

        return (partitions, num_of_groups, entropies, block_SBM_partitions)


def _nmis(partitions):
    """
    Helper function for calculating the mutual information. Take in list of partitions from 2 different models and
    computes the NMI between the two models' partitions.

    partitions: list of list
    The partitions will be two lists of lists of size N_ITER where each list corresponds to partitions of a model 
    where we retrieve N_ITER partitions each time.

    Return: list of NMIs between partitions.
    """
    n = len(partitions) # 10 * 10, depends on number of iterations to retrieve partitions
    mi_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            mi_matrix[i,j] = compute_normalised_mutual_information(partitions[i], partitions[j])
    return list(mi_matrix[np.triu_indices(n,1)]) # return upper triangle for array.


def construct_nmi_matrix(partitions, true_partition):
    """
    Compute NMI matrix for all partitions generated by models against true partition.

    partitions: list of list of partitions for each model. 
    true_partition: single list of true partitions.

    Remark: For example, we may generate 20 different partitions for the hSBM and compare it to the true partitions.
    First column is for the true partition.
    """
    num_models = len(partitions) # number of different models we are testing.
    # Store the average and standard deviation of NMIs between partitions in a n x n matrix
    nmis_avg = np.zeros((num_models+1, num_models+1))
    nmis_std = np.zeros((num_models+1, num_models+1))

    # Iterate through NMI matrix and compute NMIs between models excluding the ground truth.
    # We do not iterate through the first column.
    for i in range(1, num_models+1):
        for j in range(i, num_models+1):
            nmis = _nmis(partitions[i-1] + partitions[j-1]) # retrieve list of partitions for model i-1 and model j-1
            # Store mean and std of NMIs.
            nmis_avg[i,j] = np.average(nmis)
            nmis_std[i,j] = np.std(nmis)

    nmis_avg[0,0], nmis_std[0,0] = 1, 0 # true partition should have NMI of 1 with itself.
    # Compute the NMI for each model against ground truth. Corresponds to 1st column of NMI matrix.
    for i in range(num_models):
        # Compute NMI of model's partition with ground truth labels.
        nmis_with_true = [compute_normalised_mutual_information(p, true_partition) for p in partitions[i]]
        nmis_avg[0, i+1] = np.average(nmis_with_true)
        nmis_std[0, i+1] = np.std(nmis_with_true)
    return (nmis_avg.T, nmis_std.T)


cmaps = [matplotlib.colors.ListedColormap(
    np.vstack(plt.cm.Set3.colors[:6])), matplotlib.colors.ListedColormap(
        np.vstack(plt.cm.Set3.colors[:6][::-1]))]

def compute_nmi_plots(nmi_avg, nmi_std, methods):
    """
    Plot NMI "heat-map" across all the different methods.
    """
    #methods = ["True", "hSBM", "docWord", "wikiTag", "hyperlink", "combined"][:nmi_avg.shape[0]]
    mask = np.zeros_like(nmi_avg)
    mask[np.triu_indices_from(mask, 1)] = True
    myMatrix = np.ma.masked_where(mask, nmi_avg)

    data =  [np.ma.masked_where(mask, nmi_avg), np.ma.masked_where(mask, nmi_std)]
    text_data = [nmi_avg, nmi_std ]
    fig, ax = plt.subplots(1,2,
                           sharey=True,
                           figsize = (12,5),
                           gridspec_kw = dict(hspace=0.1))

    for k in range(2):
        ax[k].imshow(data[k], cmap = plt.cm.Blues)
        ax[k].set_xticklabels(['']+methods, rotation = 30)
        ax[k].xaxis.set_ticks_position('bottom')
        ax[k].spines['right'].set_visible(False)
        ax[k].spines['top'].set_visible(False)

        for i in range(len(methods)):
            for j in range(i, len(methods)):
                text = ax[k].text(i, j, round(text_data[k][j, i],4),
                               ha="center", va="center", color="k", size = 12)
    ax[0].set_yticklabels(['']+methods)
    fig.tight_layout()
    return fig
