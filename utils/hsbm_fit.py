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
import sbmmultilayer 
import graph_tool.all as gt


def fit_hyperlink_text_hsbm(
    edited_text, 
    titles, 
    hyperlinks, 
    N_iter, 
    results_folder, 
    filename_fit = f'results_fit_greedy_tmp.pkl.gz', 
    SEED_NUM = 1, 
    stop_at_fit = False,
    number_iterations_MC_equilibrate = 5000, 
):
    '''
        Fit N_iter iterations of doc-network sbm on dataset through agglomerative heuristic and simulated annealing.
        If stop_at_fit is True, it does only the fit and saves it in a temporary file, otherwise it does also the equilibrate.
        If the temporary file is present, it is loaded by default.

        Paramaters
        ----------
            edited_text: list of tokenized filtered texts of the filtered papers (list of lists of words)
            titles: list of IDs or titles to give as name of the docs (list of words)
            hyperlinks: list of tuples (node_1, node_2) representing links from node_1 to node_2 in the citation_layer of the model (list of tuples of two elements)
            N_iter: number of different iterations to create hsbm (int)
            results_folder: path to directory of the results_folder where to save tokenized_texts_dict (str, valid path)
            filename_fit: filename to give to the temporary fit file before the equilbrate
            SEED_NUM: seed number to use to create the sbmmultilayer.sbmmultilayer (int)
            stop_at_fit: if True, it stops the function before doing the equilibrate; if False, it continues with the equilibrate (bool)
            number_iterations_MC_equilibrate: number of iterations of markov chain montecarlo equilibrations to do (int)
        
        Returns
        ----------
            if stop_at_fit == True:
                None
            else:
                hyperlink_text_hsbm_post: list of hsbm generated from the 2-layer network after fit and mcmc_equilibrate (list of hsbm)
    '''
    hyperlink_text_hsbm_post = []
    # Repeat for all N_iter, creating a different fit at each iteration, then used to create the consensus partititions.
    for _ in range(N_iter):
        print(f'Iteration {_}')
        # Construct 2-layer network hyperlink-text model and fit multilayer SBM.
        try: 
            # If stop_at_fit==False, this function is called to do the equilibrate, and checks if the temporary fit file is done already
            print('Loading temporary results from previous run...',flush=True)
            with gzip.open(f'{results_folder}{filename_fit}', 'rb') as fp:
                hyperlink_text_hsbm,tmp = pickle.load(fp)
        except Exception as e:
            print(e)
            
            hyperlink_text_hsbm = sbmmultilayer.sbmmultilayer(random_seed=SEED_NUM)
            hyperlink_text_hsbm.make_graph(edited_text, titles, hyperlinks)
            
            start = datetime.now()
            print('Starting fit at: ',start,flush=True)
            hyperlink_text_hsbm.fit(verbose=False)
            end = datetime.now()
            print('fit took: ',end - start,flush=True)
            with gzip.open(f'{results_folder}{filename_fit}', 'wb') as fp:
                pickle.dump((hyperlink_text_hsbm,end-start),fp)
        
        if stop_at_fit == True:
            # If stop_at_fit==True, this function stops at the fit, without doing the equilibration
            return None
        
        # Retrieve state from simulated annealing hSBM
        hyperlink_text_hsbm_post_state = run_multiflip_greedy_hsbm(hyperlink_text_hsbm, number_iterations_MC_equilibrate)

        # Update hSBM model using state from simulated annealing
        updated_hsbm_model = hyperlink_text_hsbm
        updated_hsbm_model.state = hyperlink_text_hsbm_post_state
        updated_hsbm_model.mdl = hyperlink_text_hsbm_post_state.entropy()
        updated_hsbm_model.n_levels = len(hyperlink_text_hsbm_post_state.levels)
        
        # Save the results
        hyperlink_text_hsbm_post.append(updated_hsbm_model)
      
    return hyperlink_text_hsbm_post


def run_multiflip_greedy_hsbm(
    hsbm_model, 
    number_iterations_MC_equilibrate, 
):
    '''
        Run greedy merge-split on multilayer SBM, forcing number_iterations_MC_equilibrate iterations for the mcmc_equilibrate.
        
        Paramaters
        ----------
            hsbm_model: hierarchical stochastick block model to equilibrate (sbmmultilayer.sbmmultilayer)
            number_iterations_MC_equilibrate: number of iterations of markov chain montecarlo equilibrations to do (int)
        
        Returns
        ----------
            hsbm_state: state associated to SBM at the end (sbmmultilayer.sbmmultilayer.state)
    '''
    S1 = hsbm_model.mdl
    print(f'Initial entropy is {S1}')
    
    # Start equilibrate
    gt.mcmc_equilibrate(hsbm_model.state, mcmc_args=dict(beta=np.inf),history=True,verbose=True,force_niter=number_iterations_MC_equilibrate,multiflip=False)    
    
    S2 = hsbm_model.state.entropy()
    print(f'New entropy is {S2}')
    print(f'Improvement after greedy moves {S2 - S1}')
    print(f'The improvement percentage is { ((S2 - S1)/S1) * 100 }',flush=True)

    return hsbm_model.state