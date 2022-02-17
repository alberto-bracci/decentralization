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
# from nmi import *
# from doc_clustering import *
# from hsbm import sbmmultilayer 
# from hsbm.utils.nmi import *
# from hsbm.utils.doc_clustering import *

# import gi
# from gi.repository import Gtk, Gdk
import graph_tool.all as gt
# import ast # to get list comprehension from a string


# import functools, builtins # to impose flush=True on every print
# builtins.print = functools.partial(print, flush=True)


def fit_hyperlink_text_hsbm(edited_text, 
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
    Fit N_iter iterations of doc-network sbm on dataset through agglomerative heuristic
    and simulated annealing.
    
    If stop_at_fit is True, it does only the fit and saves it in a temporary file, otherwise it does also the equilibrate.
    If the temporary file is present, it is loaded by default.
    '''
    hyperlink_text_hsbm_post = []

    for _ in range(N_iter):
        print(f'Iteration {_}')
        # Construct 2-layer network hyperlink-text model and fit multilayer SBM.
        
        try: 
            print('Loading temporary results from previous run...',flush=True)
            with gzip.open(f'{results_folder}{filename_fit}', 'rb') as fp:
                hyperlink_text_hsbm,tmp = pickle.load(fp)
        except Exception as e:
            print(e)
            
            hyperlink_text_hsbm = sbmmultilayer.sbmmultilayer(random_seed=SEED_NUM)
            hyperlink_text_hsbm.make_graph(edited_text, titles, hyperlinks, multiplier=1) # TODO TOGLI MULTIPLIER
            
            start = datetime.now()
            print('Starting fit at: ',start,flush=True)
            hyperlink_text_hsbm.fit(verbose=False)
            end = datetime.now()
            print('fit took: ',end - start,flush=True)
            with gzip.open(f'{results_folder}{filename_fit}', 'wb') as fp:
                pickle.dump((hyperlink_text_hsbm,end-start),fp)
        
        if stop_at_fit == True:
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


def run_multiflip_greedy_hsbm(hsbm_model, 
                              number_iterations_MC_equilibrate, 
                             ):
    '''
    Run greedy merge-split on multilayer SBM.
    Return:
        hsbm_state - State associated to SBM at the end.
    '''
    S1 = hsbm_model.mdl
    print(f'Initial entropy is {S1}')

#     gt.mcmc_equilibrate(hsbm_model.state, force_niter=40, mcmc_args=dict(beta=np.inf),history=True,verbose=True)    
    gt.mcmc_equilibrate(hsbm_model.state, mcmc_args=dict(beta=np.inf),history=True,verbose=True,force_niter=number_iterations_MC_equilibrate,multiflip=False)    
    
    S2 = hsbm_model.state.entropy()
    print(f'New entropy is {S2}')
    print(f'Improvement after greedy moves {S2 - S1}')
    print(f'The improvement percentage is { ((S2 - S1)/S1) * 100 }',flush=True)

    return hsbm_model.state