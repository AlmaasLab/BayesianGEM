#!/usr/bin/env python
# coding: utf-8
# This script is intended to assess Supplementary Figure S2 where the PCA plot does clearly distinguish the two simulations from permuted prior 3 (2 with zero indexing)

import pickle
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import multiprocessing
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

# Note about the usage of the terms "origin" and "status": Origin is the prior used to seed the
#  simulations (unpermuted, permuted_0, permuted_1 and permuted_2), whereas the status
#  is which replicate where the difference is the random seed is used (original is Simulation 1, replicate is Simulation 2)

# Convenient pickle wrappers
def load_pickle(filename):
    return pickle.load(open(file=filename,mode='rb'))

def dump_pickle(obj,filename):
    return pickle.dump(obj=obj,file=open(file=filename, mode='wb'))

def build_a_dataframe_for_all_particles(file, n_priors = 128, r2_threshold = 0.9):
    results = load_pickle(file)
    columns = list(results.all_particles[0].keys())
    columns.sort()
    logging.info("Iterating over particles")
    data = list()
    for p in results.all_particles:
        data.append([p[k] for k in columns])
    logging.info("Creating Data Frame")
    df = pd.DataFrame(data=data,columns=columns)
    df['r2'] = results.all_distances
    logging.info(df.shape)
    
    
    logging.info("Doing filtering and labelling of Data Frame")
    df['r2'] = -df['r2']
    df["period"] = "Intermediate"
    df.loc[:n_priors,"period"] = "Prior"
    df.loc[df["r2"] > r2_threshold,"period"] = 'Posterior'
    # Remove samples with a R2 score smaller than -3
    sel_index = df.index[df['r2']>-3]    
    df = df.loc[sel_index,:]
    logging.info(df.shape)

    return df

def combine_dataframes_for_models(df_dict):
    # augmented_df_list =[ df.assign(model = lambda df: label)  for df, label in zip(df_list, index)]
    augmented_df_dict = {label: df.copy() for label, df in df_dict.items()}
    logging.info("Copying done")
    for label, df in augmented_df_dict.items():
        df["origin"] = label[0]
        df["status"] = label[1]
    logging.info("Labelling done")
    return pd.concat(augmented_df_dict.values(), ignore_index=True)

def perform_pca_on_parameters(df):
    # 1. normalize all columns to a standard normal distribution
    X = df.values[:,:-3]
    X_n = np.zeros_like(X)    
    for i in range(X_n.shape[1]): X_n[:,i] = (X[:,i]-np.mean(X[:,i]))/np.std(X[:,i])
    pca = PCA(n_components=2)
    PCS = pca.fit_transform(X_n)
    logging.info(pca.explained_variance_ratio_)
    return PCS, pca.explained_variance_ratio_

if __name__ == '__main__':
    model_frame = load_pickle("../results/permuted_smcabc_res/simulation_skeleton.pkl")
    relevant_model_frame = model_frame[model_frame["origin"] == "permuted_2"]
    relevant_model_frame.set_index(["origin","status"], inplace=True)

    with multiprocessing.Pool(2) as p:
        particle_df_map = p.map(build_a_dataframe_for_all_particles,relevant_model_frame.outfile)
        relevant_model_frame["particle_df"] = list(particle_df_map)


    full_particle_df_dict = {distribution: df for distribution, df in relevant_model_frame["particle_df"].iteritems()}
    combined_df = combine_dataframes_for_models(full_particle_df_dict)
    dump_pickle(combined_df, "../results/permuted_smcabc_res/sim_2_particles_df.pkl")
    pca_ordination = perform_pca_on_parameters(combined_df)
    dump_pickle(pca_ordination,"../results/permuted_smcabc_res/pca_sim_2_ordinarion.pkl")
