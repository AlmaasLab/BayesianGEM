#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

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
        df["model"] = label
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

bayesian_model_frame: pd.DataFrame = load_pickle("../results/permuted_smcabc_res/simulation_skeleton.pkl")
bayesian_model_frame.set_index(["origin","status"], inplace=True)
bayesian_entry = bayesian_model_frame.loc[("unpermuted", "original")]
bayesian_file = bayesian_entry["outfile"]
bayesian_df = build_a_dataframe_for_all_particles(bayesian_file, n_priors=128)

evolutionary_file = "../results/smcevo_gem_three_conditions_save_all_particles_refined.pkl"
evolutionary_df = build_a_dataframe_for_all_particles(evolutionary_file, n_priors=100)
dump_pickle(evolutionary_df,"../results/evo_particle_df")

df_dict = {'Bayesian': bayesian_df, 'Evolutionary': evolutionary_df}
combined_df = combine_dataframes_for_models(df_dict=df_dict)
dump_pickle(combined_df, "../results/evo_combined_particle_df.pkl")
pca_ordination = perform_pca_on_parameters(combined_df)
dump_pickle(pca_ordination,"../results/evo_pca_full_ordination.pkl")
logging.info("DONE")
