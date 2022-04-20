#!/usr/bin/env python
# coding: utf-8

# In[1]:


import abc_etc as abc
import numpy as np
import GEMS
import os
import pandas as pd
import pickle
import permute_parameters
import logging

# In[]

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s') 
# In[2]:


Yobs_batch = GEMS.aerobic_exp_data()
Yobs_chemo = GEMS.chemostat_exp_data()
#Yobs_batch_an = GEMS.anaerobic_exp_data()
dfae_batch,dfan_batch =GEMS.load_exp_batch_data('../data/ExpGrowth.tsv')
sel_temp = [5.0,15.0,26.3,30.0,33.0,35.0,37.5,40.0]
Yobs_batch_an = {'data':dfan_batch.loc[sel_temp,'r_an'].values}


# In[5]:


Yobs = {'rae':Yobs_batch['data'],
        'chemostat':Yobs_chemo['data'],
        'ran':Yobs_batch_an['data']}


# In[ ]:

n_permutations = 3
random_seed = 158
rng = np.random.default_rng(random_seed)
path = os.path.dirname(os.path.realpath(__file__)).replace('code','')
params = pd.read_csv(os.path.join(path,'data/model_enzyme_params.csv'),index_col=0)
permuted_params = [permute_parameters.permute_params(params, rng) for _ in range(n_permutations)]
combined_params = {'unpermuted': params}
for i, permuted_param_set in enumerate(permuted_params):
    combined_params[f'permuted_{i}'] = permuted_param_set


candidate_frame: pd.DataFrame = pd.DataFrame(index=pd.MultiIndex.from_product([combined_params.keys(), ["original", "replicate"]], names = ["origin", "status"])).reset_index()


# #### Define priors

# In[ ]:

priors = []

for origin in candidate_frame.origin:
    params = combined_params[origin]
    prior = dict()
    for ind in params.index: 
        for col in ['Tm','Topt','dCpt']: 
            prior['{0}_{1}'.format(ind,col)] = abc.RV('normal',
                                                          loc=combined_params[origin].loc[ind,col],
                                                          scale=combined_params[origin].loc[ind,col+'_std'])
    priors.append(prior)


candidate_frame['priors'] = priors



# #### Define model settings

# In[ ]:


min_epsilon = -1.0 # equivalent to r2 score of 1
population_size = 100
outdir = '../results/permuted_smcabc_res'
if not os.path.exists(outdir):
    os.makedirs(outdir)
candidate_frame['outfile'] = [f'{outdir}/smcabc_gem_three_conditions_updated_{origin}_{status}_save_all_particles.pkl' for
 origin, status in zip(candidate_frame['origin'],candidate_frame['status'])]

pickle.dump(file=open(file=f'{outdir}/simulation_skeleton.pkl',mode='wb'),obj=candidate_frame)




# In[ ]:

models = []
for priors, outfile in zip(candidate_frame['priors'], candidate_frame['outfile']):
    if not os.path.exists(outfile):
        logging.info('Initialize models')
        models.append(abc.SMCABC(GEMS.simulate_at_three_conditions_2,
                                priors=priors,
                                min_epsilon=min_epsilon,
                                population_size=population_size,
                                distance_function=GEMS.distance_2,
                                Yobs=Yobs,
                                outfile=outfile,
                                generation_size=128,
                                 maxiter=500)
        )
    else:
        models.append(pickle.load(open(outfile,'rb')))

candidate_frame['model'] = models


# #### Run simulation

# In[ ]:

for origin, status, model in zip(candidate_frame['origin'],candidate_frame['status'], candidate_frame['model']):
    logging.info(f'Start simulations with prior parameter set {origin}, {status}')
    model.run_simulation()

logging.info(f'DONE')
