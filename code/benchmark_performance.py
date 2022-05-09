#!/usr/bin/env python
# coding: utf-8
# The purpose of this script is to benchmark etcFBA model evaluation across the COBRApy and
# reframed implementations 
# In[]
import time
from timeit import timeit
import GEMS
from etcpy import etc
import os
import pandas as pd
import logging
import abc_etc as abc
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
N_ITERATIONS = 10

path = '../'
params = pd.read_csv(os.path.join(path,'data/model_enzyme_params.csv'),index_col=0)
particle = dict()
for ind in params.index: 
    for col in ['Tm','Topt','dCpt']: 
        particle['{0}_{1}'.format(ind,col)] = params.loc[ind,col]

# In[]
tic = time.perf_counter()
for _ in range(N_ITERATIONS):
    total_res = GEMS.simulate_at_three_conditions_2(particle)
toc = time.perf_counter()
print(f"Computed results at three conditions with COBRApy with an average of {(toc-tic)/N_ITERATIONS} seconds.")