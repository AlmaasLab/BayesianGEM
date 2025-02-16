#!/usr/bin/env python
# coding: utf-8

# In[1]:


from typing import Dict, Iterable, List, Optional
import pandas as pd
import numpy as np
import pickle
from etcpy import etc
import os
import logging
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
from itertools import starmap
from etcpy.thermal_parameters import format_input


candidateType = Dict[str, float]




# ### Load data

# In[ ]:


def load_exp_batch_data(infile):
    # return two dataframes for aerobic and anaerobic conditions
    # with temperature as index
    #
    data = dict()
    for line in open(infile):
        cont = line.split()
        data[cont[0]] = [float(item) for item in cont[1:]]
    
    def build_df(ind,col):
        df = pd.DataFrame()
        df[col] = data[col]
        df.index = data[ind]
        return df
    
    dfae_exp = build_df('Ts_ae','r_ae')
    dfan_exp = build_df('Ts_an','r_an')
    #print(dfae_exp,dfan_exp)
    return dfae_exp,dfan_exp


# In[ ]:


def aerobic_exp_data():
    dfae_batch,dfan_batch = load_exp_batch_data(os.path.join(path,'data/ExpGrowth.tsv'))
    return  {'data':dfae_batch['r_ae'].values}


# In[ ]:


def anaerobic_exp_data():
    dfae_batch,dfan_batch = load_exp_batch_data(os.path.join(path,'data/ExpGrowth.tsv'))
    return  {'data':dfan_batch['r_an'].values}


# In[ ]:


def chemostat_exp_data():
    dfchemo = pd.read_csv(os.path.join(path,'data/Chemostat_exp_data.txt'),sep='\t')
    rxn_lst = [
        'r_1714_REV',#Glucose
        'r_1672', #CO2
        'r_1761', # Ethanol
    ]
    columns = ['Glucose','CO2','Ethanol']
    exp_flux = []
    for i,rxn_id in enumerate(rxn_lst):
        exp_flux += list(dfchemo[columns[i]])
    return  {'data':np.array(exp_flux)}


# In[ ]:


path = os.path.dirname(os.path.realpath(__file__)).replace('code','')
params = pd.read_csv(os.path.join(path,'data/model_enzyme_params.csv'),index_col=0)
dfae_batch,dfan_batch = load_exp_batch_data(os.path.join(path,'data/ExpGrowth.tsv'))
dfchemo = pd.read_csv(os.path.join(path,'data/Chemostat_exp_data.txt'),sep='\t',index_col=0)


# ### Models 


# In[4]:


def aerobic(thermalParams):
    # thermalParams: a dictionary with ids like uniprotid_Topt 
    param_dict = format_input(params,thermalParams)
    mae = pickle.load(open(os.path.join(path,'models/aerobic.pkl'),'rb'))
    rae = etc.simulate_growth(mae,dfae_batch.index+273.15,param_dict=param_dict,sigma=0.5)
    
    rae = [0 if x is None else x for x in rae]
    rae = [0 if x<1e-3 else x for x in rae]
    logging.info(f"rae: {rae}")
    rexp = aerobic_exp_data()['data']
    
    logging.info(f'r2_batch: {r2_score(rexp,rae)}')
    logging.info(f'MSE_ae: {MSE(rexp,rae)}')
    return {'data':np.array(rae)}


# In[ ]:


def anaerobic(thermalParams):
    param_dict = format_input(params,thermalParams)
    man = pickle.load(open(os.path.join(path,'models/anaerobic.pkl'),'rb'))
    ran = etc.simulate_growth(man,dfan_batch.index+273.15,param_dict=param_dict,sigma=0.5)
    ran = [0 if x is None else x for x in ran]
    rexp = anaerobic_exp_data()['data']
    
    logging.info(f'r2_batch_an: {r2_score(rexp,ran)}')
    logging.info(f'MSE_an: {MSE(rexp,ran)}')

    return  {'data':np.array(ran)}


# In[ ]:


def anaerobic_reduced(thermalParams):
    param_dict = format_input(params,thermalParams)
    man = pickle.load(open(os.path.join(path,'models/anaerobic.pkl'),'rb'))
    
    sel_temp = [5.0,15.0,26.3,30.0,33.0,35.0,37.5,40.0]
    ran = etc.simulate_growth(man,np.array(sel_temp)+273.15,param_dict=param_dict,sigma=0.5)
    ran = [0 if x is None else x for x in ran]
    rexp = dfan_batch.loc[sel_temp,'r_an'].values
    #anaerobic_exp_data()['data']
    
    
    logging.info(f'r2_batch_an: {r2_score(rexp,ran)}')
    logging.info(f'MSE_an: {MSE(rexp,ran)}')
    logging.info(f'Model error: {len(rexp)} {len(ran)}')

    return  {'data':np.array(ran)}


# In[ ]:


def chemostat(thermalParams):
    param_dict = format_input(params,thermalParams)
    mae = pickle.load(open(os.path.join(path,'models/aerobic.pkl'),'rb'))
    exp_flux = chemostat_exp_data()['data']
    
    growth_id = 'r_2111'
    glc_up_id = 'r_1714_REV'
    prot_pool_id = 'prot_pool_exchange'
    dilut = 0.1
    sigma = 0.5
    
    solution = etc.simulate_chemostat(mae,dilut,param_dict,dfchemo.index+273.15,
                                            sigma,growth_id,glc_up_id,prot_pool_id)

    # Extract fluxes
    rxn_lst = [
            'r_1714_REV',#Glucose
            'r_1672', #CO2
            'r_1761', # Ethanol
        ]
    columns = ['Glucose','CO2','Ethanol']

    pred_flux = []
    for rxn_id in rxn_lst:
        x = [s[rxn_id] for s in solution]
        x.extend([0]*(len(dfchemo.index)-len(x)))
        pred_flux += x
    logging.info(f'Predicted flux: {pred_flux}')
    
    # pred_flux = [0 for item in exp_flux]
    
    logging.info(f'r2_flux: {r2_score(exp_flux,pred_flux)}')
    logging.info(f'MSE_chemo: {MSE(exp_flux,pred_flux)}')

    return  {'data':np.array(pred_flux)}

# In[]

def aerobic_fva(thermalParams: candidateType, reactions: Optional[List[str]]=None):
    """
    Run FVA on aerobic conditions

    Args:
        thermalParams: A dictionary of the model's thermal parameters
    """
    # thermalParams: a dictionary with ids like uniprotid_Topt
    param_dict = format_input(params,thermalParams)
    mae = pickle.load(open(os.path.join(path,'models/aerobic.pkl'),'rb'))
    rae = etc.simulate_fva(mae,dfae_batch.index+273.15,param_dict=param_dict,sigma=0.5, reactions=reactions)
    return rae


def anaerobic_reduced_fva(thermalParams: candidateType,reactions: Optional[List[str]]=None):
    """
    Run FVA under anaerobic conditions

    Args:
        thermalParams: A dictionary of the model's thermal parameters
    """
    # thermalParams: a dictionary with ids like uniprotid_Topt 
    param_dict = format_input(params,thermalParams)
    man = pickle.load(open(os.path.join(path,'models/anaerobic.pkl'),'rb'))
    sel_temp = [5.0,15.0,26.3,30.0,33.0,35.0,37.5,40.0]
    ran = etc.simulate_fva(man,np.array(sel_temp)+273.15,param_dict=param_dict,sigma=0.5,reactions=reactions)
    return ran


def chemostat_fva(thermalParams: candidateType,reactions: Optional[List[str]]=None):
    """
    Run FVA under chemostat conditions

    Args:
        thermalParams: A dictionary of the model's thermal parameters
    """
    param_dict = format_input(params,thermalParams)
    mae = pickle.load(open(os.path.join(path,'models/aerobic.pkl'),'rb'))
    growth_id = 'r_2111'
    glc_up_id = 'r_1714_REV'
    prot_pool_id = 'prot_pool_exchange'
    dilut = 0.1
    sigma = 0.5
    
    solution = etc.fva_chemostat(mae,dilut,param_dict,dfchemo.index+273.15,
                                            sigma,growth_id,glc_up_id,prot_pool_id,reactions=reactions)
    return  solution

# In[]

def run_fva_at_three_conditions(thermalParams, reactions: Optional[List[str]]=None):
    fva_functions = [aerobic_fva, anaerobic_reduced_fva, chemostat_fva]
    condition_names = ["aerobic", "anaerobic", "chemostat"]
    fva_results: Iterable[pd.DataFrame] = starmap(lambda f, x: f(thermalParams, reactions).assign(condition=x), zip(fva_functions,condition_names))
    return pd.concat(fva_results)


def run_fva_at_two_conditions(thermalParams, reactions: Optional[List[str]]=None):
    fva_functions = [aerobic_fva, anaerobic_reduced_fva]
    condition_names = ["aerobic", "anaerobic"]
    fva_results: Iterable[pd.DataFrame] = starmap(lambda f, x: f(thermalParams, reactions).assign(condition=x), zip(fva_functions,condition_names))
    return pd.concat(fva_results)


# In[ ]:


def simulate_at_three_conditions(args,distance_function,Yobs,min_epsilon):
    
    data_batch = aerobic(args)['data']
    d_ae = distance_function(Yobs['rae'],data_batch)
    if d_ae < min_epsilon['rae']: return False, {}, {}
    
    data_batch_an = anaerobic_reduced(args)['data']
    d_an = distance_function(Yobs['ran'],data_batch_an)
    if d_an < min_epsilon['ran']: return False, {}, {}
    
    data_chemo = chemostat(args)['data']
    d_c = distance_function(Yobs['chemostat'],data_chemo)
    if d_c < min_epsilon['chemostat']: return False, {}, {}
    return True,{'rae':data_batch,'chemostat':data_chemo,'ran':data_batch_an},{'rae':d_ae,'chemostat':d_c,'ran':d_an}


# In[ ]:


def simulate_at_three_conditions_2(args):
    data_batch = aerobic(args)['data']
    data_batch_an = anaerobic_reduced(args)['data']
    data_chemo = chemostat(args)['data']
    
    return {'rae':data_batch,'chemostat':data_chemo,'ran':data_batch_an}

def simulate_at_two_conditions_2(args):
    data_batch = aerobic(args)['data']
    data_batch_an = anaerobic_reduced(args)['data']
    
    return {'rae':data_batch,'ran':data_batch_an}


# ### distance

# In[2]:


def distance(x,y):
    # x: true
    # y: simulated
    
    return r2_score(x,y)


# In[ ]:


def distance_2(x,y):
    # x: True Yobs, {'rae':...,'ran'}
    # y: simulated {'rae':...,}
    
    r2s = {k:r2_score(x[k],y[k])for k in x.keys()}
    logging.info(f'Model r2: {r2s}')
    
    return -np.mean(list(r2s.values())) # maximize r2 score, minimize -r2 score


# In[ ]:


def distance_cv1(x,y):
    # x: True Yobs, {'rae':...,'ran'}
    # y: simulated {'rae':...,}
    
    keys = ['rae','chemostat']
    r2s = {k:r2_score(x[k],y[k]) for k in x.keys()}
    logging.info(f'Model r2: {r2s}')
    
    lst = [r2s[k] for k in keys]
    logging.info(f'Model distance: {-np.mean(lst)}')
    return -np.mean(lst) # maximize r2 score, minimize -r2 score


# In[ ]:


def distance_cv2(x,y):
    # x: True Yobs, {'rae':...,'ran'}
    # y: simulated {'rae':...,}
    
    keys = ['rae','ran']
    r2s = {k:r2_score(x[k],y[k]) for k in x.keys()}
    logging.info(f'Model r2: {r2s}')
    
    lst = [r2s[k] for k in keys]
    logging.info(f'Model distance: {-np.mean(lst)}')
    return -np.mean(lst) # maximize r2 score, minimize -r2 score


# In[ ]:


def distance_cv3(x,y):
    # x: True Yobs, {'rae':...,'ran'}
    # y: simulated {'rae':...,}
    
    keys = ['chemostat','ran']
    r2s = {k:r2_score(x[k],y[k]) for k in x.keys()}
    logging.info(f'Model r2: {r2s}')
    
    lst = [r2s[k] for k in keys]
    logging.info(f'Model distance: {-np.mean(lst)}')
    return -np.mean(lst) # maximize r2 score, minimize -r2 score


# def judge(distance,min_epsilon):
#     # distance = {'rae':0.3, 'ran': 4, 'chemostat': 45}
#     # min_epsilon = {'rae':0.03, 'ran': 3, 'chemostat': 5}
#     # return true if all distances smaller than targets
#     jd = True
#     for k in distance.keys():
#         if distance[k]<min_epsilon[k]: jd = False
#     return jd
