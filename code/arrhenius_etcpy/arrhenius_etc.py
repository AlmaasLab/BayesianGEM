from cobra.exceptions import OptimizationError
import numpy as np
from scipy.optimize import fsolve
import pandas as pd
import time
import logging

R = 8.314


def calculate_kcatT(T, Ea, A=1):
    """ Compute kcat value according
    to the Arrhenius equation

    Args:
        T ([float]): Temperature of reaction (K)
        Ea ([float]): Activation energy (J)
        A ([float]): Base catalytic rate at infinite temperature

    Returns:
        [float]: kcat value
    """
    return A * np.exp(- Ea / (R*T))


def u(T, dHeq, Tm):
    """
    Helper function for computing denaturation
    """
    return dHeq / R * (1 / Tm - 1 /T)


def calculate_fNT(T, dHeq, Tm):
    """
    Calculate the fraction of enzyme in native state
    """
    return 1 / (1 + np.exp(u(T, dHeq, Tm)))


def calculate_rate(T, Ea, dHeq, Tm, A=1):
    return calculate_kcatT(T=T, Ea=Ea, A=A) * calculate_fNT(T=T, dHeq=dHeq, Tm=Tm)





def get_Ea_dHeq_from_Topt_Tm_T90(Topt,Tm, T90):
    '''
    # With knowing Topt and Tm, we can compute the activation energies
    # using:
    # 
    # d / dT rate(T_opt) = 0
    # fNT(Tm) = 0.5 (already covered by the formula itself)
    # fNT(T90) = 0.1
    # Topt, Tm are in K
    '''

    dHeq = R*T90*Tm*np.log(9) / (T90 - Tm)
    # The original formula was 
    # dHeq*np.exp(dHeq/(R*Tm)) / (np.exp(dHeq/(R*Topt)) + np.exp(dHeq/ (R*Tm))),
    # but proved to be too numerically instable
    Ea = dHeq*calculate_fNT(T=T90,dHeq=dHeq,Tm=Tm)


    return Ea, dHeq 
    



def change_rxn_coeff(rxn,met,new_coeff):
    '''
    # This is based on the rxn.add_metabolites function. If there the metabolite is already in the reaction,
    # new and old coefficients will be added. For example, if the old coeff of metA is 1, use
    # rxn.add_metabolites({metA:2}), After adding, the coeff of metA is 1+2 = 3
    #
    '''

    diff_coeff = new_coeff-rxn.metabolites[met]
    rxn.add_metabolites({met:diff_coeff})





def map_fNT(model,T,df,Tadj=0):
    '''
    # apply the fraction of enzymes in native state to each protein.
    # model, cobra model
    # T, temperature, in K, float
    # Tadj, This is to adjust the orginal denaturation curve by moving to left by
    # Tadj degrees.
    # df, a dataframe containing thermal parameters of enzymes: dHTH, dSTS, dCpu
    #
    #
    # Gang Li, 2019-05-03


    # in the enzyme constrained model, protein usage was describe as reaction
    # draw_prot_P0CW41 68.69778 prot_pool --> prot_P0CW41. 68.69778 is the molecular weight of protein P0CW41
    # the total enzyme pool was constrained by the upper bound of reaction prot_pool_exchange:
    #          --> prot_pool
    # To map fNT, change the coefficient of prot_pool in reactions like MW*prot_pool --> prot_P0CW41,
    #                                MW/fNT prot_pool --> prot_P0CW41

    '''

    met = model.metabolites.prot_pool

    for rxn in met.reactions:
        # this is to ignore reaction 'prot_pool_exchange': --> prot_pool
        if len(rxn.metabolites)<2: continue

        uniprot_id = rxn.id.split('_')[-1]
        cols = ['dHeq', 'Tm']
        dHeq, Tm =df.loc[uniprot_id,cols]
        fNT = calculate_fNT(T+Tadj, dHeq,Tm=Tm)
        if fNT < 1e-32: fNT = 1e-32
        new_coeff = rxn.metabolites[met]/fNT
        
        change_rxn_coeff(rxn,met,new_coeff)
        



def map_kcatT(model,T,df):
    '''
    # Apply temperature effect on enzyme kcat.
    # based on Arrhenius equation
    # model, cobra model
    # T, temperature, in K
    # df, a dataframe containing thermal parameters of enzymes: Ea, dHeq, Topt
    # Ensure that Topt is in K. Other parameters are in standard units.
    #
    # Jakob Peder Pettersen, 2021-10-28
    #
    '''
    for met in model.metabolites:

        # look for those metabolites: prot_uniprotid
        if not met.id.startswith('prot_'): continue

        # ingore metabolite: prot_pool
        if met.id == 'prot_pool': continue
        uniprot_id = met.id.split('_')[1]

        # Change kcat value.
        # pmet_r_0001 + 1.8518518518518518e-07 prot_P00044 + 1.8518518518518518e-07 prot_P32891 -->
        # 2.0 s_0710 + s_1399
        #
        # 1.8518518518518518e-07 is correponding to 1/kcat
        # change the kcat to kcat(T)
        # In some casese, this coefficient could be 2/kcat or some other values. This doesn't matter.
        #
        # a protein could be involved in several reactions
        cols = ['Ea', 'Topt', 'dHeq', 'Tm']
        [Ea, Topt, dHeq, Tm] = df.loc[uniprot_id,cols]


        for rxn in met.reactions:
            if rxn.id.startswith('draw_prot'): continue

            # assume that Topt in the original model is measured at Topt
            kcatTopt = -1/rxn.metabolites[met]


            kcatT = calculate_kcatT(T, Ea) * kcatTopt / (calculate_kcatT(Topt, Ea) * calculate_fNT(T=Topt, dHeq=dHeq, Tm=Tm))
            if kcatT < 1e-32: kcatT = 1e-32
            new_coeff = -1/kcatT

            change_rxn_coeff(rxn,met,new_coeff)


def getNGAMT(T):
    # T is in K, a single value
    def NGAM_function(T):
        return 0.740 + 5.893/(1+np.exp(31.920-(T-273.15))) + 6.12e-6*(T-273.15-16.72)**4

    lb = 5+273.15
    ub = 40+273.15
    if T < lb: NGAM_T = NGAM_function(lb)
    elif T > ub: NGAM_T =NGAM_function(ub)
    else: NGAM_T = NGAM_function(T)

    return NGAM_T


def set_NGAMT(model,T):
    # T is in K
    NGAM_T = getNGAMT(T)
    rxn = model.reactions.NGAM
    #ori_lb,ori_ub = rxn.lower_bound,rxn.upper_bound
    rxn.lower_bound = NGAM_T
    rxn.upper_bound = NGAM_T


def set_sigma(model,sigma):
    rxn = model.reactions.prot_pool_exchange
    #ori_ub_sigma = rxn.upper_bound
    rxn.upper_bound = 0.17866*sigma


def simulate_growth(model,Ts,sigma,df,Tadj=0):
    '''
    # model, cobra model
    # Ts, a list of temperatures in K
    # sigma, enzyme saturation factor
    # df, a dataframe containing thermal parameters of enzymes: dHTH, dSTS, dCpu, Topt
    # Ensure that Topt is in K. Other parameters are in standard units.
    # Tadj, as descrbed in map_fNT
    #
    '''
    rs = list()
    for T in Ts:
        with model:
            # map temperature constraints
            map_fNT(model,T,df)
            map_kcatT(model,T,df)
            set_NGAMT(model,T)
            set_sigma(model,sigma)

            try:
                 r = model.optimize().objective_value
                 logging.info("Model solved successfully")
            except OptimizationError as err:
                logging.info(f'Failed to solve the problem, problem: {str(err)}')
                r = 0
            print(T-273.15,r)
            rs.append(r)
    return rs


def sample_data_uncertainty(params,columns=None):
    '''
    # params is a dataframe with following columns:
    # Tm,Tm_std:  melting temperature. Given in K
    #
    # Topt,Topt_std: the optimal temprature at which the specific activity is maximized. Given in K
    #
    # T90, temperature at which 90% of an enzyme is denatured. Given in K
    #
    # xx_std, corresponding uncertainty given by standard deviation.
    # 
    # columns: a list of columns to be sampled, could be any combination of ['Tm','Topt]. 
    #          If it is None, then sample all three columns
    # 
    # The script will return an new dataframe with the same columns but with randomly sampled data
    '''
    sampled_params = params.copy()
    if columns is None: columns = ['Tm','Topt']

    for col in columns:
        sampled_params[col] = np.random.normal(params[col],params[col+'_std'])
        if col == 'Tm':
                sampled_params['T90'] = sampled_params[col] + params['T90']-params[col]
    return sampled_params
    

def sample_data_uncertainty_with_constraint(inpt,columns=None):
    if type(inpt)==tuple:
        params,seed = inpt
        np.random.seed(seed+int(time.time()))
    else: params = inpt
    '''
    # params is a dataframe with following columns:
    # Tm,Tm_std:  melting temperature. Given in K
    #
    # Topt,Topt_std: the optimal temprature at which the specific activity is maximized. Given in K
    #
    # xx_std, corresponding uncertainty given by standard deviation.
    # 
    # columns: a list of columns to be sampled, could be any combination of ['Tm', 'Topt]. 
    #          If it is None, then sample all three columns
    # 
    # The script will return an new dataframe with the same columns but with randomly sampled data
    '''
    
    sampled_params = params.copy()
    if columns is None: columns = ['Tm', 'Topt']
    for col in columns:
        lst = np.random.normal(params[col], params[col+'_std'])
        sampled_params[col] = lst
          
    # resample those ones with Topt>=Tm
    for ind in sampled_params.index:
        tm,topt = sampled_params.loc[ind,'Tm'],sampled_params.loc[ind,'Topt']
        count = 0
        while topt>=tm:
            count += 1
            if 'Topt' in columns:
                topt = np.random.normal(params.loc[ind,'Topt'],params.loc[ind,'Topt_std'])
            if 'Tm' in columns: 
                tm = np.random.normal(params.loc[ind,'Tm'],params.loc[ind,'Tm_std'])
            if 'Topt' not in columns and 'Tm' not in columns:
                break
            if count>10: break
        sampled_params.loc[ind,'Tm'],sampled_params.loc[ind,'Topt'] = tm,topt
    
    sampled_params['T90'] = sampled_params['Tm']+params['T90']-params['Tm']
    return sampled_params


def calculate_thermal_params(params):
    '''
    # params, a dataframe with at least following columns: Tm, Topt. All are in standard units.
    # 
    # The script will return a dataframe with following columns: Ea, dHeq, Topt, Tm
    # 
    '''
    thermalparams = pd.DataFrame()
    thermalparams.index = params.index
    
    # step 1: calculate Ea,dHeq from tm, t90
    Tm, Topt, T90 = params[['Tm', 'Topt', 'T90']].T.to_numpy()
    Ea, dHeq = get_Ea_dHeq_from_Topt_Tm_T90(Topt=Topt, Tm=Tm, T90=T90)
    thermalparams['Ea'] = Ea
    thermalparams['dHeq'] = dHeq
        
    # step 2. copy columns Topt and Tm
    thermalparams['Tm'] = params['Tm']
    thermalparams['Topt'] = params['Topt']
    
    return thermalparams
        
        
def simulate_chomostat(model,dilu,params,Ts,sigma,growth_id,glc_up_id,prot_pool_id):
    '''
    # Do simulation on a given dilution and a list of temperatures. 
    # model, cobra model
    # dilu, dilution rate
    # params: a dataframe containing Tm, T90, Length, dCpt, Topt. All temperatures are in K.
    # Ts, a list of temperatures to simulate at. in K
    # sigma, saturation factor
    # growth_id, reaction of of growth
    # glc_up_id, reaction id of glucose uptake reaction
    # prot_pool_id, reaction id of prot_pool_exchange. --> prot_pool
    '''
    solutions = list() # corresponding to Ts. a list of solutions from model.optimize()
    df = calculate_thermal_params(params)
    
    with model as m0:
        # Step 1: fix growth rate, set objective function as minimizing glucose uptatke rate
        rxn_growth = m0.reactions.get_by_id(growth_id)
        rxn_growth.lower_bound = dilu

        m0.objective = glc_up_id
        m0.objective.direction = 'min'

        for T in Ts:
            with m0 as m1:  
                # Step 2: map temperature constraints. 
                map_fNT(m1,T,df)
                map_kcatT(m1,T,df)
                set_NGAMT(m1,T)
                set_sigma(m1,sigma)
                
                try: 
                    # Step 3: minimize the glucose uptake rate. Fix glucose uptake rate, minimize enzyme usage
                    solution1 = m1.optimize()
                    m1.reactions.get_by_id(glc_up_id).upper_bound = solution1.objective_value*1.001
                    m1.objective = prot_pool_id
                    m1.objective.direction = 'min'
                    
                    solution2 = m1.optimize()
                    solutions.append(solution2)
                    logging.info('Model solved successfully')
                except OptimizationError as err:
                    logging.info(f'Failed to solve the problem, problem: {str(err)}')
                    #solutions.append(None)
                    break # because model has been impaired. Further simulation won't give right output.
    return solutions
