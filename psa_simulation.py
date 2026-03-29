import numpy as np
import scipy.stats as stats
import pandas as pd
from markov_model import compare_strategies, STATES

def sample_gamma(mean, se):
    if mean <= 0: return 0.0
    if se <= 0: return mean
    var = se**2
    shape = (mean**2) / var
    scale = var / mean
    return np.random.gamma(shape, scale)

def sample_beta(mean, se):
    if mean <= 0: return 0.0
    if mean >= 1: return 1.0
    if se <= 0: return mean
    var = se**2
    term = (mean * (1 - mean) / var) - 1
    if term < 0: return mean
    alpha = mean * term
    beta = (1 - mean) * term
    return np.random.beta(alpha, beta)

def sample_normal(mean, se):
    if se <= 0: return mean
    val = np.random.normal(mean, se)
    return max(0.0, val)

def sample_lognormal(mean, se):
    if mean <= 0: return 0.0
    if se <= 0: return mean
    var = se**2
    mu = np.log(mean**2 / np.sqrt(var + mean**2))
    sigma = np.sqrt(np.log(var / mean**2 + 1))
    return np.random.lognormal(mu, sigma)

def sample_uniform(mean, se):
    if se <= 0: return mean
    width = np.sqrt(12 * (se**2))
    low = max(0.0, mean - width/2)
    high = mean + width/2
    return np.random.uniform(low, high)

def sample_triangular(mean, se):
    if se <= 0: return mean
    offset = se * 2.449
    low = max(0.0, mean - offset)
    high = mean + offset
    return np.random.triangular(low, mean, high)

def process_cost_df(df):
    """
    Takes the dynamic cost dataframe, samples each row according to its
    Distribution, and returns the standard cost_list of dicts.
    """
    states = ["Well", "Post-Event", "Dead"]
    cost_list = []
    subgroups = ["Medical", "Non-Medical", "Indirect"]
    
    for s in states:
        s_df = df[df['State'] == s]
        c_dict = {sub: 0.0 for sub in subgroups}
        
        for _, row in s_df.iterrows():
            base_cost = float(row.get('Cost ($/year)', 0))
            se = float(row.get('SE/SD', 0))
            dist = row.get('Distribution', 'Fixed')
            subg = row.get('Subgroup', 'Medical')
            
            # Draw sample based on distribution
            if dist == 'Gamma':
                val = sample_gamma(base_cost, se)
            elif dist == 'Lognormal':
                val = sample_lognormal(base_cost, se)
            elif dist == 'Normal':
                val = sample_normal(base_cost, se)
            elif dist == 'Uniform':
                val = sample_uniform(base_cost, se)
            elif dist == 'Triangular':
                val = sample_triangular(base_cost, se)
            elif dist == 'Beta':
                # Scale beta bound to 2*mean if cost is > 1
                if base_cost <= 1.0:
                    val = sample_beta(base_cost, se)
                else: 
                    norm_var = (se / (2*base_cost))**2
                    val = sample_beta(0.5, np.sqrt(norm_var)) * (2 * base_cost)
            else: # Fixed
                val = base_cost
                
            if subg in c_dict:
                c_dict[subg] += val
                
        cost_list.append(c_dict)
    return cost_list

def run_psa(std_base, new_base, n_cycles, wtp, n_iterations=1000):
    """Runs Probabilistic Sensitivity Analysis."""
    results = []
    
    for i in range(n_iterations):
        # Dynamically sample from tables for every iteration!
        std_costs = process_cost_df(std_base['cost_df'])
        new_costs = process_cost_df(new_base['cost_df'])
        
        std_utils = [
            sample_beta(std_base['utilities'][0], std_base['utilities'][0] * 0.05),
            sample_beta(std_base['utilities'][1], std_base['utilities'][1] * 0.05),
            0.0
        ]
        new_utils = [
            sample_beta(new_base['utilities'][0], new_base['utilities'][0] * 0.05),
            sample_beta(new_base['utilities'][1], new_base['utilities'][1] * 0.05),
            0.0
        ]
        
        def noisy_p_matrix(base_p):
            p = np.array(base_p).copy()
            noise = np.random.normal(0, 0.05, p.shape)
            p = p + noise
            p = np.clip(p, 0.001, 0.999)
            p[2, :] = [0, 0, 1]
            p[1, 0] = 0
            row_sums = p.sum(axis=1)
            return (p.T / row_sums).T

        std_p = noisy_p_matrix(std_base['p_matrix'])
        new_p = noisy_p_matrix(new_base['p_matrix'])
        
        std_params = {'p_matrix': std_p, 'costs': std_costs, 'utilities': std_utils}
        new_params = {'p_matrix': new_p, 'costs': new_costs, 'utilities': new_utils}
        
        res = compare_strategies(std_params, new_params, n_cycles, wtp)
        results.append({
            'Iteration': i + 1,
            'Inc_Cost': res['inc_cost'],
            'Inc_QALY': res['inc_qaly'],
            'ICER': res['icer'],
            'INMB': res['inmb'],
            'NMB_Std': res['nmb_std'],
            'NMB_New': res['nmb_new']
        })
        
    return pd.DataFrame(results)
