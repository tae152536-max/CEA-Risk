import numpy as np
import scipy.stats as stats
import pandas as pd
from n_state_markov import compare_n_state_strategies

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

def prepare_cost_samplers(df, state_names):
    samplers = []
    for _, row in df.iterrows():
        base_cost = float(row.get('Cost ($/year)', 0))
        se = float(row.get('SE/SD', 0))
        dist = row.get('Distribution', 'Fixed')
        state_name = row['State']
        if state_name in state_names:
            state_idx = state_names.index(state_name)
        else:
            continue
        subg = row.get('Subgroup', 'Medical')
        if subg not in ["Medical", "Non-Medical", "Indirect"]:
            subg = "Medical"
        samplers.append((state_idx, subg, dist, base_cost, se))
    return samplers

def sample_costs(samplers, n_states):
    subgroup_cost_arrays = {"Medical": np.zeros(n_states), "Non-Medical": np.zeros(n_states), "Indirect": np.zeros(n_states)}
    for state_idx, subg, dist, base_cost, se in samplers:
        if dist == 'Gamma': val = sample_gamma(base_cost, se)
        elif dist == 'Lognormal': val = sample_lognormal(base_cost, se)
        elif dist == 'Normal': val = sample_normal(base_cost, se)
        elif dist == 'Uniform': val = sample_uniform(base_cost, se)
        elif dist == 'Triangular': val = sample_triangular(base_cost, se)
        elif dist == 'Beta':
            if base_cost <= 1.0: val = sample_beta(base_cost, se)
            else: val = sample_beta(0.5, np.sqrt((se / (2*base_cost))**2)) * (2 * base_cost)
        else: val = base_cost
        subgroup_cost_arrays[subg][state_idx] += val
    return subgroup_cost_arrays

def run_n_state_psa(std_base, new_base, state_names, n_cycles, wtp, n_iterations=1000):
    """Runs Probabilistic Sensitivity Analysis for generic N-State Model."""
    results = []
    n_states = len(state_names)
    
    std_samplers = prepare_cost_samplers(std_base['cost_df'], state_names)
    new_samplers = prepare_cost_samplers(new_base['cost_df'], state_names)
    
    for i in range(n_iterations):
        # Dynamically sample from precompiled pure-python paths!
        std_precomputed_costs = sample_costs(std_samplers, n_states)
        new_precomputed_costs = sample_costs(new_samplers, n_states)
        
        std_utils = []
        new_utils = []
        for u in std_base['utilities']:
            # Assuming ~5% SE for utilities generally if not specified
            std_utils.append(sample_beta(u, max(u * 0.05, 0.001)))
        for u in new_base['utilities']:
            new_utils.append(sample_beta(u, max(u * 0.05, 0.001)))
            
        def noisy_p_matrix(base_p):
            p = np.array(base_p).copy()
            noise = np.random.normal(0, 0.05, p.shape)
            p = p + noise
            p = np.clip(p, 0.0001, 0.9999)
            # Re-normalize rows to strictly sum to 1
            row_sums = p.sum(axis=1)
            return p / row_sums[:, np.newaxis]

        std_p = noisy_p_matrix(std_base['p_matrix'])
        new_p = noisy_p_matrix(new_base['p_matrix'])
        
        std_params = {'p_matrix': std_p, 'cost_df': None, 'precomputed_cost_arrays': std_precomputed_costs, 'utilities': std_utils}
        new_params = {'p_matrix': new_p, 'cost_df': None, 'precomputed_cost_arrays': new_precomputed_costs, 'utilities': new_utils}
        
        res = compare_n_state_strategies(std_params, new_params, state_names, n_cycles, wtp)
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
