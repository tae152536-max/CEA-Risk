import numpy as np
import pandas as pd
from markov_model import compare_strategies

def build_p_matrix(p_event, p_dead_w, p_dead_p):
    p_well = max(0, 1 - p_event - p_dead_w)
    p_pe_pe = max(0, 1 - p_dead_p)
    return [
        [p_well, p_event, p_dead_w],
        [0.0, p_pe_pe, p_dead_p],
        [0.0, 0.0, 1.0]
    ]

def eval_model(params, std_base_costs, new_base_costs):
    n_cycles = int(params['time_horizon'])
    wtp = params['wtp']
    discount_rate = params['discount_rate']
    
    std_p = build_p_matrix(params['std_prob_event'], params['std_prob_dead_w'], params['std_prob_dead_p'])
    new_p = build_p_matrix(params['new_prob_event'], params['new_prob_dead_w'], params['new_prob_dead_p'])
    
    std_utilities = [params['std_qaly_well'], params['std_qaly_post'], 0.0]
    new_utilities = [params['new_qaly_well'], params['new_qaly_post'], 0.0]
    
    # Scale costs
    std_costs = []
    for c_dict in std_base_costs:
        std_costs.append({k: v * params['std_cost_multiplier'] for k, v in c_dict.items()})
        
    new_costs = []
    for c_dict in new_base_costs:
        new_costs.append({k: v * params['new_cost_multiplier'] for k, v in c_dict.items()})
        
    std_p_dict = {'p_matrix': std_p, 'costs': std_costs, 'utilities': std_utilities}
    new_p_dict = {'p_matrix': new_p, 'costs': new_costs, 'utilities': new_utilities}
    
    res = compare_strategies(std_p_dict, new_p_dict, n_cycles, wtp, discount_rate)
    return res['inmb']

def run_owsa(base_params, std_base_costs, new_base_costs, variance=0.20):
    results = []
    base_inmb = eval_model(base_params, std_base_costs, new_base_costs)
    
    for param_name, base_val in base_params.items():
        if 'cost_multiplier' in param_name:
            low_val = 1.0 - variance
            high_val = 1.0 + variance
        elif 'prob' in param_name or 'qaly' in param_name:
            low_val = max(0.001, base_val * (1 - variance))
            high_val = min(0.999, base_val * (1 + variance))
        elif 'time_horizon' in param_name:
            low_val = int(max(1, base_val * (1 - variance)))
            high_val = int(base_val * (1 + variance))
            if low_val == high_val: high_val += 1
        else: # wtp, discount rate
            low_val = base_val * (1 - variance)
            high_val = base_val * (1 + variance)
            
        p_low = base_params.copy()
        p_low[param_name] = low_val
        inmb_low = eval_model(p_low, std_base_costs, new_base_costs)
        
        p_high = base_params.copy()
        p_high[param_name] = high_val
        inmb_high = eval_model(p_high, std_base_costs, new_base_costs)
        
        swing = abs(inmb_high - inmb_low)
        
        results.append({
            'Parameter': param_name,
            'Base': base_val,
            'Low': low_val,
            'High': high_val,
            'INMB_Low': inmb_low,
            'INMB_High': inmb_high,
            'Swing': swing
        })
        
    df = pd.DataFrame(results)
    df = df.sort_values(by='Swing', ascending=False).reset_index(drop=True)
    return df, base_inmb

def find_switching_value(param_name, base_params, std_base_costs, new_base_costs, tolerance=1.0):
    low_bound = 0.001
    high_bound = 1.0 if ('prob' in param_name or 'qaly' in param_name) else 5.0
    if 'time_horizon' in param_name: high_bound = 100.0
    if 'wtp' in param_name: high_bound = base_params['wtp'] * 5.0
    
    p_low = base_params.copy()
    p_low[param_name] = low_bound
    inmb_low = eval_model(p_low, std_base_costs, new_base_costs)
    
    p_high = base_params.copy()
    p_high[param_name] = high_bound
    inmb_high = eval_model(p_high, std_base_costs, new_base_costs)
    
    if (inmb_low > 0 and inmb_high > 0) or (inmb_low < 0 and inmb_high < 0):
        return None 
        
    for _ in range(50):
        mid = (low_bound + high_bound) / 2.0
        p_mid = base_params.copy()
        p_mid[param_name] = mid
        inmb_mid = eval_model(p_mid, std_base_costs, new_base_costs)
        
        if abs(inmb_mid) < tolerance:
            return mid
            
        if (inmb_low > 0 and inmb_mid < 0) or (inmb_low < 0 and inmb_mid > 0):
            high_bound = mid
            inmb_high = inmb_mid
        else:
            low_bound = mid
            inmb_low = inmb_mid
            
    return (low_bound + high_bound) / 2.0
