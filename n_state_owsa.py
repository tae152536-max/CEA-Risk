import numpy as np
import pandas as pd
from n_state_markov import compare_n_state_strategies

def _renormalize_row(row, idx_changed, new_val):
    new_row = np.array(row).copy()
    original_val = new_row[idx_changed]
    new_val = max(0.0, min(1.0, new_val))
    new_row[idx_changed] = new_val
    
    remainder_old = 1.0 - original_val
    remainder_new = 1.0 - new_val
    
    if remainder_old <= 0.0001:
        # If the rest of the row was basically 0, distribute evenly
        n = len(row)
        if n > 1:
            dist = remainder_new / (n - 1)
            for i in range(n):
                if i != idx_changed:
                    new_row[i] = dist
    else:
        # Scale remaining elements proportionally
        ratio = remainder_new / remainder_old
        for i in range(len(row)):
            if i != idx_changed:
                new_row[i] *= ratio
    
    # Ensure strict summation to 1.0
    return new_row / new_row.sum()

def run_n_state_owsa(std_base, new_base, state_names, n_cycles, wtp, discount_rate, owsa_var=0.20):
    """
    Runs One-Way Sensitivity Analysis on a generic N-State Markov model.
    """
    results_list = []
    
    # Baseline INMB
    base_res = compare_n_state_strategies(std_base, new_base, state_names, n_cycles, wtp, discount_rate)
    baseline_inmb = base_res['inmb']
    
    n_states = len(state_names)
    
    # helper for evaluation
    def eval_model(std_p, std_costs, std_u, new_p, new_costs, new_u):
        std_prm = {'p_matrix': std_p, 'cost_df': std_costs, 'utilities': std_u}
        new_prm = {'p_matrix': new_p, 'cost_df': new_costs, 'utilities': new_u}
        return compare_n_state_strategies(std_prm, new_prm, state_names, n_cycles, wtp, discount_rate)['inmb']
        
    def add_result(p_name, base_val, low_val, high_val, inmb_low, inmb_high):
        results_list.append({
            'Parameter': p_name,
            'Base': base_val,
            'Low': low_val,
            'High': high_val,
            'INMB_Low': inmb_low,
            'INMB_High': inmb_high,
            'Swing': abs(inmb_high - inmb_low)
        })

    # 1. Vary Utilities
    for mode, base_struct in [("Std", std_base), ("New", new_base)]:
        for i, s_name in enumerate(state_names):
            base_u = base_struct['utilities'][i]
            if base_u > 0:
                low_u = max(0.0, base_u * (1 - owsa_var))
                high_u = min(1.0, base_u * (1 + owsa_var))
                
                std_u_eval = list(std_base['utilities'])
                new_u_eval = list(new_base['utilities'])
                
                if mode == "Std":
                    std_u_eval[i] = low_u
                    inmb_low = eval_model(std_base['p_matrix'], std_base['cost_df'], std_u_eval, new_base['p_matrix'], new_base['cost_df'], new_base['utilities'])
                    std_u_eval[i] = high_u
                    inmb_high = eval_model(std_base['p_matrix'], std_base['cost_df'], std_u_eval, new_base['p_matrix'], new_base['cost_df'], new_base['utilities'])
                else:
                    new_u_eval[i] = low_u
                    inmb_low = eval_model(std_base['p_matrix'], std_base['cost_df'], std_base['utilities'], new_base['p_matrix'], new_base['cost_df'], new_u_eval)
                    new_u_eval[i] = high_u
                    inmb_high = eval_model(std_base['p_matrix'], std_base['cost_df'], std_base['utilities'], new_base['p_matrix'], new_base['cost_df'], new_u_eval)
                    
                add_result(f"{mode} Utility: {s_name}", base_u, low_u, high_u, inmb_low, inmb_high)

    # 2. Vary Transition Probabilities
    for mode, base_struct in [("Std", std_base), ("New", new_base)]:
        p_mat = np.array(base_struct['p_matrix'])
        for r in range(n_states):
            for c in range(n_states):
                base_p = p_mat[r, c]
                if base_p > 0 and base_p < 1:  # Only vary if it's not deterministic 0 or 1
                    low_p = max(0.001, base_p * (1 - owsa_var))
                    high_p = min(0.999, base_p * (1 + owsa_var))
                    
                    std_p_eval = np.array(std_base['p_matrix']).copy()
                    new_p_eval = np.array(new_base['p_matrix']).copy()
                    
                    if mode == "Std":
                        std_p_eval[r] = _renormalize_row(std_p_eval[r], c, low_p)
                        inmb_low = eval_model(std_p_eval, std_base['cost_df'], std_base['utilities'], new_base['p_matrix'], new_base['cost_df'], new_base['utilities'])
                        std_p_eval[r] = _renormalize_row(std_p_eval[r], c, high_p)
                        inmb_high = eval_model(std_p_eval, std_base['cost_df'], std_base['utilities'], new_base['p_matrix'], new_base['cost_df'], new_base['utilities'])
                    else:
                        new_p_eval[r] = _renormalize_row(new_p_eval[r], c, low_p)
                        inmb_low = eval_model(std_base['p_matrix'], std_base['cost_df'], std_base['utilities'], new_p_eval, new_base['cost_df'], new_base['utilities'])
                        new_p_eval[r] = _renormalize_row(new_p_eval[r], c, high_p)
                        inmb_high = eval_model(std_base['p_matrix'], std_base['cost_df'], std_base['utilities'], new_p_eval, new_base['cost_df'], new_base['utilities'])
                        
                    add_result(f"{mode} Prob: {state_names[r]} -> {state_names[c]}", base_p, low_p, high_p, inmb_low, inmb_high)

    # 3. Vary Totals Costs (Multipliers)
    # Generic OWSA on costs can just use a global multiplier for all costs in a strategy to avoid overwhelming the tornado
    for mode, base_struct in [("Std", std_base), ("New", new_base)]:
        base_cost_df = base_struct['cost_df']
        
        low_cost_df = base_cost_df.copy()
        low_cost_df['Cost ($/year)'] *= (1 - owsa_var)
        
        high_cost_df = base_cost_df.copy()
        high_cost_df['Cost ($/year)'] *= (1 + owsa_var)
        
        if mode == "Std":
            inmb_low = eval_model(std_base['p_matrix'], low_cost_df, std_base['utilities'], new_base['p_matrix'], new_base['cost_df'], new_base['utilities'])
            inmb_high = eval_model(std_base['p_matrix'], high_cost_df, std_base['utilities'], new_base['p_matrix'], new_base['cost_df'], new_base['utilities'])
        else:
            inmb_low = eval_model(std_base['p_matrix'], std_base['cost_df'], std_base['utilities'], new_base['p_matrix'], low_cost_df, new_base['utilities'])
            inmb_high = eval_model(std_base['p_matrix'], std_base['cost_df'], std_base['utilities'], new_base['p_matrix'], high_cost_df, new_base['utilities'])
                        
        add_result(f"{mode} Total Cost Multiplier", 1.0, 1.0 - owsa_var, 1.0 + owsa_var, inmb_low, inmb_high)
                
    owsa_df = pd.DataFrame(results_list)
    owsa_df = owsa_df.sort_values(by='Swing', ascending=False).reset_index(drop=True)
    return owsa_df, baseline_inmb
