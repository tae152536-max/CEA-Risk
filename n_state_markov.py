import numpy as np
import pandas as pd

def run_n_state_markov(p_matrix, cost_df, utility_list, state_names, n_cycles, initial_cohort=None, discount_rate=0.03, precomputed_cost_arrays=None):
    """
    Runs a generic N-state Markov cohort model.
    """
    n_states = len(state_names)
    if initial_cohort is None:
        initial_cohort = np.zeros(n_states)
        initial_cohort[0] = 1.0 # Default all start in state 0
        
    trace = np.zeros((n_cycles + 1, n_states))
    trace[0, :] = initial_cohort
    
    total_cost = 0.0
    total_qaly = 0.0
    
    # Calculate costs per state by summing all subgroups for each state
    # cost_df format: State, Subgroup, Item, Cost ($/year), Distribution, SE/SD
    state_costs = np.zeros(n_states)
    cost_subtypes = {"Medical": 0.0, "Non-Medical": 0.0, "Indirect": 0.0}
    
    # Pre-compute cost arrays for each subgroup to drastically speed up simulation loops
    if precomputed_cost_arrays is not None:
        subgroup_cost_arrays = precomputed_cost_arrays
        # Compute state_costs from the subgroup arrays
        for i in range(n_states):
            state_costs[i] = sum(subgroup_cost_arrays[sub][i] for sub in cost_subtypes.keys())
    else:
        subgroup_cost_arrays = {}
        for sub in cost_subtypes.keys():
            subgroup_cost_arrays[sub] = np.array([cost_df[(cost_df['State'] == s) & (cost_df['Subgroup'] == sub)]['Cost ($/year)'].sum() for s in state_names])
        for i, state in enumerate(state_names):
            s_df = cost_df[cost_df['State'] == state]
            state_costs[i] = s_df['Cost ($/year)'].sum()
        
    # Initial Half-cycle correction (Cycle 0)
    total_qaly += np.sum(trace[0, :] * np.array(utility_list)) * 0.5
    for sub in cost_subtypes.keys():
        c_t = np.sum(trace[0, :] * subgroup_cost_arrays[sub]) * 0.5
        cost_subtypes[sub] += c_t
        total_cost += c_t
        
    p_matrix = np.array(p_matrix)
    
    for t in range(1, n_cycles + 1):
        trace[t, :] = np.dot(trace[t-1, :], p_matrix)
        df_discount = 1 / ((1 + discount_rate) ** t)
        
        q_t = np.sum(trace[t, :] * np.array(utility_list)) * df_discount
        if t == n_cycles:
            total_qaly += q_t * 0.5
        else:
            total_qaly += q_t
            
        for sub in cost_subtypes.keys():
            c_t = np.sum(trace[t, :] * subgroup_cost_arrays[sub]) * df_discount
            if t == n_cycles:
                c_t *= 0.5
            cost_subtypes[sub] += c_t
            total_cost += c_t
            
    return {
        "total_cost": total_cost,
        "cost_subtypes": cost_subtypes,
        "total_qaly": total_qaly,
        "trace": trace
    }

def calculate_nmb(cost, qaly, wtp):
    """Calculates Net Monetary Benefit"""
    return (wtp * qaly) - cost

def compare_n_state_strategies(std_params, new_params, state_names, n_cycles, wtp, discount_rate=0.03):
    """
    Runs both generic strategies and computes incremental values.
    """
    res_std = run_n_state_markov(
        std_params['p_matrix'], std_params['cost_df'], std_params['utilities'],
        state_names, n_cycles, std_params.get('initial_cohort', None), discount_rate,
        precomputed_cost_arrays=std_params.get('precomputed_cost_arrays', None)
    )
    
    res_new = run_n_state_markov(
        new_params['p_matrix'], new_params['cost_df'], new_params['utilities'],
        state_names, n_cycles, new_params.get('initial_cohort', None), discount_rate,
        precomputed_cost_arrays=new_params.get('precomputed_cost_arrays', None)
    )
    
    inc_cost = res_new['total_cost'] - res_std['total_cost']
    inc_qaly = res_new['total_qaly'] - res_std['total_qaly']
    icer = inc_cost / inc_qaly if inc_qaly > 0 else (float('inf') if inc_qaly == 0 else -1)
    # Note: negatives ICER usually means dominance. We handle display logic downstream.
    
    nmb_std = calculate_nmb(res_std['total_cost'], res_std['total_qaly'], wtp)
    nmb_new = calculate_nmb(res_new['total_cost'], res_new['total_qaly'], wtp)
    inmb = nmb_new - nmb_std
    
    return {
        "std_cost": res_std['total_cost'],
        "std_cost_subtypes": res_std['cost_subtypes'],
        "std_qaly": res_std['total_qaly'],
        "new_cost": res_new['total_cost'],
        "new_cost_subtypes": res_new['cost_subtypes'],
        "new_qaly": res_new['total_qaly'],
        "inc_cost": inc_cost,
        "inc_qaly": inc_qaly,
        "icer": icer,
        "nmb_std": nmb_std,
        "nmb_new": nmb_new,
        "inmb": inmb,
        "std_trace": res_std['trace'],
        "new_trace": res_new['trace']
    }
