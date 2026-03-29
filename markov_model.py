import numpy as np
import pandas as pd

STATES = ["Well", "Post-Event", "Dead"]

def run_markov_model(p_matrix, costs, utilities, n_cycles, initial_cohort=[1.0, 0.0, 0.0], discount_rate=0.03):
    """
    Runs a Markov cohort model with broken-down costs.
    :param p_matrix: 3x3 transition probability matrix
    :param costs: list of dicts for [Well, Post-Event, Dead]. E.g. [{"Medical": 100, "Indirect": 50}, ...]
    :param utilities: list of QALYs for [Well, Post-Event, Dead]
    :param n_cycles: number of cycles (years)
    :param initial_cohort: list of initial population distribution
    :param discount_rate: annual discount rate for costs and QALYs
    :return: dict with total cost, cost subtypes, total qaly, and trace
    """
    n_states = len(STATES)
    trace = np.zeros((n_cycles + 1, n_states))
    trace[0, :] = initial_cohort
    
    total_cost = 0.0
    total_qaly = 0.0
    
    cost_keys = costs[0].keys()
    cost_subtypes = {k: 0.0 for k in cost_keys}
    
    # Half-cycle correction for cycle 0
    total_qaly += np.sum(trace[0, :] * utilities) * 0.5
    for k in cost_keys:
        k_costs = np.array([c[k] for c in costs])
        c_t = np.sum(trace[0, :] * k_costs) * 0.5
        cost_subtypes[k] += c_t
        total_cost += c_t
    
    p_matrix = np.array(p_matrix)
    
    for t in range(1, n_cycles + 1):
        trace[t, :] = np.dot(trace[t-1, :], p_matrix)
        df = 1 / ((1 + discount_rate) ** t)
        
        q_t = np.sum(trace[t, :] * utilities) * df
        if t == n_cycles:
            total_qaly += q_t * 0.5
        else:
            total_qaly += q_t
            
        for k in cost_keys:
            k_costs = np.array([c.get(k, 0) for c in costs])
            c_t = np.sum(trace[t, :] * k_costs) * df
            if t == n_cycles:
                c_t *= 0.5
            cost_subtypes[k] += c_t
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

def compare_strategies(std_params, new_params, n_cycles, wtp, discount_rate=0.03):
    """
    Runs both strategies and computes incremental values.
    """
    res_std = run_markov_model(std_params['p_matrix'], std_params['costs'], std_params['utilities'], n_cycles, discount_rate=discount_rate)
    res_new = run_markov_model(new_params['p_matrix'], new_params['costs'], new_params['utilities'], n_cycles, discount_rate=discount_rate)
    
    inc_cost = res_new['total_cost'] - res_std['total_cost']
    inc_qaly = res_new['total_qaly'] - res_std['total_qaly']
    icer = inc_cost / inc_qaly if inc_qaly != 0 else float('inf')
    
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
