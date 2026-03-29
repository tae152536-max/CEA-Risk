import pandas as pd
from markov_model import compare_strategies, STATES
from psa_simulation import run_psa

# Set up default parameters similar to the app defaults
n_cycles = 20
wtp = 50000
discount_rate = 0.03

def build_p_matrix(p_event, p_dead_w, p_dead_p):
    p_well = max(0, 1 - p_event - p_dead_w)
    p_pe_pe = max(0, 1 - p_dead_p)
    return [
        [p_well, p_event, p_dead_w],
        [0.0, p_pe_pe, p_dead_p],
        [0.0, 0.0, 1.0]
    ]

std_params = {
    'p_matrix': build_p_matrix(0.10, 0.02, 0.15),
    'costs': [
        {'Medical': 400, 'Non-Medical': 50, 'Indirect': 50},
        {'Medical': 4000, 'Non-Medical': 500, 'Indirect': 500},
        {'Medical': 0.0, 'Non-Medical': 0.0, 'Indirect': 0.0}
    ],
    'utilities': [0.95, 0.75, 0.0]
}

new_params = {
    'p_matrix': build_p_matrix(0.05, 0.02, 0.10),
    'costs': [
        {'Medical': 2000, 'Non-Medical': 250, 'Indirect': 250},
        {'Medical': 4000, 'Non-Medical': 500, 'Indirect': 500},
        {'Medical': 0.0, 'Non-Medical': 0.0, 'Indirect': 0.0}
    ],
    'utilities': [0.95, 0.75, 0.0]
}

# Run Base case
print("Running Base Case Analysis...")
res = compare_strategies(std_params, new_params, n_cycles, wtp, discount_rate)

# Run PSA
print("Running 1,000 PSA Iterations...")
psa_df = run_psa(std_params, new_params, n_cycles, wtp, n_iterations=1000)

csv_path = "CEA_Risk_Model_Results.xlsx"
print(f"Exporting to {csv_path}...")
with pd.ExcelWriter(csv_path, engine='openpyxl') as writer:
    pd.DataFrame(res['std_trace'], columns=STATES).to_excel(writer, sheet_name='Std_Care_Trace')
    pd.DataFrame(res['new_trace'], columns=STATES).to_excel(writer, sheet_name='New_Intervention_Trace')
    
    summary_data = [
        {'Strategy': 'Standard Care', 'Cost': res['std_cost'], 'QALYs': res['std_qaly'], 'NMB': res['nmb_std'], 'ICER': 'N/A', **res['std_cost_subtypes']},
        {'Strategy': 'New Intervention', 'Cost': res['new_cost'], 'QALYs': res['new_qaly'], 'NMB': res['nmb_new'], 'ICER': res['icer'], **res['new_cost_subtypes']}
    ]
    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
    psa_df.to_excel(writer, sheet_name='PSA_Results', index=False)

print("Done! You can find your exported Results locally.")
