import pandas as pd
import io
import numpy as np

def export_parameters_to_excel(state_names, n_cycles, wtp, discount_rate, std_params, new_params):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Settings
        settings_df = pd.DataFrame([
            {"Parameter": "Time Horizon (Cycles)", "Value": n_cycles},
            {"Parameter": "WTP ($/QALY)", "Value": wtp},
            {"Parameter": "Discount Rate", "Value": discount_rate}
        ])
        settings_df.to_excel(writer, sheet_name='Settings', index=False)
        
        states_df = pd.DataFrame({"State Name": state_names})
        states_df.to_excel(writer, sheet_name='States', index=False)
        
        # Standard Care
        std_p_df = pd.DataFrame(std_params['p_matrix'], columns=state_names, index=state_names)
        std_p_df.to_excel(writer, sheet_name='SC_Transitions')
        
        std_u_df = pd.DataFrame({"State": state_names, "Utility (0-1)": std_params['utilities']})
        std_u_df.to_excel(writer, sheet_name='SC_Utilities', index=False)
        
        std_cost_df = std_params['cost_df']
        std_cost_df.to_excel(writer, sheet_name='SC_Costs', index=False)
        
        # New Intervention
        new_p_df = pd.DataFrame(new_params['p_matrix'], columns=state_names, index=state_names)
        new_p_df.to_excel(writer, sheet_name='NI_Transitions')
        
        new_u_df = pd.DataFrame({"State": state_names, "Utility (0-1)": new_params['utilities']})
        new_u_df.to_excel(writer, sheet_name='NI_Utilities', index=False)
        
        new_cost_df = new_params['cost_df']
        new_cost_df.to_excel(writer, sheet_name='NI_Costs', index=False)
        
    return output.getvalue()


def import_parameters_from_excel(file_buffer):
    xls = pd.ExcelFile(file_buffer)
    
    settings_df = pd.read_excel(xls, 'Settings')
    n_cycles = int(settings_df.loc[settings_df['Parameter'] == 'Time Horizon (Cycles)', 'Value'].values[0])
    wtp = float(settings_df.loc[settings_df['Parameter'] == 'WTP ($/QALY)', 'Value'].values[0])
    discount_rate = float(settings_df.loc[settings_df['Parameter'] == 'Discount Rate', 'Value'].values[0])
    
    states_df = pd.read_excel(xls, 'States')
    state_names = states_df['State Name'].tolist()
    n_states = len(state_names)
    
    # Read SC
    std_p_df = pd.read_excel(xls, 'SC_Transitions', index_col=0)
    std_u_df = pd.read_excel(xls, 'SC_Utilities')
    std_cost_df = pd.read_excel(xls, 'SC_Costs')
    
    std_params = {
        'p_matrix': std_p_df.values,
        'p_df': std_p_df,
        'utilities_df': std_u_df,
        'cost_df': std_cost_df
    }
    
    # Read NI
    new_p_df = pd.read_excel(xls, 'NI_Transitions', index_col=0)
    new_u_df = pd.read_excel(xls, 'NI_Utilities')
    new_cost_df = pd.read_excel(xls, 'NI_Costs')
    
    new_params = {
        'p_matrix': new_p_df.values,
        'p_df': new_p_df,
        'utilities_df': new_u_df,
        'cost_df': new_cost_df
    }
    
    return {
        'n_states': n_states,
        'state_names': state_names,
        'state_names_df': states_df,
        'n_cycles': n_cycles,
        'wtp': wtp,
        'discount_rate': discount_rate,
        'std_params': std_params,
        'new_params': new_params
    }
