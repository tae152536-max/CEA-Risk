import streamlit as st
import pandas as pd
import numpy as np
from markov_model import compare_strategies
from psa_simulation import run_psa

import importlib
import visualizations
importlib.reload(visualizations)
from visualizations import plot_ce_plane, plot_ceac, plot_inmb_distribution, plot_cost_breakdown, plot_tornado
# Note: Excel export logic moved to generate_excel_model.py for deep formula support

st.set_page_config(page_title="Cardiovascular CEA Model", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    [data-testid="stSidebar"] { background-color: #f8f9fa !important; border-right: 1px solid #e9ecef; }
    
    [data-testid="stMetric"] { background-color: #ffffff; border-radius: 8px; padding: 15px 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.04); border: 1px solid #f0f2f6; text-align: center; }
    [data-testid="stMetricValue"] { font-weight: 700; color: #2c3e50; font-size: 1.8rem !important; }
    [data-testid="stMetricLabel"] { font-weight: 600; color: #7f8c8d; font-size: 0.9rem !important; margin-bottom: 5px; }
    
    .stButton>button { border-radius: 6px; font-weight: 600; transition: all 0.2s ease; font-family: 'Inter', sans-serif !important; letter-spacing: 0.5px; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 12px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# --- INTERNAL SECURITY GATE ---
ACCESS_CODE = "HealthEcon2026"

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("🔒 Confidential Internal Dashboard")
    st.markdown("This Cost-Effectiveness Model contains proprietary baseline analytics. Please enter your Internal Access Code to decrypt the mathematical engine.")
    
    pwd = st.text_input("Enter Access Code:", type="password")
    
    if pwd == ACCESS_CODE:
        st.session_state["authenticated"] = True
        st.rerun()
    elif pwd != "":
        st.error("❌ Incorrect Access Code. Connection refused.")
    
    st.stop() # Physically halts the entire Streamlit engine from rendering the rest of the file
# ------------------------------

st.title("Cardiovascular Cost-Effectiveness Risk Analysis Model")
st.markdown("""
This application compares two strategies: **Standard Care** vs. **New Intervention** for patients with Cardiovascular Disease.
It uses a 3-State Markov Model (**Well, Post-Event, Dead**) and quantifies risk via **Probabilistic Sensitivity Analysis (PSA)**.
""")

def get_default_costs(strategy):
    if strategy == 'Standard':
         return pd.DataFrame([
             {"State": "Well", "Subgroup": "Medical", "Item": "Routine Visit", "Cost ($/year)": 400.0, "Distribution": "Gamma", "SE/SD": 60.0},
             {"State": "Well", "Subgroup": "Non-Medical", "Item": "Transport", "Cost ($/year)": 50.0, "Distribution": "Fixed", "SE/SD": 0.0},
             {"State": "Well", "Subgroup": "Indirect", "Item": "Time lost", "Cost ($/year)": 50.0, "Distribution": "Fixed", "SE/SD": 0.0},
             {"State": "Post-Event", "Subgroup": "Medical", "Item": "Hospitalization", "Cost ($/year)": 4000.0, "Distribution": "Gamma", "SE/SD": 600.0},
             {"State": "Post-Event", "Subgroup": "Non-Medical", "Item": "Rehab", "Cost ($/year)": 500.0, "Distribution": "Gamma", "SE/SD": 75.0},
             {"State": "Post-Event", "Subgroup": "Indirect", "Item": "Work lost", "Cost ($/year)": 500.0, "Distribution": "Normal", "SE/SD": 50.0},
         ])
    else:
         return pd.DataFrame([
             {"State": "Well", "Subgroup": "Medical", "Item": "New Intervention", "Cost ($/year)": 2000.0, "Distribution": "Gamma", "SE/SD": 300.0},
             {"State": "Well", "Subgroup": "Non-Medical", "Item": "Monitoring", "Cost ($/year)": 250.0, "Distribution": "Gamma", "SE/SD": 30.0},
             {"State": "Well", "Subgroup": "Indirect", "Item": "Time lost", "Cost ($/year)": 250.0, "Distribution": "Fixed", "SE/SD": 0.0},
             {"State": "Post-Event", "Subgroup": "Medical", "Item": "Hospitalization", "Cost ($/year)": 4000.0, "Distribution": "Gamma", "SE/SD": 600.0},
             {"State": "Post-Event", "Subgroup": "Non-Medical", "Item": "Rehab", "Cost ($/year)": 500.0, "Distribution": "Gamma", "SE/SD": 75.0},
             {"State": "Post-Event", "Subgroup": "Indirect", "Item": "Work lost", "Cost ($/year)": 500.0, "Distribution": "Normal", "SE/SD": 50.0},
         ])

def df_to_cost_list(df):
    states = ["Well", "Post-Event", "Dead"]
    cost_list = []
    subgroups = ["Medical", "Non-Medical", "Indirect"]
    for s in states:
        s_df = df[df['State'] == s]
        c_dict = {}
        for sub in subgroups:
            c_dict[sub] = s_df[s_df['Subgroup'] == sub]['Cost ($/year)'].sum()
        cost_list.append(c_dict)
    return cost_list

# SIDEBAR: Parameters
st.sidebar.header("Model Parameters")
n_cycles = st.sidebar.slider("Time Horizon (Years/Cycles)", min_value=1, max_value=50, value=20)
wtp = st.sidebar.number_input("Willingness To Pay (WTP/QALY)", min_value=1000, max_value=250000, value=50000, step=5000)
discount_rate = st.sidebar.slider("Discount Rate", min_value=0.0, max_value=0.10, value=0.03, step=0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("Risk Settings (CEAC)")
max_wtp_ceac = st.sidebar.slider("Max WTP on CEAC Plot", min_value=50000, max_value=500000, value=150000, step=10000)
n_simulations = st.sidebar.number_input("Number of Simulations (PSA)", min_value=100, max_value=10000, value=1000, step=100)

st.subheader("Dynamic Cost Inputs")
st.markdown("Add up to 10 lines per state/subgroup! Select a specific statistical distribution for each item's Probabilistic Risk Analysis.")

tab_sc, tab_ni = st.tabs(["Standard Care Costs", "New Intervention Costs"])

with tab_sc:
    std_cost_df = st.data_editor(
        get_default_costs('Standard'),
        num_rows="dynamic",
        use_container_width=True,
        column_config={
           "State": st.column_config.SelectboxColumn("State", options=["Well", "Post-Event"], required=True),
           "Subgroup": st.column_config.SelectboxColumn("Subgroup", options=["Medical", "Non-Medical", "Indirect"], required=True),
           "Distribution": st.column_config.SelectboxColumn("Distribution", options=["Gamma", "Lognormal", "Normal", "Uniform", "Triangular", "Beta", "Fixed"], required=True),
        },
        key="sc_costs"
    )
    
with tab_ni:
    new_cost_df = st.data_editor(
        get_default_costs('New'),
        num_rows="dynamic",
        use_container_width=True,
        column_config={
           "State": st.column_config.SelectboxColumn("State", options=["Well", "Post-Event"], required=True),
           "Subgroup": st.column_config.SelectboxColumn("Subgroup", options=["Medical", "Non-Medical", "Indirect"], required=True),
           "Distribution": st.column_config.SelectboxColumn("Distribution", options=["Gamma", "Normal", "Fixed"], required=True),
        },
        key="ni_costs"
    )

st.sidebar.subheader("Transition Probabilities")
st.sidebar.markdown("**Standard Care**")
std_prob_event = st.sidebar.slider("Prob (Well -> Post-Event)", value=0.10, min_value=0.0, max_value=1.0, key="s1")
std_prob_dead_w = st.sidebar.slider("Prob (Well -> Dead)", value=0.02, min_value=0.0, max_value=1.0, key="s2")
std_prob_dead_p = st.sidebar.slider("Prob (Post-Event -> Dead)", value=0.15, min_value=0.0, max_value=1.0, key="s3")

st.sidebar.markdown("**New Intervention**")
new_prob_event = st.sidebar.slider("Prob (Well -> Post-Event) ", value=0.05, min_value=0.0, max_value=1.0, key="n5")
new_prob_dead_w = st.sidebar.slider("Prob (Well -> Dead) ", value=0.02, min_value=0.0, max_value=1.0, key="n6")
new_prob_dead_p = st.sidebar.slider("Prob (Post-Event -> Dead) ", value=0.10, min_value=0.0, max_value=1.0, key="n7")

st.sidebar.subheader("Utilities (QALYs)")
st.sidebar.markdown("**Standard Care**")
std_qaly_well = st.sidebar.number_input("QALY - Well (Std)", value=0.95, min_value=0.0, max_value=1.0)
std_qaly_post = st.sidebar.number_input("QALY - Post-Event (Std)", value=0.75, min_value=0.0, max_value=1.0)

st.sidebar.markdown("**New Intervention**")
new_qaly_well = st.sidebar.number_input("QALY - Well (New)", value=0.95, min_value=0.0, max_value=1.0)
new_qaly_post = st.sidebar.number_input("QALY - Post-Event (New)", value=0.80, min_value=0.0, max_value=1.0)

def build_p_matrix(p_event, p_dead_w, p_dead_p):
    p_well = max(0, 1 - p_event - p_dead_w)
    p_pe_pe = max(0, 1 - p_dead_p)
    return [
        [p_well, p_event, p_dead_w],
        [0.0, p_pe_pe, p_dead_p],
        [0.0, 0.0, 1.0]
    ]

std_params = {
    'p_matrix': build_p_matrix(std_prob_event, std_prob_dead_w, std_prob_dead_p),
    'costs': df_to_cost_list(std_cost_df),
    'cost_df': std_cost_df, # pass to PSA
    'utilities': [std_qaly_well, std_qaly_post, 0.0]
}

new_params = {
    'p_matrix': build_p_matrix(new_prob_event, new_prob_dead_w, new_prob_dead_p),
    'costs': df_to_cost_list(new_cost_df),
    'cost_df': new_cost_df,
    'utilities': [new_qaly_well, new_qaly_post, 0.0]
}

st.markdown("---")
# --- Base Case Execution ---
st.header("Base Case Analysis")
res = compare_strategies(std_params, new_params, n_cycles, wtp, discount_rate)

col1, col2, col3 = st.columns(3)
col1.metric("Standard Care Cost", f"${res['std_cost']:,.0f}")
col2.metric("New Intervention Cost", f"${res['new_cost']:,.0f}")
col3.metric("Incremental Cost", f"${res['inc_cost']:,.0f}")

col1, col2, col3 = st.columns(3)
col1.metric("Standard Care QALYs", f"{res['std_qaly']:,.2f}")
col2.metric("New Intervention QALYs", f"{res['new_qaly']:,.2f}")
col3.metric("Incremental QALYs", f"{res['inc_qaly']:,.2f}")

col1, col2, col3 = st.columns(3)
col1.metric("ICER ($/QALY)", f"${res['icer']:,.0f}" if res['icer'] != float('inf') else "Dominated")
col2.metric("Incremental NMB (INMB)", f"${res['inmb']:,.0f}")
col3.markdown(f"**Decision:** {'Cost-Effective' if res['inmb'] > 0 else 'Not Cost-Effective'}")

st.subheader("Cost Distribution")
col_chart, col_text = st.columns([1, 1])
with col_chart:
    st.pyplot(plot_cost_breakdown(res))
with col_text:
    st.markdown("This bar chart sums up your dynamic line-items explicitly across Medical, Non-Medical, and Indirect subtypes for your entire horizon.")


st.markdown("---")
# --- OWSA ---
st.header("Deterministic Sensitivity Analysis (OWSA)")
st.markdown("Vary each baseline parameter independently to identify the primary drivers of model uncertainty.")
owsa_var = st.slider("Parameter Variance (%)", min_value=10, max_value=50, value=20, step=5) / 100.0

if st.button("Generate Tornado Diagram & Switching Values", type="primary"):
    from owsa_engine import run_owsa, find_switching_value
    base_eval_params = {
        'time_horizon': n_cycles,
        'discount_rate': discount_rate,
        'wtp': wtp,
        'std_prob_event': std_prob_event,
        'std_prob_dead_w': std_prob_dead_w,
        'std_prob_dead_p': std_prob_dead_p,
        'new_prob_event': new_prob_event,
        'new_prob_dead_w': new_prob_dead_w,
        'new_prob_dead_p': new_prob_dead_p,
        'std_qaly_well': std_qaly_well,
        'std_qaly_post': std_qaly_post,
        'new_qaly_well': new_qaly_well,
        'new_qaly_post': new_qaly_post,
        'std_cost_multiplier': 1.0,
        'new_cost_multiplier': 1.0,
    }
    
    with st.spinner("Running One-Way Sensitivity Analysis..."):
        owsa_df, base_inmb = run_owsa(base_eval_params, std_params['costs'], new_params['costs'], owsa_var)
        
        st.pyplot(plot_tornado(owsa_df, base_inmb))
        
        st.subheader("Threshold / Switching Values")
        st.markdown("The exact parameter value required to mathematically flip the decision (INMB = $0)")
        
        switch_data = []
        for _, row in owsa_df.head(6).iterrows():
            sv = find_switching_value(row['Parameter'], base_eval_params, std_params['costs'], new_params['costs'])
            if sv is not None:
                name = row['Parameter'].replace('_', ' ').title().replace('Multiplier', 'Total Cost')
                if 'Cost' in name:
                     sv_text = f"If cost increases by {(sv-1)*100:+.1f}%"
                else:
                     sv_text = f"{sv:.4f}"
                switch_data.append({'Parameter': name, 'Base Value': f"{row['Base']:.4f}" if 'Multiplier' not in row['Parameter'] else "Baseline", 'Switching Value (INMB=$0)': sv_text})
        
        if switch_data:
            st.table(pd.DataFrame(switch_data))
        else:
            st.info("No switching values found within realistic bounds for the top parameters.")


st.markdown("---")
# --- Risk Analysis (PSA) ---
st.header("Risk Analysis (Probabilistic Sensitivity Analysis)")
st.markdown(f"Runs **{n_simulations:,}** Monte Carlo simulations utilizing the precise statistical distributions you selected for **every single line item** in your tables!")

if st.button("Run Probabilistic Risk Analysis (PSA)", type="primary"):
    with st.spinner(f"Running {n_simulations:,} Monte Carlo Iterations with Custom Distributions..."):
        st.session_state['psa_df'] = run_psa(std_params, new_params, n_cycles, wtp, n_iterations=n_simulations)
    st.success("PSA Completed Successfully!")
    
if 'psa_df' in st.session_state:
    psa_df = st.session_state['psa_df']
    # Visualizations
    tab1, tab2, tab3 = st.tabs(["Cost-Effectiveness Acceptability Curve", "Cost-Effectiveness Plane", "NMB Distribution"])
    
    with tab1:
        st.pyplot(plot_ceac(psa_df, max_wtp_ceac))
        st.markdown(f"Shows the probability that the new intervention is cost-effective from $0 to user-selected ${max_wtp_ceac:,.0f} WTP.")
        
    with tab2:
        st.pyplot(plot_ce_plane(psa_df, wtp))
        
    with tab3:
        st.pyplot(plot_inmb_distribution(psa_df))
        prob_ce = (psa_df['INMB'] > 0).mean()
        st.info(f"**Probability Cost-Effective (INMB > 0):** {prob_ce*100:.1f}%")

st.markdown("---")
st.header("Export Native Formula-Driven Excel Model")
st.markdown("We automatically generate the actual `.xlsx` workbook containing your dynamic line-item costs inserted directly into standard Excel formulas for traceability and auditing.")

from generate_excel_model import create_excel_model
create_excel_model(std_params, new_params, n_cycles, wtp, discount_rate, filename="Formula_CEA_Model.xlsx")

with open("Formula_CEA_Model.xlsx", "rb") as f:
    st.download_button(
        label="Download Live Excel Model",
        data=f,
        file_name="Formula_CEA_Model.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary"
    )
