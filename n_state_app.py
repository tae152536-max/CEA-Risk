import streamlit as st
import pandas as pd
import numpy as np
from n_state_markov import compare_n_state_strategies
from n_state_psa import run_n_state_psa
from n_state_owsa import run_n_state_owsa

import visualizations
import importlib
importlib.reload(visualizations)
from visualizations import plot_ce_plane, plot_ceac, plot_inmb_distribution, plot_cost_breakdown, plot_tornado

st.set_page_config(page_title="Generic N-State Model", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    [data-testid="stSidebar"] { background-color: #f8f9fa !important; border-right: 1px solid #e9ecef; }
    [data-testid="stMetric"] { background-color: #ffffff; border-radius: 8px; padding: 15px 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.04); border: 1px solid #f0f2f6; text-align: center; }
    [data-testid="stMetricValue"] { font-weight: 700; color: #2c3e50; font-size: 1.8rem !important; }
    [data-testid="stMetricLabel"] { font-weight: 600; color: #7f8c8d; font-size: 0.9rem !important; margin-bottom: 5px; }
    .stButton>button { border-radius: 6px; font-weight: 600; transition: all 0.2s ease; font-family: 'Inter', sans-serif !important; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 12px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

st.title("Generic N-State Markov Risk Analysis Model")
st.markdown("Define a fully customizable dynamic Markov model by setting the states, calculating transition impacts, and running Probabilistic Sensitivity Analysis (PSA).")

# --- STEP 1: DEFINE STATES ---
st.sidebar.header("1. Define System")
n_states = st.sidebar.number_input("Number of Health States", min_value=2, max_value=10, value=3)

if 'state_names_df' not in st.session_state or len(st.session_state['state_names_df']) != n_states:
    default_states = [f"State {i+1}" for i in range(n_states)]
    if n_states == 3:
        default_states = ["Well", "Sick", "Dead"]
    st.session_state['state_names_df'] = pd.DataFrame({"State Name": default_states})
    
st.sidebar.markdown("**Name your states (Row 0 is starting state):**")
states_df = st.sidebar.data_editor(st.session_state['state_names_df'], use_container_width=True, hide_index=True)
st.session_state['state_names_df'] = states_df
state_names = states_df["State Name"].tolist()

st.sidebar.markdown("---")
st.sidebar.header("2. Global Parameters")
n_cycles = st.sidebar.slider("Time Horizon (Cycles)", min_value=1, max_value=50, value=20)
wtp = st.sidebar.number_input("Willingness To Pay (WTP/QALY)", min_value=1000, value=50000, step=5000)
discount_rate = st.sidebar.slider("Discount Rate", min_value=0.0, max_value=0.10, value=0.03, step=0.01)

# Helper for identity matrices
def build_default_matrix(n, name_list):
    df = pd.DataFrame(np.eye(n), columns=name_list, index=name_list)
    # If 3 states, provide a slightly more sensible default if it's new
    if n == 3:
        df.iloc[0,:] = [0.8, 0.15, 0.05]
        df.iloc[1,:] = [0.0, 0.8, 0.2]
        df.iloc[2,:] = [0.0, 0.0, 1.0]
    else:
        # Just simple defaults so it runs
        for i in range(n-1):
            df.iloc[i, i] = 0.9
            df.iloc[i, i+1] = 0.1
        df.iloc[-1, :] = 0.0
        df.iloc[-1, -1] = 1.0
    return df

# --- UI for Matrices and Inputs ---
st.subheader("Data Input Matrices (From Row -> To Column)")

tab_sc, tab_ni = st.tabs(["Standard Care", "New Intervention"])

with tab_sc:
    st.markdown("##### Transition Probabilities")
    # Store in session state to preserve changes across reruns if needed, but simple data_editor holds state usually
    std_p_df = st.data_editor(build_default_matrix(n_states, state_names), key="std_p", use_container_width=True)
    
    st.markdown("##### State Utilities (QALYs)")
    std_u_init = pd.DataFrame({"State": state_names, "Utility (0-1)": np.linspace(1.0, 0.0, n_states)})
    std_u_df = st.data_editor(std_u_init, hide_index=True, key="std_u_tab", use_container_width=True)

    st.markdown("##### Annual Costs")
    # Dynamic cost dataframe
    std_cost_init = pd.DataFrame([
        {"State": state_names[0], "Subgroup": "Medical", "Item": "Checkup", "Cost ($/year)": 1000.0, "Distribution": "Gamma", "SE/SD": 100.0}
    ])
    std_cost_df = st.data_editor(std_cost_init, num_rows="dynamic", hide_index=True, use_container_width=True, key="std_cost_tab",
        column_config={
           "State": st.column_config.SelectboxColumn("State", options=state_names, required=True),
           "Subgroup": st.column_config.SelectboxColumn("Subgroup", options=["Medical", "Non-Medical", "Indirect"], required=True),
           "Distribution": st.column_config.SelectboxColumn("Distribution", options=["Gamma", "Lognormal", "Normal", "Uniform", "Triangular", "Beta", "Fixed"], required=True),
        })

with tab_ni:
    st.markdown("##### Transition Probabilities")
    new_p_df = st.data_editor(build_default_matrix(n_states, state_names), key="new_p", use_container_width=True)
    
    st.markdown("##### State Utilities (QALYs)")
    new_u_init = pd.DataFrame({"State": state_names, "Utility (0-1)": np.linspace(1.0, 0.0, n_states)})
    new_u_df = st.data_editor(new_u_init, hide_index=True, key="new_u_tab", use_container_width=True)

    st.markdown("##### Annual Costs")
    new_cost_init = pd.DataFrame([
        {"State": state_names[0], "Subgroup": "Medical", "Item": "New Drug", "Cost ($/year)": 2500.0, "Distribution": "Gamma", "SE/SD": 250.0}
    ])
    new_cost_df = st.data_editor(new_cost_init, num_rows="dynamic", hide_index=True, use_container_width=True, key="new_cost_tab",
        column_config={
           "State": st.column_config.SelectboxColumn("State", options=state_names, required=True),
           "Subgroup": st.column_config.SelectboxColumn("Subgroup", options=["Medical", "Non-Medical", "Indirect"], required=True),
           "Distribution": st.column_config.SelectboxColumn("Distribution", options=["Gamma", "Lognormal", "Normal", "Uniform", "Triangular", "Beta", "Fixed"], required=True),
        })


# Enforce constraints check
def validate_matrix(df, name):
    errors = []
    mat = df.values
    if not np.allclose(mat.sum(axis=1), 1.0):
        errors.append(f"❌ Rows in {name} transition matrix must sum to 1.0!")
    if (mat < 0).any() or (mat > 1).any():
        errors.append(f"❌ Probabilities in {name} must be between 0 and 1!")
    return errors

errs = validate_matrix(std_p_df, "Standard Care") + validate_matrix(new_p_df, "New Intervention")
if errs:
    for e in errs:
        st.error(e)
    st.stop()


# Assemble parameters
std_params = {
    'p_matrix': std_p_df.values,
    'cost_df': std_cost_df,
    'utilities': std_u_df['Utility (0-1)'].values.tolist()
}

new_params = {
    'p_matrix': new_p_df.values,
    'cost_df': new_cost_df,
    'utilities': new_u_df['Utility (0-1)'].values.tolist()
}


st.markdown("---")
# --- BASE CASE ---
st.header("Base Case Analysis")
res = compare_n_state_strategies(std_params, new_params, state_names, n_cycles, wtp, discount_rate)

c1, c2, c3 = st.columns(3)
c1.metric("Standard Care Cost", f"${res['std_cost']:,.0f}")
c2.metric("New Intervention Cost", f"${res['new_cost']:,.0f}")
c3.metric("Incremental Cost", f"${res['inc_cost']:,.0f}")

c1, c2, c3 = st.columns(3)
c1.metric("Standard Care QALYs", f"{res['std_qaly']:,.2f}")
c2.metric("New Intervention QALYs", f"{res['new_qaly']:,.2f}")
c3.metric("Incremental QALYs", f"{res['inc_qaly']:,.2f}")

c1, c2, c3 = st.columns(3)
c1.metric("ICER ($/QALY)", f"${res['icer']:,.0f}" if res['icer'] != float('inf') and res['icer'] != -1 else "Dominated")
c2.metric("Incremental NMB (INMB)", f"${res['inmb']:,.0f}")
c3.markdown(f"**Decision:** {'Cost-Effective' if res['inmb'] > 0 else 'Not Cost-Effective'}")

st.subheader("Cost Breakdown")
c_chart, _ = st.columns([1, 1])
with c_chart:
    st.pyplot(plot_cost_breakdown(res))


st.markdown("---")
# --- OWSA ---
st.header("Deterministic Sensitivity Analysis (OWSA)")
owsa_var = st.slider("Parameter Variance (%)", 10, 50, 20, 5) / 100.0
if st.button("Generate One-Way Tornado", type="primary"):
    with st.spinner("Running N-State OWSA..."):
        owsa_df, base_inmb = run_n_state_owsa(std_params, new_params, state_names, n_cycles, wtp, discount_rate, owsa_var)
        st.pyplot(plot_tornado(owsa_df, base_inmb))
        st.dataframe(owsa_df[['Parameter', 'Swing', 'INMB_Low', 'INMB_High']].head(10), use_container_width=True)

st.markdown("---")
# --- PSA ---
st.header("Risk Analysis (PSA)")
n_simulations = st.number_input("Number of Simulations", min_value=100, max_value=10000, value=1000, step=100)
max_wtp_ceac = st.slider("Max WTP on CEAC Plot", 50000, 500000, 150000, 10000)

if st.button("Run Probabilistic Risk Analysis", type="primary"):
    with st.spinner(f"Running {n_simulations:,} Monte Carlo Iterations on {n_states} states..."):
        psa_df = run_n_state_psa(std_params, new_params, state_names, n_cycles, wtp, n_iterations=n_simulations)
        st.session_state['n_psa_df'] = psa_df
        st.success("PSA Completed!")

if 'n_psa_df' in st.session_state:
    psa_df = st.session_state['n_psa_df']
    t1, t2, t3 = st.tabs(["CEAC", "CE Plane", "INMB Distribution"])
    with t1:
        st.pyplot(plot_ceac(psa_df, max_wtp_ceac))
    with t2:
        st.pyplot(plot_ce_plane(psa_df, wtp))
    with t3:
        st.pyplot(plot_inmb_distribution(psa_df))
        prob_ce = (psa_df['INMB'] > 0).mean()
        st.info(f"**Probability Cost-Effective:** {prob_ce*100:.1f}%")
