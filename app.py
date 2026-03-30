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

st.set_page_config(page_title="Cardiovascular CEA Model", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Clean SaaS Dashboard Theme with Minimal Color Cards */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Inter', sans-serif !important; 
        color: #374151 !important;
        font-size: 14px !important; 
    }
    
    /* Light Grey Dashboard Canvas */
    .stApp { background-color: #f3f4f6 !important; }
    
    /* Clean Typography */
    h1 { 
        font-size: 1.8rem !important; 
        color: #111827 !important; 
        font-weight: 500 !important; 
        border-bottom: none !important; 
        margin-bottom: 20px; 
    }
    h2 { 
        font-size: 1.4rem !important; 
        color: #111827 !important; 
        font-weight: 600 !important; 
        margin-top: 30px; 
        margin-bottom: 15px; 
    }
    h3, h4, h5 { 
        font-size: 1.1rem !important; 
        color: #374151 !important; 
        font-weight: 600 !important; 
    }
    
    /* Hide Native Clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Central Dashboard Page */
    .block-container { 
        padding-top: 2rem; 
        padding-bottom: 3rem; 
        max-width: 90% !important; 
    }
    
    /* Pure White Sidebar */
    [data-testid="stSidebar"] { 
        background-color: #ffffff !important; 
        border-right: 1px solid #e5e7eb !important; 
    }
    
    /* MINIMAL CARDS REMOVED - Dynamic Inline Styles Handled in Python */
    
    /* Primary Action Buttons */
    .stButton>button { 
        background-color: #2563eb !important; /* Royal Blue */
        border: none !important;
        color: #ffffff !important;
        border-radius: 4px !important; 
        font-weight: 500 !important; 
        font-size: 0.9rem !important;
        padding: 0.4rem 1.2rem !important;
        box-shadow: none !important;
        transition: background-color 0.2s;
    }
    .stButton>button:hover { 
        background-color: #1d4ed8 !important;
    }
    
    /* Clean Minimal Tabs */
    .stTabs [data-baseweb="tab-list"] { 
        border-bottom: 1px solid #e5e7eb;
        gap: 20px; 
    }
    .stTabs [data-baseweb="tab"] { 
        background-color: transparent !important; 
        border-radius: 0; 
        border-bottom: 2px solid transparent; 
        padding: 8px 4px; 
        font-weight: 500; 
        font-size: 0.95rem;
        color: #6b7280;
    }
    .stTabs [aria-selected="true"] { 
        color: #2563eb !important; 
        border-bottom-color: #2563eb !important; 
    }
    
    /* Inputs */
    .stSlider > div[data-baseweb="slider"] { padding-top: 5px; }
    .stNumberInput > div[data-baseweb="input"] { 
        border-radius: 4px !important; 
        border: 1px solid #d1d5db !important; 
        background-color: #ffffff !important;
    }
    
    /* Data Editor Overrides for White Flat Look */
    [data-testid="stDataFrame"] {
        border-radius: 6px !important;
        border: 1px solid #e5e7eb !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
    }
    
    /* Expander Overrides */
    [data-testid="stExpander"] {
        background-color: #ffffff !important;
        border-radius: 6px !important;
        border: 1px solid #e5e7eb !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

st.title("Cardiovascular Cost-Effectiveness Risk Analysis Model")
st.markdown("""
This application compares two strategies: **Standard Care** vs. **New Intervention** for patients with Cardiovascular Disease.
It uses a 3-State Markov Model (**Well, Post-Event, Dead**) and quantifies risk via **Probabilistic Sensitivity Analysis (PSA)**.
""")

def get_default_costs(strategy, states):
    s1, s2, s3 = states
    if strategy == 'Standard':
         return pd.DataFrame([
             {"State": s1, "Subgroup": "Medical", "Item": "Routine Visit", "Cost ($/year)": 400.0, "Distribution": "Gamma", "SE/SD": 60.0},
             {"State": s1, "Subgroup": "Non-Medical", "Item": "Transport", "Cost ($/year)": 50.0, "Distribution": "Fixed", "SE/SD": 0.0},
             {"State": s1, "Subgroup": "Indirect", "Item": "Time lost", "Cost ($/year)": 50.0, "Distribution": "Fixed", "SE/SD": 0.0},
             {"State": s2, "Subgroup": "Medical", "Item": "Hospitalization", "Cost ($/year)": 4000.0, "Distribution": "Gamma", "SE/SD": 600.0},
             {"State": s2, "Subgroup": "Non-Medical", "Item": "Rehab", "Cost ($/year)": 500.0, "Distribution": "Gamma", "SE/SD": 75.0},
             {"State": s2, "Subgroup": "Indirect", "Item": "Work lost", "Cost ($/year)": 500.0, "Distribution": "Normal", "SE/SD": 50.0},
         ])
    else:
         return pd.DataFrame([
             {"State": s1, "Subgroup": "Medical", "Item": "New Intervention", "Cost ($/year)": 2000.0, "Distribution": "Gamma", "SE/SD": 300.0},
             {"State": s1, "Subgroup": "Non-Medical", "Item": "Monitoring", "Cost ($/year)": 250.0, "Distribution": "Gamma", "SE/SD": 30.0},
             {"State": s1, "Subgroup": "Indirect", "Item": "Time lost", "Cost ($/year)": 250.0, "Distribution": "Fixed", "SE/SD": 0.0},
             {"State": s2, "Subgroup": "Medical", "Item": "Hospitalization", "Cost ($/year)": 4000.0, "Distribution": "Gamma", "SE/SD": 600.0},
             {"State": s2, "Subgroup": "Non-Medical", "Item": "Rehab", "Cost ($/year)": 500.0, "Distribution": "Gamma", "SE/SD": 75.0},
             {"State": s2, "Subgroup": "Indirect", "Item": "Work lost", "Cost ($/year)": 500.0, "Distribution": "Normal", "SE/SD": 50.0},
         ])

def df_to_cost_list(df, states):
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
st.sidebar.subheader("Define States")
state_1 = st.sidebar.text_input("State 1 Name", value="Well")
state_2 = st.sidebar.text_input("State 2 Name", value="Post-Event")
state_3 = st.sidebar.text_input("State 3 Name", value="Dead")
custom_states = [state_1, state_2, state_3]

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
        get_default_costs('Standard', custom_states),
        num_rows="dynamic",
        use_container_width=True,
        column_config={
           "State": st.column_config.SelectboxColumn("State", options=custom_states[:2], required=True),
           "Subgroup": st.column_config.SelectboxColumn("Subgroup", options=["Medical", "Non-Medical", "Indirect"], required=True),
           "Distribution": st.column_config.SelectboxColumn("Distribution", options=["Gamma", "Lognormal", "Normal", "Uniform", "Triangular", "Beta", "Fixed"], required=True),
        },
        key="sc_costs_" + state_1 + state_2 # cache-break if names change heavily
    )
    
with tab_ni:
    new_cost_df = st.data_editor(
        get_default_costs('New', custom_states),
        num_rows="dynamic",
        use_container_width=True,
        column_config={
           "State": st.column_config.SelectboxColumn("State", options=custom_states[:2], required=True),
           "Subgroup": st.column_config.SelectboxColumn("Subgroup", options=["Medical", "Non-Medical", "Indirect"], required=True),
           "Distribution": st.column_config.SelectboxColumn("Distribution", options=["Gamma", "Normal", "Fixed"], required=True),
        },
        key="ni_costs_" + state_1 + state_2
    )

st.markdown("---")
st.sidebar.subheader("Transition Probabilities")
st.sidebar.markdown("**Standard Care**")
std_prob_event = st.sidebar.slider(f"Prob ({state_1} -> {state_2})", value=0.10, min_value=0.0, max_value=1.0, step=0.01, format="%.2f", key="s1")
std_prob_dead_w = st.sidebar.slider(f"Prob ({state_1} -> {state_3})", value=0.02, min_value=0.0, max_value=1.0, step=0.01, format="%.2f", key="s2")
std_prob_dead_p = st.sidebar.slider(f"Prob ({state_2} -> {state_3})", value=0.15, min_value=0.0, max_value=1.0, step=0.01, format="%.2f", key="s3")

st.sidebar.markdown("**New Intervention**")
new_prob_event = st.sidebar.slider(f"Prob ({state_1} -> {state_2})", value=0.05, min_value=0.0, max_value=1.0, step=0.01, format="%.2f", key="n5")
new_prob_dead_w = st.sidebar.slider(f"Prob ({state_1} -> {state_3})", value=0.02, min_value=0.0, max_value=1.0, step=0.01, format="%.2f", key="n6")
new_prob_dead_p = st.sidebar.slider(f"Prob ({state_2} -> {state_3})", value=0.10, min_value=0.0, max_value=1.0, step=0.01, format="%.2f", key="n7")

st.sidebar.subheader("Utilities (QALYs)")
st.sidebar.markdown("**Standard Care**")
std_qaly_well = st.sidebar.number_input(f"QALY - {state_1} (Std)", value=0.95, min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
std_qaly_post = st.sidebar.number_input(f"QALY - {state_2} (Std)", value=0.75, min_value=0.0, max_value=1.0, step=0.01, format="%.2f")

st.sidebar.markdown("**New Intervention**")
new_qaly_well = st.sidebar.number_input(f"QALY - {state_1} (New)", value=0.95, min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
new_qaly_post = st.sidebar.number_input(f"QALY - {state_2} (New)", value=0.80, min_value=0.0, max_value=1.0, step=0.01, format="%.2f")

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
    'costs': df_to_cost_list(std_cost_df, custom_states),
    'cost_df': std_cost_df, # pass to PSA
    'utilities': [std_qaly_well, std_qaly_post, 0.0]
}

new_params = {
    'p_matrix': build_p_matrix(new_prob_event, new_prob_dead_w, new_prob_dead_p),
    'costs': df_to_cost_list(new_cost_df, custom_states),
    'cost_df': new_cost_df,
    'utilities': [new_qaly_well, new_qaly_post, 0.0]
}

st.markdown("---")
# --- Base Case Execution ---
st.header("Base Case Analysis")
res = compare_strategies(std_params, new_params, n_cycles, wtp, discount_rate)

col1, col2, col3 = st.columns(3)

def render_metric(label, value, condition_val=None):
    # Minimal pastel styling with colored accent borders
    bg = "#ffffff"
    border = "#f1f5f9"
    text_color = "#1e293b"
    label_color = "#64748b"
    trend_html = ""
    accent = "#cbd5e1"
    
    if "Standard" in label:
        bg = "#f8fafc" # Slate
        border = "#e2e8f0"
        accent = "#94a3b8"
    elif "New Intervention" in label:
        bg = "#eff6ff" # Blue
        border = "#dbeafe"
        accent = "#60a5fa"
    elif "Incremental" in label:
        bg = "#fefce8" # Yellow
        border = "#fef08a"
        accent = "#facc15"
    elif "ICER" in label:
        bg = "#faf5ff" # Purple
        border = "#e9d5ff"
        accent = "#c084fc"

    if condition_val is not None and condition_val < 0:
        bg = "#fef2f2"
        border = "#fecaca"
        accent = "#f87171"
        text_color = "#991b1b"
        trend_html = """<div style="font-size: 0.8rem; color: #ef4444; font-weight: 600; margin-top: 8px;">Negative Outcome</div>"""
    elif condition_val is not None:
        bg = "#ecfdf5"
        border = "#a7f3d0"
        accent = "#34d399"
        text_color = "#065f46"
        trend_html = """<div style="font-size: 0.8rem; color: #10b981; font-weight: 600; margin-top: 8px;">Favorable Outcome</div>"""

    return f"""
    <div style="
        background-color: {bg}; 
        border-radius: 12px; 
        padding: 24px; 
        border: 1px solid {border}; 
        border-top: 4px solid {accent};
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02); 
        margin-bottom: 1rem;
    ">
        <div style="color: {label_color}; font-size: 0.95rem; font-weight: 600; margin-bottom: 8px;">{label}</div>
        <div style="color: {text_color}; font-size: 2.2rem; font-weight: 700; line-height: 1.2;">{value}</div>
        {trend_html}
    </div>
    """

col1.markdown(render_metric("Standard Care Cost", f"${res['std_cost']:,.0f}"), unsafe_allow_html=True)
col2.markdown(render_metric("New Intervention Cost", f"${res['new_cost']:,.0f}"), unsafe_allow_html=True)
col3.markdown(render_metric("Incremental Cost", f"${res['inc_cost']:,.0f}"), unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
col1.markdown(render_metric("Standard Care QALYs", f"{res['std_qaly']:,.2f}"), unsafe_allow_html=True)
col2.markdown(render_metric("New Intervention QALYs", f"{res['new_qaly']:,.2f}"), unsafe_allow_html=True)
col3.markdown(render_metric("Incremental QALYs", f"{res['inc_qaly']:,.2f}", res['inc_qaly']), unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
icer_str = f"${res['icer']:,.0f}" if res['icer'] != float('inf') else "Dominated"
col1.markdown(render_metric("ICER ($/QALY)", icer_str), unsafe_allow_html=True)
col2.markdown(render_metric("Incremental NMB (INMB)", f"${res['inmb']:,.0f}", res['inmb']), unsafe_allow_html=True)

desc_color = "#166534" if res['inmb'] > 0 else "#991b1b"
desc_text = "Cost-Effective" if res['inmb'] > 0 else "Not Cost-Effective"
col3.markdown(f'''
<div style="padding-top: 2.5rem;">
    <span style="font-weight:600; color:#475569; font-size:1.1rem;">Decision: </span>
    <span style="font-weight:700; color:{desc_color}; font-size:1.1rem;">{desc_text}</span>
</div>
''', unsafe_allow_html=True)

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
        st.plotly_chart(plot_ceac(psa_df, max_wtp_ceac), use_container_width=True)
        st.markdown(f"Shows the probability that each strategy is cost-effective from $0 to user-selected ${max_wtp_ceac:,.0f} WTP.")
        
    with tab2:
        st.pyplot(plot_ce_plane(psa_df, wtp))
        
    with tab3:
        st.plotly_chart(plot_inmb_distribution(psa_df), use_container_width=True)
        prob_ce = (psa_df['INMB'] > 0).mean()
        st.info(f"**Probability Cost-Effective (INMB > 0):** {prob_ce*100:.1f}%")

st.markdown("---")
st.header("Export Native Formula-Driven Excel Model")
st.markdown("We automatically generate the actual `.xlsx` workbook containing your dynamic line-item costs inserted directly into standard Excel formulas for traceability and auditing.")

from generate_excel_model import create_excel_model
create_excel_model(std_params, new_params, n_cycles, wtp, discount_rate, custom_states, filename="Formula_CEA_Model.xlsx")

with open("Formula_CEA_Model.xlsx", "rb") as f:
    st.download_button(
        label="Download Live Excel Model",
        data=f,
        file_name="Formula_CEA_Model.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary"
    )
