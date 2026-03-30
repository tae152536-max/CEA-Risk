import streamlit as st
import pandas as pd
import numpy as np
from n_state_markov import compare_n_state_strategies
from n_state_psa import run_n_state_psa
from n_state_owsa import run_n_state_owsa
from make_template import create_template_bytes, parse_template

import visualizations
import importlib
importlib.reload(visualizations)
from visualizations import plot_ce_plane, plot_ceac, plot_inmb_distribution, plot_cost_breakdown, plot_tornado

st.set_page_config(page_title="Generic N-State Model", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Montserrat', sans-serif !important; color: #1e293b !important; font-size: 14px !important; }
    .stApp { background-color: #f8fafc !important; }
    h1 { font-family: 'Montserrat', sans-serif !important; font-size: 1.9rem !important; color: #0f172a !important; font-weight: 700 !important; letter-spacing: -0.5px !important; margin-bottom: 4px; border-bottom: none !important;}
    h2 { font-family: 'Montserrat', sans-serif !important; font-size: 1.4rem !important; color: #1e293b !important; font-weight: 600 !important; letter-spacing: -0.3px !important; margin-top: 30px; margin-bottom: 15px; }
    h3, h4, h5 { font-family: 'Montserrat', sans-serif !important; font-size: 1.1rem !important; color: #334155 !important; font-weight: 600 !important; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 90% !important; }
    [data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #e2e8f0 !important; }
    [data-testid="stMetric"] { background-color: #ffffff; border-radius: 12px; padding: 20px 24px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; text-align: center; border-top: 4px solid #cbd5e1; transition: all 0.3s ease; }
    [data-testid="stMetric"]:hover { transform: translateY(-3px); box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); }
    [data-testid="stMetricValue"] { font-weight: 700; color: #0f172a; font-size: 2.0rem !important; }
    [data-testid="stMetricLabel"] { font-weight: 600; color: #64748b; font-size: 0.95rem !important; margin-bottom: 8px; }
    .stButton>button { background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%) !important; border: none !important; color: #ffffff !important; border-radius: 8px !important; font-weight: 600 !important; font-size: 0.95rem !important; padding: 0.5rem 1.4rem !important; box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2) !important; transition: all 0.3s ease; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3) !important; }
    .stTabs [data-baseweb="tab-list"] { border-bottom: none; gap: 10px; background-color: #f1f5f9; padding: 5px; border-radius: 12px; }
    .stTabs [data-baseweb="tab"] { background-color: transparent !important; border-radius: 8px; border: none !important; padding: 8px 16px; font-weight: 600; color: #64748b; transition: all 0.2s ease; }
    .stTabs [aria-selected="true"] { background-color: #ffffff !important; color: #0f172a !important; box-shadow: 0 1px 3px 0 rgba(0,0,0,0.1) !important; }
    [data-testid="stDataFrame"] { border-radius: 12px !important; border: 1px solid #e2e8f0 !important; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05) !important; }
    .upload-banner { background: linear-gradient(135deg, #dcfce7, #bbf7d0); border: 1px solid #4ade80; border-radius: 10px; padding: 10px 16px; margin-bottom: 12px; font-size: 0.85rem; color: #166534; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — Apply any pending upload BEFORE any widget renders
#           (sets widget keys so they initialise with uploaded values)
# ══════════════════════════════════════════════════════════════════════════════
if "pending_upload" in st.session_state:
    p = st.session_state.pop("pending_upload")

    # Sidebar widgets — set their keys so the widgets pick up new defaults
    st.session_state["_n_states_input"]       = p["n_states"]
    st.session_state["_n_cycles_slider"]      = p["n_cycles"]
    st.session_state["_wtp_input"]            = p["wtp"]
    st.session_state["_discount_rate_slider"] = p["discount_rate"]

    # State names editor
    st.session_state["state_names_df"] = pd.DataFrame(
        {"State Name": p["state_names"]}
    )

    # Uploaded matrix / utility / cost DataFrames (keyed by upload version)
    v = st.session_state.get("upload_version", 0) + 1
    st.session_state["upload_version"]      = v
    st.session_state[f"u_std_p_{v}"]        = p["std_p_df"]
    st.session_state[f"u_std_u_{v}"]        = p["std_u_df"]
    st.session_state[f"u_std_cost_{v}"]     = p["std_cost_df"]
    st.session_state[f"u_new_p_{v}"]        = p["new_p_df"]
    st.session_state[f"u_new_u_{v}"]        = p["new_u_df"]
    st.session_state[f"u_new_cost_{v}"]     = p["new_cost_df"]
    st.session_state["_upload_source_name"] = p.get("source_name", "")

version = st.session_state.get("upload_version", 0)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE TITLE
# ══════════════════════════════════════════════════════════════════════════════
st.title("Generic N-State Markov Risk Analysis Model")

if st.session_state.get("_upload_source_name"):
    st.markdown(
        f'<div class="upload-banner">📂 Loaded from: '
        f'<strong>{st.session_state["_upload_source_name"]}</strong></div>',
        unsafe_allow_html=True,
    )

st.markdown(
    "Define a fully customisable Markov model — upload an Excel template "
    "or fill in the matrices manually."
)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Section 0: Excel Import / Export
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.header("0. Excel Template")

# Download blank template (uses current n_states if already set, else 3)
current_n = st.session_state.get("_n_states_input", 3)
current_names = (
    st.session_state["state_names_df"]["State Name"].tolist()
    if "state_names_df" in st.session_state
    else None
)
st.sidebar.download_button(
    label="⬇️  Download blank template",
    data=create_template_bytes(current_n, current_names),
    file_name="CEA_Model_Template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
)

uploaded_file = st.sidebar.file_uploader(
    "⬆️  Upload filled template (.xlsx)",
    type=["xlsx"],
    help="Fill in the downloaded template, save it, then upload here.",
)

if uploaded_file is not None:
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("_last_upload_id") != file_id:
        # New file — parse it
        with st.spinner("Parsing template…"):
            params, err = parse_template(uploaded_file)
        if err:
            st.sidebar.error(f"❌ Parse error: {err}")
        else:
            params["source_name"] = uploaded_file.name
            st.session_state["_last_upload_id"] = file_id
            st.session_state["pending_upload"]   = params
            st.rerun()

st.sidebar.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Section 1: Define States
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.header("1. Define System")
n_states = st.sidebar.number_input(
    "Number of Health States",
    min_value=2, max_value=10, value=3,
    key="_n_states_input",
)

# Initialise / resize state names table
if (
    "state_names_df" not in st.session_state
    or len(st.session_state["state_names_df"]) != n_states
):
    defaults = ["Well", "Sick", "Dead", "Severe", "Critical",
                "State 6", "State 7", "State 8", "State 9", "State 10"]
    if n_states == 3:
        default_states = ["Well", "Sick", "Dead"]
    else:
        default_states = defaults[:n_states]
    st.session_state["state_names_df"] = pd.DataFrame({"State Name": default_states})

st.sidebar.markdown("**Name your states (Row 0 = starting state):**")
states_df = st.sidebar.data_editor(
    st.session_state["state_names_df"],
    use_container_width=True,
    hide_index=True,
)
st.session_state["state_names_df"] = states_df
state_names = states_df["State Name"].tolist()

st.sidebar.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Section 2: Global Parameters
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.header("2. Global Parameters")
n_cycles = st.sidebar.slider(
    "Time Horizon (Cycles)",
    min_value=1, max_value=50, value=20,
    key="_n_cycles_slider",
)
wtp = st.sidebar.number_input(
    "Willingness To Pay (WTP/QALY)",
    min_value=1000, value=50000, step=5000,
    key="_wtp_input",
)
discount_rate = st.sidebar.slider(
    "Discount Rate",
    min_value=0.0, max_value=0.10, value=0.03, step=0.01,
    key="_discount_rate_slider",
)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def build_default_matrix(n, name_list):
    df = pd.DataFrame(np.eye(n), columns=name_list, index=name_list)
    if n == 3:
        df.iloc[0, :] = [0.8, 0.15, 0.05]
        df.iloc[1, :] = [0.0, 0.8,  0.20]
        df.iloc[2, :] = [0.0, 0.0,  1.00]
    else:
        for i in range(n - 1):
            df.iloc[i, i]   = 0.90
            df.iloc[i, i+1] = 0.10
        df.iloc[-1, :] = 0.0
        df.iloc[-1,-1] = 1.0
    return df


def _safe_df(key, fallback_fn):
    """Return uploaded DF for current version, or call fallback_fn()."""
    candidate = st.session_state.get(f"{key}_{version}")
    if candidate is not None:
        # Validate shape / columns match current n_states
        try:
            if hasattr(candidate, "shape") and candidate.shape == (n_states, n_states):
                return candidate
            if hasattr(candidate, "__len__") and len(candidate) == n_states:
                return candidate
        except Exception:
            pass
    return fallback_fn()


def _safe_cost_df(key, fallback):
    candidate = st.session_state.get(f"{key}_{version}")
    return candidate if candidate is not None and not candidate.empty else fallback


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Data Input Matrices
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Data Input Matrices (From Row → To Column)")

tab_sc, tab_ni = st.tabs(["Standard Care", "New Intervention"])

# ── Standard Care ──────────────────────────────────────────────────────────
with tab_sc:
    st.markdown("##### Transition Probabilities")
    init_std_p = _safe_df("u_std_p", lambda: build_default_matrix(n_states, state_names))
    # Always rebuild with current state names as columns (handles rename)
    if list(init_std_p.columns) != state_names or list(init_std_p.index) != state_names:
        init_std_p = build_default_matrix(n_states, state_names)
    std_p_df = st.data_editor(
        init_std_p, key=f"std_p_{version}", use_container_width=True
    )

    st.markdown("##### State Utilities (QALYs)")
    init_std_u = _safe_df(
        "u_std_u",
        lambda: pd.DataFrame({
            "State": state_names,
            "Utility (0-1)": np.linspace(1.0, 0.0, n_states),
        }),
    )
    if len(init_std_u) != n_states:
        init_std_u = pd.DataFrame({
            "State": state_names,
            "Utility (0-1)": np.linspace(1.0, 0.0, n_states),
        })
    std_u_df = st.data_editor(
        init_std_u, hide_index=True, key=f"std_u_{version}",
        use_container_width=True,
    )

    st.markdown("##### Annual Costs")
    _std_cost_default = pd.DataFrame([{
        "State": state_names[0], "Subgroup": "Medical",
        "Item": "Checkup", "Cost ($/year)": 1000.0,
        "Distribution": "Gamma", "SE/SD": 100.0,
    }])
    init_std_cost = _safe_cost_df("u_std_cost", _std_cost_default)
    std_cost_df = st.data_editor(
        init_std_cost, num_rows="dynamic", hide_index=True,
        key=f"std_cost_{version}", use_container_width=True,
        column_config={
            "State":        st.column_config.SelectboxColumn(
                                "State", options=state_names, required=True),
            "Subgroup":     st.column_config.SelectboxColumn(
                                "Subgroup",
                                options=["Medical", "Non-Medical", "Indirect"],
                                required=True),
            "Distribution": st.column_config.SelectboxColumn(
                                "Distribution",
                                options=["Gamma", "Lognormal", "Normal",
                                         "Uniform", "Triangular", "Beta", "Fixed"],
                                required=True),
        },
    )

# ── New Intervention ────────────────────────────────────────────────────────
with tab_ni:
    st.markdown("##### Transition Probabilities")
    init_new_p = _safe_df("u_new_p", lambda: build_default_matrix(n_states, state_names))
    if list(init_new_p.columns) != state_names or list(init_new_p.index) != state_names:
        init_new_p = build_default_matrix(n_states, state_names)
    new_p_df = st.data_editor(
        init_new_p, key=f"new_p_{version}", use_container_width=True
    )

    st.markdown("##### State Utilities (QALYs)")
    init_new_u = _safe_df(
        "u_new_u",
        lambda: pd.DataFrame({
            "State": state_names,
            "Utility (0-1)": np.linspace(1.0, 0.0, n_states),
        }),
    )
    if len(init_new_u) != n_states:
        init_new_u = pd.DataFrame({
            "State": state_names,
            "Utility (0-1)": np.linspace(1.0, 0.0, n_states),
        })
    new_u_df = st.data_editor(
        init_new_u, hide_index=True, key=f"new_u_{version}",
        use_container_width=True,
    )

    st.markdown("##### Annual Costs")
    _new_cost_default = pd.DataFrame([{
        "State": state_names[0], "Subgroup": "Medical",
        "Item": "New Drug", "Cost ($/year)": 2500.0,
        "Distribution": "Gamma", "SE/SD": 250.0,
    }])
    init_new_cost = _safe_cost_df("u_new_cost", _new_cost_default)
    new_cost_df = st.data_editor(
        init_new_cost, num_rows="dynamic", hide_index=True,
        key=f"new_cost_{version}", use_container_width=True,
        column_config={
            "State":        st.column_config.SelectboxColumn(
                                "State", options=state_names, required=True),
            "Subgroup":     st.column_config.SelectboxColumn(
                                "Subgroup",
                                options=["Medical", "Non-Medical", "Indirect"],
                                required=True),
            "Distribution": st.column_config.SelectboxColumn(
                                "Distribution",
                                options=["Gamma", "Lognormal", "Normal",
                                         "Uniform", "Triangular", "Beta", "Fixed"],
                                required=True),
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
def validate_matrix(df, name):
    errors = []
    mat = df.values
    if not np.allclose(mat.sum(axis=1), 1.0, atol=1e-6):
        errors.append(f"❌ Rows in **{name}** must sum to 1.0  "
                      f"(current sums: {mat.sum(axis=1).round(4).tolist()})")
    if (mat < 0).any() or (mat > 1).any():
        errors.append(f"❌ All probabilities in **{name}** must be between 0 and 1.")
    return errors

errs = validate_matrix(std_p_df, "Standard Care") + validate_matrix(new_p_df, "New Intervention")
if errs:
    for e in errs:
        st.error(e)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# ASSEMBLE PARAMETER DICTS
# ══════════════════════════════════════════════════════════════════════════════
std_params = {
    "p_matrix":  std_p_df.values,
    "cost_df":   std_cost_df,
    "utilities": std_u_df["Utility (0-1)"].values.tolist(),
}
new_params = {
    "p_matrix":  new_p_df.values,
    "cost_df":   new_cost_df,
    "utilities": new_u_df["Utility (0-1)"].values.tolist(),
}


# ══════════════════════════════════════════════════════════════════════════════
# BASE CASE ANALYSIS  (runs automatically on every render)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("Base Case Analysis")

res = compare_n_state_strategies(
    std_params, new_params, state_names, n_cycles, wtp, discount_rate
)

c1, c2, c3 = st.columns(3)
c1.metric("Standard Care Cost",      f"${res['std_cost']:,.0f}")
c2.metric("New Intervention Cost",   f"${res['new_cost']:,.0f}")
c3.metric("Incremental Cost",        f"${res['inc_cost']:,.0f}")

c1, c2, c3 = st.columns(3)
c1.metric("Standard Care QALYs",     f"{res['std_qaly']:,.3f}")
c2.metric("New Intervention QALYs",  f"{res['new_qaly']:,.3f}")
c3.metric("Incremental QALYs",       f"{res['inc_qaly']:,.3f}")

c1, c2, c3 = st.columns(3)
icer_display = (
    f"${res['icer']:,.0f}"
    if res["icer"] not in (float("inf"), -1)
    else "Dominated / Equivalent"
)
c1.metric("ICER ($/QALY)", icer_display)
c2.metric("Incremental NMB (INMB)", f"${res['inmb']:,.0f}")
decision = "✅ Cost-Effective" if res["inmb"] > 0 else "❌ Not Cost-Effective"
c3.markdown(f"<br><b>Decision at WTP=${wtp:,}:</b> {decision}", unsafe_allow_html=True)

st.subheader("Cost Breakdown")
c_chart, _ = st.columns([1, 1])
with c_chart:
    st.pyplot(plot_cost_breakdown(res))


# ══════════════════════════════════════════════════════════════════════════════
# OWSA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("Deterministic Sensitivity Analysis (OWSA)")
owsa_var = st.slider("Parameter Variance (%)", 10, 50, 20, 5) / 100.0

if st.button("Generate One-Way Tornado", type="primary"):
    with st.spinner("Running N-State OWSA…"):
        owsa_df, base_inmb = run_n_state_owsa(
            std_params, new_params, state_names,
            n_cycles, wtp, discount_rate, owsa_var,
        )
    st.pyplot(plot_tornado(owsa_df, base_inmb))
    st.dataframe(
        owsa_df[["Parameter", "Swing", "INMB_Low", "INMB_High"]].head(10),
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PSA / RISK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("Risk Analysis (PSA)")

col_a, col_b = st.columns(2)
n_simulations = col_a.number_input(
    "Number of Simulations", min_value=100, max_value=10000, value=1000, step=100
)
max_wtp_ceac = col_b.slider(
    "Max WTP on CEAC Plot", 50000, 500000, 150000, 10000
)

if st.button("Run Probabilistic Risk Analysis", type="primary"):
    with st.spinner(
        f"Running {n_simulations:,} Monte Carlo iterations on {n_states} states…"
    ):
        psa_df = run_n_state_psa(
            std_params, new_params, state_names,
            n_cycles, wtp, n_iterations=n_simulations,
        )
    st.session_state["n_psa_df"] = psa_df
    st.success("PSA completed!")

if "n_psa_df" in st.session_state:
    psa_df = st.session_state["n_psa_df"]
    t1, t2, t3 = st.tabs(["CEAC", "CE Plane", "INMB Distribution"])
    with t1:
        st.plotly_chart(plot_ceac(psa_df, max_wtp_ceac), use_container_width=True)
    with t2:
        st.pyplot(plot_ce_plane(psa_df, wtp))
    with t3:
        st.plotly_chart(plot_inmb_distribution(psa_df), use_container_width=True)
        prob_ce = (psa_df["INMB"] > 0).mean()
        st.info(f"**Probability Cost-Effective at WTP=${wtp:,}:** {prob_ce*100:.1f}%")
