import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Professional aesthetic settings
sns.set_style("white")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelcolor'] = '#34495e'
plt.rcParams['text.color'] = '#2c3e50'
plt.rcParams['xtick.color'] = '#7f8c8d'
plt.rcParams['ytick.color'] = '#7f8c8d'

def plot_cost_breakdown(res):
    """Plots a stacked bar chart showing the breakdown of cost subtypes."""
    # Ensure keys always come out in the exact order: Medical, Non-Medical, Indirect
    ordered_keys = ['Medical', 'Non-Medical', 'Indirect']
    available_keys = [k for k in ordered_keys if k in res['std_cost_subtypes']]
    
    data = {
        'Strategy': ['Standard Care', 'New Intervention']
    }
    for k in available_keys:
        data[k] = [res['std_cost_subtypes'][k], res['new_cost_subtypes'][k]]
        
    df = pd.DataFrame(data).set_index('Strategy')
    
    # Exact colors from the user's reference image
    color_map = {
        'Medical': '#5c2d73',      # Deep plum/purple
        'Non-Medical': '#48a49c',  # Teal
        'Indirect': '#fae74b'      # Bright Yellow/Gold
    }
    colors = [color_map[k] for k in available_keys]
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # edgecolor='white', linewidth=1.5 perfectly matches the tiny white gaps between stacks
    df.plot(
        kind='bar', 
        stacked=True, 
        ax=ax, 
        color=colors, 
        width=0.5,
        edgecolor='white', 
        linewidth=1.5
    )
    
    title_color = '#1e293b'
    text_color = '#64748b'
    
    ax.set_title('Cost Breakdown by Subtype', fontweight='bold', fontsize=15, color=title_color, pad=20)
    ax.set_ylabel('Total Cost ($)', fontsize=12, color=text_color)
    ax.set_xlabel('')
    
    plt.xticks(rotation=0, fontsize=12, color=text_color)
    plt.yticks(fontsize=12, color=text_color)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#334155')
    ax.spines['bottom'].set_color('#334155')
    
    legend = ax.legend(title='Cost Type', frameon=False, loc='upper left', fontsize=12)
    plt.setp(legend.get_title(), fontsize=12, color=title_color)
    for text in legend.get_texts():
        text.set_color(title_color)
        
    ax.grid(axis='y', alpha=0.10, linestyle='--', color='#94a3b8')
    plt.tight_layout()
    
    return fig

def plot_ce_plane(psa_df, wtp):
    """Plots the Cost-Effectiveness Plane (Scatter plot of Inc QALY vs Inc Cost)"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='Inc_QALY', y='Inc_Cost', data=psa_df, alpha=0.6, ax=ax, color='#3498db', edgecolor='none')
    
    x_vals = np.array(ax.get_xlim())
    y_vals = wtp * x_vals
    ax.plot(x_vals, y_vals, color='#e74c3c', linestyle='--', linewidth=1.5, label=f'WTP (${wtp}/QALY)')
    
    mean_qaly = psa_df['Inc_QALY'].mean()
    mean_cost = psa_df['Inc_Cost'].mean()
    ax.scatter(mean_qaly, mean_cost, color='#2c3e50', s=100, marker='X', label='Expected Value')
    
    ax.axhline(0, color='#bdc3c7', linewidth=1)
    ax.axvline(0, color='#bdc3c7', linewidth=1)
    
    ax.set_title('Cost-Effectiveness Plane', fontweight='bold', pad=15)
    ax.set_xlabel('Incremental QALYs')
    ax.set_ylabel('Incremental Cost ($)')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.15, linestyle=':')
    sns.despine(ax=ax)
    return fig

def plot_ceac(psa_df, max_wtp=150000):
    """Plots the Cost-Effectiveness Acceptability Curve (CEAC)"""
    wtp_range = np.linspace(0, max_wtp, 151)
    new_probs = []
    std_probs = []
    for wtp in wtp_range:
        inmb = (psa_df['Inc_QALY'] * wtp) - psa_df['Inc_Cost']
        prob_new = (inmb > 0).mean()
        new_probs.append(prob_new)
        std_probs.append(1 - prob_new)
        
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=wtp_range, y=new_probs, 
        mode='lines', 
        name='New Intervention', 
        line=dict(color='#2ecc71', width=3.5),
        fill='tozeroy',
        fillcolor='rgba(46, 204, 113, 0.1)',
        hovertemplate="WTP: $%{x:,.0f}<br>Probability Cost-Effective: %{y:.1%}<extra></extra>"
    ))
    
    fig.add_trace(go.Scatter(
        x=wtp_range, y=std_probs, 
        mode='lines', 
        name='Standard Care', 
        line=dict(color='#3498db', width=3.5),
        fill='tozeroy',
        fillcolor='rgba(52, 152, 219, 0.1)',
        hovertemplate="WTP: $%{x:,.0f}<br>Probability Cost-Effective: %{y:.1%}<extra></extra>"
    ))
    
    # Highlight the cross-over point if it exists
    cross_over_wtp = None
    for i in range(1, len(wtp_range)):
        if (new_probs[i-1] - std_probs[i-1]) * (new_probs[i] - std_probs[i]) < 0:
            w1, w2 = wtp_range[i-1], wtp_range[i]
            p1_new, p2_new = new_probs[i-1], new_probs[i]
            p1_std, p2_std = std_probs[i-1], std_probs[i]
            diff1 = p1_std - p1_new
            diff2 = (p2_new - p1_new) - (p2_std - p1_std)
            if diff2 != 0:
                t = diff1 / diff2
                cross_over_wtp = w1 + t * (w2 - w1)
            break
            
    fig.update_layout(
        title=dict(text='Cost-Effectiveness Acceptability Curve', font=dict(size=20, color='#1e293b', family='sans-serif'), x=0.5, xanchor='center'),
        xaxis_title='<b>Willingness To Pay (Threshold $/QALY)</b>',
        yaxis_title='<b>Probability Cost-Effective</b>',
        yaxis=dict(range=[0, 1.05], tickformat=".0%", gridcolor='#f1f5f9', zerolinecolor='#cbd5e1', zerolinewidth=1.5),
        xaxis=dict(tickformat="$,.0f", gridcolor='#f1f5f9', zerolinecolor='#cbd5e1', zerolinewidth=1.5),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=13, color='#334155'),
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        margin=dict(l=60, r=40, t=80, b=60),
        hovermode="x unified",
        font=dict(color='#475569', family='sans-serif')
    )
    
    if cross_over_wtp is not None and cross_over_wtp > 0 and cross_over_wtp < max_wtp:
        fig.add_vline(x=cross_over_wtp, line_width=1.5, line_dash="dash", line_color="#94a3b8")
        fig.add_annotation(
            x=cross_over_wtp,
            y=0.5,
            text=f"WTP Threshold:<br>${cross_over_wtp:,.0f}",
            showarrow=True,
            arrowhead=1,
            ax=60,
            ay=0,
            bgcolor="white",
            bordercolor="#cbd5e1",
            borderwidth=1,
            borderpad=4,
            font=dict(size=11, color='#475569')
        )

    return fig

def plot_inmb_distribution(psa_df):
    """Plots the distribution of the Incremental Net Monetary Benefit (INMB) using Plotly"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=psa_df['INMB'], 
        nbinsx=30, 
        marker_color='#8b5cf6', # pastel purple
        marker_line_color='white',
        marker_line_width=1.5,
        opacity=0.8,
        name='Simulations',
        hovertemplate="INMB Range: $%{x}<br>Count: %{y}<extra></extra>"
    ))
    
    fig.add_vline(x=0, line_width=2.5, line_dash="dash", line_color="#ef4444")
    fig.add_annotation(
        x=0,
        y=1,
        yref="paper",
        text="Decision Boundary ($0)",
        showarrow=False,
        xanchor="right",
        xshift=-5,
        font=dict(size=12, color="#ef4444", weight="bold")
    )
    
    fig.update_layout(
        title=dict(text='Incremental Net Monetary Benefit (INMB) Distribution', font=dict(size=18, color='#1e293b', family='sans-serif'), x=0.5, xanchor='center'),
        xaxis_title='<b>INMB ($)</b>',
        yaxis_title='<b>Frequency</b>',
        yaxis=dict(gridcolor='#f1f5f9', zerolinecolor='#cbd5e1', zerolinewidth=1.5),
        xaxis=dict(tickformat="$,.0f", gridcolor='#f1f5f9', zerolinecolor='#cbd5e1', zerolinewidth=1.5),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        margin=dict(l=60, r=40, t=80, b=60),
        font=dict(color='#475569', family='sans-serif')
    )
    return fig

def plot_tornado(owsa_df, base_inmb):
    """Plots the Tornado Diagram for One-Way Sensitivity Analysis"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    top_df = owsa_df.head(10).copy()
    top_df = top_df.sort_values('Swing', ascending=True)
    
    y_pos = np.arange(len(top_df))
    lows = top_df['INMB_Low'].values - base_inmb
    highs = top_df['INMB_High'].values - base_inmb
    
    ax.barh(y_pos, lows, align='center', color='#3498db', alpha=0.8, label='Low Parameter Value')
    ax.barh(y_pos, highs, align='center', color='#e74c3c', alpha=0.8, label='High Parameter Value')
    
    labels = []
    for p in top_df['Parameter']:
        name = p.replace('_', ' ').title()
        if 'Multiplier' in name:
            name = name.replace('Multiplier', 'Total Cost')
        labels.append(name)
        
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Change in INMB from Base Case ($)')
    ax.set_title('Tornado Diagram (One-Way Sensitivity Analysis)', fontweight='bold', pad=15)
    
    ax.axvline(0, color='#2c3e50', linewidth=1.5)
    ax.grid(True, axis='x', alpha=0.15, linestyle=':')
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    
    return fig
