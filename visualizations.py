import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Professional aesthetic settings
sns.set_style("white")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelcolor'] = '#34495e'
plt.rcParams['text.color'] = '#2c3e50'
plt.rcParams['xtick.color'] = '#7f8c8d'
plt.rcParams['ytick.color'] = '#7f8c8d'

def plot_cost_breakdown(res):
    """Plots a stacked bar chart showing the breakdown of cost subtypes."""
    keys = list(res['std_cost_subtypes'].keys())
    data = {
        'Strategy': ['Standard Care', 'New Intervention']
    }
    for k in keys:
        data[k] = [res['std_cost_subtypes'][k], res['new_cost_subtypes'][k]]
        
    df = pd.DataFrame(data).set_index('Strategy')
    
    fig, ax = plt.subplots(figsize=(6, 5))
    df.plot(kind='bar', stacked=True, ax=ax, colormap='viridis', alpha=0.85)
    
    ax.set_title('Cost Breakdown by Subtype', fontweight='bold', pad=15)
    ax.set_ylabel('Total Cost ($)')
    ax.set_xlabel('')
    plt.xticks(rotation=0)
    ax.legend(title='Cost Type', frameon=False)
    ax.grid(axis='y', alpha=0.15, linestyle=':')
    sns.despine(ax=ax)
    
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
        
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(wtp_range, new_probs, color='#2ecc71', linewidth=2.5, label='New Intervention')
    ax.plot(wtp_range, std_probs, color='#3498db', linewidth=2.5, label='Standard Care')
    
    ax.set_title('Cost-Effectiveness Acceptability Curve', fontweight='bold', pad=15)
    ax.set_xlabel('Willingness To Pay (Threshold $/QALY)')
    ax.set_ylabel('Probability Cost-Effective')
    ax.set_ylim([0, 1.05])
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.15, linestyle=':')
    sns.despine(ax=ax)
    return fig

def plot_inmb_distribution(psa_df):
    """Plots the distribution of the Incremental Net Monetary Benefit"""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(psa_df['INMB'], kde=True, ax=ax, color='#9b59b6', bins=30, edgecolor='white')
    ax.axvline(0, color='#e74c3c', linestyle='--', linewidth=1.5, label='Decision Boundary ($0)')
    ax.set_title('Incremental Net Monetary Benefit (INMB)', fontweight='bold', pad=15)
    ax.set_xlabel('INMB ($)')
    ax.set_ylabel('Frequency')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.15, linestyle=':')
    sns.despine(ax=ax)
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
