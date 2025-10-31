"""
Policy Selection Analysis Across Defense Spending Levels

This script creates comprehensive visualizations showing:
1. Policy Selection Heatmap - Which policies are selected at each defense spending level
2. Policy Frequency Chart - How often each policy appears across all spending levels
3. Defense Policy Substitution Chart - Changes in NS (National Security) policy selections
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Configuration
output_dir = Path('outputs/defense')
output_dir.mkdir(parents=True, exist_ok=True)

# Defense spending levels (in billions)
spending_levels = list(range(-4000, 6500, 500))

def load_policy_data():
    """Load policy selection data from all defense spending CSV files."""
    policy_data = {}
    
    print("Loading policy selection data...")
    for level in spending_levels:
        file_path = output_dir / f'max_gdp_defense{level}.csv'
        try:
            df = pd.read_csv(file_path)
            # All policies in the CSV are selected policies (the file only contains selected policies)
            selected = df['Option'].tolist()
            policy_data[level] = {
                'df': df,
                'selected': selected
            }
            print(f"  [OK] Loaded {file_path.name}: {len(selected)} policies selected")
        except FileNotFoundError:
            print(f"  [MISSING] {file_path.name}")
            policy_data[level] = {'df': None, 'selected': []}
    
    return policy_data

def extract_policy_number(policy_name):
    """Extract policy number/code from policy name for sorting."""
    # Handle different policy types: numbered (1-67), S-codes (S1-S17), NS-codes (NS1-NS7)
    if policy_name.startswith('NS'):
        # Extract NS code (e.g., "NS1B" -> "NS01B" for sorting)
        parts = policy_name.split(':')[0].strip()
        num = ''.join(filter(str.isdigit, parts))
        letter = ''.join(filter(str.isalpha, parts[2:]))  # Get letter after NS
        return f"NS{int(num):02d}{letter}"
    elif policy_name.startswith('S'):
        # Extract S code (e.g., "S1" -> "S01")
        num = ''.join(filter(str.isdigit, policy_name.split(':')[0]))
        return f"S{int(num):02d}"
    else:
        # Extract regular number (e.g., "1:" -> "001")
        num = ''.join(filter(str.isdigit, policy_name.split(':')[0]))
        if num:
            return f"{int(num):03d}"
    return policy_name

def create_heatmap(policy_data):
    """Create a heatmap showing policy selections across defense spending levels."""
    print("\nCreating policy selection heatmap...")
    
    # Get all unique policies across all spending levels
    all_policies = set()
    for data in policy_data.values():
        if data['df'] is not None:
            all_policies.update(data['df']['Option'].tolist())
    
    all_policies = sorted(all_policies, key=extract_policy_number)
    
    # Create matrix: rows = policies, columns = spending levels
    matrix = np.zeros((len(all_policies), len(spending_levels)))
    
    for col_idx, level in enumerate(spending_levels):
        selected = policy_data[level]['selected']
        for row_idx, policy in enumerate(all_policies):
            if policy in selected:
                matrix[row_idx, col_idx] = 1
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 24))
    
    # Truncate policy names for display
    policy_labels = []
    for policy in all_policies:
        # Extract policy code and first few words
        if ':' in policy:
            code, desc = policy.split(':', 1)
            # Limit description to 60 characters
            if len(desc) > 60:
                desc = desc[:57] + '...'
            policy_labels.append(f"{code}:{desc}")
        else:
            policy_labels.append(policy[:70])
    
    # Create heatmap
    sns.heatmap(matrix, 
                cmap=['#f0f0f0', '#2E7D32'],  # Light gray for unselected, green for selected
                cbar_kws={'label': 'Selected'},
                yticklabels=policy_labels,
                xticklabels=[f"${l:+,}B" for l in spending_levels],
                linewidths=0.5,
                linecolor='white',
                ax=ax)
    
    ax.set_title('Policy Selection Across Defense Spending Levels\n(Green = Selected, Gray = Not Selected)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Defense Spending Change (Billions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Policy Options', fontsize=12, fontweight='bold')
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    output_file = output_dir / 'policy_selection_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Heatmap saved to '{output_file}'")
    plt.close()
    
    return all_policies, matrix

def create_frequency_chart(all_policies, matrix):
    """Create a bar chart showing how often each policy is selected."""
    print("\nCreating policy frequency chart...")
    
    # Calculate frequency for each policy (sum across spending levels)
    frequencies = matrix.sum(axis=1)
    
    # Create DataFrame for easier plotting
    freq_df = pd.DataFrame({
        'Policy': all_policies,
        'Frequency': frequencies,
        'Percentage': (frequencies / len(spending_levels)) * 100
    })
    
    # Sort by frequency
    freq_df = freq_df.sort_values('Frequency', ascending=True)
    
    # Categorize policies
    def categorize_policy(policy):
        if policy.startswith('NS'):
            return 'National Security'
        elif policy.startswith('S'):
            return 'Spending'
        else:
            return 'Tax'
    
    freq_df['Category'] = freq_df['Policy'].apply(categorize_policy)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 24))
    
    # Subplot 1: All policies
    colors = freq_df['Category'].map({
        'Tax': '#1f77b4',
        'Spending': '#ff7f0e', 
        'National Security': '#d62728'
    })
    
    # Truncate policy names
    policy_labels = []
    for policy in freq_df['Policy']:
        if ':' in policy:
            code, desc = policy.split(':', 1)
            if len(desc) > 50:
                desc = desc[:47] + '...'
            policy_labels.append(f"{code}:{desc}")
        else:
            policy_labels.append(policy[:60])
    
    y_pos = np.arange(len(freq_df))
    ax1.barh(y_pos, freq_df['Frequency'], color=colors)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(policy_labels, fontsize=7)
    ax1.set_xlabel('Number of Times Selected (out of 21 spending levels)', fontsize=11, fontweight='bold')
    ax1.set_title('Policy Selection Frequency\n(All Policies)', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Tax Policies'),
        Patch(facecolor='#ff7f0e', label='Spending Policies'),
        Patch(facecolor='#d62728', label='National Security Policies')
    ]
    ax1.legend(handles=legend_elements, loc='lower right')
    
    # Subplot 2: Category summary
    category_counts = freq_df.groupby('Category')['Frequency'].agg(['mean', 'sum', 'count'])
    category_counts['avg_percentage'] = (category_counts['mean'] / len(spending_levels)) * 100
    
    categories = category_counts.index
    x_pos = np.arange(len(categories))
    
    bars = ax2.bar(x_pos, category_counts['avg_percentage'], 
                   color=['#d62728', '#ff7f0e', '#1f77b4'])
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    ax2.set_ylabel('Average Selection Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Average Selection Rate by Policy Category', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, category_counts['avg_percentage'])):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%\n({int(category_counts.iloc[i]["count"])} policies)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / 'policy_frequency_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Frequency chart saved to '{output_file}'")
    plt.close()
    
    return freq_df

def create_defense_substitution_chart(policy_data):
    """Create a chart showing NS (National Security) policy substitutions."""
    print("\nCreating defense policy substitution chart...")
    
    # Extract NS policies across spending levels
    ns_selections = {}
    for level in spending_levels:
        selected = policy_data[level]['selected']
        ns_policies = [p for p in selected if p.startswith('NS')]
        ns_selections[level] = ns_policies
    
    # Get all unique NS policies
    all_ns_policies = set()
    for policies in ns_selections.values():
        all_ns_policies.update(policies)
    all_ns_policies = sorted(all_ns_policies, key=extract_policy_number)
    
    if not all_ns_policies:
        print("  [WARNING] No NS policies found in selections")
        return
    
    # Create matrix for NS policies only
    ns_matrix = np.zeros((len(all_ns_policies), len(spending_levels)))
    for col_idx, level in enumerate(spending_levels):
        for row_idx, policy in enumerate(all_ns_policies):
            if policy in ns_selections[level]:
                ns_matrix[row_idx, col_idx] = 1
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))
    
    # Subplot 1: Heatmap
    ns_labels = []
    for policy in all_ns_policies:
        if ':' in policy:
            code, desc = policy.split(':', 1)
            if len(desc) > 70:
                desc = desc[:67] + '...'
            ns_labels.append(f"{code}:{desc}")
        else:
            ns_labels.append(policy[:80])
    
    sns.heatmap(ns_matrix,
                cmap=['#ffebee', '#c62828'],  # Light red for unselected, dark red for selected
                cbar_kws={'label': 'Selected'},
                yticklabels=ns_labels,
                xticklabels=[f"${l:+,}B" for l in spending_levels],
                linewidths=1,
                linecolor='white',
                ax=ax1)
    
    ax1.set_title('National Security Policy Substitutions Across Defense Spending Levels', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Defense Spending Change (Billions)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('NS Policy Options', fontsize=11, fontweight='bold')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Subplot 2: Line chart showing number of NS policies selected
    ns_counts = [len(ns_selections[level]) for level in spending_levels]
    
    ax2.plot(spending_levels, ns_counts, marker='o', linewidth=2, markersize=8, color='#c62828')
    ax2.fill_between(spending_levels, ns_counts, alpha=0.3, color='#c62828')
    ax2.set_xlabel('Defense Spending Change (Billions)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of NS Policies Selected', fontsize=11, fontweight='bold')
    ax2.set_title('National Security Policy Count Across Spending Levels', 
                  fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(spending_levels[0], spending_levels[-1])
    
    # Format x-axis
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.axvline(x=0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    
    # Add annotations for key points
    max_ns = max(ns_counts)
    max_idx = ns_counts.index(max_ns)
    ax2.annotate(f'Max: {max_ns} policies\nat ${spending_levels[max_idx]:+,}B',
                xy=(spending_levels[max_idx], max_ns),
                xytext=(spending_levels[max_idx] + 1000, max_ns + 0.3),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / 'defense_policy_substitution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Defense substitution chart saved to '{output_file}'")
    plt.close()
    
    # Print summary
    print("\nDefense Policy Summary:")
    print(f"  Total unique NS policies: {len(all_ns_policies)}")
    print(f"  Max NS policies selected: {max_ns} (at ${spending_levels[max_idx]:+,}B)")
    print(f"  Min NS policies selected: {min(ns_counts)} (at ${spending_levels[ns_counts.index(min(ns_counts))]:+,}B)")

def print_policy_insights(freq_df, all_policies, matrix):
    """Print key insights about policy selections."""
    print("\n" + "="*70)
    print("POLICY SELECTION INSIGHTS")
    print("="*70)
    
    # Always selected policies
    always_selected = freq_df[freq_df['Frequency'] == len(spending_levels)]['Policy'].tolist()
    print(f"\n1. ALWAYS SELECTED ({len(always_selected)} policies):")
    for policy in always_selected[:10]:  # Show first 10
        print(f"   - {policy}")
    if len(always_selected) > 10:
        print(f"   ... and {len(always_selected) - 10} more")
    
    # Never selected policies
    never_selected = freq_df[freq_df['Frequency'] == 0]['Policy'].tolist()
    print(f"\n2. NEVER SELECTED ({len(never_selected)} policies):")
    for policy in never_selected[:10]:  # Show first 10
        print(f"   - {policy}")
    if len(never_selected) > 10:
        print(f"   ... and {len(never_selected) - 10} more")
    
    # Sometimes selected policies (most variable)
    mid_freq = freq_df[(freq_df['Frequency'] > 0) & (freq_df['Frequency'] < len(spending_levels))]
    mid_freq = mid_freq.sort_values('Frequency', ascending=False)
    print(f"\n3. SOMETIMES SELECTED ({len(mid_freq)} policies):")
    print("   Top 10 by selection frequency:")
    for i, row in mid_freq.head(10).iterrows():
        print(f"   - {row['Policy'][:70]}... ({int(row['Frequency'])}/{len(spending_levels)} times)")
    
    # Category breakdown
    print("\n4. SELECTION RATE BY CATEGORY:")
    category_stats = freq_df.groupby(freq_df['Policy'].apply(
        lambda p: 'NS' if p.startswith('NS') else ('S' if p.startswith('S') else 'Tax')
    ))['Percentage'].agg(['mean', 'count'])
    
    for category, stats in category_stats.iterrows():
        print(f"   {category:20s}: {stats['mean']:5.1f}% avg selection rate ({int(stats['count'])} policies)")
    
    print("\n" + "="*70)

def main():
    """Main execution function."""
    print("="*70)
    print("COMPREHENSIVE POLICY SELECTION ANALYSIS")
    print("="*70)
    
    # Load data
    policy_data = load_policy_data()
    
    # Create visualizations
    all_policies, matrix = create_heatmap(policy_data)
    freq_df = create_frequency_chart(all_policies, matrix)
    create_defense_substitution_chart(policy_data)
    
    # Print insights
    print_policy_insights(freq_df, all_policies, matrix)
    
    print("\n" + "="*70)
    print("All visualizations complete!")
    print(f"Check '{output_dir}' for output files:")
    print("  - policy_selection_heatmap.png")
    print("  - policy_frequency_analysis.png")
    print("  - defense_policy_substitution.png")
    print("="*70)

if __name__ == '__main__':
    main()