"""
Defense Spending Optimization Visualization

Analyzes and visualizes the economic effects of different defense spending requirements
across the optimization results from max_gdp_defense.py

This version uses percentage changes from baseline for apple-to-apples comparisons.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.ticker import FuncFormatter, PercentFormatter

# Ensure output directory exists
output_dir = Path('outputs/defense')
output_dir.mkdir(parents=True, exist_ok=True)

# Load all optimization results (-4000 to 6000 in increments of 500)
spending_levels = list(range(-4000, 6500, 500))
results = {}

print("Loading optimization results...")
for level in spending_levels:
    file_path = output_dir / f'max_gdp_defense{level}.csv'
    if file_path.exists():
        results[level] = pd.read_csv(file_path)
        print(f"  [OK] Loaded {file_path}: {len(results[level])} policies selected")
    else:
        print(f"  [MISSING] {file_path}")

# Calculate aggregate metrics for each spending level
print("\nCalculating aggregate metrics...")
metrics = {
    'spending': [],
    'gdp': [],
    'capital': [],
    'jobs': [],
    'wage': [],
    'revenue': [],
    'static_revenue': [],
    'p20': [],
    'p40_60': [],
    'p80_100': [],
    'p99': [],
    'n_policies': [],
    'n_ns_policies': [],
    'ns_spending': []
}

for level in spending_levels:
    if level not in results:
        continue
    
    df = results[level]
    metrics['spending'].append(level)
    metrics['gdp'].append(df['Long-Run Change in GDP'].sum())
    metrics['capital'].append(df['Capital Stock'].sum())
    metrics['jobs'].append(df['Full-Time Equivalent Jobs'].sum())
    metrics['wage'].append(df['Wage Rate'].sum())
    metrics['revenue'].append(df['Dynamic 10-Year Revenue (billions)'].sum())
    metrics['static_revenue'].append(df['Static 10-Year Revenue (billions)'].sum())
    metrics['p20'].append(df['P20'].sum())
    metrics['p40_60'].append(df['P40-60'].sum())
    metrics['p80_100'].append(df['P80-100'].sum())
    metrics['p99'].append(df['P99'].sum())
    metrics['n_policies'].append(len(df))
    metrics['n_ns_policies'].append(len(df[df['is_NS'] == True]))
    
    # Calculate actual NS spending (negative revenue from NS policies)
    ns_revenue = df[df['is_NS'] == True]['Dynamic 10-Year Revenue (billions)'].sum()
    metrics['ns_spending'].append(-ns_revenue)  # Convert to positive spending

# Convert to DataFrame for easier analysis
metrics_df = pd.DataFrame(metrics)

print("\nAggregate Metrics Summary:")
print(metrics_df.to_string(index=False))

# Find baseline index (spending = 0)
try:
    baseline_idx = metrics['spending'].index(0)
    baseline_exists = True
except ValueError:
    # If 0 is not in the list, use the closest value
    baseline_idx = min(range(len(metrics['spending'])), 
                      key=lambda i: abs(metrics['spending'][i]))
    baseline_exists = False
    print(f"\nWarning: $0B baseline not found. Using ${metrics['spending'][baseline_idx]:,}B as reference.")

# Helper function to normalize values to 0-100 scale
def normalize_to_100(values, reverse=False):
    """Normalize values to 0-100 scale. If reverse=True, lower values get higher scores."""
    arr = np.array(values)
    min_val = arr.min()
    max_val = arr.max()
    if max_val == min_val:
        return np.array([50.0] * len(values))
    
    if reverse:
        normalized = 100 * (max_val - arr) / (max_val - min_val)
    else:
        normalized = 100 * (arr - min_val) / (max_val - min_val)
    return normalized

# Helper function to calculate percentage point difference from baseline
def pp_diff_from_baseline(values, baseline_idx):
    """Calculate percentage point difference from baseline value."""
    baseline = values[baseline_idx]
    return [(v - baseline) for v in values]

# Calculate derived metrics for visualization
# GDP: Show absolute percentage values (already percentages)
gdp_absolute = metrics['gdp']  # These are already percentages
gdp_pp_diff = pp_diff_from_baseline(metrics['gdp'], baseline_idx)  # Percentage point difference

# Jobs: Show absolute values (actual job numbers)
jobs_absolute = metrics['jobs']
jobs_diff = pp_diff_from_baseline(metrics['jobs'], baseline_idx)  # Difference from baseline

# Revenue: Keep as absolute values (billions)
revenue_abs = metrics['revenue']

# Calculate marginal efficiency (ΔGDP / ΔSpending per $1B)
marginal_gdp = []
marginal_spending_points = []
for i in range(1, len(metrics['spending'])):
    delta_gdp = metrics['gdp'][i] - metrics['gdp'][i-1]
    delta_spending = metrics['spending'][i] - metrics['spending'][i-1]
    if delta_spending != 0:
        marginal_gdp.append(delta_gdp / delta_spending * 1000)  # per $1B
    else:
        marginal_gdp.append(0)
    # Use midpoint for x-axis
    marginal_spending_points.append((metrics['spending'][i] + metrics['spending'][i-1]) / 2)

# Calculate equity ratio (P20 / P99)
# Higher values = more equitable (bottom 20% benefits more relative to top 1%)
equity_ratio = []
for p20, p99 in zip(metrics['p20'], metrics['p99']):
    if p99 != 0 and p99 > 0:
        equity_ratio.append(p20 / p99)
    else:
        equity_ratio.append(0)

# Calculate cost-benefit ratio (GDP gain per dollar of net fiscal cost)
cost_benefit = []
for gdp, revenue in zip(metrics['gdp'], metrics['revenue']):
    net_cost = -revenue  # Negative revenue = cost
    if net_cost > 0:
        # Positive cost: calculate benefit per dollar
        cost_benefit.append(gdp / net_cost * 1000)  # per $1B cost
    elif net_cost < 0:
        # Revenue surplus: very high efficiency
        cost_benefit.append(gdp / 0.001)  # Arbitrary high value
    else:
        cost_benefit.append(0)

# Calculate Composite Economic Index with 50% equity weighting
# Weights: 50% Equity, 20% GDP, 20% Jobs, 10% Revenue
print("\nCalculating Composite Economic Index...")
print("  Weights: 50% Equity + 20% GDP + 20% Jobs + 10% Revenue")

# Normalize each component to 0-100
gdp_norm = normalize_to_100(metrics['gdp'])
jobs_norm = normalize_to_100(metrics['jobs'])
revenue_norm = normalize_to_100(metrics['revenue'])  # Higher revenue surplus = better
equity_norm = normalize_to_100(equity_ratio)  # Higher equity ratio = better

# Calculate composite index
composite_index = (
    0.50 * equity_norm +
    0.20 * gdp_norm +
    0.20 * jobs_norm +
    0.10 * revenue_norm
)

print("  GDP component range: {:.1f} to {:.1f}".format(gdp_norm.min(), gdp_norm.max()))
print("  Jobs component range: {:.1f} to {:.1f}".format(jobs_norm.min(), jobs_norm.max()))
print("  Revenue component range: {:.1f} to {:.1f}".format(revenue_norm.min(), revenue_norm.max()))
print("  Equity component range: {:.1f} to {:.1f}".format(equity_norm.min(), equity_norm.max()))
print("  Composite index range: {:.1f} to {:.1f}".format(composite_index.min(), composite_index.max()))

# Create comprehensive visualization with 2x3 grid
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Defense Spending Analysis: Apple-to-Apples Comparison', 
             fontsize=16, fontweight='bold', y=0.995)

# Define color scheme
colors = {
    'gdp': '#2E86AB',      # Blue for GDP
    'jobs': '#F18F01',     # Orange for Jobs
    'revenue': '#06A77D',  # Green for Revenue
    'efficiency': '#A23B72', # Purple for efficiency metrics
    'equity': '#C73E1D',   # Red for equity
    'baseline': '#666666'  # Gray for reference lines
}

# Row 1, Col 1: Composite Economic Index
ax1 = axes[0, 0]
ax1.plot(metrics['spending'], composite_index, 'o-', linewidth=2.5, markersize=8,
         color='#8B4789', label='Composite Index')
if baseline_exists:
    baseline_index = composite_index[baseline_idx]
    ax1.axhline(y=baseline_index, color=colors['baseline'], linestyle='--', linewidth=1.5,
                alpha=0.7, label=f'Baseline: {baseline_index:.1f}')
    ax1.axvline(x=0, color=colors['baseline'], linestyle='--', linewidth=1.5,
                alpha=0.7)
ax1.set_title('Composite Economic Index\n(50% Equity, 20% GDP, 20% Jobs, 10% Revenue)',
              fontsize=12, fontweight='bold')
ax1.set_xlabel('Defense Spending Change ($B)', fontsize=10)
ax1.set_ylabel('Index Score (0-100)', fontsize=10)
ax1.set_ylim(0, 105)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)
# Add annotation for max value
max_index_idx = np.argmax(composite_index)
ax1.annotate(f'{composite_index[max_index_idx]:.1f}',
             xy=(metrics['spending'][max_index_idx], composite_index[max_index_idx]),
             xytext=(10, -15), textcoords='offset points',
             fontsize=9, fontweight='bold', color='#8B4789',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#8B4789'))

# Row 1, Col 2: Jobs Impact (Absolute values)
ax2 = axes[0, 1]
jobs_thousands = [j / 1000 for j in jobs_absolute]
ax2.plot(metrics['spending'], jobs_thousands, 'o-', linewidth=2.5, markersize=8,
         color=colors['jobs'], label='Job Impact')
if baseline_exists:
    baseline_jobs = jobs_absolute[baseline_idx] / 1000
    ax2.axhline(y=baseline_jobs, color=colors['baseline'], linestyle='--', linewidth=1.5,
                alpha=0.7, label=f'Baseline: {baseline_jobs:,.0f}K jobs')
    ax2.axvline(x=0, color=colors['baseline'], linestyle='--', linewidth=1.5,
                alpha=0.7)
ax2.set_title('Employment Impact (FTE Jobs)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Defense Spending Change ($B)', fontsize=10)
ax2.set_ylabel('Jobs (thousands)', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:,.0f}'))

# Row 1, Col 3: Dynamic Revenue (billions)
ax3 = axes[0, 2]
ax3.plot(metrics['spending'], revenue_abs, 'o-', linewidth=2.5, markersize=8,
         color=colors['revenue'], label='Dynamic Revenue')
ax3.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5, 
            label='Revenue Neutral')
if baseline_exists:
    ax3.axvline(x=0, color=colors['baseline'], linestyle='--', linewidth=1.5, 
                alpha=0.7, label='Baseline ($0B)')
# Shade surplus/deficit areas
ax3.fill_between(metrics['spending'], 0, revenue_abs,
                  where=[r >= 0 for r in revenue_abs],
                  alpha=0.2, color='green', label='Surplus')
ax3.fill_between(metrics['spending'], 0, revenue_abs,
                  where=[r < 0 for r in revenue_abs],
                  alpha=0.2, color='red', label='Deficit')
ax3.set_title('10-Year Revenue Impact', fontsize=12, fontweight='bold')
ax3.set_xlabel('Defense Spending Change ($B)', fontsize=10)
ax3.set_ylabel('Dynamic Revenue ($B)', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=8, loc='best')
ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}B'))

# Row 2, Col 1: Marginal GDP Efficiency
ax4 = axes[1, 0]
ax4.plot(marginal_spending_points, marginal_gdp, 'o-', linewidth=2.5, markersize=8,
         color=colors['efficiency'], label='Marginal Efficiency')
ax4.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5,
            label='Zero Marginal Return')
ax4.set_title('Marginal GDP Efficiency', fontsize=12, fontweight='bold')
ax4.set_xlabel('Defense Spending Change ($B)', fontsize=10)
ax4.set_ylabel('ΔGDP per $1B Increase', fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=9)

# Row 2, Col 2: Cost-Benefit Ratio
ax5 = axes[1, 1]
ax5.plot(metrics['spending'], cost_benefit, 'o-', linewidth=2.5, markersize=8,
         color=colors['efficiency'], label='GDP per $1B Net Cost')
if baseline_exists:
    ax5.axvline(x=0, color=colors['baseline'], linestyle='--', linewidth=1.5,
                alpha=0.7, label='Baseline ($0B)')
ax5.set_title('Cost-Benefit Efficiency', fontsize=12, fontweight='bold')
ax5.set_xlabel('Defense Spending Change ($B)', fontsize=10)
ax5.set_ylabel('GDP per $1B Net Fiscal Cost', fontsize=10)
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=9)
# Set reasonable y-axis limits to avoid extreme values
max_reasonable = np.percentile([cb for cb in cost_benefit if cb > 0], 95)
ax5.set_ylim(bottom=0, top=max_reasonable * 1.2)

# Row 2, Col 3: Equity Ratio (P20 / P99)
ax6 = axes[1, 2]
ax6.plot(metrics['spending'], equity_ratio, 'o-', linewidth=2.5, markersize=8,
         color=colors['equity'], label='Equity Ratio')
ax6.axhline(y=1, color=colors['baseline'], linestyle='-', linewidth=1, alpha=0.5,
            label='Equal benefit (P20 = P99)')
if baseline_exists:
    ax6.axvline(x=0, color=colors['baseline'], linestyle='--', linewidth=1.5,
                alpha=0.7, label='Baseline ($0B)')
ax6.set_title('Income Equity Ratio (P20 / P99)', fontsize=12, fontweight='bold')
ax6.set_xlabel('Defense Spending Change ($B)', fontsize=10)
ax6.set_ylabel('P20 to P99 Ratio', fontsize=10)
ax6.grid(True, alpha=0.3)
ax6.legend(fontsize=9)
# Add annotation
ax6.text(0.02, 0.98, 'Higher = More Equitable\n(Bottom 20% benefits more)', 
         transform=ax6.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
output_file = output_dir / 'defense_spending_analysis.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n[OK] Visualization saved to '{output_file}'")
plt.close()

# Additional analysis: Print key insights
print("\n" + "="*80)
print("KEY INSIGHTS: APPLE-TO-APPLES COMPARISON")
print("="*80)

# Find spending level with maximum composite index
max_index_idx = np.argmax(composite_index)
print(f"\n1. Highest Composite Economic Index:")
print(f"   Defense Spending: ${metrics['spending'][max_index_idx]:,}B")
print(f"   Composite Index: {composite_index[max_index_idx]:.1f}/100")
print(f"   Components:")
print(f"     - Equity (50%): {equity_norm[max_index_idx]:.1f} -> {0.50 * equity_norm[max_index_idx]:.1f} points")
print(f"     - GDP (20%): {gdp_norm[max_index_idx]:.1f} -> {0.20 * gdp_norm[max_index_idx]:.1f} points")
print(f"     - Jobs (20%): {jobs_norm[max_index_idx]:.1f} -> {0.20 * jobs_norm[max_index_idx]:.1f} points")
print(f"     - Revenue (10%): {revenue_norm[max_index_idx]:.1f} -> {0.10 * revenue_norm[max_index_idx]:.1f} points")
print(f"   Actual Values:")
print(f"     - Equity Ratio: {equity_ratio[max_index_idx]:.2f}x")
print(f"     - GDP: {metrics['gdp'][max_index_idx]:+.4f}%")
print(f"     - Jobs: {metrics['jobs'][max_index_idx]:,.0f}")
print(f"     - Revenue: ${metrics['revenue'][max_index_idx]:,.2f}B")

# Find spending level with maximum GDP
max_gdp_idx = metrics['gdp'].index(max(metrics['gdp']))
baseline_gdp = metrics['gdp'][baseline_idx]
print(f"\n2. Maximum GDP Impact:")
print(f"   Defense Spending: ${metrics['spending'][max_gdp_idx]:,}B")
print(f"   GDP Change: {metrics['gdp'][max_gdp_idx]:+.4f}% ({gdp_pp_diff[max_gdp_idx]:+.4f} pp from baseline)")
print(f"   Jobs: {metrics['jobs'][max_gdp_idx]:,.0f} ({jobs_diff[max_gdp_idx]:+,.0f} from baseline)")
print(f"   Revenue Impact: ${metrics['revenue'][max_gdp_idx]:,.2f}B")
print(f"   Composite Index: {composite_index[max_gdp_idx]:.1f}/100")

# Find spending level closest to revenue neutrality
revenue_abs_values = [abs(r) for r in metrics['revenue']]
best_revenue_idx = revenue_abs_values.index(min(revenue_abs_values))
print(f"\n3. Best Revenue Neutrality:")
print(f"   Defense Spending: ${metrics['spending'][best_revenue_idx]:,}B")
print(f"   Revenue Impact: ${metrics['revenue'][best_revenue_idx]:,.2f}B")
print(f"   GDP Change: {metrics['gdp'][best_revenue_idx]:+.4f}%")
print(f"   Jobs: {metrics['jobs'][best_revenue_idx]:,.0f}")
print(f"   Composite Index: {composite_index[best_revenue_idx]:.1f}/100")

# Find most equitable distribution
max_equity_idx = equity_ratio.index(max(equity_ratio))
print(f"\n4. Most Equitable Distribution:")
print(f"   Defense Spending: ${metrics['spending'][max_equity_idx]:,}B")
print(f"   Equity Ratio (P20/P99): {equity_ratio[max_equity_idx]:.2f}x")
print(f"   P20 Benefit: {metrics['p20'][max_equity_idx]:+.4f}")
print(f"   P99 Benefit: {metrics['p99'][max_equity_idx]:+.4f}")
print(f"   Composite Index: {composite_index[max_equity_idx]:.1f}/100")

# Find best marginal efficiency (excluding extreme values)
valid_marginal = [(i, val) for i, val in enumerate(marginal_gdp) if val > 0 and val < 1e6]
if valid_marginal:
    best_marginal_idx, best_marginal_val = max(valid_marginal, key=lambda x: x[1])
    marginal_spending = marginal_spending_points[best_marginal_idx]
    print(f"\n4. Best Marginal Efficiency:")
    print(f"   Defense Spending Range: ${marginal_spending - 250:,.0f}B to ${marginal_spending + 250:,.0f}B")
    print(f"   Marginal GDP: {best_marginal_val:.6f} per $1B increase")

# Baseline comparison (if exists)
if baseline_exists:
    print(f"\n5. Baseline Metrics (${metrics['spending'][baseline_idx]:,}B):")
    print(f"   Composite Index: {composite_index[baseline_idx]:.1f}/100")
    print(f"   GDP: {metrics['gdp'][baseline_idx]:+.4f}%")
    print(f"   Jobs: {metrics['jobs'][baseline_idx]:,.0f}")
    print(f"   Revenue: ${metrics['revenue'][baseline_idx]:,.2f}B")
    print(f"   Equity Ratio: {equity_ratio[baseline_idx]:.2f}x")

# Range analysis
print(f"\n6. Overall Ranges:")
gdp_range = max(metrics['gdp']) - min(metrics['gdp'])
jobs_range = max(metrics['jobs']) - min(metrics['jobs'])
revenue_range = max(metrics['revenue']) - min(metrics['revenue'])
print(f"   GDP Range: {gdp_range:.4f}% ({min(metrics['gdp']):+.4f}% to {max(metrics['gdp']):+.4f}%)")
print(f"   Jobs Range: {jobs_range:,.0f} ({min(metrics['jobs']):,.0f} to {max(metrics['jobs']):,.0f})")
print(f"   Revenue Range: ${revenue_range:,.2f}B (${min(metrics['revenue']):,.2f}B to ${max(metrics['revenue']):,.2f}B)")

print("\n" + "="*80)
print("COMPOSITE INDEX METHODOLOGY:")
print("  • Score 0-100 combining four normalized components:")
print("    - Equity (50%): P20/P99 ratio normalized to 0-100")
print("    - GDP (20%): Long-run GDP % change normalized to 0-100")
print("    - Jobs (20%): Full-time equivalent jobs normalized to 0-100")
print("    - Revenue (10%): Dynamic 10-year revenue normalized to 0-100")
print("  • Higher scores indicate better overall economic + equity outcomes")
print("  • Heavy equity weighting prioritizes distributive fairness")
print("\nOTHER METRICS:")
print("  • GDP: Long-run % change in GDP (e.g., 0.14% = 0.14 percentage points)")
print("  • Jobs: Actual full-time equivalent jobs created")
print("  • Revenue: Dynamic 10-year revenue in billions of dollars")
print("  • Equity ratio: P20/P99 (higher = more equitable)")
print("  • 'pp' = percentage points (e.g., from 0.13% to 0.14% = +0.01 pp)")
print("="*80)
print(f"\nVisualization complete! Check '{output_dir / 'defense_spending_analysis.png'}'")