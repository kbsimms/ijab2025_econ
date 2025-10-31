"""
Defense Spending Optimization Visualization

Analyzes and visualizes the economic effects of different defense spending requirements
across the optimization results from max_gdp_defense.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.ticker import FuncFormatter

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

# Create comprehensive visualization
fig = plt.figure(figsize=(18, 12))

# Define color scheme
colors = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'tertiary': '#F18F01',
    'quaternary': '#C73E1D',
    'success': '#06A77D',
    'warning': '#F77F00'
}

# 1. GDP Impact
ax1 = plt.subplot(3, 3, 1)
ax1.plot(metrics['spending'], metrics['gdp'], 'o-', linewidth=2.5, 
         markersize=8, color=colors['primary'], label='GDP Change')
ax1.set_title('GDP Impact vs Defense Spending', fontsize=12, fontweight='bold')
ax1.set_xlabel('Defense Spending Requirement ($B)', fontsize=10)
ax1.set_ylabel('Total GDP Change', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2. Marginal Efficiency (ΔGDP / ΔSpending)
ax2 = plt.subplot(3, 3, 2)
marginal_efficiency = []
for i in range(1, len(metrics['spending'])):
    delta_gdp = metrics['gdp'][i] - metrics['gdp'][i-1]
    delta_spending = metrics['spending'][i] - metrics['spending'][i-1]
    if delta_spending != 0:
        marginal_efficiency.append(delta_gdp / delta_spending * 1000)  # per $1B
    else:
        marginal_efficiency.append(0)

# Plot with x-axis as midpoints between spending levels
midpoints = [(metrics['spending'][i] + metrics['spending'][i-1])/2 for i in range(1, len(metrics['spending']))]
ax2.plot(midpoints, marginal_efficiency, 'o-', linewidth=2.5,
         markersize=8, color=colors['secondary'], label='Marginal Efficiency')
ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Zero Marginal Return')
ax2.set_title('Marginal Efficiency\n(ΔGDP / Δ$1B Defense Spending)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Defense Spending Requirement ($B)', fontsize=10)
ax2.set_ylabel('GDP Change per $1B Increase', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)

# 3. Jobs Impact
ax3 = plt.subplot(3, 3, 3)
ax3.plot(metrics['spending'], [j/1e6 for j in metrics['jobs']], 'o-', 
         linewidth=2.5, markersize=8, color=colors['tertiary'])
ax3.set_title('Jobs Impact vs Defense Spending', fontsize=12, fontweight='bold')
ax3.set_xlabel('Defense Spending Requirement ($B)', fontsize=10)
ax3.set_ylabel('Total Jobs Created (Millions)', fontsize=10)
ax3.grid(True, alpha=0.3)

# 4. Capital Stock
ax4 = plt.subplot(3, 3, 4)
ax4.plot(metrics['spending'], metrics['capital'], 'o-', linewidth=2.5,
         markersize=8, color=colors['quaternary'])
ax4.set_title('Capital Stock Impact', fontsize=12, fontweight='bold')
ax4.set_xlabel('Defense Spending Requirement ($B)', fontsize=10)
ax4.set_ylabel('Total Capital Stock Change', fontsize=10)
ax4.grid(True, alpha=0.3)

# 5. Wage Rate
ax5 = plt.subplot(3, 3, 5)
ax5.plot(metrics['spending'], metrics['wage'], 'o-', linewidth=2.5,
         markersize=8, color=colors['secondary'])
ax5.set_title('Wage Rate Impact', fontsize=12, fontweight='bold')
ax5.set_xlabel('Defense Spending Requirement ($B)', fontsize=10)
ax5.set_ylabel('Total Wage Rate Change', fontsize=10)
ax5.grid(True, alpha=0.3)

# 6. Income Distribution Effects
ax6 = plt.subplot(3, 3, 6)
ax6.plot(metrics['spending'], metrics['p20'], 'o-', linewidth=2, markersize=7, label='P20 (Bottom 20%)')
ax6.plot(metrics['spending'], metrics['p40_60'], 's-', linewidth=2, markersize=7, label='P40-60 (Middle Class)')
ax6.plot(metrics['spending'], metrics['p80_100'], '^-', linewidth=2, markersize=7, label='P80-100 (Top 20%)')
ax6.plot(metrics['spending'], metrics['p99'], 'd-', linewidth=2, markersize=7, label='P99 (Top 1%)')
ax6.set_title('Income Distribution Effects', fontsize=12, fontweight='bold')
ax6.set_xlabel('Defense Spending Requirement ($B)', fontsize=10)
ax6.set_ylabel('Income Change by Percentile', fontsize=10)
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

# 7. Total Factor Productivity (TFP)
ax7 = plt.subplot(3, 3, 7)
# TFP = GDP Change / (Capital + Labor + Net Fiscal Cost)
# Normalize labor by dividing jobs by 1M to get comparable scale
# Net Fiscal Cost = -Static Revenue (negative revenue is cost)
tfp = []
for i in range(len(metrics['spending'])):
    capital = abs(metrics['capital'][i]) if metrics['capital'][i] != 0 else 0.01
    labor = abs(metrics['jobs'][i] / 1e6) if metrics['jobs'][i] != 0 else 0.01  # Normalize to millions
    fiscal_cost = abs(metrics['static_revenue'][i]) if metrics['static_revenue'][i] != 0 else 0.01
    
    total_inputs = capital + labor + fiscal_cost
    if total_inputs > 0:
        tfp.append(metrics['gdp'][i] / total_inputs)
    else:
        tfp.append(0)

ax7.plot(metrics['spending'], tfp, 'o-', linewidth=2.5,
         markersize=8, color=colors['quaternary'])
ax7.set_title('Total Factor Productivity\n(GDP / All Input Factors)', fontsize=12, fontweight='bold')
ax7.set_xlabel('Defense Spending Requirement ($B)', fontsize=10)
ax7.set_ylabel('TFP Index', fontsize=10)
ax7.grid(True, alpha=0.3)

# 8. 10-Year Dynamic Revenue vs Defense Spending
ax8 = plt.subplot(3, 3, 8)
ax8.plot(metrics['spending'], metrics['revenue'], 'o-', linewidth=2.5,
         markersize=8, color=colors['success'], label='Dynamic Revenue')
ax8.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Revenue Neutral')
# Shade positive revenue area
ax8.fill_between(metrics['spending'], 0, metrics['revenue'],
                  where=[r >= 0 for r in metrics['revenue']],
                  alpha=0.2, color='green', label='Surplus')
# Shade negative revenue area
ax8.fill_between(metrics['spending'], 0, metrics['revenue'],
                  where=[r < 0 for r in metrics['revenue']],
                  alpha=0.2, color='red', label='Deficit')
ax8.set_title('10-Year Dynamic Revenue vs Defense Spending', fontsize=12, fontweight='bold')
ax8.set_xlabel('Defense Spending Requirement ($B)', fontsize=10)
ax8.set_ylabel('Dynamic Revenue ($B)', fontsize=10)
ax8.legend()
ax8.grid(True, alpha=0.3)

# 9. Jobs per Net Dollar Spent
ax9 = plt.subplot(3, 3, 9)
# Net Fiscal Cost = -Dynamic Revenue (negative revenue = cost, positive = surplus)
# Jobs per $1B of net cost
jobs_per_dollar = []
for i in range(len(metrics['spending'])):
    net_cost = -metrics['revenue'][i]  # Negative revenue becomes positive cost
    if net_cost > 0:  # Only meaningful when there's actual cost
        jobs_per_billion = metrics['jobs'][i] / net_cost
        jobs_per_dollar.append(jobs_per_billion)
    elif net_cost < 0:  # Revenue surplus - infinite efficiency
        jobs_per_dollar.append(metrics['jobs'][i] / 0.001)  # Very high value for surplus scenarios
    else:
        jobs_per_dollar.append(0)

ax9.plot(metrics['spending'], jobs_per_dollar, 'o-', linewidth=2.5,
         markersize=8, color=colors['warning'])
ax9.set_title('Jobs per Net Dollar Spent\n(Jobs Created / $1B Net Fiscal Cost)', fontsize=12, fontweight='bold')
ax9.set_xlabel('Defense Spending Requirement ($B)', fontsize=10)
ax9.set_ylabel('Jobs per $1B Net Cost', fontsize=10)
ax9.grid(True, alpha=0.3)
# Format y-axis to show thousands separator
ax9.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))

plt.tight_layout()
output_file = output_dir / 'defense_spending_analysis.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n[OK] Visualization saved to '{output_file}'")

# Additional analysis: Print key insights
print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)

# Find optimal spending level for GDP
max_gdp_idx = metrics['gdp'].index(max(metrics['gdp']))
print(f"\n1. Maximum GDP achieved at ${metrics['spending'][max_gdp_idx]:,}B defense spending")
print(f"   - GDP Change: {metrics['gdp'][max_gdp_idx]:.4f}")
print(f"   - Jobs Created: {metrics['jobs'][max_gdp_idx]:,.0f}")
print(f"   - Revenue Change: ${metrics['revenue'][max_gdp_idx]:,.2f}B")

# Find spending level with best revenue neutrality
revenue_abs = [abs(r) for r in metrics['revenue']]
best_revenue_idx = revenue_abs.index(min(revenue_abs))
print(f"\n2. Closest to revenue neutrality at ${metrics['spending'][best_revenue_idx]:,}B")
print(f"   - Revenue Change: ${metrics['revenue'][best_revenue_idx]:,.2f}B")
print(f"   - GDP Change: {metrics['gdp'][best_revenue_idx]:.4f}")

# Analyze trade-offs
print(f"\n3. Trade-off Analysis:")
gdp_range = max(metrics['gdp']) - min(metrics['gdp'])
spending_range = max(metrics['spending']) - min(metrics['spending'])
print(f"   - GDP Range: {gdp_range:.4f} ({min(metrics['gdp']):.4f} to {max(metrics['gdp']):.4f})")
print(f"   - Spending Range: ${spending_range:,}B")
print(f"   - GDP Sensitivity: {gdp_range/spending_range*1000:.6f} per $1B increase")

# Equity analysis
print(f"\n4. Income Distribution Equity:")
for i, level in enumerate(metrics['spending']):
    p20 = metrics['p20'][i]
    p99 = metrics['p99'][i]
    print(f"   ${level:,}B: P20={p20:+.4f}, P99={p99:+.4f}, Ratio={p20/p99:.2f}x" if p99 != 0 else f"   ${level:,}B: P20={p20:+.4f}, P99={p99:+.4f}")

print("\n" + "="*70)
print(f"\nVisualization complete! Check '{output_dir / 'defense_spending_analysis.png'}'")