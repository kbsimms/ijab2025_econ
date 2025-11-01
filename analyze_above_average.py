"""
Objective Defense Spending Analysis: Above-Average Metric Count

This script calculates for each defense spending level how many economic
metrics perform above the average across all spending levels.

Metrics analyzed:
1. GDP Growth
2. Jobs Created
3. Revenue Surplus
4. Capital Stock
5. Wage Rate
6. P20 (Bottom 20% benefit)
7. P40-60 (Middle class benefit)
8. P80-100 (Upper middle benefit)
9. P99 (Top 1% benefit)

The spending level that performs above average on the most metrics
is considered the most balanced/optimal choice.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load the economic effects summary
print("Loading economic effects data...")
df = pd.read_csv('outputs/defense/economic_effects_summary.csv')

# Define economic metrics to analyze
metrics = {
    'GDP': 'GDP',
    'Jobs': 'Jobs',
    'Revenue': 'Revenue',
    'Capital': 'Capital',
    'Wage': 'Wage',
    'P20': 'P20',
    'P40-60': 'P40-60',
    'P80-100': 'P80-100',
    'P99': 'P99'
}

print("\nCalculating averages for each metric...")
# Calculate average for each metric
averages = {}
for metric_name, column_name in metrics.items():
    avg = df[column_name].mean()
    averages[metric_name] = avg
    print(f"  {metric_name}: {avg:.6f}")

print("\nCounting above-average metrics for each spending level...")
# For each spending level, count how many metrics are above average
results = []

for idx, row in df.iterrows():
    spending = row['Defense_Spending_B']
    above_avg_count = 0
    above_avg_metrics = []
    
    for metric_name, column_name in metrics.items():
        value = row[column_name]
        avg = averages[metric_name]
        if value > avg:
            above_avg_count += 1
            above_avg_metrics.append(metric_name)
    
    results.append({
        'Spending': spending,
        'Above_Average_Count': above_avg_count,
        'Above_Average_Metrics': ', '.join(above_avg_metrics),
        'GDP': row['GDP'],
        'Jobs': row['Jobs'],
        'Revenue': row['Revenue'],
        'Capital': row['Capital'],
        'Wage': row['Wage'],
        'P20': row['P20'],
        'P40-60': row['P40-60'],
        'P80-100': row['P80-100'],
        'P99': row['P99']
    })

# Create results DataFrame and sort by above-average count
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Above_Average_Count', ascending=False)

# Display results
print("\n" + "=" * 100)
print("RANKING: Defense Spending Levels by Above-Average Metric Count")
print("=" * 100)
print("\nTop 10 Most Balanced Options:")
print(results_df[['Spending', 'Above_Average_Count', 'Above_Average_Metrics']].head(10).to_string(index=False))

print("\n" + "=" * 100)
print("DETAILED ANALYSIS")
print("=" * 100)

# Find the top performers
top_spending = results_df.iloc[0]['Spending']
top_count = results_df.iloc[0]['Above_Average_Count']
print(f"\n🏆 MOST BALANCED OPTION:")
print(f"   Defense Spending: ${top_spending:,}B")
print(f"   Above-Average Metrics: {top_count} out of {len(metrics)}")
print(f"   Which metrics: {results_df.iloc[0]['Above_Average_Metrics']}")

# Show top 3
print(f"\n📊 TOP 3 BALANCED OPTIONS:")
for i in range(min(3, len(results_df))):
    row = results_df.iloc[i]
    print(f"\n   {i+1}. ${row['Spending']:,}B - {row['Above_Average_Count']}/{len(metrics)} metrics above average")
    print(f"      GDP: {row['GDP']*100:.2f}%, Jobs: {row['Jobs']/1e6:.2f}M, Revenue: ${row['Revenue']:.1f}B")
    print(f"      Above avg: {row['Above_Average_Metrics']}")

# Save detailed results to CSV
output_file = 'outputs/defense/above_average_analysis.csv'
results_df.to_csv(output_file, index=False)
print(f"\n💾 Detailed results saved to: {output_file}")

# Create visualization
print("\n📈 Creating visualizations...")

# Figure 1: Bar chart of above-average counts
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Defense Spending Analysis: Above-Average Metric Performance', 
             fontsize=16, fontweight='bold')

# Plot 1: Above-average count by spending level
ax1 = axes[0, 0]
spending_sorted = results_df['Spending'].values
counts_sorted = results_df['Above_Average_Count'].values
colors = plt.cm.RdYlGn(counts_sorted / len(metrics))  # Color by performance

bars = ax1.bar(range(len(spending_sorted)), counts_sorted, color=colors, edgecolor='black', linewidth=0.5)
ax1.set_xticks(range(len(spending_sorted)))
ax1.set_xticklabels([f'${s/1000:.1f}T' for s in spending_sorted], rotation=45, ha='right', fontsize=8)
ax1.set_xlabel('Defense Spending Level', fontsize=11)
ax1.set_ylabel('Number of Above-Average Metrics', fontsize=11)
ax1.set_title('Metric Count Ranking (Higher = More Balanced)', fontsize=12, fontweight='bold')
ax1.axhline(y=len(metrics)/2, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Half of metrics')
ax1.set_ylim(0, len(metrics) + 0.5)
ax1.grid(True, alpha=0.3, axis='y')
ax1.legend()

# Highlight top 3
for i in range(min(3, len(bars))):
    bars[i].set_linewidth(3)
    bars[i].set_edgecolor('gold')

# Plot 2: Defense Spending vs Above-Average Count (scatter)
ax2 = axes[0, 1]
scatter = ax2.scatter(df['Defense_Spending_B'], results_df.set_index('Spending').loc[df['Defense_Spending_B'], 'Above_Average_Count'],
                     c=results_df.set_index('Spending').loc[df['Defense_Spending_B'], 'Above_Average_Count'],
                     s=200, cmap='RdYlGn', edgecolor='black', linewidth=1.5, vmin=0, vmax=len(metrics))
ax2.axhline(y=len(metrics)/2, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Half of metrics')
ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='Baseline ($0B)')
ax2.set_xlabel('Defense Spending Change ($B)', fontsize=11)
ax2.set_ylabel('Above-Average Metric Count', fontsize=11)
ax2.set_title('Performance vs. Spending Level', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
plt.colorbar(scatter, ax=ax2, label='Metric Count')

# Plot 3: Heatmap of which metrics are above average
ax3 = axes[1, 0]
# Create matrix: rows = spending levels, columns = metrics
spending_levels = df['Defense_Spending_B'].values
metric_names = list(metrics.keys())
heatmap_data = np.zeros((len(spending_levels), len(metric_names)))

for i, spending in enumerate(spending_levels):
    row_data = df[df['Defense_Spending_B'] == spending].iloc[0]
    for j, (metric_name, column_name) in enumerate(metrics.items()):
        value = row_data[column_name]
        avg = averages[metric_name]
        heatmap_data[i, j] = 1 if value > avg else 0

im = ax3.imshow(heatmap_data.T, aspect='auto', cmap='RdYlGn', interpolation='nearest', vmin=0, vmax=1)
ax3.set_xticks(range(len(spending_levels)))
ax3.set_xticklabels([f'${s/1000:.1f}T' for s in spending_levels], rotation=45, ha='right', fontsize=8)
ax3.set_yticks(range(len(metric_names)))
ax3.set_yticklabels(metric_names, fontsize=10)
ax3.set_xlabel('Defense Spending Level', fontsize=11)
ax3.set_ylabel('Economic Metric', fontsize=11)
ax3.set_title('Above-Average Performance Map\n(Green = Above Avg, Red = Below Avg)', 
              fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax3, ticks=[0, 1], label='Above Avg')

# Plot 4: Top 5 options comparison
ax4 = axes[1, 1]
top5 = results_df.head(5)
x_pos = np.arange(len(top5))
ax4.bar(x_pos, top5['Above_Average_Count'], color='steelblue', edgecolor='black', linewidth=1.5)
ax4.set_xticks(x_pos)
ax4.set_xticklabels([f'${s/1000:.1f}T\n({c} metrics)' for s, c in zip(top5['Spending'], top5['Above_Average_Count'])],
                     fontsize=10, fontweight='bold')
ax4.set_ylabel('Above-Average Metric Count', fontsize=11)
ax4.set_title('Top 5 Most Balanced Options', fontsize=12, fontweight='bold')
ax4.axhline(y=len(metrics)/2, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Half of metrics')
ax4.set_ylim(0, len(metrics) + 0.5)
ax4.grid(True, alpha=0.3, axis='y')
ax4.legend()

# Add value labels on bars
for i, (idx, row) in enumerate(top5.iterrows()):
    ax4.text(i, row['Above_Average_Count'] + 0.2, f"{row['Above_Average_Count']}", 
             ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
output_plot = 'outputs/defense/above_average_analysis.png'
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"✅ Visualization saved to: {output_plot}")

print("\n" + "=" * 100)
print("INTERPRETATION GUIDE")
print("=" * 100)
print("""
This analysis ranks defense spending levels by how many economic metrics
perform above the average across all scenarios.

✅ Higher count = More balanced option (performs well across many dimensions)
❌ Lower count = Less balanced option (excels in some areas, weak in others)

Key Insights:
1. The top-ranked option is the most "balanced" economically
2. It doesn't mean it's optimal for security - that requires additional analysis
3. Options with 5+ above-average metrics are generally well-rounded
4. Options with <3 above-average metrics are economically weak overall

Next Step: Compare the economic ranking with security/strategic priorities
to find the true optimal balance between economy and defense.
""")