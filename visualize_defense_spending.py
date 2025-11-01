"""
Defense Spending Optimization Visualization

WHAT THIS SCRIPT DOES:
Creates comprehensive charts showing how economic outcomes change as defense spending
requirements vary from -$4,000B to +$6,000B. Think of it as answering: "What are
the economic trade-offs between different levels of defense spending?"

The visualization helps you see at a glance:
- How GDP growth changes with defense spending levels
- Effects on job creation across the spending range
- Impact on government revenue (surplus or deficit)
- How much capital investment changes
- Changes in wage rates
- Whether benefits are distributed fairly (comparing bottom 20% vs top 1%)

HOW IT WORKS:
1. Loads all CSV files from previous max_gdp_defense.py runs
2. Calculates total economic impacts for each defense spending level
3. Creates 6 charts showing different economic metrics
4. Identifies optimal spending levels for various objectives

PREREQUISITES:
Must first run: python max_gdp_defense.py
This generates the CSV files this script analyzes.

USAGE:
    python visualize_defense_spending.py

OUTPUT:
Creates 'outputs/defense/defense_spending_analysis.png' with 6 panels:
- GDP Growth Impact across spending levels
- Employment Impact (jobs created/lost)
- Revenue Impact (fiscal surplus/deficit)
- Capital Stock changes
- Wage Rate changes
- Equity Impact (P20 vs P99 distribution)

Plus detailed console output identifying:
- Spending level with best overall economic index
- Maximum GDP point
- Best revenue neutrality point
- Most equitable distribution point
- Overall ranges for all metrics
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import COLUMNS, SPENDING_RANGE
from logger import LogLevel, get_logger

# Initialize logger
logger = get_logger(__name__, level=LogLevel.INFO)

# Constants
MARGINAL_EFFICIENCY_MAX = 1_000_000.0

# Ensure output directory exists
output_dir = Path('outputs/defense')
try:
    output_dir.mkdir(parents=True, exist_ok=True)
except Exception:
    logger.exception("Cannot create output directory")
    raise

# Load all optimization results from config range
spending_levels = list(range(
    SPENDING_RANGE["min"],
    SPENDING_RANGE["max"],
    SPENDING_RANGE["step"]
))
results = {}

logger.info(f"Loading optimization results for {len(spending_levels)} spending levels...")
for level in spending_levels:
    file_path = output_dir / f'max_gdp_defense{level}.csv'
    if file_path.exists():
        results[level] = pd.read_csv(file_path)
        logger.info(f"  [OK] Loaded {file_path.name}: {len(results[level])} policies selected")
    else:
        logger.warning(f"  [MISSING] {file_path.name}")

# Calculate aggregate metrics for each spending level
logger.info("Calculating aggregate metrics...")
metrics: dict[str, list[Any]] = {
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
    metrics['gdp'].append(df[COLUMNS["gdp"]].sum())
    metrics['capital'].append(df[COLUMNS["capital"]].sum())
    metrics['jobs'].append(df[COLUMNS["jobs"]].sum())
    metrics['wage'].append(df[COLUMNS["wage"]].sum())
    metrics['revenue'].append(df[COLUMNS["dynamic_revenue"]].sum())
    metrics['static_revenue'].append(df[COLUMNS["static_revenue"]].sum())
    metrics['p20'].append(df[COLUMNS["p20"]].sum())
    metrics['p40_60'].append(df[COLUMNS["p40_60"]].sum())
    metrics['p80_100'].append(df[COLUMNS["p80_100"]].sum())
    metrics['p99'].append(df[COLUMNS["p99"]].sum())
    metrics['n_policies'].append(len(df))
    metrics['n_ns_policies'].append(len(df[df['is_NS']]))

    # Calculate actual NS spending (negative revenue from NS policies)
    ns_revenue = df[df['is_NS']][COLUMNS["dynamic_revenue"]].sum()
    metrics['ns_spending'].append(-ns_revenue)  # Convert to positive spending

# Convert to DataFrame for easier analysis
metrics_df = pd.DataFrame(metrics)

logger.info("Aggregate Metrics Summary:")
logger.info(f"\n{metrics_df.to_string(index=False)}")

# Find baseline index (spending = 0)
try:
    baseline_idx = metrics['spending'].index(0)
    baseline_exists = True
except ValueError:
    # If 0 is not in the list, use the closest value
    baseline_idx = min(range(len(metrics['spending'])),
                      key=lambda i: abs(metrics['spending'][i]))
    baseline_exists = False
    logger.warning(f"$0B baseline not found. Using ${metrics['spending'][baseline_idx]:,}B as reference.")

# Helper function to normalize values to 0-100 scale
def normalize_to_100(values: list[Any], reverse: bool = False) -> np.ndarray[Any, Any]:
    """Normalize values to 0-100 scale. If reverse=True, lower values get higher scores."""
    arr = np.array(values)
    min_val: float = float(arr.min())
    max_val: float = float(arr.max())
    if max_val == min_val:
        return np.array([50.0] * len(values))

    if reverse:
        normalized = 100 * (max_val - arr) / (max_val - min_val)
    else:
        normalized = 100 * (arr - min_val) / (max_val - min_val)
    return normalized

# Helper function to calculate percentage point difference from baseline
def pp_diff_from_baseline(values: list[Any], baseline_idx: int) -> list[Any]:
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
for p20, p99 in zip(metrics['p20'], metrics['p99'], strict=False):
    if p99 != 0 and p99 > 0:
        equity_ratio.append(p20 / p99)
    else:
        equity_ratio.append(0)

# Calculate cost-benefit ratio (GDP gain per dollar of net fiscal cost)
cost_benefit = []
for gdp, revenue in zip(metrics['gdp'], metrics['revenue'], strict=False):
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
logger.info("Calculating Composite Economic Index...")
logger.info("  Weights: 50% Equity + 20% GDP + 20% Jobs + 10% Revenue")

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

logger.debug(f"  GDP component range: {gdp_norm.min():.1f} to {gdp_norm.max():.1f}")
logger.debug(f"  Jobs component range: {jobs_norm.min():.1f} to {jobs_norm.max():.1f}")
logger.debug(f"  Revenue component range: {revenue_norm.min():.1f} to {revenue_norm.max():.1f}")
logger.debug(f"  Equity component range: {equity_norm.min():.1f} to {equity_norm.max():.1f}")
logger.info(f"  Composite index range: {composite_index.min():.1f} to {composite_index.max():.1f}")

# Create comprehensive visualization with 2x3 grid focused on key metrics
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Defense Spending Optimization: Key Economic Metrics',
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

# Row 1, Col 1: GDP Growth
ax1 = axes[0, 0]
gdp_pct = [g * 100 for g in metrics['gdp']]  # Convert to percentage
ax1.plot(metrics['spending'], gdp_pct, 'o-', linewidth=2.5, markersize=8,
         color=colors['gdp'], label='GDP Growth')
ax1.axvline(x=0, color=colors['baseline'], linestyle='--', linewidth=1.5,
            alpha=0.7, label='Baseline ($0B)')
ax1.set_title('GDP Growth Impact', fontsize=12, fontweight='bold')
ax1.set_xlabel('Defense Spending Change ($B)', fontsize=11)
ax1.set_ylabel('GDP Growth (%)', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)

# Row 1, Col 2: Jobs Impact
ax2 = axes[0, 1]
jobs_millions = [j / 1_000_000 for j in jobs_absolute]
ax2.plot(metrics['spending'], jobs_millions, 'o-', linewidth=2.5, markersize=8,
         color=colors['jobs'], label='Jobs Created')
ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax2.axvline(x=0, color=colors['baseline'], linestyle='--', linewidth=1.5,
            alpha=0.7, label='Baseline ($0B)')
ax2.set_title('Employment Impact', fontsize=12, fontweight='bold')
ax2.set_xlabel('Defense Spending Change ($B)', fontsize=11)
ax2.set_ylabel('Jobs Created (Millions)', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)

# Row 1, Col 3: Revenue Impact
ax3 = axes[0, 2]
ax3.plot(metrics['spending'], revenue_abs, 'o-', linewidth=2.5, markersize=8,
         color=colors['revenue'], label='Revenue Impact')
ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7,
            label='Revenue Neutral')
ax3.axvline(x=0, color=colors['baseline'], linestyle='--', linewidth=1.5,
            alpha=0.7, label='Baseline ($0B)')
# Shade surplus/deficit areas
ax3.fill_between(metrics['spending'], 0, revenue_abs,
                  where=[r >= 0 for r in revenue_abs],
                  alpha=0.2, color='green')
ax3.fill_between(metrics['spending'], 0, revenue_abs,
                  where=[r < 0 for r in revenue_abs],
                  alpha=0.2, color='red')
ax3.set_title('10-Year Revenue Impact', fontsize=12, fontweight='bold')
ax3.set_xlabel('Defense Spending Change ($B)', fontsize=11)
ax3.set_ylabel('Revenue Surplus ($B)', fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=9, loc='best')

# Row 2, Col 1: Capital Stock
ax4 = axes[1, 0]
capital_pct = [c * 100 for c in metrics['capital']]  # Convert to percentage
ax4.plot(metrics['spending'], capital_pct, 'o-', linewidth=2.5, markersize=8,
         color='#9B59B6', label='Capital Stock')
ax4.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax4.axvline(x=0, color=colors['baseline'], linestyle='--', linewidth=1.5,
            alpha=0.7, label='Baseline ($0B)')
ax4.set_title('Capital Stock Change', fontsize=12, fontweight='bold')
ax4.set_xlabel('Defense Spending Change ($B)', fontsize=11)
ax4.set_ylabel('Capital Stock Change (%)', fontsize=11)
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=9)

# Row 2, Col 2: Wage Rate
ax5 = axes[1, 1]
wage_pct = [w * 100 for w in metrics['wage']]  # Convert to percentage
ax5.plot(metrics['spending'], wage_pct, 'o-', linewidth=2.5, markersize=8,
         color='#E67E22', label='Wage Rate')
ax5.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax5.axvline(x=0, color=colors['baseline'], linestyle='--', linewidth=1.5,
            alpha=0.7, label='Baseline ($0B)')
ax5.set_title('Wage Rate Change', fontsize=12, fontweight='bold')
ax5.set_xlabel('Defense Spending Change ($B)', fontsize=11)
ax5.set_ylabel('Wage Rate Change (%)', fontsize=11)
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=9)

# Row 2, Col 3: Equity Impact (P20 vs P99)
ax6 = axes[1, 2]
p20_pct = [p * 100 for p in metrics['p20']]
p99_pct = [p * 100 for p in metrics['p99']]
ax6.plot(metrics['spending'], p20_pct, 'o-', linewidth=2.5, markersize=8,
         color='#27AE60', label='Bottom 20% (P20)')
ax6.plot(metrics['spending'], p99_pct, 'o-', linewidth=2.5, markersize=8,
         color='#E74C3C', label='Top 1% (P99)')
ax6.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax6.axvline(x=0, color=colors['baseline'], linestyle='--', linewidth=1.5,
            alpha=0.7, label='Baseline ($0B)')
ax6.set_title('Equity: Income Distribution Impact', fontsize=12, fontweight='bold')
ax6.set_xlabel('Defense Spending Change ($B)', fontsize=11)
ax6.set_ylabel('After-Tax Income Change (%)', fontsize=11)
ax6.grid(True, alpha=0.3)
ax6.legend(fontsize=9, loc='best')
# Add shaded area where P20 > P99 (progressive)
ax6.fill_between(metrics['spending'], p20_pct, p99_pct,
                  where=[p20 >= p99 for p20, p99 in zip(p20_pct, p99_pct, strict=False)],
                  alpha=0.2, color='green', label='Progressive')

plt.tight_layout()
output_file = output_dir / 'defense_spending_analysis.png'
try:
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"[OK] Visualization saved to '{output_file}'")
except Exception:
    logger.exception("Failed to save visualization")
    raise
finally:
    plt.close()

# Additional analysis: Print key insights
logger.section("KEY INSIGHTS: APPLE-TO-APPLES COMPARISON")

# Find spending level with maximum composite index
max_index_idx = np.argmax(composite_index)
logger.info("\n1. Highest Composite Economic Index:")
logger.info(f"   Defense Spending: ${metrics['spending'][max_index_idx]:,}B")
logger.info(f"   Composite Index: {composite_index[max_index_idx]:.1f}/100")
logger.info("   Components:")
logger.info(f"     - Equity (50%): {equity_norm[max_index_idx]:.1f} -> {0.50 * equity_norm[max_index_idx]:.1f} points")
logger.info(f"     - GDP (20%): {gdp_norm[max_index_idx]:.1f} -> {0.20 * gdp_norm[max_index_idx]:.1f} points")
logger.info(f"     - Jobs (20%): {jobs_norm[max_index_idx]:.1f} -> {0.20 * jobs_norm[max_index_idx]:.1f} points")
logger.info(f"     - Revenue (10%): {revenue_norm[max_index_idx]:.1f} -> {0.10 * revenue_norm[max_index_idx]:.1f} points")
logger.info("   Actual Values:")
logger.info(f"     - Equity Ratio: {equity_ratio[max_index_idx]:.2f}x")
logger.info(f"     - GDP: {metrics['gdp'][max_index_idx]:+.4f}%")
logger.info(f"     - Jobs: {metrics['jobs'][max_index_idx]:,.0f}")
logger.info(f"     - Revenue: ${metrics['revenue'][max_index_idx]:,.2f}B")

# Find spending level with maximum GDP
max_gdp_idx = metrics['gdp'].index(max(metrics['gdp']))
baseline_gdp = metrics['gdp'][baseline_idx]
logger.info("\n2. Maximum GDP Impact:")
logger.info(f"   Defense Spending: ${metrics['spending'][max_gdp_idx]:,}B")
logger.info(f"   GDP Change: {metrics['gdp'][max_gdp_idx]:+.4f}% ({gdp_pp_diff[max_gdp_idx]:+.4f} pp from baseline)")
logger.info(f"   Jobs: {metrics['jobs'][max_gdp_idx]:,.0f} ({jobs_diff[max_gdp_idx]:+,.0f} from baseline)")
logger.info(f"   Revenue Impact: ${metrics['revenue'][max_gdp_idx]:,.2f}B")
logger.info(f"   Composite Index: {composite_index[max_gdp_idx]:.1f}/100")

# Find spending level closest to revenue neutrality
revenue_abs_values = [abs(r) for r in metrics['revenue']]
best_revenue_idx = revenue_abs_values.index(min(revenue_abs_values))
logger.info("\n3. Best Revenue Neutrality:")
logger.info(f"   Defense Spending: ${metrics['spending'][best_revenue_idx]:,}B")
logger.info(f"   Revenue Impact: ${metrics['revenue'][best_revenue_idx]:,.2f}B")
logger.info(f"   GDP Change: {metrics['gdp'][best_revenue_idx]:+.4f}%")
logger.info(f"   Jobs: {metrics['jobs'][best_revenue_idx]:,.0f}")
logger.info(f"   Composite Index: {composite_index[best_revenue_idx]:.1f}/100")

# Find most equitable distribution
max_equity_idx = equity_ratio.index(max(equity_ratio))
logger.info("\n4. Most Equitable Distribution:")
logger.info(f"   Defense Spending: ${metrics['spending'][max_equity_idx]:,}B")
logger.info(f"   Equity Ratio (P20/P99): {equity_ratio[max_equity_idx]:.2f}x")
logger.info(f"   P20 Benefit: {metrics['p20'][max_equity_idx]:+.4f}")
logger.info(f"   P99 Benefit: {metrics['p99'][max_equity_idx]:+.4f}")
logger.info(f"   Composite Index: {composite_index[max_equity_idx]:.1f}/100")

# Find best marginal efficiency (excluding extreme values)
valid_marginal = [(i, val) for i, val in enumerate(marginal_gdp) if val > 0 and val < MARGINAL_EFFICIENCY_MAX]
if valid_marginal:
    best_marginal_idx, best_marginal_val = max(valid_marginal, key=lambda x: x[1])
    marginal_spending = marginal_spending_points[best_marginal_idx]
    logger.info("\n5. Best Marginal Efficiency:")
    logger.info(f"   Defense Spending Range: ${marginal_spending - 250:,.0f}B to ${marginal_spending + 250:,.0f}B")
    logger.info(f"   Marginal GDP: {best_marginal_val:.6f} per $1B increase")

# Baseline comparison (if exists)
if baseline_exists:
    logger.info(f"\n6. Baseline Metrics (${metrics['spending'][baseline_idx]:,}B):")
    logger.info(f"   Composite Index: {composite_index[baseline_idx]:.1f}/100")
    logger.info(f"   GDP: {metrics['gdp'][baseline_idx]:+.4f}%")
    logger.info(f"   Jobs: {metrics['jobs'][baseline_idx]:,.0f}")
    logger.info(f"   Revenue: ${metrics['revenue'][baseline_idx]:,.2f}B")
    logger.info(f"   Equity Ratio: {equity_ratio[baseline_idx]:.2f}x")

# Range analysis
logger.info("\n7. Overall Ranges:")
gdp_range = max(metrics['gdp']) - min(metrics['gdp'])
jobs_range = max(metrics['jobs']) - min(metrics['jobs'])
revenue_range = max(metrics['revenue']) - min(metrics['revenue'])
logger.info(f"   GDP Range: {gdp_range:.4f}% ({min(metrics['gdp']):+.4f}% to {max(metrics['gdp']):+.4f}%)")
logger.info(f"   Jobs Range: {jobs_range:,.0f} ({min(metrics['jobs']):,.0f} to {max(metrics['jobs']):,.0f})")
logger.info(f"   Revenue Range: ${revenue_range:,.2f}B (${min(metrics['revenue']):,.2f}B to ${max(metrics['revenue']):,.2f}B)")

logger.section("COMPOSITE INDEX METHODOLOGY")
logger.info("  • Score 0-100 combining four normalized components:")
logger.info("    - Equity (50%): P20/P99 ratio normalized to 0-100")
logger.info("    - GDP (20%): Long-run GDP % change normalized to 0-100")
logger.info("    - Jobs (20%): Full-time equivalent jobs normalized to 0-100")
logger.info("    - Revenue (10%): Dynamic 10-year revenue normalized to 0-100")
logger.info("  • Higher scores indicate better overall economic + equity outcomes")
logger.info("  • Heavy equity weighting prioritizes distributive fairness")
logger.info("\nOTHER METRICS:")
logger.info("  • GDP: Long-run % change in GDP (e.g., 0.14% = 0.14 percentage points)")
logger.info("  • Jobs: Actual full-time equivalent jobs created")
logger.info("  • Revenue: Dynamic 10-year revenue in billions of dollars")
logger.info("  • Equity ratio: P20/P99 (higher = more equitable)")
logger.info("  • 'pp' = percentage points (e.g., from 0.13% to 0.14% = +0.01 pp)")
logger.info(f"\n[OK] Visualization complete! Check '{output_dir / 'defense_spending_analysis.png'}'")
