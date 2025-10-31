"""
Policy Optimization Script with Distributional Equality

This script maximizes GDP growth subject to:
- Revenue neutrality constraint (no increase in deficit)
- Distributional equality constraint (all income groups within 1% of each other)
- National Security (NS) constraints: Mutual exclusivity within NS policy groups

The distributional equality constraint ensures that policy impacts are
distributed fairly across all income groups, with after-tax income changes
for P20, P40-60, P80-100, and P99 differing by no more than 1 percentage point.

Key Features:
1. Distributional Equality: All income groups within 1% of each other
2. Revenue Neutrality: Ensures total dynamic revenue is non-negative
3. National Security (NS) Constraints:
   - NS policy groups (e.g., NS1A, NS1B, NS1C): Only one option per group can be selected
   - Prevents selecting conflicting national security policies
4. Two-stage optimization with tiebreaking on lower-income benefit
"""

import pandas as pd
from gurobipy import Model, GRB, quicksum

# Configuration
FILE_PATH = "tax reform & spending menu options (v8) template.xlsx"
SUPPRESS_GUROBI_OUTPUT = True


def load_and_clean_data(file_path):
    """
    Load and clean policy data from Excel file.
    
    This function handles data preprocessing including:
    - Loading raw Excel data
    - Extracting proper column headers
    - Converting numeric columns
    - Identifying National Security (NS) policy groups
    
    Returns:
        tuple: (df_clean, ns_groups)
            - df_clean: Cleaned DataFrame with all policies
            - ns_groups: Dict mapping NS group names to policy indices
    """
    print("Loading policy data...")
    
    # Load the Excel file
    xls = pd.ExcelFile(file_path)
    df = xls.parse('Sheet1')
    
    # Extract headers from row 2 (index 1)
    headers = df.iloc[1]
    df_clean = df[2:].copy()
    df_clean.columns = headers
    df_clean = df_clean.reset_index(drop=True)
    
    # Drop rows that are not actual policy options
    df_clean = df_clean[df_clean["Option"].notna()]
    df_clean = df_clean[df_clean["Long-Run Change in GDP"].notna()]
    
    # Convert all numeric columns
    numeric_cols = [
        "Long-Run Change in GDP",
        "Capital Stock",
        "Full-Time Equivalent Jobs",
        "Wage Rate",
        "P20",
        "P40-60",
        "P80-100",
        "P99",
        "Static 10-Year Revenue (billions)",
        "Dynamic 10-Year Revenue (billions)"
    ]
    df_clean[numeric_cols] = df_clean[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    # Fill NaN values in distributional columns with 0
    # This handles spending policies that may not have distributional data
    df_clean["P20"] = df_clean["P20"].fillna(0)
    df_clean["P40-60"] = df_clean["P40-60"].fillna(0)
    df_clean["P80-100"] = df_clean["P80-100"].fillna(0)
    df_clean["P99"] = df_clean["P99"].fillna(0)
    
    # Add indicator for NS-prefixed policies (National Security policies)
    # Matches patterns like: NS1A:, NS2B:, NS7C:, etc.
    # Note: This uses a strict pattern to only match NSxY where x=digits, Y=letter
    df_clean["is_NS"] = df_clean["Option"].str.contains(r"^NS\d+[A-Z]:", case=False, na=False, regex=True)
    
    # Extract NS groupings for mutual exclusivity constraints
    # Example: NS1A, NS1B, NS1C all belong to group "NS1"
    # Only one option from each NS group can be selected
    # IMPORTANT: Use positional index (0-based) not DataFrame index label
    ns_groups = {}
    for pos_idx, (label_idx, row) in enumerate(df_clean[df_clean["is_NS"]].iterrows()):
        code = row["Option"].split(":")[0].strip()  # Extract "NS1A" from "NS1A: Description"
        group = code[:-1]  # Extract "NS1" from "NS1A"
        # Find positional index in the full df_clean DataFrame
        pos_in_df = df_clean.index.get_loc(label_idx)
        ns_groups.setdefault(group, []).append(pos_in_df)
    
    print(f"✓ Loaded {len(df_clean)} policy options")
    if ns_groups:
        print(f"✓ Identified {len(ns_groups)} NS policy groups:")
        for group, idxs in sorted(ns_groups.items()):
            policies = [df_clean.iloc[idx]["Option"].split(":")[0] for idx in idxs]
            print(f"   {group}: {', '.join(policies)} ({len(idxs)} options)")
        print()
    else:
        print("⚠ WARNING: No NS policy groups detected\n")
    
    return df_clean, ns_groups


def optimize_policy_selection(df_clean, ns_groups, verbose=True):
    """
    Two-stage optimization with distributional equality and NS mutual exclusivity constraints.
    
    Stage 1: Maximize GDP subject to revenue neutrality, distributional equality, and NS constraints
        - Finds the maximum achievable GDP growth
        - Ensures total dynamic revenue is non-negative
        - Enforces that all income groups have similar after-tax income changes
        - Enforces NS mutual exclusivity: at most one policy per NS group
        
    Stage 2: Maximize lower-income benefit while maintaining optimal GDP and equality
        - Among solutions achieving optimal GDP from Stage 1
        - Prioritizes policies that benefit lower-income groups
        - Uses lexicographic weighting to prefer P20 > P40-60 > P80-100 > P99
    
    National Security (NS) Mutual Exclusivity:
        For each NS group (e.g., NS1 with options NS1A, NS1B, NS1C),
        at most one option can be selected. This prevents selecting
        conflicting national security policies within the same category.
    
    Args:
        df_clean: DataFrame containing policy options and their impacts
        ns_groups: Dict mapping NS group names to lists of policy indices
        verbose: If True, prints progress messages
        
    Returns:
        tuple: (selected_df, gdp_impact, revenue_impact)
            - selected_df: DataFrame of selected policies
            - gdp_impact: Total GDP impact achieved
            - revenue_impact: Total revenue impact achieved
    """
    # Extract data arrays from DataFrame for optimization
    n = len(df_clean)
    gdp = df_clean["Long-Run Change in GDP"].values
    revenue = df_clean["Dynamic 10-Year Revenue (billions)"].values
    policy_names = df_clean["Option"].values
    
    # Income percentile impact arrays
    # P20 = bottom 20%, P40-60 = middle class, P80-100 = top 20%, P99 = top 1%
    p20 = df_clean["P20"].values
    p40_60 = df_clean["P40-60"].values
    p80_100 = df_clean["P80-100"].values
    p99 = df_clean["P99"].values
    
    # === Stage 1: Maximize GDP with distributional equality ===
    if verbose:
        print("Running optimization (Stage 1: Maximize GDP with Distributional Equality)...")
    
    model = Model("GDP_Maximization_Equal_Distribution")
    if SUPPRESS_GUROBI_OUTPUT:
        model.setParam('OutputFlag', 0)
    
    # Decision variables: x[i] = 1 if policy i is selected, 0 otherwise
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    
    # Constraint: Revenue neutrality (same as basic model)
    model.addConstr(quicksum(revenue[i] * x[i] for i in range(n)) >= 0, name="RevenueNeutrality")
    
    # Calculate total after-tax income change for each percentile group
    p20_total = quicksum(p20[i] * x[i] for i in range(n))
    p40_60_total = quicksum(p40_60[i] * x[i] for i in range(n))
    p80_100_total = quicksum(p80_100[i] * x[i] for i in range(n))
    p99_total = quicksum(p99[i] * x[i] for i in range(n))
    
    # Distributional equality constraints: enforce |group1 - group2| <= 0.01
    # This ensures all income groups experience similar after-tax income changes
    # (within 1 percentage point of each other)
    distribution_pairs = [
        (p20_total, p40_60_total, "P20_P40"),      # Bottom 20% vs Middle class
        (p20_total, p80_100_total, "P20_P80"),     # Bottom 20% vs Top 20%
        (p20_total, p99_total, "P20_P99"),         # Bottom 20% vs Top 1%
        (p40_60_total, p80_100_total, "P40_P80"),  # Middle class vs Top 20%
        (p40_60_total, p99_total, "P40_P99"),      # Middle class vs Top 1%
        (p80_100_total, p99_total, "P80_P99"),     # Top 20% vs Top 1%
    ]
    
    # For each pair of income groups, enforce the absolute difference constraint
    # We model |lhs - rhs| <= 0.01 using: lhs - rhs <= diff AND rhs - lhs <= diff
    for lhs, rhs, name in distribution_pairs:
        # Create auxiliary variable to represent absolute difference
        diff = model.addVar(lb=0.0, name=f"abs_diff_{name}")
        # Enforce: lhs - rhs <= diff (handles case when lhs > rhs)
        model.addConstr(lhs - rhs <= diff, name=f"{name}_pos")
        # Enforce: rhs - lhs <= diff (handles case when rhs > lhs)
        model.addConstr(rhs - lhs <= diff, name=f"{name}_neg")
        # Limit the absolute difference to 1 percentage point
        model.addConstr(diff <= 0.01, name=f"{name}_limit")
    
    # NS mutual exclusivity constraints
    # For each NS group (e.g., NS1 with options NS1A, NS1B, NS1C),
    # at most one option can be selected
    for group, idxs in ns_groups.items():
        model.addConstr(quicksum(x[i] for i in idxs) <= 1, name=f"NS_{group}_mutual_exclusivity")
    
    # Objective: Maximize total GDP impact
    model.setObjective(quicksum(gdp[i] * x[i] for i in range(n)), GRB.MAXIMIZE)
    model.optimize()
    
    # Store the optimal GDP value for use in Stage 2
    best_gdp = model.ObjVal
    
    # === Stage 2: Tiebreaking - prioritize lower-income benefit ===
    # Among all solutions that achieve optimal GDP and satisfy equality constraints,
    # select the one that maximizes benefits for lower-income groups
    if verbose:
        print("Running optimization (Stage 2: Maximize Lower-Income Benefit)...")
    
    model_2 = Model("Lower_Income_Prioritization")
    if SUPPRESS_GUROBI_OUTPUT:
        model_2.setParam('OutputFlag', 0)
    
    # New decision variables for Stage 2
    x2 = model_2.addVars(n, vtype=GRB.BINARY, name="x")
    
    # Constraint: Revenue neutrality (same as Stage 1)
    model_2.addConstr(quicksum(revenue[i] * x2[i] for i in range(n)) >= 0, name="RevenueNeutrality")
    
    # Constraint: Must achieve exactly the optimal GDP from Stage 1
    model_2.addConstr(quicksum(gdp[i] * x2[i] for i in range(n)) == best_gdp, name="GDPMatch")
    
    # Calculate distributional totals for Stage 2
    p20_sum = quicksum(p20[i] * x2[i] for i in range(n))
    p40_sum = quicksum(p40_60[i] * x2[i] for i in range(n))
    p80_sum = quicksum(p80_100[i] * x2[i] for i in range(n))
    p99_sum = quicksum(p99[i] * x2[i] for i in range(n))
    
    # Distributional equality constraints (same as Stage 1)
    for (lhs, rhs, name) in [
        (p20_sum, p40_sum, "P20_P40"),
        (p20_sum, p80_sum, "P20_P80"),
        (p20_sum, p99_sum, "P20_P99"),
        (p40_sum, p80_sum, "P40_P80"),
        (p40_sum, p99_sum, "P40_P99"),
        (p80_sum, p99_sum, "P80_P99"),
    ]:
        diff = model_2.addVar(lb=0.0, name=f"abs_diff_{name}")
        model_2.addConstr(lhs - rhs <= diff, name=f"{name}_pos")
        model_2.addConstr(rhs - lhs <= diff, name=f"{name}_neg")
        model_2.addConstr(diff <= 0.01, name=f"{name}_limit")
    
    # NS mutual exclusivity constraints (same as Stage 1)
    for group, idxs in ns_groups.items():
        model_2.addConstr(quicksum(x2[i] for i in idxs) <= 1, name=f"NS_{group}_mutual_exclusivity")
    
    # Objective: Prioritize lower-income brackets using lexicographic weighting
    # Weights: P20 gets highest priority (1e6), then P40-60 (1e3), then P80-100 (1e1), then P99 (1)
    # This ensures we maximize P20 benefit first, then P40-60, etc.
    model_2.setObjective(
        p20_sum * 1e6 + p40_sum * 1e3 + p80_sum * 1e1 + p99_sum,
        GRB.MAXIMIZE
    )
    model_2.optimize()
    
    # Extract solution: policies where x2[i] > 0.5 are selected
    # Using 0.5 threshold handles numerical precision issues in binary variables
    selected_indices = [i for i in range(n) if x2[i].X > 0.5]
    selected_df = df_clean.iloc[selected_indices].copy()
    
    return selected_df, best_gdp, model_2.ObjVal


def display_results(result_df, gdp_impact, revenue_impact):
    """Display optimization results in a readable format."""
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS (WITH DISTRIBUTIONAL EQUALITY)".center(80))
    print("="*80)
    
    # Separate policies by positive/negative revenue impact
    positive_revenue = result_df[result_df["Dynamic 10-Year Revenue (billions)"] >= 0].copy()
    negative_revenue = result_df[result_df["Dynamic 10-Year Revenue (billions)"] < 0].copy()
    
    # Sort by absolute impact
    positive_revenue = positive_revenue.sort_values("Dynamic 10-Year Revenue (billions)", ascending=False)
    negative_revenue = negative_revenue.sort_values("Dynamic 10-Year Revenue (billions)", ascending=True)
    
    if len(positive_revenue) > 0:
        print(f"\n{'REVENUE RAISING POLICIES':^80}")
        print("-"*80)
        for _, row in positive_revenue.iterrows():
            print(f"  {row['Option'][:70]:<70}")
            print(f"    GDP: {row['Long-Run Change in GDP'] * 100:>+7.4f}%  |  Revenue: ${row['Dynamic 10-Year Revenue (billions)']:>8.2f}B")
    
    if len(negative_revenue) > 0:
        print(f"\n{'REVENUE REDUCING POLICIES':^80}")
        print("-"*80)
        for _, row in negative_revenue.iterrows():
            print(f"  {row['Option'][:70]:<70}")
            print(f"    GDP: {row['Long-Run Change in GDP'] * 100:>+7.4f}%  |  Revenue: ${row['Dynamic 10-Year Revenue (billions)']:>8.2f}B")
    
    # Calculate totals for all metrics
    print("\n" + "="*80)
    print("FINAL SUMMARY - TOTAL IMPACT OF SELECTED POLICIES".center(80))
    print("="*80)
    print(f"\n{'Economic Impacts':^80}")
    print("-"*80)
    print(f"  Long-Run Change in GDP:              {result_df['Long-Run Change in GDP'].sum() * 100:>+8.4f}%")
    print(f"  Capital Stock:                       {result_df['Capital Stock'].sum() * 100:>+8.4f}%")
    print(f"  Full-Time Equivalent Jobs:           {result_df['Full-Time Equivalent Jobs'].sum():>+10,.0f}")
    print(f"  Wage Rate:                           {result_df['Wage Rate'].sum() * 100:>+8.4f}%")
    
    print(f"\n{'After-Tax Income Changes (by Income Percentile)':^80}")
    print("-"*80)
    p20_total = result_df['P20'].sum() * 100
    p40_total = result_df['P40-60'].sum() * 100
    p80_total = result_df['P80-100'].sum() * 100
    p99_total = result_df['P99'].sum() * 100
    print(f"  P20 (Bottom 20%):                    {p20_total:>+8.4f}%")
    print(f"  P40-60 (Middle Class):               {p40_total:>+8.4f}%")
    print(f"  P80-100 (Top 20%):                   {p80_total:>+8.4f}%")
    print(f"  P99 (Top 1%):                        {p99_total:>+8.4f}%")
    
    # Calculate and display the range (max difference between any two groups)
    distro_values = [p20_total, p40_total, p80_total, p99_total]
    max_diff = max(distro_values) - min(distro_values)
    print(f"\n  Distributional Range (max - min):    {max_diff:>+8.4f}%")
    print(f"  (Constraint: must be ≤ 1.00%)")
    
    print(f"\n{'Revenue Impacts':^80}")
    print("-"*80)
    print(f"  Static 10-Year Revenue:              ${result_df['Static 10-Year Revenue (billions)'].sum():>10.2f} billion")
    print(f"  Dynamic 10-Year Revenue:             ${result_df['Dynamic 10-Year Revenue (billions)'].sum():>10.2f} billion")
    
    print(f"\n{'Policy Count':^80}")
    print("-"*80)
    print(f"  Number of Selected Policies:         {len(result_df):>10}")
    
    print("\n" + "="*80 + "\n")


def main():
    """Main execution function."""
    # Load and clean data
    df_clean, ns_groups = load_and_clean_data(FILE_PATH)
    
    # Run optimization
    result_df, gdp_impact, revenue_impact = optimize_policy_selection(df_clean, ns_groups)
    
    # Display results
    display_results(result_df, gdp_impact, revenue_impact)
    
    # Save to CSV
    result_df.to_csv("max_gdp_equal_distro.csv", index=False)
    print("✓ Results saved to 'max_gdp_equal_distro.csv'\n")


if __name__ == "__main__":
    main()
