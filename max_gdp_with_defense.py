"""
Policy Optimization Script with Distributional Equality and Defense Spending
Maximizes GDP growth subject to:
- Revenue constraint >= 270 billion (to fund defense spending)
- Distributional equality constraint (all income groups within 1% of each other)
"""

import pandas as pd
from gurobipy import Model, GRB, quicksum

# Configuration
FILE_PATH = "tax reform & spending menu options (v8) template.xlsx"
SUPPRESS_GUROBI_OUTPUT = True


def load_and_clean_data(file_path):
    """Load and clean policy data from Excel file."""
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
    df_clean["P20"] = df_clean["P20"].fillna(0)
    df_clean["P40-60"] = df_clean["P40-60"].fillna(0)
    df_clean["P80-100"] = df_clean["P80-100"].fillna(0)
    df_clean["P99"] = df_clean["P99"].fillna(0)
    
    print(f"✓ Loaded {len(df_clean)} policy options\n")
    return df_clean


def optimize_policy_selection(df_clean, verbose=True):
    """
    Two-stage optimization with distributional equality:
    1. Maximize GDP subject to revenue >= 270 and distributional equality
    2. Maximize lower-income benefit while maintaining optimal GDP and equality
    """
    # Extract data
    n = len(df_clean)
    gdp = df_clean["Long-Run Change in GDP"].values
    revenue = df_clean["Dynamic 10-Year Revenue (billions)"].values
    policy_names = df_clean["Option"].values
    p20 = df_clean["P20"].values
    p40_60 = df_clean["P40-60"].values
    p80_100 = df_clean["P80-100"].values
    p99 = df_clean["P99"].values
    
    # Stage 1: Maximize GDP subject to revenue >= 270 and distributional equality
    if verbose:
        print("Running optimization (Stage 1: Maximize GDP with Distributional Equality)...")
    
    model = Model("GDP_Maximization_Defense_Equal_Distribution")
    if SUPPRESS_GUROBI_OUTPUT:
        model.setParam('OutputFlag', 0)
    
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    
    # Revenue constraint >= 270 (for defense spending)
    model.addConstr(quicksum(revenue[i] * x[i] for i in range(n)) >= 270, name="RevenueConstraint270")
    
    # Distributional totals
    p20_total = quicksum(p20[i] * x[i] for i in range(n))
    p40_60_total = quicksum(p40_60[i] * x[i] for i in range(n))
    p80_100_total = quicksum(p80_100[i] * x[i] for i in range(n))
    p99_total = quicksum(p99[i] * x[i] for i in range(n))
    
    # Distributional equality constraints (max 1 percentage point difference)
    # We need to enforce |group1 - group2| <= 0.01 for all pairs
    distribution_pairs = [
        (p20_total, p40_60_total, "P20_P40"),
        (p20_total, p80_100_total, "P20_P80"),
        (p20_total, p99_total, "P20_P99"),
        (p40_60_total, p80_100_total, "P40_P80"),
        (p40_60_total, p99_total, "P40_P99"),
        (p80_100_total, p99_total, "P80_P99"),
    ]
    
    for lhs, rhs, name in distribution_pairs:
        diff = model.addVar(lb=0.0, name=f"abs_diff_{name}")
        model.addConstr(lhs - rhs <= diff, name=f"{name}_pos")
        model.addConstr(rhs - lhs <= diff, name=f"{name}_neg")
        model.addConstr(diff <= 0.01, name=f"{name}_limit")
    
    # Objective: Maximize GDP
    model.setObjective(quicksum(gdp[i] * x[i] for i in range(n)), GRB.MAXIMIZE)
    model.optimize()
    
    best_gdp = model.ObjVal
    
    # Stage 2: Tiebreaking - prioritize lower-income brackets while maintaining GDP and equality
    if verbose:
        print("Running optimization (Stage 2: Maximize Lower-Income Benefit)...")
    
    model_2 = Model("Lower_Income_Prioritization_Defense")
    if SUPPRESS_GUROBI_OUTPUT:
        model_2.setParam('OutputFlag', 0)
    
    x2 = model_2.addVars(n, vtype=GRB.BINARY, name="x")
    
    # Revenue constraint >= 270 (for defense spending)
    model_2.addConstr(quicksum(revenue[i] * x2[i] for i in range(n)) >= 270, name="RevenueConstraint270")
    
    # GDP match constraint
    model_2.addConstr(quicksum(gdp[i] * x2[i] for i in range(n)) == best_gdp, name="GDPMatch")
    
    # Distributional totals
    p20_sum = quicksum(p20[i] * x2[i] for i in range(n))
    p40_sum = quicksum(p40_60[i] * x2[i] for i in range(n))
    p80_sum = quicksum(p80_100[i] * x2[i] for i in range(n))
    p99_sum = quicksum(p99[i] * x2[i] for i in range(n))
    
    # Distributional equality constraints
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
    
    # Objective: prioritize lower-income brackets (lexicographic with weights)
    model_2.setObjective(
        p20_sum * 1e6 + p40_sum * 1e3 + p80_sum * 1e1 + p99_sum, 
        GRB.MAXIMIZE
    )
    model_2.optimize()
    
    # Extract solution and all metrics for selected policies
    selected_indices = [i for i in range(n) if x2[i].X > 0.5]
    selected_df = df_clean.iloc[selected_indices].copy()
    
    return selected_df, best_gdp, model_2.ObjVal


def display_results(result_df, gdp_impact, revenue_impact):
    """Display optimization results in a readable format."""
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS (WITH DISTRIBUTIONAL EQUALITY & DEFENSE FUNDING)".center(80))
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
    
    # Calculate and display the range (max difference)
    distro_values = [p20_total, p40_total, p80_total, p99_total]
    max_diff = max(distro_values) - min(distro_values)
    print(f"\n  Distributional Range (max - min):    {max_diff:>+8.4f}%")
    print(f"  (Constraint: must be ≤ 1.00%)")
    
    print(f"\n{'Revenue Impacts':^80}")
    print("-"*80)
    print(f"  Static 10-Year Revenue:              ${result_df['Static 10-Year Revenue (billions)'].sum():>10.2f} billion")
    print(f"  Dynamic 10-Year Revenue:             ${result_df['Dynamic 10-Year Revenue (billions)'].sum():>10.2f} billion")
    print(f"  (Constraint: must be ≥ $270.00 billion)")
    
    print(f"\n{'Policy Count':^80}")
    print("-"*80)
    print(f"  Number of Selected Policies:         {len(result_df):>10}")
    
    print("\n" + "="*80 + "\n")


def main():
    """Main execution function."""
    # Load and clean data
    df_clean = load_and_clean_data(FILE_PATH)
    
    # Run optimization
    result_df, gdp_impact, revenue_impact = optimize_policy_selection(df_clean)
    
    # Display results
    display_results(result_df, gdp_impact, revenue_impact)
    
    # Optionally save to CSV
    result_df.to_csv("max_gdp_with_defense.csv", index=False)
    print("✓ Results saved to 'max_gdp_with_defense.csv'\n")


if __name__ == "__main__":
    main()
