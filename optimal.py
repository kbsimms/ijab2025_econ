"""
Policy Optimization Script with Equity and National Security Constraints
Two-stage optimization:
1. Maximize GDP subject to fiscal, economic, equity, and NS constraints
2. Maximize revenue while maintaining optimal GDP
"""

import pandas as pd
from gurobipy import Model, GRB, quicksum

# Configuration
FILE_PATH = "tax reform & spending menu options (v8) template.xlsx"
SUPPRESS_GUROBI_OUTPUT = True


def load_and_clean_data(file_path):
    """Load and clean policy data from Excel file."""
    print("Loading policy data...")
    
    df_raw = pd.read_excel(file_path, sheet_name=0)
    
    # Define columns
    columns = [
        "Option", "LongRunGDP", "CapitalStock", "Jobs", "WageRate",
        "P20", "P40_60", "P80_100", "P99", "StaticRevenue", "DynamicRevenue", "Select"
    ]
    
    df = df_raw.iloc[3:].reset_index(drop=True)
    df.columns = columns[:len(df.columns)]
    
    # Drop non-policy rows and convert numeric
    df = df[df["Option"].notna()].reset_index(drop=True)
    numeric_cols = columns[1:-1]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    
    # Drop rows with missing GDP or dynamic revenue
    df = df[df["LongRunGDP"].notna() & df["DynamicRevenue"].notna()].reset_index(drop=True)
    
    # Add indicator for NS-prefixed policies
    df["is_NS"] = df["Option"].str.contains(r"NS[1-7][A-Z]:", case=False, na=False)
    
    # Extract NS groupings (NS1A, NS1B, NS1C → group "NS1")
    ns_groups = {}
    for idx, row in df[df["is_NS"]].iterrows():
        code = row["Option"].split(":")[0].strip()
        group = code[:-1]
        ns_groups.setdefault(group, []).append(idx)
    
    # Strict NS1–NS7 indices
    ns_strict_indices = df[df["Option"].str.match(r"NS[1-7][A-Z]:", na=False)].index.tolist()
    
    print(f"✓ Loaded {len(df)} policy options")
    print(f"✓ Identified {len(ns_groups)} NS policy groups\n")
    
    return df, ns_groups, ns_strict_indices


def optimize_policy_selection(df, ns_groups, ns_strict_indices, verbose=True):
    """
    Two-stage optimization with equity and NS constraints:
    1. Maximize GDP subject to all constraints
    2. Maximize revenue while maintaining optimal GDP
    """
    indices = range(len(df))
    
    # === Stage 1: Maximize GDP ===
    if verbose:
        print("Running optimization (Stage 1: Maximize GDP)...")
    
    m_gdp = Model("GDP_Max")
    if SUPPRESS_GUROBI_OUTPUT:
        m_gdp.setParam("OutputFlag", 0)
    
    x = m_gdp.addVars(indices, vtype=GRB.BINARY, name="x")
    
    # Objective: Maximize GDP
    m_gdp.setObjective(quicksum(x[i] * df.loc[i, "LongRunGDP"] for i in indices), GRB.MAXIMIZE)
    
    # Constraints
    # Fiscal
    m_gdp.addConstr(quicksum(x[i] * df.loc[i, "DynamicRevenue"] for i in indices) >= 0)
    
    # Economic
    m_gdp.addConstr(quicksum(x[i] * df.loc[i, "CapitalStock"] for i in indices) >= 0)
    m_gdp.addConstr(quicksum(x[i] * df.loc[i, "Jobs"] for i in indices) >= 0)
    m_gdp.addConstr(quicksum(x[i] * df.loc[i, "WageRate"] for i in indices) >= 0)
    
    # Equity
    p20 = quicksum(x[i] * df.loc[i, "P20"] for i in indices)
    p40 = quicksum(x[i] * df.loc[i, "P40_60"] for i in indices)
    p80 = quicksum(x[i] * df.loc[i, "P80_100"] for i in indices)
    p99 = quicksum(x[i] * df.loc[i, "P99"] for i in indices)
    
    m_gdp.addConstr(p20 - p40 <= 0.01)
    m_gdp.addConstr(p40 - p20 <= 0.01)
    m_gdp.addConstr(p20 - p99 >= 1e-5)
    m_gdp.addConstr(p40 - p99 >= 1e-5)
    m_gdp.addConstr(p20 - p80 >= 1e-5)
    m_gdp.addConstr(p40 - p80 >= 1e-5)
    m_gdp.addConstr(p20 + p40 - p80 - p99 >= 1e-5)
    
    # NS mutual exclusivity
    for group, idxs in ns_groups.items():
        m_gdp.addConstr(quicksum(x[i] for i in idxs) <= 1)
    
    # NS spending (only from NS1–NS7)
    m_gdp.addConstr(quicksum(x[i] * df.loc[i, "DynamicRevenue"] for i in ns_strict_indices) <= -3000)
    
    # Solve first pass
    m_gdp.optimize()
    gdp_star = m_gdp.objVal
    
    # === Stage 2: Maximize Revenue under optimal GDP ===
    if verbose:
        print("Running optimization (Stage 2: Maximize Revenue)...")
    
    m_rev = Model("Revenue_Tiebreak")
    if SUPPRESS_GUROBI_OUTPUT:
        m_rev.setParam("OutputFlag", 0)
    
    x2 = m_rev.addVars(indices, vtype=GRB.BINARY, name="x")
    
    # Objective: Maximize dynamic revenue
    m_rev.setObjective(quicksum(x2[i] * df.loc[i, "DynamicRevenue"] for i in indices), GRB.MAXIMIZE)
    
    # Constraints (same as first pass)
    m_rev.addConstr(quicksum(x2[i] * df.loc[i, "DynamicRevenue"] for i in indices) >= 0)
    m_rev.addConstr(quicksum(x2[i] * df.loc[i, "CapitalStock"] for i in indices) >= 0)
    m_rev.addConstr(quicksum(x2[i] * df.loc[i, "Jobs"] for i in indices) >= 0)
    m_rev.addConstr(quicksum(x2[i] * df.loc[i, "WageRate"] for i in indices) >= 0)
    
    p20 = quicksum(x2[i] * df.loc[i, "P20"] for i in indices)
    p40 = quicksum(x2[i] * df.loc[i, "P40_60"] for i in indices)
    p80 = quicksum(x2[i] * df.loc[i, "P80_100"] for i in indices)
    p99 = quicksum(x2[i] * df.loc[i, "P99"] for i in indices)
    
    m_rev.addConstr(p20 - p40 <= 0.01)
    m_rev.addConstr(p40 - p20 <= 0.01)
    m_rev.addConstr(p20 - p99 >= 1e-5)
    m_rev.addConstr(p40 - p99 >= 1e-5)
    m_rev.addConstr(p20 - p80 >= 1e-5)
    m_rev.addConstr(p40 - p80 >= 1e-5)
    m_rev.addConstr(p20 + p40 - p80 - p99 >= 1e-5)
    
    for group, idxs in ns_groups.items():
        m_rev.addConstr(quicksum(x2[i] for i in idxs) <= 1)
    
    m_rev.addConstr(quicksum(x2[i] * df.loc[i, "DynamicRevenue"] for i in ns_strict_indices) <= -3000)
    
    # Fix GDP from first pass
    m_rev.addConstr(quicksum(x2[i] * df.loc[i, "LongRunGDP"] for i in indices) == gdp_star)
    
    # Solve second pass
    m_rev.optimize()
    
    # Extract solution
    selected_indices = [i for i in indices if x2[i].X > 0.5]
    selected_df = df.iloc[selected_indices].copy()
    
    return selected_df, gdp_star, m_rev.ObjVal


def display_results(result_df, gdp_impact, revenue_impact):
    """Display optimization results in a readable format."""
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS".center(80))
    print("="*80)
    
    # Separate policies by positive/negative revenue impact
    positive_revenue = result_df[result_df["DynamicRevenue"] >= 0].copy()
    negative_revenue = result_df[result_df["DynamicRevenue"] < 0].copy()
    
    # Sort by absolute impact
    positive_revenue = positive_revenue.sort_values("DynamicRevenue", ascending=False)
    negative_revenue = negative_revenue.sort_values("DynamicRevenue", ascending=True)
    
    if len(positive_revenue) > 0:
        print(f"\n{'REVENUE RAISING POLICIES':^80}")
        print("-"*80)
        for _, row in positive_revenue.iterrows():
            print(f"  {row['Option'][:70]:<70}")
            print(f"    GDP: {row['LongRunGDP'] * 100:>+7.4f}%  |  Revenue: ${row['DynamicRevenue']:>8.2f}B")
    
    if len(negative_revenue) > 0:
        print(f"\n{'REVENUE REDUCING POLICIES':^80}")
        print("-"*80)
        for _, row in negative_revenue.iterrows():
            print(f"  {row['Option'][:70]:<70}")
            print(f"    GDP: {row['LongRunGDP'] * 100:>+7.4f}%  |  Revenue: ${row['DynamicRevenue']:>8.2f}B")
    
    # Calculate totals for all metrics
    print("\n" + "="*80)
    print("FINAL SUMMARY - TOTAL IMPACT OF SELECTED POLICIES".center(80))
    print("="*80)
    print(f"\n{'Economic Impacts':^80}")
    print("-"*80)
    print(f"  Long-Run Change in GDP:              {result_df['LongRunGDP'].sum() * 100:>+8.4f}%")
    print(f"  Capital Stock:                       {result_df['CapitalStock'].sum() * 100:>+8.4f}%")
    print(f"  Full-Time Equivalent Jobs:           {result_df['Jobs'].sum():>+10,.0f}")
    print(f"  Wage Rate:                           {result_df['WageRate'].sum() * 100:>+8.4f}%")
    
    print(f"\n{'After-Tax Income Changes (by Income Percentile)':^80}")
    print("-"*80)
    print(f"  P20 (Bottom 20%):                    {result_df['P20'].sum() * 100:>+8.4f}%")
    print(f"  P40-60 (Middle Class):               {result_df['P40_60'].sum() * 100:>+8.4f}%")
    print(f"  P80-100 (Top 20%):                   {result_df['P80_100'].sum() * 100:>+8.4f}%")
    print(f"  P99 (Top 1%):                        {result_df['P99'].sum() * 100:>+8.4f}%")
    
    print(f"\n{'Revenue Impacts':^80}")
    print("-"*80)
    print(f"  Static 10-Year Revenue:              ${result_df['StaticRevenue'].sum():>10.2f} billion")
    print(f"  Dynamic 10-Year Revenue:             ${result_df['DynamicRevenue'].sum():>10.2f} billion")
    
    print(f"\n{'Policy Count':^80}")
    print("-"*80)
    print(f"  Number of Selected Policies:         {len(result_df):>10}")
    
    print("\n" + "="*80 + "\n")


def main():
    """Main execution function."""
    # Load and clean data
    df, ns_groups, ns_strict_indices = load_and_clean_data(FILE_PATH)
    
    # Run optimization
    result_df, gdp_impact, revenue_impact = optimize_policy_selection(
        df, ns_groups, ns_strict_indices
    )
    
    # Display results
    display_results(result_df, gdp_impact, revenue_impact)
    
    # Save to CSV
    result_df.to_csv("optimal.csv", index=False)
    print("✓ Results saved to 'optimal.csv'\n")


if __name__ == "__main__":
    main()
