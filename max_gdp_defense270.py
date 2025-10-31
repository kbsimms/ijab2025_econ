"""
Policy Optimization Script with Equity and National Security Constraints

This script maximizes GDP growth subject to:
- Fiscal constraint: Revenue neutrality (no increase in deficit)
- Economic constraints: Non-negative capital stock, jobs, and wage rate
- Equity constraints: Progressive distribution favoring lower/middle income groups
- National security constraints: Minimum $3,000B spending, mutual exclusivity

Key Features:
1. Equity Constraints: Ensures lower and middle income groups (P20, P40-60) benefit
   at least as much as upper income groups (P80-100, P99)
2. National Security (NS) Constraints:
   - NS1-NS7 groups (A/B/C options): Only one option per group can be selected
   - Minimum $3,000B total spending across all NS1-NS7 selections
3. Two-stage optimization to find best GDP with tiebreaking on revenue
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
        tuple: (df, ns_groups, ns_strict_indices)
            - df: Cleaned DataFrame with all policies
            - ns_groups: Dict mapping NS group names to policy indices
            - ns_strict_indices: List of indices for NS1-NS7 policies only
    """
    print("Loading policy data...")
    
    df_raw = pd.read_excel(file_path, sheet_name=0)
    
    # Define column names (matching Excel structure)
    columns = [
        "Option", "LongRunGDP", "CapitalStock", "Jobs", "WageRate",
        "P20", "P40_60", "P80_100", "P99", "StaticRevenue", "DynamicRevenue", "Select"
    ]
    
    # Skip first 3 rows (headers) and reset index
    df = df_raw.iloc[3:].reset_index(drop=True)
    df.columns = columns[:len(df.columns)]
    
    # Drop non-policy rows and convert numeric columns
    df = df[df["Option"].notna()].reset_index(drop=True)
    numeric_cols = columns[1:-1]  # All columns except Option and Select
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    
    # Drop rows with missing GDP or dynamic revenue (incomplete data)
    df = df[df["LongRunGDP"].notna() & df["DynamicRevenue"].notna()].reset_index(drop=True)
    
    # Add indicator for NS-prefixed policies (National Security policies)
    # Matches patterns like: NS1A:, NS2B:, NS7C:, etc.
    df["is_NS"] = df["Option"].str.contains(r"NS[1-7][A-Z]:", case=False, na=False)
    
    # Extract NS groupings for mutual exclusivity constraints
    # Example: NS1A, NS1B, NS1C all belong to group "NS1"
    # Only one option from each NS group can be selected
    ns_groups = {}
    for idx, row in df[df["is_NS"]].iterrows():
        code = row["Option"].split(":")[0].strip()  # Extract "NS1A" from "NS1A: Description"
        group = code[:-1]  # Extract "NS1" from "NS1A"
        ns_groups.setdefault(group, []).append(idx)
    
    # Get indices of strict NS1–NS7 policies (for spending constraint)
    # These are the policies that count toward the $3,000B minimum NS spending
    ns_strict_indices = df[df["Option"].str.match(r"NS[1-7][A-Z]:", na=False)].index.tolist()
    
    print(f"✓ Loaded {len(df)} policy options")
    print(f"✓ Identified {len(ns_groups)} NS policy groups\n")
    
    return df, ns_groups, ns_strict_indices


def optimize_policy_selection(df, ns_groups, ns_strict_indices, verbose=True):
    """
    Two-stage optimization with equity and national security constraints.
    
    Stage 1: Maximize GDP subject to all constraints
        - Fiscal: Revenue neutrality (sum of dynamic revenue >= 0)
        - Economic: Non-negative capital stock, jobs, wage rate
        - Equity: Progressive distribution (P20, P40-60 >= P80-100, P99)
        - NS mutual exclusivity: At most one policy per NS group
        - NS spending minimum: At least $3,000B spending on NS1-NS7 policies
        
    Stage 2: Maximize revenue while maintaining optimal GDP
        - Same constraints as Stage 1
        - Additionally constrains GDP to equal the optimal value from Stage 1
        - Breaks ties by maximizing revenue surplus
    
    Equity Constraints Explained:
        The equity constraints ensure progressive policy impacts:
        1. P20 and P40-60 must be within 1% of each other (protect both groups equally)
        2. P20 and P40-60 must benefit at least as much as P80-100 and P99
        3. Combined P20+P40-60 must exceed combined P80-100+P99
        
    Args:
        df: DataFrame containing policy options and their impacts
        ns_groups: Dict mapping NS group names to lists of policy indices
        ns_strict_indices: List of indices for NS1-NS7 policies
        verbose: If True, prints progress messages
        
    Returns:
        tuple: (selected_df, gdp_impact, revenue_impact)
            - selected_df: DataFrame of selected policies
            - gdp_impact: Total GDP impact achieved
            - revenue_impact: Total revenue impact achieved
    """
    indices = range(len(df))
    
    # === Stage 1: Maximize GDP ===
    if verbose:
        print("Running optimization (Stage 1: Maximize GDP)...")
    
    m_gdp = Model("GDP_Max")
    if SUPPRESS_GUROBI_OUTPUT:
        m_gdp.setParam("OutputFlag", 0)
    
    # Decision variables: x[i] = 1 if policy i is selected, 0 otherwise
    x = m_gdp.addVars(indices, vtype=GRB.BINARY, name="x")
    
    # Objective: Maximize total GDP impact
    m_gdp.setObjective(quicksum(x[i] * df.loc[i, "LongRunGDP"] for i in indices), GRB.MAXIMIZE)
    
    # === Constraints ===
    
    # Fiscal constraint: Total dynamic revenue must be non-negative
    # Ensures the policy package doesn't increase the deficit
    m_gdp.addConstr(quicksum(x[i] * df.loc[i, "DynamicRevenue"] for i in indices) >= 0)
    
    # Economic constraints: Ensure positive economic impacts
    # Capital stock change must be non-negative
    m_gdp.addConstr(quicksum(x[i] * df.loc[i, "CapitalStock"] for i in indices) >= 0)
    # Job creation must be non-negative
    m_gdp.addConstr(quicksum(x[i] * df.loc[i, "Jobs"] for i in indices) >= 0)
    # Wage rate change must be non-negative
    m_gdp.addConstr(quicksum(x[i] * df.loc[i, "WageRate"] for i in indices) >= 0)
    
    # Equity constraints: Ensure progressive distribution
    # Calculate total after-tax income change for each percentile group
    p20 = quicksum(x[i] * df.loc[i, "P20"] for i in indices)
    p40 = quicksum(x[i] * df.loc[i, "P40_60"] for i in indices)
    p80 = quicksum(x[i] * df.loc[i, "P80_100"] for i in indices)
    p99 = quicksum(x[i] * df.loc[i, "P99"] for i in indices)
    
    # Constraint 1: P20 and P40-60 within 1% of each other (protect both equally)
    m_gdp.addConstr(p20 - p40 <= 0.01)  # P20 at most 1% higher than P40-60
    m_gdp.addConstr(p40 - p20 <= 0.01)  # P40-60 at most 1% higher than P20
    
    # Constraint 2: Lower/middle income groups must benefit at least as much as upper groups
    # P20 >= P99 (bottom 20% benefits at least as much as top 1%)
    m_gdp.addConstr(p20 - p99 >= 1e-5)  # Small epsilon to ensure strict inequality
    # P40-60 >= P99 (middle class benefits at least as much as top 1%)
    m_gdp.addConstr(p40 - p99 >= 1e-5)
    # P20 >= P80-100 (bottom 20% benefits at least as much as top 20%)
    m_gdp.addConstr(p20 - p80 >= 1e-5)
    # P40-60 >= P80-100 (middle class benefits at least as much as top 20%)
    m_gdp.addConstr(p40 - p80 >= 1e-5)
    
    # Constraint 3: Combined lower/middle income benefit exceeds combined upper income
    # (P20 + P40-60) >= (P80-100 + P99)
    m_gdp.addConstr(p20 + p40 - p80 - p99 >= 1e-5)
    
    # NS mutual exclusivity constraints
    # For each NS group (e.g., NS1 with options NS1A, NS1B, NS1C),
    # at most one option can be selected
    for group, idxs in ns_groups.items():
        m_gdp.addConstr(quicksum(x[i] for i in idxs) <= 1)
    
    # NS spending constraint
    # Total spending (negative revenue) from NS1-NS7 policies must be at least $3,000B
    # Note: Revenue is negative for spending, so we use <= -3000
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
    
    # New decision variables for Stage 2
    x2 = m_rev.addVars(indices, vtype=GRB.BINARY, name="x")
    
    # Objective: Maximize dynamic revenue (break ties from Stage 1)
    m_rev.setObjective(quicksum(x2[i] * df.loc[i, "DynamicRevenue"] for i in indices), GRB.MAXIMIZE)
    
    # Constraints (same as first pass, but with x2 variables)
    m_rev.addConstr(quicksum(x2[i] * df.loc[i, "DynamicRevenue"] for i in indices) >= 0)
    m_rev.addConstr(quicksum(x2[i] * df.loc[i, "CapitalStock"] for i in indices) >= 0)
    m_rev.addConstr(quicksum(x2[i] * df.loc[i, "Jobs"] for i in indices) >= 0)
    m_rev.addConstr(quicksum(x2[i] * df.loc[i, "WageRate"] for i in indices) >= 0)
    
    # Equity constraints (same as Stage 1)
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
    
    # NS constraints (same as Stage 1)
    for group, idxs in ns_groups.items():
        m_rev.addConstr(quicksum(x2[i] for i in idxs) <= 1)
    
    m_rev.addConstr(quicksum(x2[i] * df.loc[i, "DynamicRevenue"] for i in ns_strict_indices) <= -3000)
    
    # Additional constraint: Fix GDP to the optimal value from Stage 1
    # This ensures we don't sacrifice GDP to gain more revenue
    m_rev.addConstr(quicksum(x2[i] * df.loc[i, "LongRunGDP"] for i in indices) == gdp_star)
    
    # Solve second pass
    m_rev.optimize()
    
    # Extract solution: policies where x2[i] > 0.5 are selected
    # Using 0.5 threshold handles numerical precision issues in binary variables
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
    result_df.to_csv("max_gdp_defense270.csv", index=False)
    print("✓ Results saved to 'max_gdp_defense270.csv'\n")


if __name__ == "__main__":
    main()
