"""
Policy Optimization Script
Maximizes GDP growth subject to revenue neutrality constraint and National Security constraints

This script maximizes GDP growth subject to:
- Fiscal constraint: Revenue neutrality (no increase in deficit)
- National Security (NS) constraints: Mutual exclusivity within NS policy groups

Key Features:
1. Revenue Neutrality: Ensures total dynamic revenue is non-negative
2. National Security (NS) Constraints:
   - NS policy groups (e.g., NS1A, NS1B, NS1C): Only one option per group can be selected
   - Prevents selecting conflicting national security policies
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
    Two-stage optimization to find optimal policy package.
    
    Stage 1: Maximize GDP subject to revenue neutrality and NS mutual exclusivity
        - Finds the maximum achievable GDP growth
        - Ensures total dynamic revenue is non-negative (revenue neutral or positive)
        - Enforces NS mutual exclusivity: at most one policy per NS group
        
    Stage 2: Maximize revenue while maintaining optimal GDP
        - Among all solutions that achieve the optimal GDP from Stage 1
        - Selects the one with the highest revenue surplus
        - This breaks ties when multiple policy combinations achieve the same GDP
    
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
    
    # === Stage 1: Maximize GDP subject to revenue constraint ===
    if verbose:
        print("Running optimization (Stage 1: Maximize GDP)...")
    
    # Create Gurobi optimization model
    model = Model("GDP_Maximization")
    if SUPPRESS_GUROBI_OUTPUT:
        model.setParam('OutputFlag', 0)
    
    # Decision variables: x[i] = 1 if policy i is selected, 0 otherwise
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    
    # Constraint: Total dynamic revenue must be non-negative (revenue neutral)
    # This ensures the policy package doesn't increase the deficit
    model.addConstr(quicksum(revenue[i] * x[i] for i in range(n)) >= 0, name="RevenueNeutrality")
    
    # NS mutual exclusivity constraints
    # For each NS group (e.g., NS1 with options NS1A, NS1B, NS1C),
    # at most one option can be selected
    if verbose and ns_groups:
        print(f"  Adding {len(ns_groups)} NS mutual exclusivity constraints...")
    for group, idxs in ns_groups.items():
        model.addConstr(quicksum(x[i] for i in idxs) <= 1, name=f"NS_{group}_mutual_exclusivity")
    
    # Objective: Maximize total GDP impact
    model.setObjective(quicksum(gdp[i] * x[i] for i in range(n)), GRB.MAXIMIZE)
    model.optimize()
    
    # Store the optimal GDP value for use in Stage 2
    best_gdp = model.ObjVal
    
    # === Stage 2: Maximize revenue while maintaining optimal GDP ===
    # This stage breaks ties when multiple solutions achieve the same GDP
    if verbose:
        print("Running optimization (Stage 2: Maximize Revenue)...")
    
    model_2 = Model("Revenue_Maximization")
    if SUPPRESS_GUROBI_OUTPUT:
        model_2.setParam('OutputFlag', 0)
    
    # New decision variables for Stage 2
    x2 = model_2.addVars(n, vtype=GRB.BINARY, name="x")
    
    # Constraint: Revenue neutrality (same as Stage 1)
    model_2.addConstr(quicksum(revenue[i] * x2[i] for i in range(n)) >= 0, name="RevenueNeutrality")
    
    # Constraint: Must achieve exactly the optimal GDP from Stage 1
    # This ensures we don't sacrifice GDP to gain more revenue
    model_2.addConstr(quicksum(gdp[i] * x2[i] for i in range(n)) == best_gdp, name="GDPMatch")
    
    # NS mutual exclusivity constraints (same as Stage 1)
    for group, idxs in ns_groups.items():
        model_2.addConstr(quicksum(x2[i] for i in idxs) <= 1, name=f"NS_{group}_mutual_exclusivity")
    
    # Objective: Maximize revenue surplus (among optimal GDP solutions)
    model_2.setObjective(quicksum(revenue[i] * x2[i] for i in range(n)), GRB.MAXIMIZE)
    model_2.optimize()
    
    # Extract solution: policies where x2[i] > 0.5 are selected
    # Using 0.5 threshold handles numerical precision issues in binary variables
    selected_indices = [i for i in range(n) if x2[i].X > 0.5]
    selected_df = df_clean.iloc[selected_indices].copy()
    
    # Verify NS mutual exclusivity in solution
    if verbose and ns_groups:
        print("\nVerifying NS mutual exclusivity in solution:")
        violations = []
        for group, idxs in sorted(ns_groups.items()):
            selected_in_group = [i for i in idxs if i in selected_indices]
            if len(selected_in_group) > 1:
                policies = [df_clean.iloc[i]["Option"].split(":")[0] for i in selected_in_group]
                violations.append(f"  ✗ {group}: {len(selected_in_group)} policies selected ({', '.join(policies)})")
            elif len(selected_in_group) == 1:
                policy = df_clean.iloc[selected_in_group[0]]["Option"].split(":")[0]
                print(f"  ✓ {group}: 1 policy selected ({policy})")
            else:
                print(f"  ✓ {group}: 0 policies selected")
        
        if violations:
            print("\n⚠ WARNING: NS MUTUAL EXCLUSIVITY VIOLATIONS DETECTED:")
            for v in violations:
                print(v)
        else:
            print("  All NS constraints satisfied! ✓")
    
    return selected_df, best_gdp, model_2.ObjVal


def display_results(result_df, gdp_impact, revenue_impact):
    """Display optimization results in a readable format."""
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS".center(80))
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
    print(f"  P20 (Bottom 20%):                    {result_df['P20'].sum() * 100:>+8.4f}%")
    print(f"  P40-60 (Middle Class):               {result_df['P40-60'].sum() * 100:>+8.4f}%")
    print(f"  P80-100 (Top 20%):                   {result_df['P80-100'].sum() * 100:>+8.4f}%")
    print(f"  P99 (Top 1%):                        {result_df['P99'].sum() * 100:>+8.4f}%")
    
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
    
    # Optionally save to CSV
    result_df.to_csv("max_gdp.csv", index=False)
    print("✓ Results saved to 'max_gdp.csv'\n")


if __name__ == "__main__":
    main()
