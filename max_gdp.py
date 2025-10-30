"""
Policy Optimization Script
Maximizes GDP growth subject to revenue neutrality constraint
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
    
    # Convert relevant columns to numeric
    cols_to_convert = [
        "Long-Run Change in GDP",
        "Dynamic 10-Year Revenue (billions)"
    ]
    df_clean[cols_to_convert] = df_clean[cols_to_convert].apply(pd.to_numeric, errors='coerce')
    
    print(f"✓ Loaded {len(df_clean)} policy options\n")
    return df_clean


def optimize_policy_selection(df_clean, verbose=True):
    """
    Two-stage optimization:
    1. Maximize GDP subject to revenue neutrality
    2. Maximize revenue while maintaining optimal GDP
    """
    # Extract data
    n = len(df_clean)
    gdp = df_clean["Long-Run Change in GDP"].values
    revenue = df_clean["Dynamic 10-Year Revenue (billions)"].values
    policy_names = df_clean["Option"].values
    
    # Stage 1: Maximize GDP subject to revenue constraint
    if verbose:
        print("Running optimization (Stage 1: Maximize GDP)...")
    
    model = Model("GDP_Maximization")
    if SUPPRESS_GUROBI_OUTPUT:
        model.setParam('OutputFlag', 0)
    
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    model.addConstr(quicksum(revenue[i] * x[i] for i in range(n)) >= 0, name="RevenueNeutrality")
    model.setObjective(quicksum(gdp[i] * x[i] for i in range(n)), GRB.MAXIMIZE)
    model.optimize()
    
    best_gdp = model.ObjVal
    
    # Stage 2: Maximize revenue while maintaining optimal GDP
    if verbose:
        print("Running optimization (Stage 2: Maximize Revenue)...")
    
    model_2 = Model("Revenue_Maximization")
    if SUPPRESS_GUROBI_OUTPUT:
        model_2.setParam('OutputFlag', 0)
    
    x2 = model_2.addVars(n, vtype=GRB.BINARY, name="x")
    model_2.addConstr(quicksum(revenue[i] * x2[i] for i in range(n)) >= 0, name="RevenueNeutrality")
    model_2.addConstr(quicksum(gdp[i] * x2[i] for i in range(n)) == best_gdp, name="GDPMatch")
    model_2.setObjective(quicksum(revenue[i] * x2[i] for i in range(n)), GRB.MAXIMIZE)
    model_2.optimize()
    
    # Extract solution
    selected_policies = []
    for i in range(n):
        if x2[i].X > 0.5:
            selected_policies.append({
                "Policy": policy_names[i],
                "GDP Impact (%)": gdp[i],
                "Revenue Impact ($B)": revenue[i]
            })
    
    return pd.DataFrame(selected_policies), best_gdp, model_2.ObjVal


def display_results(result_df, gdp_impact, revenue_impact):
    """Display optimization results in a readable format."""
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS".center(80))
    print("="*80)
    
    print(f"\n{'SUMMARY':^80}")
    print("-"*80)
    print(f"  Maximum GDP Impact:        {gdp_impact:>8.4f}%")
    print(f"  Total Revenue Impact:      ${revenue_impact:>8.2f} billion")
    print(f"  Number of Policies:        {len(result_df):>8}")
    print("-"*80)
    
    # Separate policies by positive/negative revenue impact
    positive_revenue = result_df[result_df["Revenue Impact ($B)"] >= 0].copy()
    negative_revenue = result_df[result_df["Revenue Impact ($B)"] < 0].copy()
    
    # Sort by absolute impact
    positive_revenue = positive_revenue.sort_values("Revenue Impact ($B)", ascending=False)
    negative_revenue = negative_revenue.sort_values("Revenue Impact ($B)", ascending=True)
    
    if len(positive_revenue) > 0:
        print(f"\n{'REVENUE RAISING POLICIES':^80}")
        print("-"*80)
        for _, row in positive_revenue.iterrows():
            print(f"  {row['Policy'][:70]:<70}")
            print(f"    GDP: {row['GDP Impact (%)']:>+7.4f}%  |  Revenue: ${row['Revenue Impact ($B)']:>8.2f}B")
    
    if len(negative_revenue) > 0:
        print(f"\n{'REVENUE REDUCING POLICIES':^80}")
        print("-"*80)
        for _, row in negative_revenue.iterrows():
            print(f"  {row['Policy'][:70]:<70}")
            print(f"    GDP: {row['GDP Impact (%)']:>+7.4f}%  |  Revenue: ${row['Revenue Impact ($B)']:>8.2f}B")
    
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
    result_df.to_csv("max_gdp.csv", index=False)
    print("✓ Results saved to 'max_gdp.csv'\n")


if __name__ == "__main__":
    main()
