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
    
    # Extract solution and all metrics for selected policies
    selected_indices = [i for i in range(n) if x2[i].X > 0.5]
    selected_df = df_clean.iloc[selected_indices].copy()
    
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
