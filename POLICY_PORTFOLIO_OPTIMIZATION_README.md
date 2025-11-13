# Policy Portfolio Optimization Model

## Overview

This optimization model finds the best combination of economic policies to maximize GDP growth while ensuring progressive distribution of benefits, including/excluding specific policies, and maintaining adequate national defense funding across multiple spending scenarios.

## Key Features

### 1. Progressive Distribution Without Positivity Requirement
Unlike the standard [`max_gdp_defense.py`](max_gdp_defense.py:1) model, this optimization:
- **Does NOT require** all income groups to have positive after-tax income changes
- **Does require** that lower-income groups benefit MORE than higher-income groups
- Implements true progressive policy where working families gain relatively more than the wealthy
- Allows all groups to potentially have negative changes, as long as decreases are progressive

### 2. Required Policies
The following policies **MUST** be included in any solution:
- **S14**: Extend expanded ACA premiums & tax credits from the American Rescue Plan
- **28**: Tax Carried Interest as Ordinary Income
- **29**: Raise the Top Capital Gains and Dividend Tax Rate to 30%
- **42**: Eliminate 1031 Like-Kind Exchanges
- **S5**: Two-years of subsidized tuition for students from families earning less than $125,000 enrolled in a four-year HBCU, TCU, or MSI
- **54**: Make Child Tax Credit First Dollar Refundable
- **S9**: Create a national comprehensive paid family and medical leave program

### 3. Prohibited Policies
The following policies **CANNOT** be included in any solution:

**Additional Prohibited Policies (specific to this model):**
- **5**: Lower the Top Rate on Capital Gains and Dividends to 15 Percent
- **19**: Eliminate the Child Tax Credit
- **64**: Eliminate the Net Investment Income Tax
- **63**: Implement $2,500 per Year Roth-style Universal Savings Accounts

**Excluded Policies (from [`max_gdp_defense.py`](max_gdp_defense.py:213-217)):**
- **37**: Corporate Surtax of 5%
- **43**: Enact a 5% VAT
- **49**: Reinstate the Cadillac Tax
- **68**: Replace CIT with 5% VAT

**Total: 8 prohibited policies** (4 additional + 4 excluded)

### 4. Defense Spending Analysis
Like [`max_gdp_defense.py`](max_gdp_defense.py:1), this model:
- Analyzes the full range of defense spending levels: -$4,000B to +$6,000B
- Uses $500B increments for comprehensive scenario analysis
- Generates separate results for each spending level
- Can also run single-level optimization with `--spending` flag

## Mathematical Formulation

### Objective Function
```
Maximize: Σ(GDP_i × x_i) for all policies i
```

### Decision Variables
```
x_i ∈ {0, 1} for each policy i
where x_i = 1 if policy i is selected, 0 otherwise
```

### Constraints

#### 1. Fiscal Constraint
```
Σ(Revenue_i × x_i) ≥ 600
```
Total dynamic revenue must generate at least $600B surplus.

#### 2. Economic Constraints
```
Σ(Capital_i × x_i) ≥ 0
Σ(Jobs_i × x_i) ≥ 0
Σ(Wage_i × x_i) ≥ 0
```
Non-negative impacts on capital stock, job creation, and wage rates.

#### 3. Progressive Distribution Constraints (MODIFIED)
```
Δ After-Tax Income(P20) > Δ After-Tax Income(P80-100) + ε
Δ After-Tax Income(P20) > Δ After-Tax Income(P99) + ε
Δ After-Tax Income(P40-60) > Δ After-Tax Income(P80-100) + ε
Δ After-Tax Income(P40-60) > Δ After-Tax Income(P99) + ε
```
where ε = 1e-5 ensures strict inequality.

**Key Difference**: Unlike [`max_gdp_defense.py`](max_gdp_defense.py:220-225), these constraints do NOT require:
- `Δ After-Tax Income(P20) ≥ 0`
- `Δ After-Tax Income(P40-60) ≥ 0`
- `Δ After-Tax Income(P80-100) ≥ 0`
- `Δ After-Tax Income(P99) ≥ 0`

This allows for solutions where ALL groups may see decreases in after-tax income, as long as the decreases are progressive (lower income groups are hurt less than higher income groups).

#### 4. Required Policy Constraints (NEW)
```
x_i = 1 for i ∈ {S14, 28, 29, 42, S5, 54, S9}
```

#### 5. Prohibited Policy Constraints (EXPANDED)
```
x_i = 0 for i ∈ {5, 19, 64, 63, 37, 43, 49, 68}
```
Combines additional prohibited policies with excluded policies from [`config.py`](config.py:109-114).

#### 6. Policy Mutual Exclusivity (FROM max_gdp_defense.py)
At most one policy can be selected from each of 15 competing policy groups:
1. Corporate Tax Rate/Structure: {11, 36, 68}
2. Gas Tax Increases: {47, 48}
3. Estate Tax: {12, 44, 46, 69}
4. Child Tax Credit - Refundability: {53, 54}
5. Social Security Payroll Tax Cap: {34, 35}
6. Payroll Tax Rate Changes: {4, 33}
7. EITC Reforms: {21, 51, 52, 55, S15}
8. Individual Income Tax Structure: {1, 2, 3, 14, 59}
9. Child Tax Credit - Comprehensive: {19, 20, 55, S13}
10. Section 199A Deduction: {10, 38}
11. Home Mortgage Interest Deduction: {23, 24}
12. Charitable Deduction: {25, 58}
13. Capital Gains Tax Rate: {5, 29, 30}
14. Depreciation/Expensing: {7, 40, 65}
15. Value Added Tax (VAT): {43, 68}

#### 7. Policy Co-Exclusion Rules (FROM max_gdp_defense.py)
```
If x_68 = 1, then x_37 = 0
(If VAT replacement selected, corporate surtax excluded)
```

#### 8. National Security Mutual Exclusivity (FROM max_gdp_defense.py)
At most one policy can be selected from each NS group (NS1, NS2, ..., NS7).

#### 9. National Security Spending Constraint (FROM max_gdp_defense.py)
```
Σ(Revenue_i × x_i) for i ∈ NS1-NS7 = -min_ns_spending
```
Total spending on NS1-NS7 policies must equal exactly the specified defense spending level.
Note: Revenue is negative for spending policies.

## Optimization Approach

### Three-Stage Lexicographic Optimization

**Stage 1: Maximize GDP**
```
Maximize: Σ(GDP_i × x_i)
Subject to: All constraints above
```
Finds the maximum achievable GDP growth.

**Stage 2: Maximize Jobs (Given Optimal GDP)**
```
Maximize: Σ(Jobs_i × x_i)
Subject to: All constraints above
           Σ(GDP_i × x_i) = GDP* (from Stage 1)
```
Among all solutions achieving optimal GDP, finds the one creating the most jobs.

**Stage 3: Maximize Revenue (Given Optimal GDP and Jobs)**
```
Maximize: Σ(Revenue_i × x_i)
Subject to: All constraints above
           Σ(GDP_i × x_i) = GDP* (from Stage 1)
           Σ(Jobs_i × x_i) = Jobs* (from Stage 2)
```
Final tiebreaker ensuring unique solution with highest fiscal surplus.

## Usage

### Full Range Analysis (Default)
```bash
python policy_portfolio_optimization.py
```
Runs optimization for all defense spending levels from -$4,000B to +$6,000B in $500B increments.

### Single Spending Level
```bash
python policy_portfolio_optimization.py --spending 3000
```

### Verbose Output
```bash
python policy_portfolio_optimization.py --verbose
```

### Explicit Full Range
```bash
python policy_portfolio_optimization.py --all
```

## Output

### CSV Files (Per Spending Level)
Results saved to `outputs/portfolio/policy_portfolio_{spending_level}.csv`:
- All selected policies for each defense spending level
- Full economic and distributional impacts for each policy

### Summary Files
- `outputs/portfolio/policy_decisions_matrix.csv`: Policy selection across all spending levels
- `outputs/portfolio/economic_effects_summary.csv`: Economic KPIs for each spending level

### Console Output (Per Spending Level)
- List of required and prohibited policies
- Three-stage optimization progress
- Selected policies with revenue raising vs. reducing classification
- Economic impact summary:
  - Long-Run Change in GDP
  - Capital Stock change
  - Full-Time Equivalent Jobs created
  - Wage Rate change
- After-tax income changes by percentile:
  - P20 (Bottom 20%)
  - P40-60 (Middle Class)
  - P80-100 (Top 20%)
  - P99 (Top 1%)
- Progressive distribution verification
- Revenue impacts (Static and Dynamic)

## Comparison with Other Models

| Feature | [`max_gdp.py`](max_gdp.py:1) | [`max_gdp_defense.py`](max_gdp_defense.py:1) | **policy_portfolio_optimization.py** |
|---------|---------|-----------------|--------------------------------------|
| Maximize GDP | ✓ | ✓ | ✓ |
| Revenue Surplus | ✓ ($600B) | ✓ ($600B) | ✓ ($600B) |
| Economic Non-Negativity | ✗ | ✓ | ✓ |
| Progressive Distribution | ✗ | ✓ | ✓ |
| Positive Income Requirement | ✗ | ✓ (All groups ≥ 0) | ✗ (Progressive only) |
| NS Spending Constraint | ✗ | ✓ (Customizable) | ✓ (Customizable) |
| Required Policies | ✗ | ✗ | ✓ (7 policies) |
| Prohibited Policies | ✗ | ✓ (4 excluded) | ✓ (8 total: 4 additional + 4 excluded) |
| Policy Mutual Exclusivity | ✗ | ✓ (15 groups) | ✓ (15 groups) |
| Policy Co-Exclusions | ✗ | ✓ | ✓ |
| Multi-Level Analysis | ✗ | ✓ (-$4T to +$6T) | ✓ (-$4T to +$6T) |

## Key Differences from max_gdp_defense.py

### **1. No Positive Income Requirement** (CRITICAL DIFFERENCE)
**[`max_gdp_defense.py`](max_gdp_defense.py:220-225)**:
```python
# Non-negative after-tax income for all groups (everyone must be better off)
model.addConstr(p20 >= 0, name="P20_NonNegative")
model.addConstr(p40 >= 0, name="P40_NonNegative")
model.addConstr(p80 >= 0, name="P80_NonNegative")
model.addConstr(p99 >= 0, name="P99_NonNegative")
```

**This model**: These constraints are **REMOVED**. Only progressive distribution is enforced:
```python
# Progressive distribution only (no positivity requirement)
model.addConstr(p20 - p80 >= epsilon, name="P20_gt_P80")
model.addConstr(p20 - p99 >= epsilon, name="P20_gt_P99")
model.addConstr(p40 - p80 >= epsilon, name="P40_gt_P80")
model.addConstr(p40 - p99 >= epsilon, name="P40_gt_P99")
```

**Impact**: Allows solutions where ALL income groups may have negative after-tax income changes, as long as:
- P20 is hurt less than P80-100 and P99
- P40-60 is hurt less than P80-100 and P99

### **2. Required Policies** (NEW CONSTRAINT)
**[`max_gdp_defense.py`](max_gdp_defense.py:1)**: No required policies

**This model**: 7 specific policies must be included:
```python
x_i = 1 for i ∈ {S14, 28, 29, 42, S5, 54, S9}
```

### **3. Expanded Prohibited Policies**
**[`max_gdp_defense.py`](max_gdp_defense.py:213-217)**: Excludes 4 policies (37, 43, 49, 68)

**This model**: Excludes 8 policies total:
- Original 4 excluded: {37, 43, 49, 68}
- Additional 4 prohibited: {5, 19, 64, 63}

### **4. Defense Spending Analysis**
Both models analyze the same range:
- Range: -$4,000B to +$6,000B
- Increment: $500B
- Total scenarios: 21 different spending levels
- Same as [`max_gdp_defense.py`](max_gdp_defense.py:509-512)

## Dependencies

- Python 3.8+
- gurobipy (Gurobi Optimizer - license required)
- pandas
- Standard library: argparse, sys, traceback, typing, pathlib

## Configuration

### Policy Requirements (in script)
```python
REQUIRED_POLICIES = ["S14", "28", "29", "42", "S5", "54", "S9"]
ADDITIONAL_PROHIBITED_POLICIES = ["5", "19", "64", "63"]
```

### Global Settings (from [`config.py`](config.py:1))
```python
REVENUE_SURPLUS_REQUIREMENT = 600  # $600B minimum
EPSILON = 1e-5  # For strict inequality
EXCLUDED_POLICIES = ["37", "43", "49", "68"]
POLICY_CO_EXCLUSIONS = [("68", "37")]
```

### Defense Spending Range (from [`config.py`](config.py:89-95))
```python
SPENDING_RANGE = {
    "min": -4000,   # -$4,000B
    "max": 6500,    # +$6,500B (exclusive)
    "step": 500,    # $500B increments
}
```

## Data Requirements

Requires [`tax reform & spending menu options (v8) template.xlsx`](tax reform & spending menu options (v8) template.xlsx:1) with columns:
- Option (policy name/code)
- Long-Run Change in GDP
- Capital Stock
- Full-Time Equivalent Jobs
- Wage Rate
- P20, P40-60, P80-100, P99 (distributional effects)
- Static 10-Year Revenue (billions)
- Dynamic 10-Year Revenue (billions)

## Complete Constraint Set

This model inherits **ALL** constraints from [`max_gdp_defense.py`](max_gdp_defense.py:1) with modifications:

### From max_gdp_defense.py (UNCHANGED):
1. ✓ Fiscal: Revenue surplus ≥ $600B
2. ✓ Economic: Non-negative capital, jobs, wages
3. ✓ Policy exclusions: {37, 43, 49, 68}
4. ✓ Policy mutual exclusivity: 15 groups
5. ✓ Policy co-exclusions: (68, 37)
6. ✓ NS mutual exclusivity: One per NS group
7. ✓ NS spending constraint: Exact spending on NS1-NS7

### NEW Constraints (This Model):
8. ✓ Required policies: {S14, 28, 29, 42, S5, 54, S9}
9. ✓ Additional prohibited: {5, 19, 64, 63}

### MODIFIED Constraint (Key Difference):
10. **Progressive distribution WITHOUT positivity**:
   - REMOVED: `p20 >= 0, p40 >= 0, p80 >= 0, p99 >= 0`
   - KEPT: `p20 > p80, p20 > p99, p40 > p80, p40 > p99`

## Error Handling

The model includes comprehensive error handling:
- **Input Validation**: Checks for required data columns and valid policy codes
- **Optimization Status**: Reports infeasibility, unboundedness, or other solver issues
- **Progressive Constraint Failures**: Indicates if progressive distribution cannot be achieved
- **Required Policy Conflicts**: Detects if required policies cannot be satisfied
- **NS Spending Infeasibility**: Reports when defense spending level cannot be met
- **Gurobi License**: Catches and reports license-related errors

## Example Output

```
Policy Portfolio Optimization
  NS spending requirement: $3,000B
  Required policies: 7 policies must be included
    - S14: Extend expanded ACA premiums & tax credits from the American Rescue Plan
    - 28: Tax Carried Interest as Ordinary Income
    - 29: Raise the Top Capital Gains and Dividend Tax Rate to 30%
    - 42: Eliminate 1031 Like-Kind Exchanges
    - S5: Two-years of subsidized tuition for students from families earning less than $125,000...
    - 54: Make Child Tax Credit First Dollar Refundable
    - S9: Create a national comprehensive paid family and medical leave program

  Prohibited policies: 8 policies cannot be included
    - 5: Lower the Top Rate on Capital Gains and Dividends to 15 Percent
    - 19: Eliminate the Child Tax Credit
    - 64: Eliminate the Net Investment Income Tax
    - 63: Implement $2,500 per Year Roth-style Universal Savings Accounts
    - 37: Corporate Surtax of 5%
    - 43: Enact a 5% VAT
    - 49: Reinstate the Cadillac Tax
    - 68: Replace CIT with 5% VAT

Running optimization (Stage 1: Maximize GDP)...
Running optimization (Stage 2: Maximize Jobs)...
Running optimization (Stage 3: Maximize Revenue)...

[Three-Stage Optimization Results]
  Stage 1 - Optimal GDP: +0.2345%
  Stage 2 - Optimal Jobs: 456,789
  Stage 3 - Optimal Revenue: $987.65B

[Progressive Distribution Verification]
  P20 (Bottom 20%):    -0.0023%  ← Can be negative
  P40-60 (Middle):     -0.0045%  ← Can be negative
  P80-100 (Top 20%):   -0.0078%  ← MORE negative than lower groups
  P99 (Top 1%):        -0.0120%  ← MOST negative (wealthy hurt most)

  [Progressive Constraints Check]
  P20 > P80-100: -0.0023% > -0.0078% = True  ✓ (P20 less hurt)
  P20 > P99:     -0.0023% > -0.0120% = True  ✓ (P20 less hurt)
  P40 > P80-100: -0.0045% > -0.0078% = True  ✓ (P40 less hurt)
  P40 > P99:     -0.0045% > -0.0120% = True  ✓ (P40 less hurt)

[OK] Results saved to 'outputs/senate/policy_portfolio_3000.csv'
```

## Full Range Output Structure

When running without `--spending` flag, generates:

### Individual Result Files (21 files)
```
outputs/senate/policy_portfolio_-4000.csv
outputs/senate/policy_portfolio_-3500.csv
...
outputs/senate/policy_portfolio_3000.csv
...
outputs/senate/policy_portfolio_6000.csv
```

### Summary Files
```
outputs/senate/policy_decisions_matrix.csv
outputs/senate/economic_effects_summary.csv
```

### Visualization Files
```
outputs/senate/defense_spending_analysis.png
outputs/senate/policy_selection_heatmap.png
```

## Technical Notes

1. **Epsilon for Strict Inequality**: ε = 1e-5 ensures numerical precision doesn't allow equality when strict inequality is required

2. **Lexicographic Optimization**: The three-stage approach ensures:
   - First priority: Maximum GDP
   - Second priority: Maximum jobs (among max GDP solutions)
   - Third priority: Maximum revenue (among max GDP, max jobs solutions)

3. **Binary Threshold**: Solutions use 0.5 threshold for binary variables to handle numerical precision

4. **Defense Spending Encoding**: NS policies have negative revenue values. A $3,000B spending requirement means `Σ(Revenue_i × x_i) = -3000` for NS1-NS7 policies.

5. **Progressive vs. Positive**: The model enforces **progressive** distribution (relative fairness) but NOT **positive** changes (absolute improvement). This allows the optimizer more flexibility to find GDP-maximizing solutions.

## Interpreting Results

### Progressive Distribution Examples

**Scenario A: All Positive, Progressive**
- P20: +2.5% ✓ (best off)
- P40-60: +1.8% ✓
- P80-100: +0.9% ✓
- P99: +0.5% ✓ (least benefit)
- **Result**: Valid - everyone benefits, lower groups benefit more

**Scenario B: Mixed, Progressive**
- P20: +1.2% ✓ (best off)
- P40-60: +0.8% ✓
- P80-100: -0.3% ✓
- P99: -0.9% ✓ (worst off)
- **Result**: Valid - lower groups benefit, wealthy hurt

**Scenario C: All Negative, Progressive**
- P20: -0.5% ✓ (least hurt)
- P40-60: -0.8% ✓
- P80-100: -1.5% ✓
- P99: -2.3% ✓ (most hurt)
- **Result**: Valid - everyone worse off, but progressively so

**Scenario D: Non-Progressive (INVALID)**
- P20: +0.5%
- P40-60: +0.3%
- P80-100: +1.2% ✗ (higher than P20 and P40-60)
- P99: +2.0% ✗ (highest benefit)
- **Result**: Invalid - violated progressive constraints

## License

Part of the IJAB Economic Scenario Analysis project.

## Related Files

- [`config.py`](config.py:1): Global configuration settings
- [`optimizer_utils.py`](optimizer_utils.py:1): Shared optimization utilities
- [`utils.py`](utils.py:1): Data loading and display utilities
- [`logger.py`](logger.py:1): Logging infrastructure
- [`validation.py`](validation.py:1): Input validation functions
- [`max_gdp_defense.py`](max_gdp_defense.py:1): Base model this is derived from
