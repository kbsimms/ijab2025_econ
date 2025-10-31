# IJAB Economic Scenario Analysis

Optimization scripts for analyzing economic policy scenarios using linear programming. These scripts use the Gurobi optimizer to select optimal policy combinations that maximize GDP growth while satisfying various constraints.

## Overview

This project provides three optimization scripts, each with different constraint configurations:

1. **max_gdp.py** - Basic GDP maximization with revenue neutrality
2. **max_gdp_equal_distro.py** - GDP maximization with distributional equality
3. **max_gdp_defense270.py** - GDP maximization with national security and equity requirements

All scripts share common features including revenue neutrality and National Security (NS) mutual exclusivity constraints.

## Common Features

### 1. Revenue Neutrality
All scripts ensure fiscal responsibility by requiring that the total dynamic revenue impact is non-negative (≥ $0 billion). This prevents selecting policy packages that would increase the federal deficit.

### 2. National Security (NS) Mutual Exclusivity
**NEW:** All scripts now include NS mutual exclusivity constraints to prevent selecting conflicting national security policies.

- **How it works:** Policies are grouped by their NS prefix (e.g., NS1, NS2, NS3, etc.)
- **Constraint:** At most one policy can be selected from each NS group
- **Example:** If policies NS1A, NS1B, and NS1C exist, only one of these three can be selected
- **Purpose:** Ensures coherent national security strategy by preventing contradictory policy selections

### 3. Two-Stage Optimization
All scripts use a sophisticated two-stage approach:
- **Stage 1:** Find the maximum achievable GDP growth given all constraints
- **Stage 2:** Among solutions achieving optimal GDP, select the best according to secondary criteria (varies by script)

## Scripts

### max_gdp.py - Basic GDP Maximization

**Objective:** Maximize GDP growth

**Constraints:**
- Revenue neutrality (total dynamic revenue ≥ $0)
- NS mutual exclusivity (at most one policy per NS group)

**Tiebreaking:** Maximizes revenue surplus among optimal GDP solutions

**Use Case:** Finding the highest GDP growth achievable while maintaining fiscal responsibility and coherent national security policies

**Run:**
```bash
python max_gdp.py
```

**Output:** `max_gdp.csv`

---

### max_gdp_equal_distro.py - Distributional Equality

**Objective:** Maximize GDP growth with fair distribution across income groups

**Constraints:**
- Revenue neutrality (total dynamic revenue ≥ $0)
- NS mutual exclusivity (at most one policy per NS group)
- Distributional equality: All income groups (P20, P40-60, P80-100, P99) must experience after-tax income changes within 1 percentage point of each other

**Tiebreaking:** Prioritizes lower-income benefits using lexicographic weighting
- P20 (bottom 20%) gets highest priority
- Then P40-60 (middle class)
- Then P80-100 (top 20%)
- Finally P99 (top 1%)

**Use Case:** Finding GDP-maximizing policies that benefit all income groups equally

**Run:**
```bash
python max_gdp_equal_distro.py
```

**Output:** `max_gdp_equal_distro.csv`

---

### max_gdp_defense270.py - National Security & Equity Focus

**Objective:** Maximize GDP growth with strong national security funding and progressive distribution

**Constraints:**
- Fiscal: Revenue neutrality (total dynamic revenue ≥ $0)
- Economic: Non-negative capital stock, jobs, and wage rate
- Equity: Progressive distribution
  - P20 and P40-60 must be within 1% of each other
  - Lower/middle income groups (P20, P40-60) must benefit at least as much as upper groups (P80-100, P99)
  - Combined lower/middle income benefit must exceed combined upper income benefit
- National Security:
  - NS mutual exclusivity (at most one policy per NS group)
  - Minimum $3,000 billion spending on NS1-NS7 policies

**Tiebreaking:** Maximizes revenue surplus among optimal GDP solutions

**Use Case:** Finding GDP-maximizing policies that ensure robust national security funding while maintaining progressive distributional impacts

**Run:**
```bash
python max_gdp_defense270.py
```

**Output:** `max_gdp_defense270.csv`

## Installation

### Prerequisites
- Python 3.8+
- Gurobi Optimizer (requires license)
- Required packages: `pandas`, `gurobipy`, `openpyxl`

### Setup
```bash
# Install dependencies
pip install pandas gurobipy openpyxl

# Or using uv (if available)
uv sync
```

## Input Data

All scripts read from: `tax reform & spending menu options (v8) template.xlsx`

**Expected Excel Format:**
- Sheet name: `Sheet1`
- Headers in row 2 (index 1)
- Columns:
  - `Option`: Policy name/description
  - `Long-Run Change in GDP`: GDP impact (decimal, e.g., 0.05 for 5%)
  - `Capital Stock`: Capital stock impact (decimal)
  - `Full-Time Equivalent Jobs`: Job creation (number)
  - `Wage Rate`: Wage rate impact (decimal)
  - `P20`, `P40-60`, `P80-100`, `P99`: After-tax income changes by percentile (decimal)
  - `Static 10-Year Revenue (billions)`: Static revenue estimate
  - `Dynamic 10-Year Revenue (billions)`: Dynamic revenue estimate

**National Security Policies:**
- Must follow naming convention: `NSxY: Description`
  - Where `x` is one or more digits (e.g., 1, 2, 7, 15)
  - Where `Y` is a letter (A, B, C, etc.)
  - Example: `NS1A: Increase defense spending by 10%`
  - Example: `NS1B: Increase defense spending by 20%`
  - Example: `NS2A: Modernize nuclear arsenal`

## Output

Each script generates:
1. **CSV file** with selected policies and their impacts
2. **Console output** showing:
   - Revenue raising policies
   - Revenue reducing policies
   - Total economic impacts (GDP, capital, jobs, wages)
   - Distributional impacts by income group
   - Revenue impacts (static and dynamic)
   - Number of selected policies

## Understanding the Results

### Economic Impacts
- **GDP Impact:** Long-run change in GDP (percentage)
- **Capital Stock:** Change in capital stock (percentage)
- **Jobs:** Full-time equivalent jobs created (number)
- **Wage Rate:** Change in wage rate (percentage)

### Distributional Impacts
- **P20:** After-tax income change for bottom 20% of earners
- **P40-60:** After-tax income change for middle 40% of earners
- **P80-100:** After-tax income change for top 20% of earners
- **P99:** After-tax income change for top 1% of earners

### Revenue Impacts
- **Static Revenue:** Revenue estimate without behavioral responses
- **Dynamic Revenue:** Revenue estimate including economic behavioral responses
- **Positive values:** Revenue-raising (reduces deficit)
- **Negative values:** Revenue-reducing (increases deficit)

## National Security (NS) Mutual Exclusivity

### How It Works

The NS mutual exclusivity constraint ensures coherent national security policy by preventing selection of conflicting options within the same category.

**Example:**
```
NS1A: Increase defense spending by 10%
NS1B: Increase defense spending by 20%
NS1C: Maintain current defense spending
```
Only one of these three can be selected (they're mutually exclusive).

**Another Example:**
```
NS2A: Modernize nuclear arsenal (aggressive timeline)
NS2B: Modernize nuclear arsenal (moderate timeline)
NS2C: Delay nuclear modernization
```
Only one option from the NS2 group can be selected.

### Implementation Details

- Automatically detects NS policies based on naming pattern: `NSxY:` where x=digits, Y=letter
- Groups policies by their numeric prefix (NS1, NS2, NS3, etc.)
- Adds constraint: `sum(selected_policies_in_group) ≤ 1`
- Applied in both optimization stages for consistency

### Benefits

1. **Prevents Contradictions:** Can't select both "increase spending" and "decrease spending"
2. **Ensures Coherence:** National security strategy remains logically consistent
3. **Realistic Scenarios:** Mirrors real-world policy-making where mutually exclusive choices must be made
4. **Flexible Grouping:** Automatically handles any number of NS groups and options per group

## Technical Details

### Optimization Framework
- **Solver:** Gurobi (commercial-grade integer programming solver)
- **Problem Type:** Binary Integer Linear Programming (BILP)
- **Variables:** Binary (0 or 1) for each policy
- **Objective:** Linear (weighted sum of policy impacts)
- **Constraints:** Linear inequalities and equalities

### Two-Stage Approach Benefits
1. **Stage 1** finds the frontier of what's possible
2. **Stage 2** intelligently breaks ties using secondary objectives
3. Ensures globally optimal solutions (not just locally optimal)

## Troubleshooting

### Common Issues

**"Gurobi license not found"**
- Ensure you have a valid Gurobi license installed
- Academic licenses are free: https://www.gurobi.com/academia/

**"Model is infeasible"**
- One or more constraints cannot be satisfied simultaneously
- Try relaxing constraints (e.g., increase distributional tolerance)
- Check that NS policies are properly formatted

**"No NS policy groups detected"**
- Verify policy names follow `NSxY:` pattern (e.g., `NS1A:`, `NS2B:`)
- Check that colon (`:`) is present after NS code
- Ensure policies exist in the Excel file

## Contributing

When adding new optimization scripts:
1. Include NS mutual exclusivity constraints
2. Use two-stage optimization approach
3. Document all constraints clearly
4. Follow existing code structure and naming conventions

## License

[Specify license here]

## Contact

[Add contact information]

## Version History

- **v2.0** (Current): Added NS mutual exclusivity constraints to all scripts
- **v1.0**: Initial release with three optimization scripts