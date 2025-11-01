# IJAB Economic Scenario Analysis

Optimization scripts for analyzing economic policy scenarios using linear programming. These scripts use the Gurobi optimizer to select optimal policy combinations that maximize GDP growth while satisfying various constraints.

## Overview

This project provides optimization scripts for analyzing economic policy scenarios:

1. **max_gdp.py** - Basic GDP maximization with revenue neutrality
2. **max_gdp_defense.py** - GDP maximization with national security and equity requirements (parameterized for various spending levels)

Additionally, two visualization scripts are included for analyzing and comparing results:

3. **visualize_defense_spending.py** - Generates comprehensive charts analyzing economic effects across defense spending levels
4. **visualize_policy_selection.py** - Creates heatmaps showing which policies are selected at different spending levels

All scripts share common infrastructure through centralized configuration and utility modules for consistency and maintainability.

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Gurobi Optimizer (requires license - free academic licenses available)
- Excel file: `tax reform & spending menu options (v8) template.xlsx`

### Installation

```bash
# Install dependencies
pip install pandas gurobipy openpyxl matplotlib seaborn

# Or using uv (if available)
uv sync
```

### Basic Usage

**1. Run basic GDP optimization:**

```bash
python max_gdp.py
```

Output: `max_gdp.csv` with selected policies

**2. Run defense spending analysis (full range):**

```bash
python max_gdp_defense.py
```

Output: Multiple CSV files in `outputs/defense/` directory + visualization

**3. Run single defense spending level:**

```bash
python max_gdp_defense.py --spending 3000
```

Output: `outputs/defense/max_gdp_defense3000.csv`

**4. Visualize existing defense results:**

```bash
python visualize_defense_spending.py
```

Output: `outputs/defense/defense_spending_analysis.png`

**5. Analyze policy selections across spending levels:**

```bash
python visualize_policy_selection.py
```

Output: `outputs/defense/policy_selection_heatmap.png`

## Project Structure

```
ijab_econ_scenario_analysis/
├── Core Modules
│   ├── config.py                  # Configuration and constants
│   ├── validation.py              # Input validation framework
│   ├── logger.py                  # Structured logging
│   ├── optimizer_utils.py         # Gurobi constraint builders
│   └── utils.py                   # Data loading and display
│
├── Optimization Scripts
│   ├── max_gdp.py                 # Basic GDP maximization
│   └── max_gdp_defense.py         # GDP with defense & equity
│
├── Visualization Scripts
│   ├── visualize_defense_spending.py  # Economic effects charts
│   └── visualize_policy_selection.py  # Policy selection heatmap
│
├── Configuration
│   ├── pyproject.toml             # Python dependencies
│   └── .gitignore                 # Git ignore rules
│
├── Data
│   └── tax reform & spending menu options (v8) template.xlsx
│
├── Documentation
│   └── README.md                  # This file
│
└── outputs/defense/               # Generated results
    ├── max_gdp_defense*.csv       # Optimization results
    ├── defense_spending_analysis.png
    └── policy_selection_heatmap.png
```

## Common Features

### 1. Revenue Neutrality

All scripts ensure fiscal responsibility by requiring that the total dynamic revenue impact is non-negative (≥ $0 billion). This prevents selecting policy packages that would increase the federal deficit.

### 2. National Security (NS) Mutual Exclusivity

All scripts include NS mutual exclusivity constraints to prevent selecting conflicting national security policies.

- **How it works:** Policies are grouped by their NS prefix (e.g., NS1, NS2, NS3, etc.)
- **Constraint:** At most one policy can be selected from each NS group
- **Example:** If policies NS1A, NS1B, and NS1C exist, only one of these three can be selected
- **Purpose:** Ensures coherent national security strategy by preventing contradictory policy selections

### 3. Two-Stage Optimization

All scripts use a sophisticated two-stage approach:

- **Stage 1:** Find the maximum achievable GDP growth given all constraints
- **Stage 2:** Among solutions achieving optimal GDP, select the best according to secondary criteria (varies by script)

### 4. Consistent Column Naming

All scripts now use standardized column names from `config.py`, ensuring consistency and making maintenance easier.

## Scripts

### max_gdp.py - Basic GDP Maximization

**Purpose:** Finds the policy combination that maximizes GDP growth while maintaining fiscal responsibility and coherent national security policies.

**Objective:** Maximize GDP growth

**Constraints:**

- Revenue neutrality (total dynamic revenue ≥ $0)
- NS mutual exclusivity (at most one policy per NS group)

**Tiebreaking:** Maximizes revenue surplus among optimal GDP solutions

**How to Run:**

```bash
python max_gdp.py
```

**What it Does:**

1. Loads policy data from Excel file
2. Identifies National Security policy groups
3. Stage 1: Finds maximum achievable GDP growth
4. Stage 2: Among optimal solutions, selects one with highest revenue surplus
5. Displays results to console
6. Saves selected policies to CSV

**Output:** `max_gdp.csv` containing:

- All selected policy options
- Their individual impacts (GDP, revenue, jobs, etc.)
- Distributional effects by income group

**Console Output Includes:**

- Revenue raising vs. reducing policies
- Total GDP impact
- Capital stock change
- Jobs created
- Wage rate change
- After-tax income changes for all income groups (P20, P40-60, P80-100, P99)
- Static and dynamic revenue impacts
- Number of policies selected

---

### max_gdp_defense.py - National Security & Equity Focus

**Purpose:** Finds the policy combination that maximizes GDP while ensuring robust national security funding and progressive distribution of benefits.

**Objective:** Maximize GDP growth with strong national security funding and progressive distribution

**Constraints:**

- **Fiscal:** Revenue neutrality (total dynamic revenue ≥ $0)
- **Economic:** Non-negative capital stock, jobs, and wage rate
- **Equity:** Progressive distribution
  - P20 and P40-60 must benefit at least as much as P80-100 and P99
  - All income groups must have non-negative after-tax income effects
- **Policy:** Certain policies excluded, mutual exclusivity groups enforced
- **National Security:**
  - NS mutual exclusivity (at most one policy per NS group)
  - Configurable minimum spending on NS1-NS7 policies

**Tiebreaking:** Maximizes revenue surplus among optimal GDP solutions

**How to Run:**

```bash
# Default: Run full range (-4000 to 6000 in 500B increments) + visualization
python max_gdp_defense.py

# Single optimization with specific spending requirement
python max_gdp_defense.py --spending 3000

# Explicit full range (same as default)
python max_gdp_defense.py --all
```

**What it Does:**

**Single Run Mode** (`--spending AMOUNT`):

1. Loads policy data from Excel
2. Identifies NS policy groups and NS1-NS7 policies
3. Defines policy mutual exclusivity groups (15 groups)
4. Stage 1: Finds maximum GDP with all constraints
5. Stage 2: Maximizes revenue while maintaining optimal GDP
6. Displays detailed results to console
7. Saves results to `outputs/defense/max_gdp_defense{spending}.csv`

**Full Range Mode** (default):

1. Runs optimization for each spending level from -$4,000B to +$6,000B in $500B increments
2. Saves individual CSV files for each spending level
3. Generates summary files:
   - `policy_decisions_matrix.csv` - Which policies selected at each level
   - `economic_effects_summary.csv` - KPI values across all levels
4. Automatically calls visualization script to generate charts

**Outputs:**

- **CSV files:** `outputs/defense/max_gdp_defense{spending}.csv` for each spending level
- **Summary files:** Policy decision matrix and economic effects summary
- **Visualization:** `outputs/defense/defense_spending_analysis.png` (when running full range)

**Default Behavior:** When run without arguments, the script automatically:

1. Runs optimizations for all defense spending levels (-4000B to 6000B in 500B increments)
2. Saves results to `outputs/defense/` directory
3. Generates comprehensive visualization showing trade-offs across spending levels

**Policy Exclusions:** The following policies are specifically excluded from selection:

- Policy 37: Corporate Surtax of 5%
- Policy 43: Enact a 5% VAT
- Policy 49: Reinstate the Cadillac Tax
- Policy 68: Replace CIT with 5% VAT

**Policy Mutual Exclusivity Groups:** 15 groups ensure incompatible policies aren't selected together (e.g., competing corporate tax structures, estate tax options, etc.)

---

### visualize_defense_spending.py - Economic Effects Visualization

**Purpose:** Creates comprehensive visualizations analyzing how economic outcomes vary across different defense spending requirements.

**How to Run:**

```bash
python visualize_defense_spending.py
```

**Prerequisites:** Must have already run `max_gdp_defense.py` in full range mode to generate the CSV files.

**What it Does:**

1. Loads all optimization results from `outputs/defense/max_gdp_defense*.csv`
2. Calculates aggregate metrics for each spending level:
   - GDP growth, capital stock, jobs, wages
   - Revenue impact (static and dynamic)
   - Distributional effects (P20, P40-60, P80-100, P99)
   - Number of policies selected
3. Generates 2x3 grid of charts showing:
   - **GDP Growth Impact** across spending levels
   - **Employment Impact** (jobs created)
   - **Revenue Impact** (surplus/deficit)
   - **Capital Stock Change**
   - **Wage Rate Change**
   - **Equity Impact** (P20 vs P99 distribution)
4. Prints detailed insights to console including:
   - Spending level with highest composite economic index
   - Maximum GDP impact point
   - Best revenue neutrality point
   - Most equitable distribution point

**Output:** `outputs/defense/defense_spending_analysis.png` - A comprehensive 6-panel chart

**Key Insights Provided:**

- Trade-offs between defense spending and economic growth
- Relationship between spending levels and job creation
- Revenue neutrality points across the spending range
- Distributional equity across different spending scenarios
- Optimal spending levels for various policy objectives

---

### visualize_policy_selection.py - Policy Selection Analysis

**Purpose:** Creates heatmap visualizations showing which specific tax and spending policies are selected at each defense spending level.

**How to Run:**

```bash
python visualize_policy_selection.py
```

**Prerequisites:** Must have already run `max_gdp_defense.py` in full range mode.

**What it Does:**

1. Loads all optimization results from `outputs/defense/`
2. Extracts which policies were selected at each spending level
3. Creates a heatmap with:
   - Rows: Individual policy options (excluding NS policies)
   - Columns: Defense spending levels
   - Colors: Green = selected, Gray = not selected
4. Shows patterns of policy substitution as spending requirements change

**Output:** `outputs/defense/policy_selection_heatmap.png`

**Use Cases:**

- Identify which policies are always selected (robust across all scenarios)
- Find which policies are never selected (dominated options)
- See how policy selections change as defense spending varies
- Understand policy substitution patterns

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

Each optimization script generates:

1. **CSV file** with selected policies and their impacts
2. **Console output** showing:
   - Revenue raising policies
   - Revenue reducing policies
   - Total economic impacts (GDP, capital, jobs, wages)
   - Distributional impacts by income group
   - Revenue impacts (static and dynamic)
   - Number of selected policies

Visualization scripts generate:

1. **PNG image files** with charts and heatmaps
2. **Console output** with key insights and analysis

## Understanding the Results

### Economic Impacts

- **GDP Impact:** Long-run change in GDP (percentage). Example: 0.14 means 0.14% GDP growth
- **Capital Stock:** Change in capital stock (percentage)
- **Jobs:** Full-time equivalent jobs created (actual number, e.g., 150,000 jobs)
- **Wage Rate:** Change in wage rate (percentage)

### Distributional Impacts

- **P20:** After-tax income change for bottom 20% of earners (percentage)
- **P40-60:** After-tax income change for middle 40% of earners (percentage)
- **P80-100:** After-tax income change for top 20% of earners (percentage)
- **P99:** After-tax income change for top 1% of earners (percentage)

**Progressive Distribution:** When P20 and P40-60 benefit more than P80-100 and P99

### Revenue Impacts

- **Static Revenue:** Revenue estimate without behavioral responses (billions)
- **Dynamic Revenue:** Revenue estimate including economic behavioral responses (billions)
- **Positive values:** Revenue-raising (reduces deficit)
- **Negative values:** Revenue-reducing (increases deficit)

### Reading the Visualizations

**defense_spending_analysis.png:**

- Each panel shows a different economic metric
- X-axis: Defense spending change from -$4,000B to +$6,000B
- Vertical dashed line at $0B: Baseline reference point
- Look for trade-offs between GDP, jobs, revenue, and equity

**policy_selection_heatmap.png:**

- Rows: Individual policy options
- Columns: Defense spending levels
- Green: Policy selected at that spending level
- Gray: Policy not selected
- Patterns show how policy choices change with spending requirements

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

## Code Architecture & Maintainability

### Centralized Configuration (`config.py`)

All configuration settings, constants, and column mappings are centralized in one place:

- Change column names once, affects all scripts
- Adjust optimization parameters globally
- Easy to add new constants or modify existing ones

### Shared Utilities (`utils.py`)

Common functionality extracted to eliminate code duplication:

- Standardized data loading across all scripts
- Consistent NS group detection
- Uniform output formatting
- Easy to test and maintain

### Benefits of Current Structure

- **DRY Principle:** Eliminated ~600 lines of duplicate code
- **Consistency:** All scripts use same column names and patterns
- **Maintainability:** Changes in one place propagate everywhere
- **Testability:** Shared utilities can be unit tested
- **Readability:** Clear separation of concerns
- **Type Safety:** Added type hints for better IDE support

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

## Testing & Validation

### Running Tests

Test each script individually to ensure everything works:

```bash
# 1. Test basic optimization
python max_gdp.py

# 2. Test single defense level
python max_gdp_defense.py --spending 3000

# 3. Test full range (will take several minutes)
python max_gdp_defense.py

# 4. Test visualization (requires step 3 output)
python visualize_defense_spending.py
python visualize_policy_selection.py
```

### Expected Outputs

- **max_gdp.py**: `max_gdp.csv` in project root
- **max_gdp_defense.py**: CSV files in `outputs/defense/`
- **Visualization scripts**: PNG images in `outputs/defense/`

### Performance

- **Single optimization:** Typically completes in seconds
- **Full range (21 scenarios):** Takes several minutes depending on system
- **Large Excel files:** Loading time may increase with more policies
- **Gurobi license:** Academic licenses are free and fully functional

## Troubleshooting

### Common Issues

**"Gurobi license not found"**

- Ensure you have a valid Gurobi license installed
- Academic licenses are free: <https://www.gurobi.com/academia/>
- Follow Gurobi's installation guide for your platform

**"Model is infeasible"**

- One or more constraints cannot be satisfied simultaneously
- For defense scripts: Try a different spending level (some levels may be infeasible)
- Check that NS policies are properly formatted
- Verify Excel file has all required columns

**"No NS policy groups detected"**

- Verify policy names follow `NSxY:` pattern (e.g., `NS1A:`, `NS2B:`)
- Check that colon (`:`) is present after NS code
- Ensure policies exist in the Excel file
- Check Sheet1 has data starting from row 3

**"ImportError: No module named 'config' or 'utils'"**

- Ensure `config.py` and `utils.py` are in the same directory as the scripts
- Check that you're running from the correct directory
- Try running: `python -c "import config; import utils"` to verify

**"FileNotFoundError: tax reform & spending menu options (v8) template.xlsx"**

- Verify the Excel file exists in the same directory
- Check the exact filename (including version number)
- Ensure the file isn't open in Excel (may cause read errors on some systems)

**"KeyError" when reading Excel****

- Check that column names in Excel match expected names
- Verify headers are in row 2 (index 1)
- Ensure no extra spaces in column names

**Visualization scripts show "No files found"**

- Run `max_gdp_defense.py` first to generate CSV files
- Check that `outputs/defense/` directory exists and contains CSV files
- Verify file naming matches expected pattern: `max_gdp_defense{number}.csv`

## Development & Contributing

When adding new optimization scripts or modifying existing ones:

1. Import from `config` and `utils` modules
2. Use standardized column names from `COLUMNS` dictionary
3. Use shared utility functions for data loading and display
4. Include NS mutual exclusivity constraints
5. Use two-stage optimization approach
6. Follow existing code structure and naming conventions
7. Add comprehensive docstrings with type hints
8. Document all constraints clearly
9. Add inline comments for complex logic (but avoid overcrowding)
10. Update this README with any new features or changes

### Code Documentation Standards

- **File-level docstrings:** Explain what the file does in plain language
- **Function docstrings:** Include purpose, parameters, returns, and any exceptions
- **Inline comments:** Use sparingly, only for complex logic that isn't self-explanatory
- **Type hints:** Include for all function parameters and return values

## Code Architecture

### Module Responsibilities

**Core Infrastructure:**
- **config.py** - All constants, column mappings, patterns, thresholds
- **validation.py** - Pre-flight checks, input validation, actionable errors
- **logger.py** - Structured logging (ERROR/WARNING/INFO/DEBUG)

**Optimization:**
- **optimizer_utils.py** - Gurobi constraint building (fiscal, equity, policy exclusions)
- **utils.py** - Data loading, NS group extraction, result formatting

**Separation of Concerns:**
- `optimizer_utils.py` handles Gurobi model construction (requires gurobipy)
- `utils.py` handles data I/O and display (no Gurobi dependency)
- These serve different purposes and should NOT be merged

### Key Design Principles

1. **DRY (Don't Repeat Yourself)** - Eliminated 80% of code duplication
2. **Single Responsibility** - Each module has one clear purpose
3. **Centralized Configuration** - All constants in `config.py`
4. **Comprehensive Validation** - Validate early, fail fast with clear messages
5. **Structured Logging** - Replaceable logging levels for debugging
6. **Type Safety** - Type hints throughout for IDE support

## Testing & Validation

### Quick Test

```bash
# Test all scripts sequentially
python max_gdp.py
python max_gdp_defense.py --spending 3000
python visualize_defense_spending.py
python visualize_policy_selection.py
```

### Validation Features

The validation framework provides:
- Excel file existence and format checks
- DataFrame structure validation
- NS policy naming convention validation
- Spending level range validation
- Policy index validation
- Pre-flight optimization checks

### Error Handling

All scripts include comprehensive error handling:
- **ValidationError** - Input data issues (actionable messages)
- **GurobiError** - Optimization failures (license, infeasibility)
- **FileNotFoundError** - Missing files
- **KeyboardInterrupt** - Graceful user cancellation

## Version History

- **v3.1** (Current): Enterprise-grade refactoring for robustness and maintainability
  - Created comprehensive validation framework ([`validation.py`](validation.py))
  - Implemented structured logging system ([`logger.py`](logger.py))
  - Extracted optimizer utilities ([`optimizer_utils.py`](optimizer_utils.py))
  - Enhanced error handling with actionable error messages
  - Eliminated 80% of code duplication (~500 lines removed)
  - Fixed Windows compatibility issues (Unicode encoding)
  - Streamlined documentation (single README)
- **v3.0**: Major refactoring for consistency
  - Created centralized `config.py` and `utils.py` modules
  - Standardized column naming across all scripts
  - Consolidated defense scripts into single `max_gdp_defense.py`
  - Added type hints and comprehensive documentation
  - Added visualization scripts
- **v2.0**: Added NS mutual exclusivity constraints
- **v1.0**: Initial release

## License

[Specify license here]

## Contact

[Add contact information]
