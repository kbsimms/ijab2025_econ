"""
Configuration file for IJAB Economic Scenario Analysis.

This module centralizes all configuration settings, constants, and column name
mappings used across all optimization scripts. This ensures consistency and
makes it easy to update settings in one place.
"""


# ============================================================================
# File Paths
# ============================================================================

EXCEL_FILE_PATH = "tax reform & spending menu options (v8) template.xlsx"
SHEET_NAME = "Sheet1"

# ============================================================================
# Column Name Mappings
# ============================================================================
# Standardized column names that match the Excel file headers.
# All scripts should use these consistent names.

COLUMNS = {
    "option": "Option",
    "gdp": "Long-Run Change in GDP",
    "capital": "Capital Stock",
    "jobs": "Full-Time Equivalent Jobs",
    "wage": "Wage Rate",
    "p20": "P20",
    "p40_60": "P40-60",
    "p80_100": "P80-100",
    "p99": "P99",
    "static_revenue": "Static 10-Year Revenue (billions)",
    "dynamic_revenue": "Dynamic 10-Year Revenue (billions)",
}

# List of numeric columns (for type conversion)
NUMERIC_COLUMNS = [
    COLUMNS["gdp"],
    COLUMNS["capital"],
    COLUMNS["jobs"],
    COLUMNS["wage"],
    COLUMNS["p20"],
    COLUMNS["p40_60"],
    COLUMNS["p80_100"],
    COLUMNS["p99"],
    COLUMNS["static_revenue"],
    COLUMNS["dynamic_revenue"],
]

# ============================================================================
# National Security (NS) Policy Patterns
# ============================================================================

# Regex pattern to match NS policies: NS1A, NS2B, NS15C, etc.
# Matches: NS<digits><letter>:
NS_PATTERN = r"^NS\d+[A-Z]:"

# Regex pattern for strict NS1-NS7 policies only (used in defense scripts)
NS_STRICT_PATTERN = r"^NS[1-7][A-Z]:"

# ============================================================================
# Optimization Settings
# ============================================================================

# Suppress Gurobi solver output to console
SUPPRESS_GUROBI_OUTPUT = True

# Revenue surplus requirement (in billions)
# All policy packages must generate at least this amount in dynamic revenue
REVENUE_SURPLUS_REQUIREMENT = 600

# Small epsilon value for strict inequality constraints (e.g., >= becomes > epsilon)
EPSILON = 1e-5

# Tolerance for distributional equality constraints (1 percentage point)
DISTRIBUTIONAL_TOLERANCE = 0.01

# ============================================================================
# Defense Spending Configuration
# ============================================================================
# Minimum spending requirements for national security policies (in billions)

DEFENSE_SPENDING = {
    "baseline": 3000,  # $3,000B requirement
    "increased": 4000,  # $4,000B requirement
}

# Defense spending range for full analysis (in billions)
# These values define the range and granularity of defense spending scenarios
SPENDING_RANGE = {
    "min": -4000,  # Minimum defense spending change: -$4,000B
    "max": 6500,  # Maximum defense spending change: +$6,500B (exclusive)
    "step": 500,  # Increment between spending levels: $500B
}

# Feasible spending validation limits (in billions)
FEASIBLE_SPENDING_LIMITS = {
    "min": -10000,  # Absolute minimum for validation
    "max": 10000,  # Absolute maximum for validation
}

# ============================================================================
# Policy Exclusions and Constraints
# ============================================================================
# These policies are excluded from selection to implement the "no new taxes" constraint
# as required by the analysis framework

EXCLUDED_POLICIES = [
    "37",  # Corporate Surtax of 5% - excluded per "no new taxes" requirement
    "43",  # Enact a 5% VAT - new tax, therefore excluded
    "49",  # Reinstate the Cadillac Tax - new tax, therefore excluded
    "68",  # Replace CIT with 5% VAT - new tax structure, therefore excluded
]

# Special policy co-exclusion rules
# If policy A is selected, policy B cannot be selected (and vice versa)
# Format: List of tuples (policy_A, policy_B)
# Note: These may be redundant if both policies are in EXCLUDED_POLICIES,
# but kept for future flexibility if exclusion list changes
POLICY_CO_EXCLUSIONS = [
    ("68", "37")  # If VAT replacement (68) selected, corporate surtax (37) excluded
]

# ============================================================================
# Display Formatting
# ============================================================================

# Width for console output formatting
DISPLAY_WIDTH = 80

# Number of characters to display for policy names
POLICY_NAME_MAX_LENGTH = 70

# ============================================================================
# Constraint Documentation
# ============================================================================
# This section documents the business logic behind key constraints

# EPSILON Usage:
# EPSILON (1e-5) is added to certain inequality constraints to enforce
# strict inequalities in the optimization:
# - Use EPSILON when we need STRICT inequality (P20 > P99, not P20 >= P99)
# - Without EPSILON, numerical precision issues in solvers can allow
#   effectively equal values to satisfy >= constraints
# - Not needed for simple non-negativity constraints (>= 0)
#
# Example: p20 - p99 >= EPSILON ensures P20 strictly benefits more than P99

# Equity Constraints Logic:
# The equity constraints ensure that lower and middle-income groups benefit
# at least as much as upper-income groups. Specifically:
# - P20 (bottom 20%) must benefit >= P99 (top 1%) AND >= P80-100 (top 20%)
# - P40-60 (middle class) must benefit >= P99 (top 1%) AND >= P80-100 (top 20%)
# - All groups (P20, P40-60, P80-100, P99) must have non-negative benefits
#
# This implements a progressive distribution requirement where working families
# and middle class benefit at least as much as wealthy households.

# NS1-NS7 "Strict" Distinction:
# NS policies are numbered NS1, NS2, ... NS7, NS8, etc.
# "Strict" refers to NS1-NS7 which are the core defense spending policies
# that count toward the minimum national security spending requirement.
# Other NS policies (NS8+) may exist but don't count toward the spending floor.
