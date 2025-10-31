"""
Configuration file for IJAB Economic Scenario Analysis.

This module centralizes all configuration settings, constants, and column name
mappings used across all optimization scripts. This ensures consistency and
makes it easy to update settings in one place.
"""

from typing import Dict, List

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
    "dynamic_revenue": "Dynamic 10-Year Revenue (billions)"
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
    COLUMNS["dynamic_revenue"]
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

# Small epsilon value for strict inequality constraints (e.g., >= becomes > epsilon)
EPSILON = 1e-5

# Tolerance for distributional equality constraints (1 percentage point)
DISTRIBUTIONAL_TOLERANCE = 0.01

# ============================================================================
# Defense Spending Thresholds
# ============================================================================
# Minimum spending requirements for national security policies (in billions)

DEFENSE_SPENDING = {
    "baseline": 3000,    # $3,000B requirement
    "increased": 4000    # $4,000B requirement
}

# ============================================================================
# Display Formatting
# ============================================================================

# Width for console output formatting
DISPLAY_WIDTH = 80

# Number of characters to display for policy names
POLICY_NAME_MAX_LENGTH = 70