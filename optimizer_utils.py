"""
Optimization Utility Functions for IJAB Economic Scenario Analysis.

This module provides reusable functions for building optimization models,
reducing code duplication across optimization scripts.

Key functions:
- add_excluded_policy_constraints(): Force certain policies to not be selected
- add_fiscal_constraints(): Revenue surplus requirements (>= $600B)
- add_economic_constraints(): Capital, jobs, wage non-negativity
- add_equity_constraints(): Progressive distribution requirements
- add_policy_mutual_exclusivity(): Competing policy groups
- add_ns_constraints(): National security mutual exclusivity and spending
"""

from typing import Any, Protocol

from gurobipy import Model, quicksum
import pandas as pd

from config import (
    COLUMNS,
    EPSILON,
    EXCLUDED_POLICIES,
    POLICY_CO_EXCLUSIONS,
    REVENUE_SURPLUS_REQUIREMENT,
)


class SupportsDebug(Protocol):
    """Protocol for objects that support debug logging."""

    def debug(self, message: str) -> None: ...


def get_policy_indices_by_codes(df: pd.DataFrame, policy_codes: list[str]) -> list[int]:
    """
    Get positional indices for policies by their option codes.

    This is a utility function used throughout the optimization scripts
    to find policies by their numeric or alphanumeric codes.

    Args:
        df: DataFrame containing policy data
        policy_codes: List of policy codes (e.g., ['11', '36', '68'])

    Returns:
        List of positional indices for matching policies

    Examples:
        >>> indices = get_policy_indices_by_codes(df, ["11", "36"])
        >>> # Returns indices of policies starting with "11:" or "36:"
    """
    indices: list[int] = []
    for code in policy_codes:
        # Match policies that start with the code followed by ':'
        # This handles both numeric codes like "11:" and alphanumeric like "S15:"
        matching = df[df[COLUMNS["option"]].str.match(f"^{code}:", na=False)]
        if len(matching) > 0:
            # Get positional index (not DataFrame label index)
            label_idx = matching.index[0]
            pos_idx = df.index.get_loc(label_idx)
            if isinstance(pos_idx, int):
                indices.append(pos_idx)
    return indices


def add_excluded_policy_constraints(
    model: Model,
    x: Any,  # Gurobi tupledict[int, Var]
    df: pd.DataFrame,
    excluded_codes: list[str] | None = None,
    logger: SupportsDebug | None = None,
) -> int:
    """
    Add constraints to exclude certain policies from selection.

    Implements the "no new taxes" constraint by forcing excluded
    policies to have value 0 in the optimization.

    Args:
        model: Gurobi model to add constraints to
        x: Decision variables dict (policy index -> variable)
        df: DataFrame containing policy data
        excluded_codes: List of policy codes to exclude (uses config default if None)
        logger: Optional logger for progress messages

    Returns:
        Number of constraints added
    """
    if excluded_codes is None:
        excluded_codes = EXCLUDED_POLICIES

    excluded_indices = get_policy_indices_by_codes(df, excluded_codes)

    for idx in excluded_indices:
        model.addConstr(x[idx] == 0, name=f"Exclude_policy_{idx}")

    if logger and len(excluded_indices) > 0:
        logger.debug(f"Added {len(excluded_indices)} excluded policy constraints")

    return len(excluded_indices)


def add_fiscal_constraints(
    model: Model,
    x: Any,  # Gurobi tupledict[int, Var]
    revenue: Any,  # ArrayLike (list, np.ndarray, or pandas Series)
    indices: range,
    logger: SupportsDebug | None = None,
) -> None:
    """
    Add fiscal responsibility constraint (revenue surplus requirement).

    Requires that total dynamic revenue is at least $600B, ensuring
    the policy package generates a substantial revenue surplus.

    Args:
        model: Gurobi model to add constraints to
        x: Decision variables dict
        revenue: List of dynamic revenue values for each policy
        indices: Range of policy indices
        logger: Optional logger for progress messages
    """
    model.addConstr(
        quicksum(x[i] * revenue[i] for i in indices) >= REVENUE_SURPLUS_REQUIREMENT,
        name="RevenueSurplus",
    )

    if logger:
        logger.debug(f"Added revenue surplus constraint: >= ${REVENUE_SURPLUS_REQUIREMENT}B")


def add_economic_constraints(
    model: Model,
    x: Any,  # Gurobi tupledict[int, Var]
    capital: Any,  # Array like (list, np.ndarray, or pandas Series)
    jobs: Any,  # ArrayLike
    wage: Any,  # ArrayLike
    indices: range,
    logger: SupportsDebug | None = None,
) -> None:
    """
    Add economic impact constraints.

    Ensures non-negative impacts on:
    - Capital stock
    - Job creation
    - Wage rates

    Args:
        model: Gurobi model to add constraints to
        x: Decision variables dict
        capital: List of capital stock values
        jobs: List of job creation values
        wage: List of wage rate values
        indices: Range of policy indices
        logger: Optional logger for progress messages
    """
    # Capital stock change must be non-negative
    model.addConstr(quicksum(x[i] * capital[i] for i in indices) >= 0, name="CapitalStock")

    # Job creation must be non-negative
    model.addConstr(quicksum(x[i] * jobs[i] for i in indices) >= 0, name="Jobs")

    # Wage rate change must be non-negative
    model.addConstr(quicksum(x[i] * wage[i] for i in indices) >= 0, name="WageRate")

    if logger:
        logger.debug("Added economic constraints (capital, jobs, wage)")


def add_equity_constraints(
    model: Model,
    x: Any,  # Gurobi tupledict[int, Var]
    p20_arr: Any,  # ArrayLike
    p40_arr: Any,  # ArrayLike
    p80_arr: Any,  # ArrayLike
    p99_arr: Any,  # ArrayLike
    indices: range,
    logger: SupportsDebug | None = None,
) -> None:
    """
    Add progressive distribution equity constraints.

    Implements the requirement that lower and middle-income groups
    benefit at least as much as upper-income groups. Specifically:

    1. P20 (bottom 20%) must benefit >= P99 (top 1%) AND >= P80-100 (top 20%)
    2. P40-60 (middle class) must benefit >= P99 (top 1%) AND >= P80-100 (top 20%)
    3. All groups must have non-negative after-tax income effects

    EPSILON is used to ensure STRICT inequality (P20 > P99, not just P20 >= P99).
    Without EPSILON, numerical precision issues could allow effectively equal
    values to satisfy the constraints, which doesn't meet the progressive requirement.

    Args:
        model: Gurobi model to add constraints to
        x: Decision variables dict
        p20_arr: After-tax income effects for bottom 20%
        p40_arr: After-tax income effects for middle 40%
        p80_arr: After-tax income effects for top 20%
        p99_arr: After-tax income effects for top 1%
        indices: Range of policy indices
        logger: Optional logger for progress messages
    """
    # Calculate total after-tax income change for each percentile group
    p20 = quicksum(x[i] * p20_arr[i] for i in indices)
    p40 = quicksum(x[i] * p40_arr[i] for i in indices)
    p80 = quicksum(x[i] * p80_arr[i] for i in indices)
    p99 = quicksum(x[i] * p99_arr[i] for i in indices)

    # Progressive distribution: Lower/middle income groups must benefit
    # at least as much as upper groups (with EPSILON for strict inequality)
    model.addConstr(p20 - p99 >= EPSILON, name="P20_ge_P99")
    model.addConstr(p40 - p99 >= EPSILON, name="P40_ge_P99")
    model.addConstr(p20 - p80 >= EPSILON, name="P20_ge_P80")
    model.addConstr(p40 - p80 >= EPSILON, name="P40_ge_P80")

    # Non-negative after-tax income for all groups (everyone must be better off)
    # No EPSILON needed here - simple non-negativity constraint
    model.addConstr(p20 >= 0, name="P20_NonNegative")
    model.addConstr(p40 >= 0, name="P40_NonNegative")
    model.addConstr(p80 >= 0, name="P80_NonNegative")
    model.addConstr(p99 >= 0, name="P99_NonNegative")

    if logger:
        logger.debug("Added equity constraints (progressive distribution)")


def add_policy_mutual_exclusivity(
    model: Model,
    x: Any,  # Gurobi tupledict[int, Var]
    policy_groups: dict[str, list[int]],
    logger: SupportsDebug | None = None,
) -> int:
    """
    Add policy mutual exclusivity constraints.

    For each policy group (e.g., competing corporate tax structures),
    at most one option can be selected. This prevents selecting
    incompatible policies together.

    Args:
        model: Gurobi model to add constraints to
        x: Decision variables dict
        policy_groups: Dict mapping group names to policy indices
        logger: Optional logger for progress messages

    Returns:
        Number of constraints added
    """
    count = 0
    for group_name, idxs in policy_groups.items():
        if len(idxs) > 1:  # Only add constraint if group has multiple options
            model.addConstr(
                quicksum(x[i] for i in idxs) <= 1, name=f"Policy_{group_name}_mutual_exclusivity"
            )
            count += 1

    if logger and count > 0:
        logger.debug(f"Added {count} policy mutual exclusivity constraints")

    return count


def add_policy_co_exclusion_constraints(
    model: Model,
    x: Any,  # Gurobi tupledict[int, Var]
    df: pd.DataFrame,
    co_exclusions: list[tuple[str, str]] | None = None,
    logger: SupportsDebug | None = None,
) -> int:
    """
    Add special policy co-exclusion constraints.

    If policy A is selected, policy B cannot be selected (and vice versa).
    This is used for special cases beyond normal mutual exclusivity groups.

    Args:
        model: Gurobi model to add constraints to
        x: Decision variables dict
        df: DataFrame containing policy data
        co_exclusions: List of (policy_A_code, policy_B_code) tuples
        logger: Optional logger for progress messages

    Returns:
        Number of constraints added
    """
    if co_exclusions is None:
        co_exclusions = POLICY_CO_EXCLUSIONS

    count = 0
    for code_a, code_b in co_exclusions:
        idx_a = get_policy_indices_by_codes(df, [code_a])
        idx_b = get_policy_indices_by_codes(df, [code_b])

        if len(idx_a) > 0 and len(idx_b) > 0:
            # If x[A] = 1, then x[B] must be 0
            # Equivalent to: x[A] + x[B] <= 1
            model.addConstr(
                x[idx_a[0]] + x[idx_b[0]] <= 1, name=f"Policy_{code_a}_excludes_{code_b}"
            )
            count += 1

    if logger and count > 0:
        logger.debug(f"Added {count} policy co-exclusion constraints")

    return count


def add_ns_mutual_exclusivity(
    model: Model,
    x: Any,  # Gurobi tupledict[int, Var]
    ns_groups: dict[str, list[int]],
    logger: SupportsDebug | None = None,
) -> int:
    """
    Add National Security (NS) mutual exclusivity constraints.

    For each NS group (e.g., NS1 with options NS1A, NS1B, NS1C),
    at most one option can be selected. This ensures coherent
    national security policy by preventing conflicting selections.

    Args:
        model: Gurobi model to add constraints to
        x: Decision variables dict
        ns_groups: Dict mapping NS group names to policy indices
        logger: Optional logger for progress messages

    Returns:
        Number of constraints added
    """
    count = 0
    for group, idxs in ns_groups.items():
        model.addConstr(quicksum(x[i] for i in idxs) <= 1, name=f"NS_{group}_mutual_exclusivity")
        count += 1

    if logger and count > 0:
        logger.debug(f"Added {count} NS mutual exclusivity constraints")

    return count


def add_ns_spending_constraint(
    model: Model,
    x: Any,  # Gurobi tupledict[int, Var]
    revenue: Any,  # ArrayLike
    ns_strict_indices: list[int],
    min_ns_spending: int,
    logger: SupportsDebug | None = None,
) -> None:
    """
    Add National Security spending requirement constraint.

    Total spending (negative revenue) from NS1-NS7 policies must
    equal exactly the specified minimum spending level.

    Note: Revenue is negative for spending, so we use == -min_ns_spending

    Args:
        model: Gurobi model to add constraints to
        x: Decision variables dict
        revenue: List of revenue values for each policy
        ns_strict_indices: Indices of NS1-NS7 policies (count toward spending)
        min_ns_spending: Required NS spending in billions
        logger: Optional logger for progress messages
    """
    model.addConstr(
        quicksum(x[i] * revenue[i] for i in ns_strict_indices) == -min_ns_spending,
        name="ExactNSSpending",
    )

    if logger:
        logger.debug(f"Added NS spending constraint: ${min_ns_spending:,}B")


def add_all_constraints(
    model: Model,
    x: Any,  # Gurobi tupledict[int, Var]
    df: pd.DataFrame,
    ns_groups: dict[str, list[int]],
    policy_groups: dict[str, list[int]],
    ns_strict_indices: list[int],
    min_ns_spending: int,
    logger: SupportsDebug | None = None,
) -> None:
    """
    Add all standard constraints to the optimization model.

    This is a convenience function that adds all the standard constraints
    used in the defense optimization scripts. Reduces code duplication
    and ensures consistency across optimization stages.

    Args:
        model: Gurobi model to add constraints to
        x: Decision variables dict
        df: DataFrame containing policy data
        ns_groups: NS mutual exclusivity groups
        policy_groups: Policy mutual exclusivity groups
        ns_strict_indices: Indices of NS1-NS7 policies
        min_ns_spending: Required NS spending in billions
        logger: Optional logger for progress messages
    """
    # Extract data arrays
    indices = range(len(df))
    revenue = df[COLUMNS["dynamic_revenue"]].values
    capital = df[COLUMNS["capital"]].values
    jobs = df[COLUMNS["jobs"]].values
    wage = df[COLUMNS["wage"]].values
    p20_arr = df[COLUMNS["p20"]].values
    p40_arr = df[COLUMNS["p40_60"]].values
    p80_arr = df[COLUMNS["p80_100"]].values
    p99_arr = df[COLUMNS["p99"]].values

    # Add all constraint types
    add_excluded_policy_constraints(model, x, df, logger=logger)
    add_fiscal_constraints(model, x, revenue, indices, logger=logger)
    add_economic_constraints(model, x, capital, jobs, wage, indices, logger=logger)
    add_equity_constraints(model, x, p20_arr, p40_arr, p80_arr, p99_arr, indices, logger=logger)
    add_policy_mutual_exclusivity(model, x, policy_groups, logger=logger)
    add_policy_co_exclusion_constraints(model, x, df, logger=logger)
    add_ns_mutual_exclusivity(model, x, ns_groups, logger=logger)
    add_ns_spending_constraint(model, x, revenue, ns_strict_indices, min_ns_spending, logger=logger)
