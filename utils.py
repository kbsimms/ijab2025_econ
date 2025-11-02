"""
Utility functions for IJAB Economic Scenario Analysis.

This module provides shared functionality used across all optimization scripts,
including data loading, NS group extraction, result formatting, and validation.
"""

import pandas as pd

from config import (
    COLUMNS,
    DISPLAY_WIDTH,
    EXCEL_FILE_PATH,
    NS_PATTERN,
    NS_STRICT_PATTERN,
    NUMERIC_COLUMNS,
    POLICY_NAME_MAX_LENGTH,
    SHEET_NAME,
)
from logger import get_logger
from validation import (
    ValidationError,
    validate_dataframe,
    validate_excel_file,
    validate_ns_groups,
    validate_sheet_exists,
)

# Module logger
logger = get_logger(__name__)


def load_policy_data(file_path: str = EXCEL_FILE_PATH) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    """
    Load and clean policy data from Excel file.

    This function handles data preprocessing including:
    - Loading raw Excel data
    - Extracting proper column headers
    - Converting numeric columns
    - Identifying National Security (NS) policy groups

    Args:
        file_path: Path to Excel file (default: from config)

    Returns:
        tuple: (df_clean, ns_groups)
            - df_clean: Cleaned DataFrame with all policies
            - ns_groups: Dict mapping NS group names to policy indices

    Raises:
        FileNotFoundError: If Excel file doesn't exist
        ValueError: If required columns are missing
    """
    logger.info("Loading policy data from Excel file...")

    # Validate Excel file exists and is readable
    try:
        validate_excel_file(file_path)
        validate_sheet_exists(file_path, SHEET_NAME)
    except ValidationError:
        logger.exception("Validation failed")
        raise

    # Load the Excel file
    try:
        xls = pd.ExcelFile(file_path)
    except Exception:
        logger.exception("Failed to load Excel file")
        raise
    df = xls.parse(SHEET_NAME)

    # Extract headers from row 2 (index 1)
    headers = df.iloc[1]
    # Skip first 3 rows (0=title, 1=headers, 2=section header) to get actual data
    df_clean = df[3:].copy()
    df_clean.columns = headers
    df_clean = df_clean.reset_index(drop=True)

    # Drop rows that are not actual policy options
    df_clean = df_clean[df_clean[COLUMNS["option"]].notna()]
    df_clean = df_clean[df_clean[COLUMNS["gdp"]].notna()]

    # Convert all numeric columns
    df_clean[NUMERIC_COLUMNS] = df_clean[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")

    # Validate DataFrame structure and content
    try:
        validate_dataframe(df_clean, require_ns_policies=True)
    except ValidationError:
        logger.exception("DataFrame validation failed")
        raise

    # Extract NS groupings for mutual exclusivity constraints
    ns_groups = extract_ns_groups(df_clean)

    # Validate NS groups
    try:
        validate_ns_groups(ns_groups, df_clean, min_groups=1)
    except ValidationError as e:
        logger.warning(f"NS group validation warning: {e}")

    logger.info(f"Loaded {len(df_clean)} policy options")
    print_ns_groups(df_clean, ns_groups)

    return df_clean, ns_groups


def extract_ns_groups(df: pd.DataFrame) -> dict[str, list[int]]:
    """
    Extract NS policy groupings for mutual exclusivity constraints.

    Example: NS1A, NS1B, NS1C all belong to group "NS1"
    Only one option from each NS group can be selected.

    Args:
        df: DataFrame containing policy data

    Returns:
        Dict mapping NS group names (e.g., "NS1") to lists of policy indices
    """
    # Add indicator for NS-prefixed policies (National Security policies)
    # Matches patterns like: NS1A:, NS2B:, NS7C:, etc.
    df["is_NS"] = df[COLUMNS["option"]].str.contains(NS_PATTERN, case=False, na=False, regex=True)

    # Build NS groups dictionary
    # IMPORTANT: Use positional index (0-based) not DataFrame index label
    ns_groups: dict[str, list[int]] = {}
    for label_idx, row in df[df["is_NS"]].iterrows():
        code = (
            row[COLUMNS["option"]].split(":")[0].strip()
        )  # Extract "NS1A" from "NS1A: Description"
        group = code[:-1]  # Extract "NS1" from "NS1A"
        # Find positional index in the full df DataFrame
        pos_in_df = df.index.get_loc(label_idx)
        if isinstance(pos_in_df, int):
            ns_groups.setdefault(group, []).append(pos_in_df)

    return ns_groups


def get_ns_strict_indices(df: pd.DataFrame) -> list[int]:
    """
    Get positional indices of strict NS1-NS7 policies (for defense spending constraint).

    These are the policies that count toward minimum NS spending requirements.
    IMPORTANT: Returns positional indices (0-based position in DataFrame),
    not DataFrame index labels.

    Args:
        df: DataFrame containing policy data

    Returns:
        List of positional indices for NS1-NS7 policies only
    """
    # Get DataFrame index labels that match NS1-NS7 pattern
    matching_labels = df[
        df[COLUMNS["option"]].str.match(NS_STRICT_PATTERN, na=False)
    ].index.tolist()

    # Convert DataFrame index labels to positional indices
    # Ensure we always get integer indices
    ns_strict_indices: list[int] = []
    for label in matching_labels:
        loc = df.index.get_loc(label)
        # get_loc can return int, slice, or boolean array
        # We expect int for standard index lookup
        if isinstance(loc, int):
            ns_strict_indices.append(loc)
        else:
            # This shouldn't happen with unique index, but handle it
            logger.warning(f"Unexpected index type for label {label}: {type(loc)}")

    return ns_strict_indices


def print_ns_groups(df: pd.DataFrame, ns_groups: dict[str, list[int]]) -> None:
    """
    Print NS group information to console.

    Args:
        df: DataFrame containing policy data
        ns_groups: Dict mapping NS group names to policy indices
    """
    if ns_groups:
        logger.info(f"Identified {len(ns_groups)} NS policy groups:")
        for group, idxs in sorted(ns_groups.items()):
            policies = [df.iloc[idx][COLUMNS["option"]].split(":")[0] for idx in idxs]
            logger.info(f"   {group}: {', '.join(policies)} ({len(idxs)} options)")
    else:
        logger.warning("No NS policy groups detected")


def verify_ns_exclusivity(
    df: pd.DataFrame,
    ns_groups: dict[str, list[int]],
    selected_indices: list[int],
    verbose: bool = True,
) -> bool:
    """
    Verify NS mutual exclusivity in solution.

    Checks that at most one policy per NS group is selected.

    Args:
        df: DataFrame containing policy data
        ns_groups: Dict mapping NS group names to policy indices
        selected_indices: List of selected policy indices
        verbose: If True, prints verification details

    Returns:
        True if all NS constraints are satisfied, False otherwise
    """
    if not verbose or not ns_groups:
        return True

    logger.info("Verifying NS mutual exclusivity in solution:")
    violations: list[str] = []
    all_satisfied = True

    for group, idxs in sorted(ns_groups.items()):
        selected_in_group = [i for i in idxs if i in selected_indices]

        if len(selected_in_group) > 1:
            policies = [df.iloc[i][COLUMNS["option"]].split(":")[0] for i in selected_in_group]
            violations.append(
                f"  X {group}: {len(selected_in_group)} policies selected ({', '.join(policies)})"
            )
            all_satisfied = False
        elif len(selected_in_group) == 1:
            policy = df.iloc[selected_in_group[0]][COLUMNS["option"]].split(":")[0]
            logger.info(f"  [OK] {group}: 1 policy selected ({policy})")
        else:
            logger.debug(f"  [OK] {group}: 0 policies selected")

    if violations:
        logger.error("NS MUTUAL EXCLUSIVITY VIOLATIONS DETECTED:")
        for v in violations:
            logger.error(v)
    else:
        logger.info("  [OK] All NS constraints satisfied!")

    return all_satisfied


def display_results(result_df: pd.DataFrame, _gdp_impact: float, _revenue_impact: float) -> None:
    """
    Display optimization results in a readable format.

    Args:
        result_df: DataFrame of selected policies
        _gdp_impact: Total GDP impact achieved (unused, kept for API compatibility)
        _revenue_impact: Total revenue impact achieved (unused, kept for API compatibility)
    """
    logger.section("OPTIMIZATION RESULTS")

    # Separate policies by positive/negative revenue impact
    positive_revenue = result_df[result_df[COLUMNS["dynamic_revenue"]] >= 0].copy()
    negative_revenue = result_df[result_df[COLUMNS["dynamic_revenue"]] < 0].copy()

    # Sort by absolute impact
    positive_revenue = positive_revenue.sort_values(COLUMNS["dynamic_revenue"], ascending=False)
    negative_revenue = negative_revenue.sort_values(COLUMNS["dynamic_revenue"], ascending=True)

    if len(positive_revenue) > 0:
        print(f"\n{'REVENUE RAISING POLICIES':^{DISPLAY_WIDTH}}")  # noqa: T201
        print("-" * DISPLAY_WIDTH)  # noqa: T201
        for _, row in positive_revenue.iterrows():
            print(f"  {row[COLUMNS['option']][:POLICY_NAME_MAX_LENGTH]:<{POLICY_NAME_MAX_LENGTH}}")  # noqa: T201
            print(
                f"    GDP: {row[COLUMNS['gdp']] * 100:>+7.4f}%  |  Revenue: ${row[COLUMNS['dynamic_revenue']]:>8.2f}B"
            )

    if len(negative_revenue) > 0:
        print(f"\n{'REVENUE REDUCING POLICIES':^{DISPLAY_WIDTH}}")  # noqa: T201
        print("-" * DISPLAY_WIDTH)  # noqa: T201
        for _, row in negative_revenue.iterrows():
            print(f"  {row[COLUMNS['option']][:POLICY_NAME_MAX_LENGTH]:<{POLICY_NAME_MAX_LENGTH}}")  # noqa: T201
            print(
                f"    GDP: {row[COLUMNS['gdp']] * 100:>+7.4f}%  |  Revenue: ${row[COLUMNS['dynamic_revenue']]:>8.2f}B"
            )

    # Calculate totals for all metrics
    print("\n" + "=" * DISPLAY_WIDTH)  # noqa: T201
    print("FINAL SUMMARY - TOTAL IMPACT OF SELECTED POLICIES".center(DISPLAY_WIDTH))  # noqa: T201
    print("=" * DISPLAY_WIDTH)  # noqa: T201
    print(f"\n{'Economic Impacts':^{DISPLAY_WIDTH}}")  # noqa: T201
    print("-" * DISPLAY_WIDTH)  # noqa: T201
    print(f"  Long-Run Change in GDP:              {result_df[COLUMNS['gdp']].sum() * 100:>+8.4f}%")  # noqa: T201
    print(
        f"  Capital Stock:                       {result_df[COLUMNS['capital']].sum() * 100:>+8.4f}%"
    )
    print(f"  Full-Time Equivalent Jobs:           {result_df[COLUMNS['jobs']].sum():>+10,.0f}")  # noqa: T201
    print(
        f"  Wage Rate:                           {result_df[COLUMNS['wage']].sum() * 100:>+8.4f}%"
    )

    print(f"\n{'After-Tax Income Changes (by Income Percentile)':^{DISPLAY_WIDTH}}")  # noqa: T201
    print("-" * DISPLAY_WIDTH)  # noqa: T201
    print(f"  P20 (Bottom 20%):                    {result_df[COLUMNS['p20']].sum() * 100:>+8.4f}%")  # noqa: T201
    print(
        f"  P40-60 (Middle Class):               {result_df[COLUMNS['p40_60']].sum() * 100:>+8.4f}%"
    )
    print(
        f"  P80-100 (Top 20%):                   {result_df[COLUMNS['p80_100']].sum() * 100:>+8.4f}%"
    )
    print(f"  P99 (Top 1%):                        {result_df[COLUMNS['p99']].sum() * 100:>+8.4f}%")  # noqa: T201

    print(f"\n{'Revenue Impacts':^{DISPLAY_WIDTH}}")  # noqa: T201
    print("-" * DISPLAY_WIDTH)  # noqa: T201
    print(
        f"  Static 10-Year Revenue:              ${result_df[COLUMNS['static_revenue']].sum():>10.2f} billion"
    )
    print(
        f"  Dynamic 10-Year Revenue:             ${result_df[COLUMNS['dynamic_revenue']].sum():>10.2f} billion"
    )

    print(f"\n{'Policy Count':^{DISPLAY_WIDTH}}")  # noqa: T201
    print("-" * DISPLAY_WIDTH)  # noqa: T201
    print(f"  Number of Selected Policies:         {len(result_df):>10}")  # noqa: T201

    print("\n" + "=" * DISPLAY_WIDTH + "\n")  # noqa: T201


def display_results_with_distribution(
    result_df: pd.DataFrame, _gdp_impact: float, _revenue_impact: float
) -> None:
    """
    Display optimization results with distributional equality info.

    Same as display_results() but adds distributional range information.

    Args:
        result_df: DataFrame of selected policies
        _gdp_impact: Total GDP impact achieved (unused, kept for API compatibility)
        _revenue_impact: Total revenue impact achieved (unused, kept for API compatibility)
    """
    logger.section("OPTIMIZATION RESULTS (WITH DISTRIBUTIONAL EQUALITY)")

    # Separate policies by positive/negative revenue impact
    positive_revenue = result_df[result_df[COLUMNS["dynamic_revenue"]] >= 0].copy()
    negative_revenue = result_df[result_df[COLUMNS["dynamic_revenue"]] < 0].copy()

    # Sort by absolute impact
    positive_revenue = positive_revenue.sort_values(COLUMNS["dynamic_revenue"], ascending=False)
    negative_revenue = negative_revenue.sort_values(COLUMNS["dynamic_revenue"], ascending=True)

    if len(positive_revenue) > 0:
        logger.info(f"\n{'REVENUE RAISING POLICIES':^{DISPLAY_WIDTH}}")
        logger.info("-" * DISPLAY_WIDTH)
        for _, row in positive_revenue.iterrows():
            logger.info(
                f"  {row[COLUMNS['option']][:POLICY_NAME_MAX_LENGTH]:<{POLICY_NAME_MAX_LENGTH}}"
            )
            logger.info(
                f"    GDP: {row[COLUMNS['gdp']] * 100:>+7.4f}%  |  Revenue: ${row[COLUMNS['dynamic_revenue']]:>8.2f}B"
            )

    if len(negative_revenue) > 0:
        logger.info(f"\n{'REVENUE REDUCING POLICIES':^{DISPLAY_WIDTH}}")
        logger.info("-" * DISPLAY_WIDTH)
        for _, row in negative_revenue.iterrows():
            logger.info(
                f"  {row[COLUMNS['option']][:POLICY_NAME_MAX_LENGTH]:<{POLICY_NAME_MAX_LENGTH}}"
            )
            logger.info(
                f"    GDP: {row[COLUMNS['gdp']] * 100:>+7.4f}%  |  Revenue: ${row[COLUMNS['dynamic_revenue']]:>8.2f}B"
            )

    # Calculate totals for all metrics
    logger.section("FINAL SUMMARY - TOTAL IMPACT OF SELECTED POLICIES")
    logger.info(f"{'Economic Impacts':^{DISPLAY_WIDTH}}")
    logger.info("-" * DISPLAY_WIDTH)
    logger.info(
        f"  Long-Run Change in GDP:              {result_df[COLUMNS['gdp']].sum() * 100:>+8.4f}%"
    )
    logger.info(
        f"  Capital Stock:                       {result_df[COLUMNS['capital']].sum() * 100:>+8.4f}%"
    )
    logger.info(
        f"  Full-Time Equivalent Jobs:           {result_df[COLUMNS['jobs']].sum():>+10,.0f}"
    )
    logger.info(
        f"  Wage Rate:                           {result_df[COLUMNS['wage']].sum() * 100:>+8.4f}%"
    )

    logger.info(f"\n{'After-Tax Income Changes (by Income Percentile)':^{DISPLAY_WIDTH}}")
    logger.info("-" * DISPLAY_WIDTH)
    p20_total = result_df[COLUMNS["p20"]].sum() * 100
    p40_total = result_df[COLUMNS["p40_60"]].sum() * 100
    p80_total = result_df[COLUMNS["p80_100"]].sum() * 100
    p99_total = result_df[COLUMNS["p99"]].sum() * 100
    logger.info(f"  P20 (Bottom 20%):                    {p20_total:>+8.4f}%")
    logger.info(f"  P40-60 (Middle Class):               {p40_total:>+8.4f}%")
    logger.info(f"  P80-100 (Top 20%):                   {p80_total:>+8.4f}%")
    logger.info(f"  P99 (Top 1%):                        {p99_total:>+8.4f}%")

    # Calculate and display the range (max difference between any two groups)
    distro_values = [p20_total, p40_total, p80_total, p99_total]
    max_diff = max(distro_values) - min(distro_values)
    logger.info(f"\n  Distributional Range (max - min):    {max_diff:>+8.4f}%")
    logger.info("  (Constraint: must be ≤ 1.00%)")

    logger.info(f"\n{'Revenue Impacts':^{DISPLAY_WIDTH}}")
    logger.info("-" * DISPLAY_WIDTH)
    logger.info(
        f"  Static 10-Year Revenue:              ${result_df[COLUMNS['static_revenue']].sum():>10.2f} billion"
    )
    logger.info(
        f"  Dynamic 10-Year Revenue:             ${result_df[COLUMNS['dynamic_revenue']].sum():>10.2f} billion"
    )

    logger.info(f"\n{'Policy Count':^{DISPLAY_WIDTH}}")
    logger.info("-" * DISPLAY_WIDTH)
    logger.info(f"  Number of Selected Policies:         {len(result_df):>10}\n")
