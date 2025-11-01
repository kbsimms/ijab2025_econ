"""
Input Validation Module for IJAB Economic Scenario Analysis.

This module provides comprehensive validation for data inputs, configuration,
and policy structures to ensure robustness and early error detection.

Key validation functions:
- validate_excel_file(): Check file existence and structure
- validate_dataframe(): Verify DataFrame has required columns and data types
- validate_ns_policy_name(): Ensure NS policies follow naming conventions
- validate_policy_indices(): Check policy index lists are valid
- validate_spending_level(): Ensure spending levels are within feasible range
"""

from typing import List, Optional, Set, Tuple
from pathlib import Path
import pandas as pd
import re

from config import (
    EXCEL_FILE_PATH,
    SHEET_NAME,
    COLUMNS,
    NUMERIC_COLUMNS,
    NS_PATTERN,
    NS_STRICT_PATTERN
)


class ValidationError(Exception):
    """Custom exception for validation failures with actionable messages."""
    pass


def validate_excel_file(file_path: str = EXCEL_FILE_PATH) -> None:
    """
    Validate that Excel file exists and is readable.
    
    Args:
        file_path: Path to Excel file
        
    Raises:
        ValidationError: If file doesn't exist or can't be read
    """
    path = Path(file_path)
    
    if not path.exists():
        raise ValidationError(
            f"Excel file not found: {file_path}\n"
            f"Please ensure the file exists in the current directory.\n"
            f"Expected location: {path.absolute()}"
        )
    
    if not path.is_file():
        raise ValidationError(
            f"Path exists but is not a file: {file_path}\n"
            f"Please provide a valid Excel file path."
        )
    
    if path.suffix.lower() not in ['.xlsx', '.xls']:
        raise ValidationError(
            f"File does not appear to be an Excel file: {file_path}\n"
            f"Expected .xlsx or .xls extension, got: {path.suffix}"
        )
    
    # Try to open the file to ensure it's readable
    try:
        _ = pd.ExcelFile(file_path)
    except Exception as e:
        raise ValidationError(
            f"Excel file exists but cannot be read: {file_path}\n"
            f"Error: {str(e)}\n"
            f"The file may be corrupted or currently open in Excel."
        )


def validate_sheet_exists(file_path: str, sheet_name: str = SHEET_NAME) -> None:
    """
    Validate that specified sheet exists in Excel file.
    
    Args:
        file_path: Path to Excel file
        sheet_name: Name of sheet to validate
        
    Raises:
        ValidationError: If sheet doesn't exist
    """
    try:
        xls = pd.ExcelFile(file_path)
        if sheet_name not in xls.sheet_names:
            raise ValidationError(
                f"Sheet '{sheet_name}' not found in {file_path}\n"
                f"Available sheets: {', '.join(str(s) for s in xls.sheet_names)}\n"
                f"Please check the SHEET_NAME configuration in config.py"
            )
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(f"Error reading Excel file sheets: {str(e)}")


def validate_dataframe(df: pd.DataFrame, require_ns_policies: bool = True) -> None:
    """
    Validate that DataFrame has required structure and data.
    
    Args:
        df: DataFrame to validate
        require_ns_policies: If True, require at least one NS policy
        
    Raises:
        ValidationError: If DataFrame is invalid
    """
    if df is None or df.empty:
        raise ValidationError("DataFrame is empty or None")
    
    # Check required columns exist
    required_cols = list(COLUMNS.values())
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValidationError(
            f"Missing required columns in DataFrame:\n"
            f"  Missing: {', '.join(missing_cols)}\n"
            f"  Available: {', '.join(df.columns.tolist())}\n"
            f"Please check that Excel file has correct column headers in row 2."
        )
    
    # Check that we have data rows
    if len(df) == 0:
        raise ValidationError(
            "No policy data found in DataFrame.\n"
            "Please check that Excel file has data starting from row 4."
        )
    
    # Validate numeric columns have numeric data
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
            
        non_numeric = df[col].apply(lambda x: not isinstance(x, (int, float)) and pd.notna(x))
        if non_numeric.any():
            bad_rows = df[non_numeric].index.tolist()[:5]  # Show first 5
            raise ValidationError(
                f"Column '{col}' contains non-numeric values.\n"
                f"First few problematic rows: {bad_rows}\n"
                f"Please ensure all numeric columns contain only numbers."
            )
    
    # Check for NS policies if required
    if require_ns_policies:
        ns_pattern = re.compile(NS_PATTERN, re.IGNORECASE)
        ns_count = df[COLUMNS["option"]].str.contains(ns_pattern, na=False).sum()
        
        if ns_count == 0:
            raise ValidationError(
                "No National Security (NS) policies found.\n"
                "NS policies must follow pattern: NSxY: (e.g., NS1A:, NS2B:)\n"
                "Please check policy naming in Excel file."
            )


def validate_ns_policy_name(policy_name: str) -> Tuple[bool, Optional[str]]:
    """
    Validate NS policy naming convention.
    
    Args:
        policy_name: Policy name to validate
        
    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message is None
        
    Examples:
        >>> validate_ns_policy_name("NS1A: Increase defense")
        (True, None)
        >>> validate_ns_policy_name("NS1: Missing letter")
        (False, "NS policy must include letter after number")
    """
    if not policy_name.startswith("NS"):
        return False, "Policy name must start with 'NS'"
    
    # Extract the code part before colon
    if ":" not in policy_name:
        return False, "NS policy must include colon after code (e.g., 'NS1A:')"
    
    code = policy_name.split(":")[0].strip()
    
    # Check pattern: NS + digits + letter
    pattern = re.compile(r"^NS(\d+)([A-Z])$")
    match = pattern.match(code)
    
    if not match:
        return False, (
            f"NS policy code '{code}' is invalid.\n"
            f"Expected format: NS<digits><letter> (e.g., NS1A, NS2B, NS15C)"
        )
    
    return True, None


def validate_policy_indices(
    indices: List[int],
    df_length: int,
    context: str = "policy"
) -> None:
    """
    Validate that policy indices are within valid range.
    
    Args:
        indices: List of indices to validate
        df_length: Length of DataFrame (max valid index is df_length - 1)
        context: Description of what these indices represent (for error messages)
        
    Raises:
        ValidationError: If any index is out of range
    """
    if not indices:
        return  # Empty list is valid
    
    invalid = [idx for idx in indices if idx < 0 or idx >= df_length]
    
    if invalid:
        raise ValidationError(
            f"Invalid {context} indices: {invalid}\n"
            f"Indices must be in range [0, {df_length - 1}]\n"
            f"DataFrame length: {df_length}"
        )


def validate_spending_level(
    spending: int,
    min_spending: int = -10000,
    max_spending: int = 10000
) -> None:
    """
    Validate that spending level is within reasonable range.
    
    Args:
        spending: Spending level in billions
        min_spending: Minimum allowed spending (default: -10,000B)
        max_spending: Maximum allowed spending (default: +10,000B)
        
    Raises:
        ValidationError: If spending is out of range
    """
    if not isinstance(spending, (int, float)):
        raise ValidationError(
            f"Spending level must be numeric, got: {type(spending).__name__}"
        )
    
    if spending < min_spending or spending > max_spending:
        raise ValidationError(
            f"Spending level ${spending:,}B is outside valid range.\n"
            f"Valid range: ${min_spending:,}B to ${max_spending:,}B\n"
            f"Please adjust spending requirement or update validation limits."
        )


def validate_ns_groups(
    ns_groups: dict,
    df: pd.DataFrame,
    min_groups: int = 1
) -> None:
    """
    Validate NS group structure and indices.
    
    Args:
        ns_groups: Dict mapping group names to policy indices
        df: DataFrame containing policies
        min_groups: Minimum number of NS groups expected
        
    Raises:
        ValidationError: If NS groups are invalid
    """
    if not ns_groups:
        if min_groups > 0:
            raise ValidationError(
                f"Expected at least {min_groups} NS policy groups, found none.\n"
                f"Please check that NS policies exist and follow naming convention."
            )
        return
    
    if len(ns_groups) < min_groups:
        raise ValidationError(
            f"Expected at least {min_groups} NS policy groups, found {len(ns_groups)}.\n"
            f"Groups found: {list(ns_groups.keys())}"
        )
    
    # Validate each group's indices
    for group_name, indices in ns_groups.items():
        if not indices:
            raise ValidationError(f"NS group '{group_name}' has no policies")
        
        validate_policy_indices(indices, len(df), f"NS group '{group_name}'")
        
        # Validate that all policies in group actually match the NS pattern
        for idx in indices:
            policy_name = df.iloc[idx][COLUMNS["option"]]
            if not policy_name.startswith("NS"):
                raise ValidationError(
                    f"Policy at index {idx} in NS group '{group_name}' "
                    f"doesn't start with 'NS': {policy_name}"
                )


def validate_optimization_inputs(
    df: pd.DataFrame,
    ns_groups: dict,
    ns_strict_indices: List[int],
    min_ns_spending: int
) -> None:
    """
    Comprehensive validation before running optimization.
    
    This is a pre-flight check that validates all inputs to ensure
    the optimization is likely to succeed.
    
    Args:
        df: Policy DataFrame
        ns_groups: NS mutual exclusivity groups
        ns_strict_indices: Indices of NS1-NS7 policies
        min_ns_spending: Required NS spending level
        
    Raises:
        ValidationError: If any validation fails
    """
    # Validate DataFrame structure
    validate_dataframe(df, require_ns_policies=True)
    
    # Validate NS groups
    validate_ns_groups(ns_groups, df, min_groups=1)
    
    # Validate NS strict indices
    validate_policy_indices(ns_strict_indices, len(df), "NS strict")
    
    if not ns_strict_indices:
        raise ValidationError(
            "No NS1-NS7 policies found for defense spending constraint.\n"
            "At least one NS1-NS7 policy is required for defense optimization."
        )
    
    # Validate spending level
    validate_spending_level(min_ns_spending)
    
    # Check that NS strict policies have revenue data
    if COLUMNS["dynamic_revenue"] in df.columns:
        ns_policies = df.iloc[ns_strict_indices]
        missing_revenue = ns_policies[COLUMNS["dynamic_revenue"]].isna()
        
        if missing_revenue.any():
            bad_policies = ns_policies[missing_revenue][COLUMNS["option"]].tolist()
            raise ValidationError(
                f"NS1-NS7 policies missing revenue data:\n"
                f"  {', '.join(bad_policies)}\n"
                f"All NS policies must have revenue values for spending constraints."
            )


def validate_output_directory(directory: Path) -> None:
    """
    Validate and create output directory if needed.
    
    Args:
        directory: Path to output directory
        
    Raises:
        ValidationError: If directory cannot be created or accessed
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise ValidationError(
            f"Cannot create output directory: {directory}\n"
            f"Error: {str(e)}\n"
            f"Please check directory permissions."
        )
    
    if not directory.is_dir():
        raise ValidationError(
            f"Output path exists but is not a directory: {directory}"
        )
    
    # Test write permissions
    test_file = directory / ".write_test"
    try:
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        raise ValidationError(
            f"No write permission in output directory: {directory}\n"
            f"Error: {str(e)}"
        )