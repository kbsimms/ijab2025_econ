"""
Policy Selection Analysis Across Defense Spending Levels

WHAT THIS SCRIPT DOES:
Creates a visual "heatmap" showing exactly which tax and spending policies are
selected at each defense spending requirement level. Think of it as answering:
"Which specific policies get chosen when we require different defense spending amounts?"

The heatmap makes it easy to see:
- Which policies are ALWAYS selected (robust across all scenarios)
- Which policies are NEVER selected (dominated by better options)
- Which policies substitute for each other as spending requirements change
- How policy choices shift between defense spending levels

This is extremely useful for understanding:
- Policy stability: Which recommendations are consistent?
- Policy flexibility: Where do choices change?
- Substitution patterns: Which policies compete with each other?

HOW IT WORKS:
1. Loads all CSV files from max_gdp_defense.py runs
2. Extracts which policies were selected at each spending level
3. Creates a grid: rows=policies, columns=spending levels
4. Colors: Green=selected, Gray=not selected

PREREQUISITES:
Must first run: python max_gdp_defense.py
This generates the CSV files this script analyzes.

USAGE:
    python visualize_policy_selection.py

OUTPUT:
Creates 'outputs/defense/policy_selection_heatmap.png'
- Each row is one policy option
- Each column is one defense spending level
- Green cells show when a policy is selected
- Patterns reveal policy substitution as requirements change

National Security (NS) policies are excluded from the main heatmap since
they're the primary variable being changed across scenarios.

INTERPRETING THE HEATMAP:
- Solid green row = Policy selected at all spending levels (robust recommendation)
- Solid gray row = Policy never selected (dominated by alternatives)
- Striped pattern = Policy selected only at certain spending levels (conditional choice)
- Vertical patterns = Major policy substitution at specific spending thresholds
"""

import argparse
from pathlib import Path
from typing import Any

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore[import-untyped]

from config import COLUMNS, SPENDING_RANGE
from logger import LogLevel, get_logger

# Initialize logger
logger = get_logger(__name__, level=LogLevel.INFO)

# Constants
POLICY_DESC_LONG = 70
POLICY_DESC_MEDIUM = 60
POLICY_DESC_SHORT = 50
TOP_N_POLICIES = 10

# Global configuration (set by main)
output_dir: Path
file_prefix: str
spending_levels = list(range(SPENDING_RANGE["min"], SPENDING_RANGE["max"], SPENDING_RANGE["step"]))


def load_policy_data() -> dict[int, dict[str, Any]]:
    """Load policy selection data from all defense spending CSV files."""
    policy_data: dict[int, dict[str, Any]] = {}

    logger.info(f"Loading policy selection data for {len(spending_levels)} spending levels...")
    for level in spending_levels:
        file_path = output_dir / f"{file_prefix}{level}.csv"
        try:
            df = pd.read_csv(file_path)
            # All policies in the CSV are selected policies (the file only contains selected policies)
            selected = df[COLUMNS["option"]].tolist()
            policy_data[level] = {"df": df, "selected": selected}
            logger.info(f"  [OK] Loaded {file_path.name}: {len(selected)} policies selected")
        except FileNotFoundError:
            logger.warning(f"  [MISSING] {file_path.name}")
            policy_data[level] = {"df": None, "selected": []}

    return policy_data


def extract_policy_number(policy_name: str) -> str:
    """Extract policy number/code from policy name for sorting."""
    # Handle different policy types: numbered (1-67), S-codes (S1-S17), NS-codes (NS1-NS7)
    if policy_name.startswith("NS"):
        # Extract NS code (e.g., "NS1B" -> "NS01B" for sorting)
        parts = policy_name.split(":")[0].strip()
        num = "".join(filter(str.isdigit, parts))
        letter = "".join(filter(str.isalpha, parts[2:]))  # Get letter after NS
        return f"NS{int(num):02d}{letter}"
    if policy_name.startswith("S"):
        # Extract S code (e.g., "S1" -> "S01")
        num = "".join(filter(str.isdigit, policy_name.split(":")[0]))
        return f"S{int(num):02d}"
    # Extract regular number (e.g., "1:" -> "001")
    num = "".join(filter(str.isdigit, policy_name.split(":")[0]))
    if num:
        return f"{int(num):03d}"
    return policy_name


def create_heatmap(
    policy_data: dict[int, dict[str, Any]],
) -> tuple[list[Any], np.ndarray[Any, Any]]:
    """Create a heatmap showing policy selections across defense spending levels (excluding NS policies)."""
    logger.info("Creating policy selection heatmap (excluding National Security policies)...")

    # Get all unique policies across all spending levels, excluding NS policies
    all_policies: set[Any] = set()
    for data in policy_data.values():
        if data["df"] is not None:
            # Filter out NS policies
            non_ns_policies = [p for p in data["df"]["Option"].tolist() if not p.startswith("NS")]
            all_policies.update(non_ns_policies)

    all_policies_sorted: list[Any] = sorted(all_policies, key=extract_policy_number)

    # Create matrix: rows = policies, columns = spending levels
    # Filter out NS policies from selected policies
    matrix = np.zeros((len(all_policies_sorted), len(spending_levels)))

    for col_idx, level in enumerate(spending_levels):
        selected = policy_data[level]["selected"]
        # Filter out NS policies from selections
        selected_non_ns = [p for p in selected if not p.startswith("NS")]
        for row_idx, policy in enumerate(all_policies_sorted):
            if policy in selected_non_ns:
                matrix[row_idx, col_idx] = 1

    # Create figure
    _fig, ax = plt.subplots(figsize=(20, 24))

    # Truncate policy names for display
    policy_labels = []
    for policy in all_policies_sorted:
        # Extract policy code and first few words
        if ":" in policy:
            code, desc = policy.split(":", 1)
            # Limit description to POLICY_DESC_MEDIUM characters
            if len(desc) > POLICY_DESC_MEDIUM:
                desc = desc[:57] + "..."
            policy_labels.append(f"{code}:{desc}")
        else:
            policy_labels.append(policy[:POLICY_DESC_LONG])

    # Create heatmap
    sns.heatmap(
        matrix,
        cmap=["#f0f0f0", "#2E7D32"],  # Light gray for unselected, green for selected
        cbar_kws={"label": "Selected"},
        yticklabels=policy_labels,
        xticklabels=[f"${level:+,}B" for level in spending_levels],
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )

    ax.set_title(
        "Tax & Spending Policy Selection Across Defense Spending Levels\n(Green = Selected, Gray = Not Selected, NS Policies Excluded)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Defense Spending Change (Billions)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Policy Options", fontsize=12, fontweight="bold")

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Adjust layout
    plt.tight_layout()

    # Save
    output_file = output_dir / "policy_selection_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"[OK] Heatmap saved to '{output_file}'")
    plt.close()

    return all_policies_sorted, matrix


def create_frequency_chart(all_policies: list[Any], matrix: np.ndarray[Any, Any]) -> pd.DataFrame:
    """Create a bar chart showing how often each policy is selected."""
    logger.info("Creating policy frequency chart...")

    # Calculate frequency for each policy (sum across spending levels)
    frequencies = matrix.sum(axis=1)

    # Create DataFrame for easier plotting
    freq_df = pd.DataFrame(
        {
            "Policy": all_policies,
            "Frequency": frequencies,
            "Percentage": (frequencies / len(spending_levels)) * 100,
        }
    )

    # Sort by frequency
    freq_df = freq_df.sort_values("Frequency", ascending=True)

    # Categorize policies
    def categorize_policy(policy: str) -> str:
        if policy.startswith("NS"):
            return "National Security"
        if policy.startswith("S"):
            return "Spending"
        return "Tax"

    freq_df["Category"] = freq_df["Policy"].apply(categorize_policy)

    # Create figure with two subplots
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 24))

    # Subplot 1: All policies
    colors = freq_df["Category"].map(
        {"Tax": "#1f77b4", "Spending": "#ff7f0e", "National Security": "#d62728"}
    )

    # Truncate policy names
    policy_labels = []
    for policy in freq_df["Policy"]:
        if ":" in policy:
            code, desc = policy.split(":", 1)
            if len(desc) > POLICY_DESC_SHORT:
                desc = desc[:47] + "..."
            policy_labels.append(f"{code}:{desc}")
        else:
            policy_labels.append(policy[:60])

    y_pos = np.arange(len(freq_df))
    ax1.barh(y_pos, freq_df["Frequency"], color=colors)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(policy_labels, fontsize=7)
    ax1.set_xlabel(
        "Number of Times Selected (out of 21 spending levels)", fontsize=11, fontweight="bold"
    )
    ax1.set_title(
        "Policy Selection Frequency\n(All Policies)", fontsize=14, fontweight="bold", pad=15
    )
    ax1.grid(axis="x", alpha=0.3)

    # Add legend
    legend_elements = [
        Patch(facecolor="#1f77b4", label="Tax Policies"),
        Patch(facecolor="#ff7f0e", label="Spending Policies"),
        Patch(facecolor="#d62728", label="National Security Policies"),
    ]
    ax1.legend(handles=legend_elements, loc="lower right")

    # Subplot 2: Category summary
    category_counts = freq_df.groupby("Category")["Frequency"].agg(["mean", "sum", "count"])
    category_counts["avg_percentage"] = (category_counts["mean"] / len(spending_levels)) * 100

    categories = category_counts.index
    x_pos = np.arange(len(categories))

    bars = ax2.bar(
        x_pos, category_counts["avg_percentage"], color=["#d62728", "#ff7f0e", "#1f77b4"]
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(categories, rotation=45, ha="right")
    ax2.set_ylabel("Average Selection Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_title(
        "Average Selection Rate by Policy Category", fontsize=14, fontweight="bold", pad=15
    )
    ax2.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, category_counts["avg_percentage"], strict=True)):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.1f}%\n({int(category_counts.iloc[i]['count'])} policies)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()

    # Save
    output_file = output_dir / "policy_frequency_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"[OK] Frequency chart saved to '{output_file}'")
    plt.close()

    return freq_df


def create_defense_substitution_chart(policy_data: dict[int, dict[str, Any]]) -> None:
    """Create a heatmap showing NS (National Security) policy substitutions."""
    logger.info("Creating defense policy substitution heatmap...")

    # Extract NS policies across spending levels
    ns_selections: dict[int, list[Any]] = {}
    for level in spending_levels:
        selected = policy_data[level]["selected"]
        ns_policies = [p for p in selected if p.startswith("NS")]
        ns_selections[level] = ns_policies

    # Get all unique NS policies
    all_ns_policies_set: set[Any] = set()
    for policies in ns_selections.values():
        all_ns_policies_set.update(policies)
    all_ns_policies: list[Any] = sorted(all_ns_policies_set, key=extract_policy_number)

    if not all_ns_policies:
        logger.warning("No NS policies found in selections")
        return

    # Create matrix for NS policies only
    ns_matrix = np.zeros((len(all_ns_policies), len(spending_levels)))
    for col_idx, level in enumerate(spending_levels):
        for row_idx, policy in enumerate(all_ns_policies):
            if policy in ns_selections[level]:
                ns_matrix[row_idx, col_idx] = 1

    # Create figure with single subplot
    _fig, ax = plt.subplots(figsize=(18, 10))

    # Create labels for NS policies
    ns_labels = []
    for policy in all_ns_policies:
        if ":" in policy:
            code, desc = policy.split(":", 1)
            if len(desc) > POLICY_DESC_LONG:
                desc = desc[:67] + "..."
            ns_labels.append(f"{code}:{desc}")
        else:
            ns_labels.append(policy[:80])

    # Create heatmap
    sns.heatmap(
        ns_matrix,
        cmap=["#ffebee", "#c62828"],  # Light red for unselected, dark red for selected
        cbar_kws={"label": "Selected"},
        yticklabels=ns_labels,
        xticklabels=[f"${level:+,}B" for level in spending_levels],
        linewidths=1,
        linecolor="white",
        ax=ax,
    )

    ax.set_title(
        "National Security Policy Substitutions Across Defense Spending Levels",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Defense Spending Change (Billions)", fontsize=12, fontweight="bold")
    ax.set_ylabel("NS Policy Options", fontsize=12, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    # Save
    output_file = output_dir / "defense_policy_substitution.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"[OK] Defense substitution heatmap saved to '{output_file}'")
    plt.close()

    # Print summary
    ns_counts = [len(ns_selections[level]) for level in spending_levels]
    logger.info("Defense Policy Summary:")
    logger.info(f"  Total unique NS policies: {len(all_ns_policies)}")
    logger.info(
        f"  Max NS policies selected: {max(ns_counts)} (at ${spending_levels[ns_counts.index(max(ns_counts))]:+,}B)"
    )
    logger.info(
        f"  Min NS policies selected: {min(ns_counts)} (at ${spending_levels[ns_counts.index(min(ns_counts))]:+,}B)"
    )


def print_policy_insights(
    freq_df: pd.DataFrame, _all_policies: list[Any], _matrix: np.ndarray[Any, Any]
) -> None:
    """Print key insights about policy selections."""
    logger.section("POLICY SELECTION INSIGHTS")

    # Always selected policies
    always_selected = freq_df[freq_df["Frequency"] == len(spending_levels)]["Policy"].tolist()
    logger.info(f"\n1. ALWAYS SELECTED ({len(always_selected)} policies):")
    for policy in always_selected[:TOP_N_POLICIES]:  # Show first TOP_N_POLICIES
        logger.info(f"   - {policy}")
    if len(always_selected) > TOP_N_POLICIES:
        logger.info(f"   ... and {len(always_selected) - TOP_N_POLICIES} more")

    # Never selected policies
    never_selected = freq_df[freq_df["Frequency"] == 0]["Policy"].tolist()
    logger.info(f"\n2. NEVER SELECTED ({len(never_selected)} policies):")
    for policy in never_selected[:TOP_N_POLICIES]:  # Show first TOP_N_POLICIES
        logger.info(f"   - {policy}")
    if len(never_selected) > TOP_N_POLICIES:
        logger.info(f"   ... and {len(never_selected) - TOP_N_POLICIES} more")

    # Sometimes selected policies (most variable)
    mid_freq = freq_df[(freq_df["Frequency"] > 0) & (freq_df["Frequency"] < len(spending_levels))]
    mid_freq = mid_freq.sort_values("Frequency", ascending=False)
    logger.info(f"\n3. SOMETIMES SELECTED ({len(mid_freq)} policies):")
    logger.info("   Top 10 by selection frequency:")
    for _i, row in mid_freq.head(TOP_N_POLICIES).iterrows():
        logger.info(
            f"   - {row['Policy'][:70]}... ({int(row['Frequency'])}/{len(spending_levels)} times)"
        )

    # Category breakdown
    logger.info("\n4. SELECTION RATE BY CATEGORY:")
    category_stats = freq_df.groupby(
        freq_df["Policy"].apply(
            lambda p: "NS" if p.startswith("NS") else ("S" if p.startswith("S") else "Tax")
        )
    )["Percentage"].agg(["mean", "count"])

    for category, stats in category_stats.iterrows():
        logger.info(
            f"   {category:20s}: {stats['mean']:5.1f}% avg selection rate ({int(stats['count'])} policies)"
        )


def main() -> None:
    """Main execution function."""
    global output_dir, file_prefix

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Visualize policy selection patterns across defense spending levels."
    )
    parser.add_argument(
        "--senate",
        action="store_true",
        help="Use Senate portfolio optimization results from outputs/senate/",
    )
    args = parser.parse_args()

    # Configure paths based on mode
    if args.senate:
        output_dir = Path("outputs/senate")
        file_prefix = "policy_portfolio_"
        logger.info("Using Senate portfolio optimization results")
    else:
        output_dir = Path("outputs/defense")
        file_prefix = "max_gdp_defense"
        logger.info("Using defense optimization results")

    # Ensure output directory exists
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.exception("Cannot create output directory")
        raise

    logger.section("POLICY SELECTION ANALYSIS")

    # Load data
    policy_data = load_policy_data()

    # Create policy selection heatmap (excluding NS policies)
    _all_policies, _matrix = create_heatmap(policy_data)

    logger.info("\n" + "=" * 70)
    logger.info("Visualization complete!")
    logger.info(f"Check '{output_dir}' for output file:")
    logger.info("  - policy_selection_heatmap.png")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
