"""
IJAB Economic Scenario Analysis - Main Entry Point

This repository contains optimization scripts for analyzing economic policy scenarios
using linear programming. The scripts use the Gurobi optimizer to select optimal
policy combinations that maximize GDP growth while satisfying various constraints.

Available Scripts:
- max_gdp.py: Basic GDP maximization with revenue neutrality and NS mutual exclusivity
- max_gdp_equal_distro.py: GDP maximization with distributional equality and NS mutual exclusivity
- max_gdp_defense270.py: GDP maximization with national security spending, equity, and NS mutual exclusivity

Common Features Across All Scripts:
1. Revenue Neutrality: Ensures fiscal responsibility (no deficit increase)
2. National Security (NS) Mutual Exclusivity: Prevents selecting conflicting NS policies
   - Example: If policies NS1A, NS1B, and NS1C exist, only one can be selected
   - Applies to all NS policy groups (NS1-NS7, etc.)
3. Two-stage optimization for optimal solutions with intelligent tiebreaking

For more information, see README.md
"""


def main():
    """
    Main entry point for the IJAB Economic Scenario Analysis project.
    
    This is a placeholder main function. To run the optimization scripts,
    execute them directly:
    
    Examples:
        python max_gdp.py
        python max_gdp_equal_distro.py
        python max_gdp_defense270.py
    """
    print("Hello from ijab-econ-scenario-analysis!")
    print("\nAvailable optimization scripts:")
    print("  - max_gdp.py: Basic GDP maximization with revenue neutrality and NS mutual exclusivity")
    print("  - max_gdp_equal_distro.py: GDP maximization with distributional equality and NS mutual exclusivity")
    print("  - max_gdp_defense270.py: GDP maximization with NS spending requirements, equity, and NS mutual exclusivity")
    print("\nAll scripts include National Security (NS) mutual exclusivity constraints:")
    print("  - Prevents selecting conflicting policies within the same NS group")
    print("  - Example: Only one of NS1A, NS1B, or NS1C can be selected")
    print("\nRun any script with: python <script_name>")


if __name__ == "__main__":
    main()
