"""
IJAB Economic Scenario Analysis - Main Entry Point

This repository contains optimization scripts for analyzing economic policy scenarios
using linear programming. The scripts use the Gurobi optimizer to select optimal
policy combinations that maximize GDP growth while satisfying various constraints.

Available Scripts:
- max_gdp.py: Basic GDP maximization with revenue neutrality
- max_gdp_equal_distro.py: GDP maximization with distributional equality constraints
- max_gdp_defense270.py: GDP maximization with national security and equity constraints

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
    print("  - max_gdp.py: Basic GDP maximization")
    print("  - max_gdp_equal_distro.py: GDP maximization with distributional equality")
    print("  - max_gdp_defense270.py: GDP maximization with national security constraints")
    print("\nRun any script with: python <script_name>")


if __name__ == "__main__":
    main()
