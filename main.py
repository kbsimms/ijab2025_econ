"""
IJAB Economic Scenario Analysis - Main Entry Point

This repository contains optimization scripts for analyzing economic policy scenarios
using linear programming. The scripts use the Gurobi optimizer to select optimal
policy combinations that maximize GDP growth while satisfying various constraints.

Project Structure:
- config.py: Centralized configuration and constants
- utils.py: Shared utility functions for data loading and display
- max_gdp.py: Basic GDP maximization with revenue neutrality and NS mutual exclusivity
- max_gdp_equal_distro.py: GDP maximization with distributional equality and NS mutual exclusivity
- max_gdp_defense.py: GDP maximization with NS spending, equity, and NS mutual exclusivity (parameterized)

Common Features Across All Scripts:
1. Revenue Neutrality: Ensures fiscal responsibility (no deficit increase)
2. National Security (NS) Mutual Exclusivity: Prevents selecting conflicting NS policies
   - Example: If policies NS1A, NS1B, and NS1C exist, only one can be selected
   - Applies to all NS policy groups (NS1-NS7, etc.)
3. Two-stage optimization for optimal solutions with intelligent tiebreaking
4. Consistent column naming and code structure for maintainability

New in v3.0:
- Centralized configuration (config.py) for all settings and constants
- Shared utilities (utils.py) eliminating code duplication
- Consolidated defense scripts into single parameterized max_gdp_defense.py
- Type hints and comprehensive documentation throughout
- Standardized column naming across all scripts

For more information, see README.md
"""


def main() -> None:
    """
    Main entry point for the IJAB Economic Scenario Analysis project.
    
    This is a placeholder main function. To run the optimization scripts,
    execute them directly:
    
    Examples:
        # Basic GDP maximization
        python max_gdp.py
        
        # GDP maximization with distributional equality
        python max_gdp_equal_distro.py
        
        # GDP maximization with national security and equity constraints
        python max_gdp_defense.py                    # Default: $3,000B NS spending
        python max_gdp_defense.py --spending 3000    # Explicit: $3,000B
        python max_gdp_defense.py --spending 4000    # Increased: $4,000B
    """
    print("="*80)
    print("IJAB Economic Scenario Analysis".center(80))
    print("="*80)
    print("\nVersion 3.0 - Refactored for Consistency & Maintainability")
    print("\nProject Structure:")
    print("  config.py         - Centralized configuration and constants")
    print("  utils.py          - Shared utility functions")
    print("  max_gdp.py        - Basic GDP maximization")
    print("  max_gdp_equal_distro.py - GDP with distributional equality")
    print("  max_gdp_defense.py      - GDP with defense & equity (parameterized)")
    
    print("\nAvailable optimization scripts:")
    print("\n1. max_gdp.py")
    print("   Purpose: Basic GDP maximization with revenue neutrality and NS mutual exclusivity")
    print("   Run: python max_gdp.py")
    
    print("\n2. max_gdp_equal_distro.py")
    print("   Purpose: GDP maximization with distributional equality and NS mutual exclusivity")
    print("   Run: python max_gdp_equal_distro.py")
    
    print("\n3. max_gdp_defense.py")
    print("   Purpose: GDP maximization with NS spending requirements, equity, and NS mutual exclusivity")
    print("   Run: python max_gdp_defense.py [--spending AMOUNT]")
    print("   Examples:")
    print("     python max_gdp_defense.py                # Default: $3,000B")
    print("     python max_gdp_defense.py --spending 4000  # Increased: $4,000B")
    
    print("\nAll scripts include National Security (NS) mutual exclusivity constraints:")
    print("  - Prevents selecting conflicting policies within the same NS group")
    print("  - Example: Only one of NS1A, NS1B, or NS1C can be selected")
    
    print("\nKey Improvements in v3.0:")
    print("  ✓ Centralized configuration for easy maintenance")
    print("  ✓ Shared utilities eliminating ~600 lines of duplicate code")
    print("  ✓ Consistent column naming across all scripts")
    print("  ✓ Type hints for better IDE support")
    print("  ✓ Comprehensive documentation")
    
    print("\nFor detailed information, see README.md")
    print("="*80)


if __name__ == "__main__":
    main()
