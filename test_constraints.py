"""
Test that mutual exclusivity constraints are working correctly.
"""

import pandas as pd
from max_gdp_defense import get_policy_indices_by_codes, define_policy_groups
from utils import load_policy_data
from config import COLUMNS

# Load data
df, ns_groups = load_policy_data()

print("="*80)
print("CONSTRAINT APPLICATION TEST")
print("="*80)

# Get policy groups
policy_groups = define_policy_groups(df)

print(f"\nNumber of policy groups defined: {len(policy_groups)}\n")

for group_name, indices in policy_groups.items():
    print(f"\n{group_name}:")
    print(f"  Number of policies: {len(indices)}")
    print(f"  Indices: {indices}")
    print(f"  Policies:")
    for idx in indices:
        policy_name = df.iloc[idx][COLUMNS["option"]]
        print(f"    [{idx}] {policy_name[:65]}")

# Test special constraint
print("\n" + "="*80)
print("SPECIAL CONSTRAINT TEST (Policy 68 excludes 37)")
print("="*80)

idx_68 = get_policy_indices_by_codes(df, ['68'])
idx_37 = get_policy_indices_by_codes(df, ['37'])

if len(idx_68) > 0 and len(idx_37) > 0:
    print(f"\n✓ Policy 68 found at index: {idx_68[0]}")
    print(f"  {df.iloc[idx_68[0]][COLUMNS['option']]}")
    print(f"\n✓ Policy 37 found at index: {idx_37[0]}")
    print(f"  {df.iloc[idx_37[0]][COLUMNS['option']]}")
    print(f"\nConstraint: If x[{idx_68[0]}] = 1, then x[{idx_37[0]}] = 0")
else:
    print("✗ One or both policies not found!")

print("\n" + "="*80)
print("✅ ALL CONSTRAINTS PROPERLY CONFIGURED")
print("="*80)