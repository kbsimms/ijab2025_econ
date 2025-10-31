"""
Final comprehensive verification of all mutual exclusivity constraints.
Checks multiple output files to ensure constraints work across different spending levels.
"""

import pandas as pd
from config import COLUMNS

def check_constraints(filename):
    """Check all mutual exclusivity constraints for a given output file."""
    df = pd.read_csv(filename)
    
    # Define constraint groups
    groups = {
        'Corporate Tax {11,36,68}': ['^11:', '^36:', '^68:'],
        'Gas Tax {47,48}': ['^47:', '^48:'],
        'Estate Tax {12,44,46,69}': ['^12:', '^44:', '^46:', '^69:'],
        'CTC Refund {53,54}': ['^53:', '^54:'],
        'SS Payroll {34,35}': ['^34:', '^35:'],
        'Payroll Rate {4,33}': ['^4:', '^33:'],
        'EITC {21,51,52,55,S15}': ['^21:', '^51:', '^52:', '^55:', '^S15:'],
        'Individual Tax {1,3,14,59}': ['^1:', '^3:', '^14:', '^59:'],
        'CTC Comp {19,20,55,S13}': ['^19:', '^20:', '^55:', '^S13:'],
        'Sec199A {10,38}': ['^10:', '^38:'],
        'Mortgage {23,24}': ['^23:', '^24:'],
        'Charitable {25,58}': ['^25:', '^58:'],
        'Cap Gains {5,29,30}': ['^5:', '^29:', '^30:'],
        'Depreciation {7,40,65}': ['^7:', '^40:', '^65:'],
        'VAT {43,68}': ['^43:', '^68:']
    }
    
    violations = []
    for group_name, patterns in groups.items():
        count = 0
        selected = []
        for pattern in patterns:
            matches = df[df['Option'].str.match(pattern, na=False)]
            if len(matches) > 0:
                count += len(matches)
                selected.extend(matches['Option'].str[:30].tolist())
        
        if count > 1:
            violations.append(f"  ✗ {group_name}: {count} policies - {selected}")
    
    # Check special constraint
    has_68 = len(df[df['Option'].str.match('^68:', na=False)]) > 0
    has_37 = len(df[df['Option'].str.match('^37:', na=False)]) > 0
    if has_68 and has_37:
        violations.append(f"  ✗ SPECIAL: Both 68 and 37 present!")
    
    return violations

# Test multiple spending levels
test_files = [
    'outputs/defense/max_gdp_defense-4000.csv',
    'outputs/defense/max_gdp_defense0.csv',
    'outputs/defense/max_gdp_defense3000.csv',
    'outputs/defense/max_gdp_defense6000.csv'
]

print("="*80)
print("COMPREHENSIVE CONSTRAINT VERIFICATION")
print("="*80)

all_passed = True
for filename in test_files:
    spending_level = filename.split('defense')[-1].replace('.csv', '')
    print(f"\nChecking {spending_level}B spending level...")
    
    violations = check_constraints(filename)
    
    if violations:
        print(f"  ❌ VIOLATIONS FOUND:")
        for v in violations:
            print(v)
        all_passed = False
    else:
        print(f"  ✅ All constraints satisfied")

print("\n" + "="*80)
if all_passed:
    print("✅ PERFECT! ALL CONSTRAINTS WORKING CORRECTLY ACROSS ALL SPENDING LEVELS")
else:
    print("❌ CONSTRAINT VIOLATIONS DETECTED")
print("="*80)