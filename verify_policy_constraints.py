"""
Verification script to check policy mutual exclusivity constraints.
This will load the data and verify that all policy codes are found correctly.
"""

import pandas as pd
from config import COLUMNS, EXCEL_FILE_PATH, SHEET_NAME, NUMERIC_COLUMNS
from utils import load_policy_data

def verify_policy_codes():
    """Verify that all policy codes in the constraints actually exist in the data."""
    
    # Load data
    df, ns_groups = load_policy_data()
    
    print("="*80)
    print("POLICY CODE VERIFICATION")
    print("="*80)
    
    # Define all policy codes we're looking for
    policy_code_groups = {
        'Corporate Tax': ['11', '36', '68'],
        'Gas Tax': ['47', '48'],
        'Estate Tax': ['12', '44', '46', '69'],
        'CTC Refundability': ['53', '54'],
        'SS Payroll Cap': ['34', '35'],
        'Payroll Rate': ['4', '33'],
        'EITC Reforms': ['21', '51', '52', '55', 'S15'],
        'Individual Tax Structure': ['1', '3', '14', '59'],
        'CTC Comprehensive': ['19', '20', '55', 'S13'],
        'Section 199A': ['10', '38'],
        'Mortgage Deduction': ['23', '24'],
        'Charitable Deduction': ['25', '58'],
        'Capital Gains': ['5', '29', '30'],
        'Depreciation': ['7', '40', '65'],
        'VAT': ['43', '68']
    }
    
    print(f"\nTotal policies in dataset: {len(df)}\n")
    
    all_found = True
    for group_name, codes in policy_code_groups.items():
        print(f"\n{group_name}: {codes}")
        for code in codes:
            # Search for this code
            matching = df[df[COLUMNS["option"]].str.match(f"^{code}:", na=False)]
            if len(matching) > 0:
                policy_name = matching.iloc[0][COLUMNS["option"]]
                idx = matching.index[0]
                pos_idx = df.index.get_loc(idx)
                print(f"  ✓ {code}: Found at position {pos_idx}")
                print(f"    Full name: {policy_name[:70]}")
            else:
                print(f"  ✗ {code}: NOT FOUND!")
                all_found = False
    
    print("\n" + "="*80)
    if all_found:
        print("✅ ALL POLICY CODES FOUND SUCCESSFULLY")
    else:
        print("❌ SOME POLICY CODES NOT FOUND - CHECK DATA!")
    print("="*80)
    
    # Also check the special constraint policies
    print("\nSPECIAL CONSTRAINT VERIFICATION:")
    for code in ['68', '37']:
        matching = df[df[COLUMNS["option"]].str.match(f"^{code}:", na=False)]
        if len(matching) > 0:
            policy_name = matching.iloc[0][COLUMNS["option"]]
            print(f"  ✓ {code}: {policy_name[:70]}")
        else:
            print(f"  ✗ {code}: NOT FOUND!")
    
    return all_found

if __name__ == "__main__":
    verify_policy_codes()