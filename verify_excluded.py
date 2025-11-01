"""Verify that excluded policies are not in the output."""

import pandas as pd

df = pd.read_csv('outputs/defense/max_gdp_defense5000.csv')

excluded_codes = ['37', '43', '49', '68']

print('Verifying Excluded Policies (5000B Spending):\n')

all_excluded = True
for code in excluded_codes:
    found = df[df['Option'].str.match(f'^{code}:', na=False)]
    if len(found) > 0:
        print(f'  Policy {code}: FOUND (ERROR!)')
        print(f'    {found.iloc[0]["Option"]}')
        all_excluded = False
    else:
        print(f'  Policy {code}: Excluded (correct)')

if all_excluded:
    print('\n✅ All 4 policies successfully excluded from selection!')
else:
    print('\n❌ Some policies were not excluded!')