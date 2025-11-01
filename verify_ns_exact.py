"""Verify NS spending is exactly the target amount."""

import pandas as pd

df = pd.read_csv('outputs/defense/max_gdp_defense5000.csv')

ns_policies = df[df['Option'].str.contains('NS[1-7][A-Z]:', na=False, regex=True)]

print('NS Policies Selected:')
for _, row in ns_policies.iterrows():
    print(f'  {row["Option"][:60]} -> ${row["Dynamic 10-Year Revenue (billions)"]:.2f}B')

total_ns = ns_policies['Dynamic 10-Year Revenue (billions)'].sum()
print(f'\nTotal NS Spending: ${total_ns:.2f}B')
print(f'Target: $-5,000.00B')
print(f'Match: {"✅ EXACT!" if abs(total_ns - (-5000)) < 0.01 else "❌ MISMATCH"}')