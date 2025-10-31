"""
Generate optimization results for negative defense spending levels (-4000 to -500)
Negative values represent defense spending cuts/reductions
"""

import subprocess
import sys

# Negative defense spending levels (representing cuts)
spending_levels = [-4000, -3500, -3000, -2500, -2000, -1500, -1000, -500]

print("Generating optimization results for negative defense spending levels...")
print("(Negative values represent defense spending cuts/reductions)")
print("=" * 70)

for level in spending_levels:
    print(f"\nRunning optimization for ${level:,}B defense spending...")
    try:
        result = subprocess.run(
            [sys.executable, "max_gdp_defense.py", "--spending", str(level)],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ Successfully generated max_gdp_defense{level}.csv")
        # Print key output lines
        for line in result.stdout.split('\n'):
            if 'GDP' in line or 'Revenue' in line or 'saved' in line:
                print(f"  {line}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to generate results for ${level:,}B")
        print(f"  Error: {e.stderr}")

print("\n" + "=" * 70)
print("All optimization runs complete!")
print("\nGenerated files:")
for level in spending_levels:
    print(f"  - max_gdp_defense{level}.csv")