"""
Generate optimization results for all defense spending levels in increments of 500
"""

import subprocess
import sys

# Defense spending levels to generate (in billions)
spending_levels = [500, 1500, 2500, 3500, 4500, 5500]

print("Generating optimization results for defense spending levels...")
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