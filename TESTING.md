# Testing Guide for Refactored Codebase

## 🧪 Testing Status

**Current Status**: Code refactoring complete, ready for testing

**Test Priority**: HIGH - Comprehensive refactoring requires validation

---

## ⚠️ Known Issues (Non-Critical)

### Mypy Type Stub Warnings
The following Mypy warnings are **informational only** and do not affect functionality:

```
Library stubs not installed for "pandas"
```

**Cause**: pandas type stubs not installed (optional development dependency)  
**Impact**: None - code runs correctly  
**Fix** (optional): `pip install pandas-stubs` or ignore

**Other Mypy warnings**:
- Missing type annotations in some dicts - These are intentional for flexibility
- No impact on runtime behavior

---

## ✅ Pre-Testing Checklist

Before running tests, ensure:
- [ ] Gurobi license is installed and valid
- [ ] Excel file exists: `tax reform & spending menu options (v8) template.xlsx`
- [ ] Python 3.8+ is installed
- [ ] All dependencies installed: `pip install pandas gurobipy openpyxl matplotlib seaborn`
- [ ] Directory `outputs/defense/` is writable (will be created if missing)

---

## 🎯 Test Suite

### Test 1: Basic Validation
**Purpose**: Verify validation module catches errors

```bash
# Test 1a: Missing Excel file
mv "tax reform & spending menu options (v8) template.xlsx" temp.xlsx
python max_gdp.py
# Expected: Clear error about missing file

# Restore file
mv temp.xlsx "tax reform & spending menu options (v8) template.xlsx"
```

**Expected Result**:
```
ValidationError: Excel file not found: tax reform & spending menu options (v8) template.xlsx
Please ensure the file exists in the current directory.
```

### Test 2: Basic GDP Optimization
**Purpose**: Verify max_gdp.py works with new modules

```bash
python max_gdp.py
```

**Expected Results**:
- ✓ Loads policy data successfully
- ✓ Identifies NS policy groups
- ✓ Completes Stage 1 and Stage 2 optimization
- ✓ Displays formatted results with logger
- ✓ Saves `max_gdp.csv`
- ✓ No errors or warnings

**Validation**:
- Check console output uses new logger format with timestamps
- Verify `max_gdp.csv` contains selected policies
- Compare GDP/revenue totals with previous runs (should match)

### Test 3: Single Defense Optimization
**Purpose**: Verify max_gdp_defense.py with single spending level

```bash
python max_gdp_defense.py --spending 3000
```

**Expected Results**:
- ✓ Validates spending level (3000)
- ✓ Loads and validates policy data
- ✓ Runs Stage 1 and Stage 2 optimization
- ✓ Uses `add_all_constraints()` from optimizer_utils
- ✓ Displays results
- ✓ Saves `outputs/defense/max_gdp_defense3000.csv`

**Validation**:
- Verify fewer lines of code generated same constraints
- Check logger output format
- Compare results with previous runs

### Test 4: Verbose Mode
**Purpose**: Verify debug logging works

```bash
python max_gdp_defense.py --spending 3000 --verbose
```

**Expected Results**:
- ✓ Shows DEBUG level messages
- ✓ Shows constraint addition details
- ✓ Shows Stage 1 optimal GDP value
- ✓ More detailed progress information

### Test 5: Full Range Optimization
**Purpose**: Verify batch processing with new SPENDING_RANGE config

```bash
python max_gdp_defense.py
```

**Expected Results**:
- ✓ Uses SPENDING_RANGE from config (-4000 to 6500, step 500)
- ✓ Runs 21 optimizations sequentially
- ✓ Handles failures gracefully (if any)
- ✓ Generates summary matrices
- ✓ Calls visualization script
- ✓ Creates all output files in `outputs/defense/`

**Validation**:
- Count CSV files: Should be 21 files
- Check `policy_decisions_matrix.csv` exists
- Check `economic_effects_summary.csv` exists
- Verify `defense_spending_analysis.png` generated

### Test 6: Visualization Scripts
**Purpose**: Verify visualizations use config constants

```bash
python visualize_defense_spending.py
```

**Expected Results**:
- ✓ Uses SPENDING_RANGE from config
- ✓ Uses COLUMNS from config
- ✓ Loads all CSV files
- ✓ Generates 6-panel chart
- ✓ Saves `defense_spending_analysis.png`

```bash
python visualize_policy_selection.py
```

**Expected Results**:
- ✓ Uses SPENDING_RANGE and COLUMNS from config
- ✓ Creates policy selection heatmap
- ✓ Saves `policy_selection_heatmap.png`

### Test 7: Error Handling
**Purpose**: Verify graceful error handling

```bash
# Test 7a: Invalid spending level
python max_gdp_defense.py --spending 999999

# Expected: Validation error with clear message
```

```bash
# Test 7b: Keyboard interrupt
python max_gdp_defense.py
# Press CTRL+C during execution

# Expected: Clean exit with status 130
```

**Expected Results**:
- ✓ Validation errors show actionable messages
- ✓ Keyboard interrupt exits cleanly
- ✓ No stack traces for expected errors

---

## 🔍 Regression Testing

### Compare with Previous Version

**For each test, compare**:
1. **Numerical Results**: GDP, revenue, jobs should match exactly
2. **Policy Selections**: Same policies selected for same inputs
3. **File Outputs**: CSV files have same content (except formatting)

**Key Comparisons**:
```bash
# Run both versions
python max_gdp.py                    # New version
# python max_gdp_old.py              # Old version (if available)

# Compare csvs (numerical values should match)
diff max_gdp.csv max_gdp_old.csv
```

---

## 📋 Test Results Log

| Test | Status | Notes |
|------|--------|-------|
| 1. Validation (missing file) | ⏳ Pending | |
| 2. Basic GDP optimization | ⏳ Pending | |
| 3. Single defense optimization | ⏳ Pending | |
| 4. Verbose mode | ⏳ Pending | |
| 5. Full range optimization | ⏳ Pending | |
| 6a. Defense viz script | ⏳ Pending | |
| 6b. Policy selection viz | ⏳ Pending | |
| 7a. Invalid input handling | ⏳ Pending | |
| 7b. Keyboard interrupt | ⏳ Pending | |

**Update this table as tests are completed**

---

## 🐛 Bug Report Template

If issues are found during testing:

```markdown
**Test**: [Test name/number]
**Expected**: [What should happen]
**Actual**: [What actually happened]
**Error Message**: [Full error if any]
**Steps to Reproduce**:
1. Step 1
2. Step 2

**Environment**:
- Python version: 
- Gurobi version:
- OS:
```

---

## ✅ Sign-Off Criteria

Code is ready for production when:
- [ ] All tests pass
- [ ] No regression in numerical results
- [ ] Error handling works as expected
- [ ] Logging output is clear and informative
- [ ] Documentation is accurate
- [ ] No critical bugs found

---

## 📞 Next Steps After Testing

1. **If all tests pass**: 
   - Mark TODO list complete
   - Update README.md with v3.1 changes
   - Commit changes with descriptive message
   - Consider code review

2. **If bugs found**:
   - Document in bug report template
   - Prioritize by severity
   - Fix critical bugs first
   - Re-run affected tests

3. **Performance issues**:
   - Profile slow operations
   - Consider optimizations
   - Document any trade-offs

---

## 🚀 Performance Benchmarks

Expected performance (approximate):

| Operation | Time | Notes |
|-----------|------|-------|
| Single optimization | 5-15 seconds | Depends on Gurobi license type |
| Full range (21 runs) | 2-5 minutes | Linear scaling with # of runs |
| Validation overhead | <100ms | One-time cost at startup |
| Logging overhead | <1% total | Minimal impact |

**If significantly slower**: Check Gurobi license type (free licenses are slower)

---

## 💡 Debugging Tips

### If optimization fails:
1. Run with `--verbose` flag to see debug logs
2. Check Gurobi status messages
3. Verify Excel file hasn't changed
4. Try simpler spending level (e.g., 0 or 3000)

### If validation fails:
1. Check Excel file location and name
2. Verify column headers match config.COLUMNS
3. Check for extra/missing spaces in column names
4. Ensure data starts at row 4

### If imports fail:
1. Ensure all new modules (validation.py, logger.py, optimizer_utils.py) are present
2. Check Python path includes current directory
3. Verify no circular import issues

---

## 📚 Reference

- **Main documentation**: [`IMPROVEMENTS.md`](IMPROVEMENTS.md:1-241)
- **Summary**: [`SUMMARY.md`](SUMMARY.md:1-227)  
- **Configuration**: [`config.py`](config.py:1-159)
- **README**: [`README.md`](README.md:1-545)

**Last Updated**: 2025-11-01