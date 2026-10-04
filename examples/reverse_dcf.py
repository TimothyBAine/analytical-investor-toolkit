"""Reproduces the Month 5 Quant Lab sensitivity table and reverse DCF."""
from analytical_investor import valuation as V

G = [0.12, 0.10, 0.09, 0.07, 0.06, 0.05, 0.05, 0.04, 0.04, 0.03]
table = V.sensitivity_table(100, G, [0.10, 0.12, 0.14], [0.02, 0.03, 0.04], 150, 50)
print("Per-share value (rows r = 10/12/14%, cols g = 2/3/4%):\n", table.round(2))
for price in (15, 25, 35):
    print(f"Price {price}: implies {V.implied_growth(price, 100, 0.12, 0.03, 150, 50):.1%} growth")
