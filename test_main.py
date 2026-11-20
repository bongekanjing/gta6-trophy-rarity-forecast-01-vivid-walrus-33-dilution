#!/usr/bin/env python3
"""Regression tests for the forecast entry.  Run: python3 test_main.py

Each test pins a failure that actually happened, so it cannot come back quietly.
"""
import importlib.util
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("entry_main", HERE / "main.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

KIT = Path("/root/makefun/trf/gta6-trophy-rarity-forecast-01-kit/input")

fails = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        fails.append(name)


# 1. The release origin.  A hand-computed constant was nine days out, which
#    pushed every read's clock negative and clamped them all to one x.
check("hours: release is 0", abs(m.hours("2026-11-19T00:00:00Z")) < 1e-9,
      m.hours("2026-11-19T00:00:00Z"))
check("hours: a day later is 24",
      abs(m.hours("2026-11-20T00:00:00Z") - 24.0) < 1e-9,
      m.hours("2026-11-20T00:00:00Z"))
check("hours: the final target is 168",
      abs(m.hours("2026-11-26T00:00:00Z") - 168.0) < 1e-9,
      m.hours("2026-11-26T00:00:00Z"))
check("hours: strictly increasing over the whole week",
      m.hours("2026-11-18T23:50:00Z") < m.hours("2026-11-19T00:00:00Z")
      < m.hours("2026-11-21T11:30:00Z") < m.hours("2026-11-26T00:00:00Z"))

# 2. The growth axis must actually separate reads.  When the clock was broken
#    every read mapped to x=0, the fit had no slope, and the population ratio
#    collapsed to 1.0 — the forecast became the baseline on every real case.
check("growth_axis separates hours 0 and 24",
      m.growth_axis(0.0) < m.growth_axis(24.0) < m.growth_axis(168.0))

reads = [
    {"at": "2026-11-18T23:50:00Z", "slot": "2026-11-18T23:50:00Z",
     "players_tracked": 638, "trophies_earned": 943, "rates": [1.0]},
    {"at": "2026-11-19T23:50:00Z", "slot": "2026-11-19T23:50:00Z",
     "players_tracked": 14028, "trophies_earned": 14882, "rates": [1.0]},
]
at_h, t_h = m.hours("2026-11-19T23:50:00Z"), m.hours("2026-11-20T23:50:00Z")
ratio = m.population_ratio(reads, at_h, t_h)
check("population ratio over a growing history exceeds 1", ratio > 1.05, ratio)
check("population ratio is capped", ratio <= 30.0, ratio)

# 3. Saturation.  A rate of 100% means earners equals players exactly, so it
#    cannot dilute; reading it as ordinary dilution took a 99.99% trophy to 6%.
alpha, trust = m.alpha_for(2, "bronze", 100.0, 1, {})
check("a saturated trophy cannot dilute", abs(alpha - 1.0) < 1e-12, alpha)
alpha, trust = m.alpha_for(2, "bronze", 0.5, 1, {})
check("a rare trophy does dilute", alpha < 0.5, alpha)

# 4. The output contract, on the real practice cases.
for name in ("p00", "p01"):
    case = json.loads((KIT / f"{name}.in").read_text())
    out = m.forecast_case(case, name)
    rows = out["trophies"]
    check(f"{name}: one row per trophy", len(rows) == len(case["trophies"]),
          f"{len(rows)} rows for {len(case['trophies'])} trophies")
    check(f"{name}: ids match exactly",
          sorted(r["id"] for r in rows) == sorted(t["id"] for t in case["trophies"]))
    check(f"{name}: five quantiles each", all(len(r["q"]) == 5 for r in rows))
    check(f"{name}: non-decreasing", all(all(a <= b for a, b in zip(r["q"], r["q"][1:]))
                                        for r in rows))
    check(f"{name}: in range", all(0.0 <= q <= 100.0 for r in rows for q in r["q"]))
    check(f"{name}: JSON serialises under the size cap",
          len(json.dumps(out, separators=(",", ":")).encode()) < 65536)

print()
if fails:
    print(f"FAILED: {len(fails)} — {', '.join(fails)}")
    sys.exit(1)
print("all tests pass")
