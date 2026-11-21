#!/usr/bin/env python3
"""Out-of-sample test of the forecast on a case shaped like the real final.

The practice cannot test this: every launch-night read is the same frozen page at
638 players, so its history is flat, there is no growth to fit, and the model's
growth path is never exercised.  The final is the opposite — a week of reads with
a 4.5-day extrapolation, and the prize is decided on it alone.

So this builds cases whose truth we know, from the growth curve the rules publish,
and scores what the entry would have written.

Run: python3 simulate.py
"""
import importlib.util
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("entry_main", HERE / "main.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# ---- The world: the published anchors ------------------------------------------
P0, K = 671.0, 0.964


def players(t_h):
    return P0 * (1.0 + max(0.0, t_h)) ** K


def logit(p):
    p = min(max(p, 0.01), 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(z):
    return 100.0 / (1.0 + math.exp(-z))


def pinball(q, y, tau):
    d = logit(y) - logit(q)
    return 2.0 * max(tau * d, (tau - 1.0) * d)


LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# trophy id -> (rate at release, alpha).  The set spans the real case's shape: a
# platinum nearly nobody has, a saturated one at 100, commons, and rares.
TROPHIES = {
    1: (0.30, 0.02),    # platinum: earners almost fixed while the base grows
    2: (100.0, 1.00),   # saturated: 100% is definitional, cannot dilute
    3: (30.0, 0.35),    # a common early story trophy
    4: (0.05, 0.10),    # a rare one
    5: (5.0, 0.20),     # mid
}


def rate_at(t_h, r0, alpha):
    """The rate the page would show at t_h, given release-rate r0 and alpha."""
    if r0 >= 99.99:
        return 100.0
    ratio = players(t_h) / players(0.0)
    return min(99.99, max(0.01, inv_logit(logit(r0) + (alpha - 1.0) * math.log(ratio))))


def make_case(ckpt_h, target_h, refresh_h, seed=0):
    rng = random.Random(seed)
    trophies = [{"id": i, "name": f"T{i}", "tier": "bronze", "points": 15, "secret": False}
                for i in sorted(TROPHIES)]

    def stamp(h):
        day = 19 + int(h // 24)
        return "2026-11-%02dT%02d:%02d:00Z" % (day, int(h % 24), int(round((h % 1) * 60)) % 60)

    history = []
    step = max(1, int(refresh_h * 6))
    for i in range(0, int(ckpt_h * 6) + 1):
        h = i / 6.0
        # the page only moves when Exophase refreshes: hold the last refresh's hour
        shown = (i // step) * step / 6.0
        rates = [round(rate_at(shown, r0, a), 2) for r0, a in
                 (TROPHIES[t] for t in sorted(TROPHIES))]
        history.append({"slot": stamp(h), "at": stamp(h),
                        "players_tracked": int(players(shown)),
                        "trophies_earned": int(sum(r / 100.0 for r in rates) * players(shown)),
                        "rates": rates})
    truth = {t: rate_at(target_h, r0, a) for t, (r0, a) in TROPHIES.items()}
    case = {"id": "s00", "at": stamp(ckpt_h), "target": stamp(target_h),
            "horizon_minutes": int(round((target_h - ckpt_h) * 60)),
            "levels": LEVELS, "game": {"name": "x", "platform": "PS5"},
            "trophies": trophies, "history": history}
    return case, truth, players(target_h) / players(ckpt_h)


def score(rows, truth):
    tot, n = 0.0, 0
    for tid, q in rows.items():
        for tau, qi in zip(LEVELS, q):
            tot += pinball(qi, truth[tid], tau)
            n += 1
    return tot / n


def baseline_rows(case):
    """The house baseline exactly as RULES describes it."""
    d = case["horizon_minutes"] / 1440.0
    z = [-1.645, -0.674, 0.0, 0.674, 1.645]
    last = case["history"][-1]["rates"]
    return {t["id"]: [round(inv_logit(logit(last[t["id"] - 1]) + zz * 0.5 * math.sqrt(d)), 4)
                      for zz in z] for t in case["trophies"]}


print("%-24s %7s %7s %9s %9s %8s" % ("scenario", "ratio", "horizon", "entry", "baseline", "better?"))
scenarios = [
    ("final, ckpt at deadline", 41.75, 168.0, 6.0),
    ("final, ckpt a day early", 47.75, 168.0, 6.0),
    ("final, refresh every 3h", 41.75, 168.0, 3.0),
    ("final, refresh every 12h", 41.75, 168.0, 12.0),
    ("mid-week ckpt", 65.75, 168.0, 6.0),
    ("practice-shaped 90min", 4.5, 6.0, 0.5),
]
wins = 0
for name, ck, tg, rh in scenarios:
    case, truth, ratio = make_case(ck, tg, rh)
    f = m.forecast_case(case, "s00")
    rows = {r["id"]: r["q"] for r in f["trophies"]}
    e, b = score(rows, truth), score(baseline_rows(case), truth)
    ok = e < b
    wins += ok
    print("%-24s %6.2fx %6.0fh %9.4f %9.4f %8s"
          % (name, ratio, tg - ck, e, b, "yes" if ok else "NO"))

print()
print(f"{wins}/{len(scenarios)} scenarios beat the house baseline")

print()
case, truth, ratio = make_case(41.75, 168.0, 6.0)
f = m.forecast_case(case, "s00")
rows = {r["id"]: r["q"] for r in f["trophies"]}
print("the final, trophy by trophy (rate at 168h, ratio %.2fx):" % ratio)
print("  id  truth%     q05      q50      q95   covered")
cov = 0
for tid in sorted(truth):
    q = rows[tid]
    inside = q[0] <= truth[tid] <= q[4] or (truth[tid] >= 99.99 and q[4] >= 99.99)
    cov += inside
    print("  %2d %7.3f %8.5f %8.5f %8.5f   %s"
          % (tid, truth[tid], q[0], q[2], q[4], "yes" if inside else "NO"))
print("  coverage %d/%d" % (cov, len(truth)))
