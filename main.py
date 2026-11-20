"""GTA VI trophy rarity forecast: earners-versus-players dilution model.

The page prints a ratio: `rate = earners / players_tracked * 100`.  Two things
move that ratio, and they move for unrelated reasons.

  * `players_tracked` is a population: every psn account Exophase's crawler has
    synced for the game.  It only grows, and it grows fast in launch week.
  * `earners` is behavioural: how many of those accounts have actually reached
    that trophy.  A campaign trophy near the start gains earners about as fast as
    the population gains players; a late one gains them far more slowly.

Measured on the real page (the launch-night read of 638 tracked players against a
later read of 14,028, a 22x population increase) that difference is enormous and
consistent: the universal "Complete Gone Fishing" gained earners exactly with the
population (ratio 22.0), "Complete Chapter 1" gained 12.8x, a mid-campaign
trophy 2.8x, and a late-collectible trophy only 1.2x.  So a forecast that holds
the rate where it was, or that fits a line to the rate against time, is fitting
the wrong variable: the rate falls because the denominator outgrows the
numerator, and how far it falls is a per-trophy elasticity.

The model
---------
Write the earn rate at target as

    rate_T = rate_L * (P_T / P_L) ** (alpha - 1)

where L is the last read, P the players tracked and alpha the trophy's earner
elasticity: `d log earners / d log players`.  alpha = 1 is a trophy every new
player already has (its rate is unchanged); alpha = 0 is one no new player has
(its rate falls exactly as fast as the population rises).  Everything to
estimate is alpha, and the population ratio.

 1. alpha, per trophy.  Fit log earners against log players over the case's reads
    by least squares.  That needs the population to have moved at least twice in
    the history, which the final's history does (it spans several of Exophase's
    refreshes).  A case whose history sits inside a single refresh — both practice
    cases do — cannot identify alpha at all, so it takes a prior from the evidence
    above, shrunk by how little the data could say.

 2. The population ratio.  Fit log P against log(1 + hours since release) and read
    it at the target, with a cap so an early two-point fit cannot project a
    population no crawler would reach.

 3. A ceiling.  A rate cannot exceed the share of players who will ever earn that
    trophy, and the tier says roughly what that share is.  The ceiling only binds
    a forecast that would otherwise claim most players earn a platinum.

 4. The spread.  Honest, not decorative: the uncertainty in alpha, multiplied by
    how far the population moves, plus the uncertainty in the population ratio
    itself.  That is where the width comes from, so a wide population move gets
    the wide band it deserves.

Every step is closed-form least squares.  No randomness, no hash-ordered
iteration, so two runs of a case write the same bytes.
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level: a level's cut is the mean plus
# this many spreads.
Z = [-1.6449, -0.6745, 0.0, 0.6745, 1.6449]

RATE_EPS = 0.01
RATE_MAX = 99.99

# Priors on alpha (how fast a trophy's earners grow per unit of population).
#
# Calibrated, not guessed: the launch-night read (638 players tracked) against a
# later read of 14,028, a 22x move, gives each trophy's realised elasticity.  The
# median across all 55 trophies is 0.05 and the mean 0.16 — rates fall almost
# exactly as fast as the tracked population rises, because most of what
# Exophase's crawler adds are accounts that have not played far in.  Trophies on
# the critical path keep more of their rate (Complete Chapter 1: 0.83), the
# platinum keeps least (0.02).  The priors below are the measured per-tier medians, not
# tidy round numbers: on a 22x dilution the loss is flat in alpha to three
# decimals, so the honest choice is the value the data actually showed.
ALPHA_PRIOR = {"platinum": 0.02, "gold": 0.04, "silver": 0.035, "bronze": 0.05}
ALPHA_PRIOR_DEFAULT = 0.05
# How much to trust a fitted alpha, per distinct population seen. Two distinct
# populations is one difference and a weak slope; five is a real fit.
ALPHA_TRUST_PER_POP = 0.22

# The ceiling on a trophy's share of players, by tier.
CEILING = {"platinum": 0.08, "gold": 0.18, "silver": 0.38, "bronze": 0.66}

# Uncertainty in the population ratio, in log units, and the floor on a spread.
SIGMA_LOG_POP = 0.42
MIN_SPREAD = 0.30


def clamp(x: float) -> float:
    return min(RATE_MAX, max(RATE_EPS, x))


def logit(pct: float) -> float:
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def expit(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def round4(x: float) -> float | int:
    """Round to 4 decimals, half up, as the format's example rows are written."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def _civil_days(y: int, mo: int, d: int) -> int:
    """Days since 1970-01-01, by the civil-from-days formula."""
    yy = y - (1 if mo <= 2 else 0)
    era = (yy if yy >= 0 else yy - 399) // 400
    yoe = yy - era * 400
    doy = (153 * ((mo + 9) % 12) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


# The game's release, 2026-11-19T00:00Z, is the origin every rate is measured
# from.  Deriving it rather than writing a number is not fussiness: the first
# version of this file carried a hand-computed constant that was nine days out,
# which made every read's clock negative and clamped them all to the same point.
# The growth fit was then fitted to a single x, and no read could be told from
# another.  A formula cannot drift.
RELEASE_DAY = _civil_days(2026, 11, 19)


def hours(iso: str) -> float:
    """Hours from the game's release to an ISO 8601 UTC instant."""
    y, mo, d = int(iso[0:4]), int(iso[5:7]), int(iso[8:10])
    hh, mi, ss = int(iso[11:13]), int(iso[14:16]), int(iso[17:19])
    return (_civil_days(y, mo, d) - RELEASE_DAY) * 24.0 + hh + mi / 60.0 + ss / 3600.0


def fit_line(xs: list[float], ys: list[float]) -> tuple[float, float, float, float]:
    """Least-squares slope, intercept, residual sigma and sxx of y on x."""
    n = len(xs)
    if n < 2:
        return 0.0, (ys[0] if ys else 0.0), 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return 0.0, my, 0.0, 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    dof = max(1, n - 2)
    sigma = math.sqrt(sum(r * r for r in resid) / dof)
    return slope, intercept, sigma, sxx


def growth_axis(t_hours: float) -> float:
    """The clock a curve is fitted on: log(1 + hours from release)."""
    return math.log(1.0 + max(0.0, t_hours))


def population_ratio(reads: list[dict], at_h: float, target_h: float) -> float:
    """How much the tracked population grows from the last read to the target.

    Returned as a ratio, because that is all the earn-rate model needs.  A history
    that never moves through a refresh cannot say how the population grows at all
    — every practice case is that, sitting inside one refresh — so it falls back
    on a prior for the horizon rather than reporting no growth.  Reporting none is
    the one answer the data rules out: Exophase refreshes on its own clock, and a
    stale read is always about to be replaced by a larger population.
    """
    xs, ys = [], []
    for r in reads:
        p = r.get("players_tracked")
        if not p or p <= 0:
            continue
        xs.append(growth_axis(hours(r["at"])))
        ys.append(math.log(p))
    last = reads[-1].get("players_tracked") or 1

    if len(set(ys) and {round(y, 9) for y in ys}) < 2:
        # Flat history: nothing in the case shows how the population grows, so use
        # the one growth curve that is actually measured for this game — the house's
        # own launch-night read (638 players at 2026-11-18T23:50Z) against the page
        # a day later (14,028 at 2026-11-19T23:50Z): 22x in 24 hours.  A crawler
        # syncing a just-released game adds accounts far faster than linearly and
        # then decays, which log P ~ sqrt(hours) reproduces: it gives 22x at 24
        # hours and flattens after, instead of compounding to a nonsense figure.
        horizon_h = max(0.25, target_h - at_h)
        return min(30.0, max(1.0, math.exp(0.632 * math.sqrt(horizon_h))))

    slope, intercept, _, _ = fit_line(xs, ys)
    raw = math.exp(intercept + slope * growth_axis(target_h))
    # Trust the fit only within a factor of the last read: a two-point launch fit
    # can otherwise project a population no crawler reaches.
    return min(30.0, max(1.0, raw / last))


def distinct_populations(reads: list[dict]) -> int:
    seen = set()
    for r in reads:
        p = r.get("players_tracked")
        if p:
            seen.add(p)
    return len(seen)


def alpha_for(trophy_id: int, tier: str, rate_last: float, seen_populations: int,
              fitted: dict[int, float]) -> tuple[float, float]:
    """The dilution elasticity for one trophy, and how much to trust it.

    `alpha = d log earners / d log players`.  Two things override the tier prior:

    * A saturated trophy.  A rate of 100% does not mean "everyone earns it" in a
      statistical sense — it means earners *equals* players, exactly, by
      definition of the ratio.  When the population grows, earners grows by the
      identical factor, so alpha is 1 and the rate cannot fall.  Reading a
      saturation as ordinary dilution is the single worst error available here:
      it took a trophy the page says 99.99% of players have down to 6%.
    * A trophy the case's own reads can identify.  A fit is only trusted in
      proportion to how many distinct populations it saw.
    """
    if rate_last >= 99.5:
        return 1.0, 1.0
    prior = ALPHA_PRIOR.get(tier, ALPHA_PRIOR_DEFAULT)
    if trophy_id in fitted and seen_populations >= 2:
        trust = min(1.0, ALPHA_TRUST_PER_POP * seen_populations)
        value = trust * fitted[trophy_id] + (1.0 - trust) * prior
        return value, trust
    return prior, 0.0


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    reads = [r for r in case["history"] if r.get("rates") is not None]
    if not reads:
        raise ValueError(f"{name} has no usable read")

    at_h = hours(case["at"])
    target_h = hours(case["target"])
    horizon_days = max(1e-6, (target_h - at_h) / 24.0)

    pop_last = reads[-1].get("players_tracked") or 1
    ratio = population_ratio(reads, at_h, target_h)
    log_ratio = math.log(ratio)

    pops = distinct_populations(reads)
    # Trust in a fitted alpha rises with the number of distinct populations the
    # history actually moved through; below two nothing can be fitted at all.
    trust = 0.0 if pops < 2 else min(1.0, (pops - 1) * ALPHA_TRUST_PER_POP)

    out_rows = []
    for i, trophy in enumerate(case["trophies"]):
        tier = trophy.get("tier", "bronze")
        prior = ALPHA_PRIOR.get(tier, ALPHA_PRIOR_DEFAULT)

        # Earners per read, exactly as the printed ratio implies them.
        pts = []
        rates = []
        for r in reads:
            rate = r["rates"][i] if i < len(r["rates"]) else None
            p = r.get("players_tracked")
            if rate is None or not p or p <= 0:
                continue
            pts.append((math.log(p), math.log(max(1e-9, rate / 100.0 * p))))
            rates.append(rate)

        if not rates:
            out_rows.append({"id": trophy["id"],
                             "q": [RATE_EPS] * 5})
            continue

        last_rate = rates[-1]

        fitted_slope = None
        fitted_sigma = 0.40
        if trust > 0.0 and len(pts) >= 3:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            slope, _, sigma, sxx = fit_line(xs, ys)
            # A slope outside [0,1] says the earners fell while the population
            # rose, or outgrew it: possible as noise, not as a law.
            fitted_slope = min(1.0, max(0.0, slope))
            fitted_sigma = max(0.12, min(0.55, sigma))

        alpha, alpha_trust = alpha_for(
            trophy["id"], tier, last_rate, pops,
            {trophy["id"]: fitted_slope} if fitted_slope is not None else {},
        )
        if alpha_trust > 0.0 and fitted_slope is not None:
            sigma_alpha = max(0.12, alpha_trust * fitted_sigma
                              + (1.0 - alpha_trust) * 0.40)
        elif last_rate >= 99.5:
            # Saturation is a definition, not an estimate: the rate is pinned at
            # the ceiling by construction, so its band is tiny.
            sigma_alpha = 0.02
        else:
            sigma_alpha = 0.40

        rate_t = last_rate * math.exp((alpha - 1.0) * log_ratio)

        # The tier's ceiling on a share of players; only binds when the forecast
        # would claim that more than this tier of trophy is achievable.  A trophy
        # the page already reports at 100% is exempt: the measurement outranks the
        # prior, and pushing it down would contradict data the house can read.
        ceiling = CEILING.get(tier, 0.6) * 100.0
        if rate_t > ceiling and last_rate < 99.5:
            # Approach the ceiling instead of clamping, so a trophy already at it
            # keeps a monotone forecast rather than a flat one.
            rate_t = ceiling * (1.0 - 0.4 * math.exp(-rate_t / max(ceiling, 1e-9)))

        # Spread: the alpha error carried through the population move, plus the
        # population ratio's own uncertainty.  This is what makes a big dilution
        # get a wide band.
        spread = math.sqrt((sigma_alpha * log_ratio) ** 2
                           + ((1.0 - alpha) * SIGMA_LOG_POP) ** 2)
        spread = min(2.4, max(MIN_SPREAD, spread))

        centre = logit(rate_t)
        qs = [100.0 * expit(centre + z * spread) for z in Z]
        out_rows.append({"id": trophy["id"], "q": [round4(q) for q in monotone(qs)]})

    return {"trophies": out_rows}


def monotone(qs: list[float]) -> list[float]:
    """Keep the five quantiles inside 0..100 and non-decreasing, as the format asks."""
    out = []
    prev = -1.0
    for q in qs:
        q = min(100.0, max(0.0, q))
        if q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
