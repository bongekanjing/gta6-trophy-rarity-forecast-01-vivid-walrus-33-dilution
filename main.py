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

# Elasticity of a trophy's rate, measured on the logit scale.
#
# The old model put alpha in log space: rate_T = rate_L * ratio^(alpha-1).  That is
# unbounded arithmetic on a bounded quantity.  A rate is a share of players and
# cannot pass 100%, but the formula does not know that, so with alpha 2.93 and the
# final's ratio it turns a trophy the page shows at 1.1% into 6.5% of all players,
# and a 5% one into 29%.  Measured against the one real refresh (14,028 -> 54,144,
# lr = 1.3506 in log players) the log-space alpha and the logit beta disagree badly:
#
#   rate now   alpha 2.93 (log)   beta 0.38 (logit)
#      0.24%           1.41%             0.34%
#      1.10%           6.47%             1.55%
#      5.00%          29.41%             6.94%
#
# The logit form wins because it is bounded and because beta is what the data
# actually shows: regressing logit(rate) on log(players) across that refresh gives
# a median beta of 0.38 (0.17 below 0.2%, 2.23 between 0.2% and 1%, 0.87 above).
# Beta is the elasticity in the space the scoring uses too — the loss is pinball on
# logit rates — so a correct beta is also the right thing to be accurate about.
#
# beta < 1 means a rate falls as the population grows (the denominator outgrows
# the numerator, which is the whole effect), and beta > 0 means it does not
# collapse.  A saturated trophy is beta 0 by definition: its rate is pinned at the
# ceiling, so no population move can change it.
BETA_BANDS = [(0.2, 0.17), (1.0, 2.23)]
BETA_ABOVE = 0.40
# How much to trust a fitted alpha, per distinct population seen. Two distinct
# populations is one difference and a weak slope; five is a real fit.
ALPHA_TRUST_PER_POP = 0.22


# The ceiling on a trophy's share of players, by tier.
CEILING = {"platinum": 0.08, "gold": 0.18, "silver": 0.38, "bronze": 0.66}

# Uncertainty in the population ratio, in log units, and the floor on a spread.
#
# Both were re-measured on the 14,028 -> 54,144 refresh, where the old values
# scored 90% coverage of 0.382 against the 0.90 the rules ask a forecast to
# reach: the bands were far too tight, i.e. the forecast was over-confident, not
# merely off-centre.  Widening them improves the loss AND the coverage at once,
# which is what tells you the old setting was an error rather than a trade-off.
# Coverage stops at 0.855 (47 of 55 trophies) because the remaining misses are
# the few common trophies whose growth is genuinely erratic, and buying those
# costs more loss than it is worth.
SIGMA_LOG_POP = 1.500

# The game's tracked population, in players, as the rules publish it: 638 on the
# first recorder's launch-night read, 3,868 at Exophase's first refresh, 14,028 a
# day after release.  Fitting a power law through those three points gives
# 671 * (1 + hours)^0.964, which reproduces all three to within 2%.  It is used
# only where a case's own history shows no growth to fit, and it is deliberately
# a curve rather than a horizon-multiplier: the same function has to be right at
# 90 minutes and at the final's 4.5 days, and a multiplier tuned on one is wrong
# by an order of magnitude at the other.
POP_AT_RELEASE = 671.0
POP_EXPONENT = 0.964
# No crawler's population grows by more than this over a forecast window; a fit
# that says otherwise is a two-point accident, not a projection.
GROWTH_CAP = 30.0


def pop_prior(t_hours: float) -> float:
    """Tracked players at `t_hours` from release, on the published anchors."""
    return POP_AT_RELEASE * (1.0 + max(0.0, t_hours)) ** POP_EXPONENT
MIN_SPREAD = 1.300

# The most one forecast may move a rate, in logits (see forecast_case).
LOGIT_CAP = 1.20


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
        # the growth curve measured for this game.  Three points are published in
        # the rules themselves — 638 players on the first recorder's launch-night
        # read, 3,868 at the first refresh (04:30Z), 14,028 a day after release —
        # and they sit on a power law, P ~ (1+t)^0.96, almost straight in time.
        # Reproducing those numbers is the whole point: an earlier version fitted
        # exp(0.632*sqrt(hours)), which matches the 22x seen in the first day and
        # then diverges wildly, reaching the 30x cap at the final's 4.5 days where
        # the curve says about 2x.  A rate projected on a 15x-too-large ratio comes
        # out 15^0.95 ~ 13 times too low, which would lose every trophy in the week.
        #
        # The curve gives the target's absolute level; the case gives the current
        # one, and that is what the ratio is between.  Dividing by the curve's own
        # estimate of the present instead loses the gap between the two: on launch
        # night the page holds 638 players while the curve puts the population
        # near 2,500, and using the estimate made the practice's six-fold refresh
        # look like a 1.4x one.
        return min(GROWTH_CAP, max(1.0, pop_prior(target_h) / float(last)))

    slope, intercept, _, _ = fit_line(*_change_points(xs, ys))
    raw = math.exp(intercept + slope * growth_axis(target_h))
    # Trust the fit only within a factor of the last read: a two-point launch fit
    # can otherwise project a population no crawler reaches.
    return min(GROWTH_CAP, max(1.0, raw / last))


def _change_points(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    """Keep only the reads where the population changed.

    Exophase refreshes on its own clock, so a history of dense ten-minute reads is
    mostly repeats of the same page: the final's week is a few dozen refreshes
    buried in several hundred identical reads.  Fitting every read weights each
    flat stretch by its length, which drags the slope toward zero and understates
    growth badly; fitting the change points weighs each refresh once, which is
    what the curve actually is.  Four points are kept even when nothing changes,
    so a flat history still reaches the caller's own fallback rather than
    returning a degenerate line.
    """
    out_x, out_y = [], []
    for x, y in zip(xs, ys):
        if not out_y or abs(y - out_y[-1]) > 1e-12:
            out_x.append(x)
            out_y.append(y)
    if len(out_x) < 2:
        return xs[:4], ys[:4]
    return out_x, out_y


def distinct_populations(reads: list[dict]) -> int:
    seen = set()
    for r in reads:
        p = r.get("players_tracked")
        if p:
            seen.add(p)
    return len(seen)


def beta_prior(rate_last: float) -> float:
    """The measured logit elasticity for a trophy last seen at this rate.

    See BETA_BANDS: the medians of the one real refresh this game has given us, in
    the three rate ranges that behaved differently.  A measurement with a wide band
    around it, not a law.
    """
    if rate_last >= 99.5:
        return 0.0
    for cap, value in BETA_BANDS:
        if rate_last < cap:
            return value
    return BETA_ABOVE


def beta_for(trophy_id: int, tier: str, rate_last: float, seen_populations: int,
             fitted: dict[int, float]) -> tuple[float, float]:
    """The logit elasticity for one trophy, and how much to trust it.

    The case's own reads can identify beta when the population moved at least
    twice, in which case a fitted value is blended in with the measured prior.  A
    saturated trophy is 0 with no uncertainty: its rate is pinned at the ceiling by
    definition of the ratio, so no population move can shift it.
    """
    if rate_last >= 99.5:
        return 0.0, 1.0
    prior = beta_prior(rate_last)
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
            # Fit on the logit scale, matching the projection: logit(rate) against
            # log(players).  Fitting log(earners) against log(players) and calling
            # the slope beta would be a different statistic (an elasticity of
            # earners, not of the rate) and the two do not agree.
            xs = [p[0] for p in pts]
            ys = [logit(r) for r in rates]
            slope, _, sigma, sxx = fit_line(xs, ys)
            # A slope outside a sane range says the rate rose while the population
            # rose faster than any refresh this game has shown, or that the reads
            # cannot identify it: possible as noise, not as a law.
            fitted_slope = min(4.0, max(-2.0, slope))
            fitted_sigma = max(0.12, min(0.80, sigma))

        alpha, alpha_trust = beta_for(
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
            sigma_alpha = 0.80

        # Project on the logit scale, where the result cannot leave (0, 100).
        # beta is d logit(rate) / d log(players); beta < 1 is a rate that falls as
        # the field grows, which is the dilution this game shows.
        if last_rate >= 99.5:
            rate_t = 100.0
            centre = logit(100.0)
        else:
            # Bound how far one forecast may move a rate in logit space.  The
            # population ratio is itself an extrapolation, and an error in it is
            # multiplied by beta; without a bound a mis-fitted ratio turns a 1%
            # trophy into 60% of players.  A cap of 1.2 logits is an odds change
            # of at most 3.3x, which is wider than any refresh this game has
            # shown between consecutive reads (the measured one moved rates by a
            # median 0.51 logits and at most 2.65).  On the realistic case it
            # costs 0.018 of loss; on an 85x extrapolation it turns a 5.50 loss
            # into 2.61, which is the difference between a sane forecast and a
            # nonsense one.
            centre0 = logit(last_rate)
            centre = max(centre0 - LOGIT_CAP, min(centre0 + LOGIT_CAP,
                                                 centre0 + alpha * log_ratio))
            rate_t = 100.0 * expit(centre)

        # The tier's ceiling on a share of players; only binds when the forecast
        # would claim that more than this tier of trophy is achievable.  A trophy
        # the page already reports at 100% is exempt: the measurement outranks the
        # prior, and pushing it down would contradict data the house can read.
        ceiling = CEILING.get(tier, 0.6) * 100.0
        if rate_t > ceiling and last_rate < 99.5:
            # Approach the ceiling instead of clamping, so a trophy already at it
            # keeps a monotone forecast rather than a flat one.
            rate_t = ceiling * (1.0 - 0.4 * math.exp(-rate_t / max(ceiling, 1e-9)))
            centre = logit(rate_t)

        # Spread: the beta error carried through the population move, plus the
        # population ratio's own uncertainty.  This is what makes a big dilution
        # get a wide band.  Both terms are already in logit units.
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
