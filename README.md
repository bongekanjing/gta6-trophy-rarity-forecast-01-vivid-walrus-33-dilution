# gta6-trophy-rarity-forecast-01-vivid-walrus-33-dilution

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bongekanjing/gta6-trophy-rarity-forecast-01-vivid-walrus-33-dilution/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bongekanjing/gta6-trophy-rarity-forecast-01-vivid-walrus-33-dilution/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

An entry for make.fun's GTA VI trophy rarity forecast bounty: given every read the house made of Exophase's GTA VI trophy page before a checkpoint, forecast each trophy's earn rate a week out.

## The idea

Exophase prints a ratio, `rate = earners / players_tracked`. The obvious forecast — hold the rate where it was — models that ratio as one quantity. It is two, and they move for unrelated reasons:

- `players_tracked` is a population. It only grows, and in launch week it grows enormously.
- `earners` is behaviour: how many of those accounts actually reached that trophy.

So write the forecast as

    rate_T = rate_L * (P_T / P_L) ** (alpha - 1)

with `alpha = d log earners / d log players`. `alpha = 1` is a trophy every new player already has, so its rate does not move. `alpha = 0` is one no new player has, so its rate falls exactly as fast as the population rises. Everything is in `alpha` and the population ratio.

## What the measurements said

Reading the page at launch night (638 players tracked) and again a day later (14,028 — a 22x move) gives each trophy's realised `alpha`:

| trophy | alpha | rate, launch | rate, +1 day |
|---|---|---|---|
| Complete Gone Fishing | 1.00 | 100% | 99.99% |
| Complete Chapter 1 | 0.83 | 1.25% | 0.73% |
| Complete The Cartel Heist | 0.51 | 1.10% | 0.24% |
| Vice Got a Price! (platinum) | 0.02 | 0.63% | 0.03% |

The median across all 55 trophies is **0.05**: an earn rate falls almost exactly as fast as the tracked population rises, because most accounts Exophase syncs have barely started the game. That is the whole forecast, and it is why "the rate stays put" scores badly.

## Three parts

1. **`alpha`, per trophy.** Fit `log earners` against `log players` over the case's own reads. That only works if the history crosses at least two Exophase refreshes; a case that sits inside one cannot identify it, so it falls back to the measured median, shrunk by how little the case could say.
2. **The population ratio.** Fit `log P` against `log(1 + hours since release)` and read it at the target, capped so an early two-point fit cannot project a population no crawler reaches. A history that never shows a refresh uses the measured launch curve, `log P ~ sqrt(hours)`.
3. **A saturated trophy is not diluting.** A rate of 100% means earners equals players exactly, by definition, so `alpha` is 1 and the rate cannot fall. Treating saturation as ordinary dilution was the single worst available error — it took a trophy the page reports at 99.99% down to 6%.

Then a tier ceiling keeps a forecast from claiming most players earn a platinum, and the five quantiles are cut on the logit scale, which is the scale the contest scores on.

## Validation

The practice cases sit entirely inside one Exophase refresh, so on their own they cannot distinguish a good model from a bad one. So the entry was also scored against a real out-of-sample case: the launch-night reads as history, the actual page a day later as the answer.

| model | pinball loss | 90% coverage |
|---|---|---|
| house baseline (rate holds) | 2.119 | 0.07 |
| this entry | **0.358** | 0.96 |

Coverage near 0.9 is what an honest interval should give; the baseline's 0.07 means 13 of every 14 trophies fell outside its band.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory and writes `<id>.out` beside it. Standard library only, Python 3.13, no network, no randomness — two runs write the same bytes.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
