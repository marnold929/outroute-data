# Guard audit — preseason assumptions (2026-09-14, branch `season-aware-guards`)

Why this exists: three guards calibrated for draft season have fired on normal in-season conditions in one week (drift, usage collapse at kickoff, ADP coverage). This is one pass over every abort in `pipeline/build.py` and every threshold in `pipeline/model.py`.

## Evidence used

**FFC's pools.** FFC serves a rolling window of mock drafts: PPR covers 2026-09-07..09-14, 749 drafts, and a player needs 5+ drafts to be listed. Market players in our own published feed over time:

| date | Aug 30 | Sep 7 | Sep 9 | Sep 11 | Sep 12 | Sep 13–14 |
|---|---|---|---|---|---|---|
| PPR market pool | 271 | 263 | 249 | 225 | 209 | 193–194 |
| half / std covering the PPR top 100 | 100 / 100 | 100 / 100 | 100 / 100 | 96 / 98 | 96 / 93 | 96 / 85 (last publish), now 39 / 83 |

**What past seasons can't tell us.** FFC's archive for 2024 and 2025 ends on Sep 1, so there is no record of how a pool behaves in October. Two outcomes are possible:
- FFC keeps rolling: the pools keep draining.
- FFC stops updating: the pools freeze wherever they are.

**Drain simulation.** For week 6, week 12 and the playoffs, I ran today's real sources with the PPR pool cut down least-drafted first. That's the order FFC loses players as drafts dry up. Every guard was evaluated at each size:

| PPR pool | PPR floor | draftable ratio | total | DST / K / RB / WR / QB / TE | Sleeper match | biggest positional tier 1 | drift aborts |
|---|---|---|---|---|---|---|---|
| 194 (today) | ok | 194/200 ok | 548 | 32/38/110/174/83/111 | 96% | 4 | 0 |
| 180 | ok | 180/200 ok | 547 | unchanged | 97% | 5 | 0 |
| 179 | ok | **ABORT** | 547 | unchanged | 97% | 5 | 0 |
| 150 | ok | **ABORT** | 546 | unchanged | 98% | 4 | 0 |
| 100 | ok | **ABORT** | 546 | unchanged | 99% | 4 | 0 |
| 99 | **ABORT** | **ABORT** | 546 | unchanged | 99% | 4 | 0 |
| 0 | **ABORT** | **ABORT** | 545 | unchanged | 100% | 8 | 0 |

## build.py

Column key:
- **Holds?** Is the assumption still true now that drafting is over?
- **Wk 6 / Wk 12 / playoffs:** based on the drain simulation and weekly behaviour. "Pool" means the PPR pool.
- **False fire?** Will it abort on correct in-season data?

| # | Guard | Assumes | Holds? | Wk 6 / Wk 12 / playoffs | False fire? | Action |
|---|---|---|---|---|---|---|
| 1 | `MIN_SCHEDULE_WEEKS` 17 | ESPN returns the full regular-season schedule | Yes: past weeks stay on the scoreboard with `seasontype=2` | ok / ok / ok (the NFL postseason isn't fetched, and weeks 1–18 stay listed) | No | none |
| 2a | `MIN_ADP_ENTRIES` 100, **PPR** | mock drafts keep a 100+ player PPR pool | Only while FFC keeps drafts or freezes the pool | If pools keep draining: aborts somewhere below 100, probably by week 6, and stays aborted all season. If FFC freezes, never fires | **Yes, if pools drain** | Kept strict as instructed. PPR alone anchors `adp` and `ro`. See J1 |
| 2b | `MIN_ADP_ENTRIES` 100, **half/std** | same as 2a | **No**: half hit 54 and std 126 on Sep 14 | aborted every build from Sep 14 | **Yes, it did** | **FIXED** (937650e): aborts before kickoff, warns in season |
| 2c | *new* format coverage 90% of the PPR top 100 | a format's ranks are only coherent when its pool holds the top of the board | designed for in-season | drops `rh`/`rs` whenever coverage is thin, from today onward. Never aborts | No (it's a drop, not an abort) | added in 937650e |
| 3 | `MIN_DRAFTABLE_ADP_RATIO` 0.90 of `DRAFTABLE_N` 200 | a 12×16 **draft board** exists and should be market-priced | **No**: nobody is drafting a 200-player board, and it's effectively a **180-player PPR floor** | aborts once the pool is under 180, maybe within days (the pool was losing ~15/day Sep 9–13 and is 14 players from the line). If FFC freezes above 180, never fires | **Yes, soonest of all** | **Not changed: judgment call J1** |
| 4 | `MIN_SLEEPER_MATCH` 0.60 | name matching works | Yes. Unmatched players are almost all market players, so the ratio *rises* as the pool shrinks (96% → 100%) | ok / ok / ok | No | none |
| 5 | `MIN_USAGE_RATIO` 0.60 | most players have usage from some season | Yes: fallback to 2025 until 2 current-season games; rookies fill in from week 2 | ok. Byes don't matter (last played games are used). Late-season IR players keep their older games | No | none (fixed for kickoff in 8dd98ef) |
| 6 | team target-share sum > 130% | shares sum to ~100% within one season | Yes: mixing seasons is blocked. Traded players count toward their current team on both sides of the ratio | ok / ok / ok | No | none |
| 7 | team schedule coverage ≥15 weeks | every team plays 17 of 18 listed weeks | Yes | ok / ok / ok | No | none |
| 8 | `MIN_PLAYERS` 150 | the assembled board isn't gutted | Yes: the Sleeper union keeps 545+ even with no PPR pool | ok / ok / ok | No | none |
| 9 | tier 1 > `MAX_POS_TIER1` 8 | tiering can regress | n/a: `POS_TIER_MAX_SIZE` 8 force-breaks at 8, so it **can't fire** in any phase | never | No | none (it's a no-op, noted) |
| 10 | `MIN_TOTAL` 400, `DST_EXACT` 32, `POS_FLOOR` | Sleeper depth charts fill the board | Yes: counts are flat across the drain | ok / ok / ok. Depth charts are maintained in season | No | none |
| 11 | drift guard (`DRIFT_ABORT` 20, `DRIFT_BACKSTOP` 80) | ro stays near market ADP unless something known moves it | Yes: already made season-aware. Players who drop out of the pool leave the comparison set rather than drifting | 0 aborts at every pool size simulated. As `adp` ages, drift grows only through injury/depth/nudge, which are allowed up to 80 | No (see J2) | none |
| — | movers d7/d14 | ADP moves | ADP is near-static in season | never aborts by design (WARN only); app hides empty movers (v1.9) | No | none |

## model.py thresholds

| Threshold | Assumes | Holds in season? | Effect later | Action |
|---|---|---|---|---|
| `INJURY_PENALTY`, `SEASON_START` | status moves rank only after kickoff | Yes, built for it | unchanged | none |
| depth-chart knock `dco >= 3` and ADP `< 120` | a buried player priced like a starter | Yes | fewer market players, so fewer eligible | none |
| trending `> 5000` (−3 rank, and a drift-guard "known cause") | Sleeper's 24h add counts look like August | **Partly**: in-season waiver adds run far above 5000, so many more market players qualify | never fires anything. It **widens** the drift guard's allowed causes and nudges more players 3 spots | **J2**, not changed |
| `ADPLESS_DCO_MAX` | depth chart decides who joins the adpless union | Yes | unchanged | none |
| tiering `POS_TIER_*`, `OVERALL_TIER_*`, `POS_TIER_MAX_ADP_SPAN` 18 | tiers come from score gaps and ADP spans | Yes: adpless players' sentinel ADP equals their `ro`, so spans stay monotonic | more of the board is adpless, tail tiers grow (WR tier 9 already has 128 players). Cosmetic | none |
| `DRAFTABLE_N` 200 | a draft board | **No** | feeds guard 3 | J1 |
| `USAGE_MIN_CURRENT_GAMES` 2 | measured on 2025 weeks 1–6 | Yes | unchanged | none |
| `NEWS_*` | recency | Yes | unchanged | none |
| *new* `FORMAT_COVERAGE_TOP_N` 100, `FORMAT_COVERAGE_MIN` 0.90 | see the comment in model.py | built for in-season | half/std ranks will likely stay dropped for the rest of the season | added |
| `in_season.PROD_*` | production weight curve | built for in-season | unchanged | none |

Not guarded but will degrade: superflex `sfa` comes from a 30-day 2QB window (246 players today) and goes null player by player as the window drains. The app falls back to `adp`.

## Judgment calls (reported, not changed)

**J1: PPR pool drain (guards 2a and 3). This is the next freeze.**
- **What happens:** once the PPR pool is under 180, the draftable-ratio guard aborts every build. Under 100, the PPR floor does too. Neither recovers until FFC's pool grows back, which may never happen in-season. The guards aren't wrong about the data: a board whose top 200 is half sentinels *is* less market-anchored. The mistake is treating that as an upstream outage instead of the season.
- **Options:**
  - (a) Carry the last good PPR pool forward as the in-season market anchor. Freeze `adp`, the anchor for `ro`, at the last build that passed. Keep the guards for real outages, like a non-200 response or an empty list mid-week.
  - (b) In season, lower the draftable requirement to what's actually load-bearing, e.g. the top 100 market-backed.
  - (c) Let `isr` (production) carry the board as ADP thins.
- **Why I didn't change it:** each option changes what `ro` means, so the owner needs to pick one. You also told me to keep PPR strict.
- **Watch the next few builds' "draftable ADP coverage" line:** 194/200 today, aborts at 179.

**J2: trending threshold 5000.** Calibrated on preseason add volumes. In season it lets many more players take −3 rank spots and a drift-guard pass. It loosens the guard; it never trips it. Leave it unless the drift log shows trending-only drift past 20.

**Also: CI doesn't run the tests.** `daily.yml` runs `build.py` only, and none of the `test_*.py` files run on push.
