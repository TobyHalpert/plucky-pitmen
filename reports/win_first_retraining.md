# Win-first retraining comparison

Generated **2026-11-10**.

## What changed

- Warm start: `checkpoints/selfplay4_balance_s7003.npz`.
- Retrained checkpoint: `checkpoints/selfplay4_winfirst_s8111.npz`.
- Training objective: **terminal_win_share**, with `gamma=1.0`.
- The previous trainer rewarded score deltas and intermediate gem/card/escape/death shaping, then added a winner bonus.
- The new mode gives only the final official winners reward, split as `1 / number_of_winners`; all other seats get zero.
  It disables intermediate rewards and forces `gamma=1`, so runner-up placement and points do not receive positive reward.
- Batched REINFORCE returns now reset at each game's episode boundary instead of leaking outcomes between games.
- Training run: 60 generations × 64 games = 3840 games; seed 8111, 4 players, mirror probability 0.25, league probability 0.50.

## Evaluation method

- Player counts tested: **3, 4, 5** seats.
- Opponent styles: `rule`, `greedy`, `cautious`, `reader`, `rusher`, `leaver`.
- **180** games per opponent and player count for each model; seats rotate evenly.
- Candidate and baseline use the same held-out seeds, opponent style, and seat schedule; neural action temperature is **0.70**.
- Primary metric is official win share: a co-win is shared evenly, after applying the official tie-break. Score/margin are diagnostics only.
- Paired differences use 95% percentile bootstrap intervals over games completed by both models. Overall panels equally weight the opponent styles.
- Aborted games are excluded: baseline 0, retrained 1; table N is the matched completed count.

## Primary result: official win share

| Players | Matched games | Baseline | Win-only retrain | Paired Δ [95% CI] | Unique win rate, before → after |
|--:|--:|:--|:--|:--|:--|
| 3 | 1079 | +76.64% [+74.32%, +79.05%] | +75.43% [+72.93%, +77.84%] | -1.21% [-3.15%, +0.74%] | 76.6% → 75.4% |
| 4 | 1080 | +71.67% [+69.17%, +74.26%] | +72.13% [+69.63%, +74.54%] | +0.46% [-1.67%, +2.50%] | 71.7% → 72.1% |
| 5 | 1080 | +65.46% [+62.96%, +68.06%] | +66.57% [+64.07%, +69.17%] | +1.11% [-1.02%, +3.24%] | 65.5% → 66.6% |

**Reading:** The paired 95% win-share interval crosses zero at every player count. The point estimates are mixed (3p -1.2%, 4p +0.5%, 5p +1.1%), so this run does not establish a reliable win-share improvement. Mean score-margin changes were 3p -0.08 points, 4p -0.05 points, 5p +0.00 points; those remain diagnostics, not the objective.

## Per-opponent win-share deltas

| Players | Opponent | Matched games | Baseline win share | Win-only win share | Paired Δ [95% CI] | Mean-margin Δ (points) |
|--:|:--|--:|--:|--:|:--|--:|
| 3 | rule | 179 | 70.9% | 67.6% | -3.35% [-9.50%, +3.35%] | -0.20 |
| 3 | greedy | 180 | 66.1% | 68.9% | +2.78% [-1.11%, +7.22%] | +0.04 |
| 3 | cautious | 180 | 69.4% | 67.2% | -2.22% [-6.67%, +2.22%] | +0.18 |
| 3 | reader | 180 | 72.8% | 70.0% | -2.78% [-8.89%, +3.33%] | -0.39 |
| 3 | rusher | 180 | 98.9% | 99.4% | +0.56% [+0.00%, +1.67%] | +0.11 |
| 3 | leaver | 180 | 81.7% | 79.4% | -2.22% [-6.67%, +2.22%] | -0.21 |
| 4 | rule | 180 | 66.7% | 63.9% | -2.78% [-9.44%, +4.44%] | -0.48 |
| 4 | greedy | 180 | 61.7% | 65.6% | +3.89% [-1.11%, +9.44%] | +0.12 |
| 4 | cautious | 180 | 52.8% | 52.8% | +0.00% [-5.56%, +5.00%] | +0.02 |
| 4 | reader | 180 | 65.0% | 65.6% | +0.56% [-5.00%, +6.11%] | -0.27 |
| 4 | rusher | 180 | 98.9% | 98.9% | +0.00% [+0.00%, +0.00%] | +0.11 |
| 4 | leaver | 180 | 85.0% | 86.1% | +1.11% [-3.33%, +5.56%] | +0.19 |
| 5 | rule | 180 | 63.9% | 61.1% | -2.78% [-10.00%, +5.00%] | -0.47 |
| 5 | greedy | 180 | 50.6% | 52.2% | +1.67% [-3.89%, +6.67%] | +0.03 |
| 5 | cautious | 180 | 39.4% | 40.0% | +0.56% [-5.56%, +6.67%] | +0.08 |
| 5 | reader | 180 | 61.1% | 62.8% | +1.67% [-3.89%, +7.22%] | +0.03 |
| 5 | rusher | 180 | 97.8% | 98.3% | +0.56% [+0.00%, +1.67%] | +0.08 |
| 5 | leaver | 180 | 80.0% | 85.0% | +5.00% [+1.11%, +9.44%] | +0.26 |

## Behavior and score diagnostics

These describe play in this bot panel; none supersedes the official win outcome.

| Players | Mean score, before → after | Mean point margin, before → after | Escapes/game, before → after | Gems per escaped haul, before → after | Deaths/game, before → after | Blasts/game, before → after |
|--:|:--|:--|:--|:--|:--|:--|
| 3 | 4.97 → 4.91 | +4.49 → +4.41 | 2.40 → 2.37 | 1.00 → 1.01 | 0.60 → 0.63 | 0.20 → 0.20 |
| 4 | 4.33 → 4.29 | +4.02 → +3.97 | 2.29 → 2.26 | 0.82 → 0.83 | 0.71 → 0.74 | 0.12 → 0.13 |
| 5 | 3.94 → 3.94 | +3.77 → +3.77 | 2.18 → 2.17 | 0.75 → 0.76 | 0.82 → 0.83 | 0.11 → 0.12 |

## Limits

This is a controlled comparison against scripted bots, not an estimate of human-relative strength. It tests one warm start and one training seed; it cannot establish that the new objective is generally superior across independent training runs.
The checkpoint-level result is still useful as a before/after check of this specific retraining run. See the accompanying JSON for exact seeds, intervals, hashes, and cell summaries.
