# Current-rules balance study

Generated **2026-11-10**. This report is separate from the historical `BALANCE.md` measurements.

## Scope and method

This is a bot/self-play stress test, **not evidence that human games are balanced**. All games use the same current rules: `n-1` consecutive-pass rule, solo-escape haul bonus **on**, and blind sequential blast-card choice **on**. The target's seat is rotated; each benchmark cell pits one target against identical copies of one named opponent policy. All reported intervals are 95% bootstrap intervals over completed games; engine-aborted games are excluded, not scored as losses, and their counts are reported per cell. The tables show the range of seed-block mean margins, and the JSON retains per-seed summaries. These intervals do not capture variation over human players or a population of independently trained models.

- Evaluation seeds: `1729, 8191, 65537`
- Games per seed and cell: **60** (seat-rotated; 180 attempted per cell)
- Neural action temperature: **0.70**
- Complete/aborted games: **16,689 / 51**
- Total simulated games (including mirror/control games): **16,740**
- Aborted by block: mirror 0, neural benchmark 0, scripted control 51
- Abort reasons: `step_limit` 51

### Checkpoints

| policy | checkpoint | SHA-256 (prefix) | role in this study |
|:--|:--|:--|:--|
| `selfplay4` | `checkpoints/selfplay4.npz` | `009a051ed366` | shipped reference |
| `selfplay4_tactical` | `checkpoints/selfplay4_tactical.npz` | `e90b1f2a83a4` | downstream tactical policy (not an independent training replicate) |
| `selfplay4_balance_s7002` | `checkpoints/selfplay4_balance_s7002.npz` | `27b58b5e3916` | 60-generation continuation from `selfplay4`, training seed 7002 |
| `selfplay4_balance_s7003` | `checkpoints/selfplay4_balance_s7003.npz` | `37c7302d2ea1` | 60-generation continuation from `selfplay4`, training seed 7003 |

## Quick read

For each player count, the model-panel margin range is the min-to-max of the four checkpoint means against the six selected bot styles; it is not a confidence interval. Paired deltas compare each new continuation with the shipped checkpoint on the same game seeds and opponent cells.

| players | model-panel margin range | seed 7002 Δ margin [95% CI] | seed 7003 Δ margin [95% CI] | mirror mean score-range range |
|--:|:--|:--|:--|:--|
| 3 | +3.92 to +4.46 | +0.17 [-0.01, +0.36] | +0.53 [+0.28, +0.79] | 4.84 to 5.02 |
| 4 | +3.52 to +4.18 | +0.16 [-0.01, +0.34] | +0.65 [+0.43, +0.87] | 4.87 to 5.17 |
| 5 | +2.82 to +3.98 | +0.51 [+0.35, +0.68] | +1.16 [+0.95, +1.38] | 4.39 to 4.96 |

## 3-player games

Each row has **180 attempted games** (60 per evaluation seed) unless aborted; margins are target score minus the mean score of the other seats. The seed range is the minimum-to-maximum of the 3 seed-block mean margins, not a confidence interval.

| target policy | opponent style | completed N | win rate [95% CI] | top-score rate | mean margin [95% CI] | mean score [95% CI] | seed margin range | aborted |
|:--|:--|--:|:--|--:|:--|:--|:--|--:|
| `selfplay4` | `rule` | 180 | 65.0% [58.3%, 71.7%] | 67.8% | +3.84 [+3.21, +4.46] | +4.55 [+4.09, +5.02] | +3.51 to +4.18 | 0 |
| `selfplay4` | `greedy` | 180 | 60.6% [53.3%, 67.2%] | 65.0% | +2.17 [+1.62, +2.71] | +2.13 [+1.69, +2.58] | +1.68 to +3.02 | 0 |
| `selfplay4` | `cautious` | 180 | 66.7% [59.4%, 73.3%] | 71.1% | +2.95 [+2.42, +3.47] | +3.13 [+2.69, +3.56] | +2.46 to +3.24 | 0 |
| `selfplay4` | `reader` | 180 | 69.4% [62.2%, 76.1%] | 73.9% | +4.01 [+3.42, +4.58] | +4.42 [+3.96, +4.88] | +3.88 to +4.15 | 0 |
| `selfplay4` | `rusher` | 180 | 97.2% [95.0%, 99.4%] | 98.3% | +6.89 [+6.49, +7.31] | +7.80 [+7.50, +8.10] | +6.63 to +7.03 | 0 |
| `selfplay4` | `leaver` | 180 | 71.1% [64.4%, 77.8%] | 77.8% | +3.69 [+3.16, +4.21] | +4.91 [+4.51, +5.31] | +2.99 to +4.33 | 0 |
| `selfplay4_tactical` | `rule` | 180 | 68.9% [62.2%, 75.6%] | 74.4% | +4.18 [+3.51, +4.88] | +4.73 [+4.23, +5.21] | +3.47 to +4.58 | 0 |
| `selfplay4_tactical` | `greedy` | 180 | 68.9% [61.7%, 75.6%] | 72.8% | +2.79 [+2.28, +3.31] | +2.51 [+2.07, +2.96] | +2.30 to +3.27 | 0 |
| `selfplay4_tactical` | `cautious` | 180 | 67.8% [60.6%, 73.9%] | 72.2% | +3.38 [+2.79, +3.98] | +3.43 [+2.97, +3.89] | +3.00 to +3.81 | 0 |
| `selfplay4_tactical` | `reader` | 180 | 67.2% [60.6%, 73.9%] | 70.0% | +3.72 [+3.08, +4.33] | +4.19 [+3.72, +4.69] | +3.48 to +4.00 | 0 |
| `selfplay4_tactical` | `rusher` | 180 | 97.8% [95.6%, 99.4%] | 99.4% | +6.95 [+6.53, +7.34] | +7.84 [+7.53, +8.12] | +6.68 to +7.28 | 0 |
| `selfplay4_tactical` | `leaver` | 180 | 76.1% [70.0%, 81.7%] | 79.4% | +3.76 [+3.19, +4.32] | +4.95 [+4.46, +5.41] | +2.90 to +4.40 | 0 |
| `selfplay4_balance_s7002` | `rule` | 180 | 68.3% [61.7%, 75.0%] | 71.1% | +4.17 [+3.56, +4.84] | +4.87 [+4.38, +5.34] | +3.50 to +4.91 | 0 |
| `selfplay4_balance_s7002` | `greedy` | 180 | 67.2% [60.6%, 73.9%] | 71.7% | +2.85 [+2.26, +3.45] | +2.78 [+2.31, +3.26] | +2.26 to +3.92 | 0 |
| `selfplay4_balance_s7002` | `cautious` | 180 | 63.3% [56.1%, 70.0%] | 67.2% | +3.11 [+2.54, +3.71] | +3.27 [+2.79, +3.75] | +2.87 to +3.50 | 0 |
| `selfplay4_balance_s7002` | `reader` | 180 | 70.6% [63.9%, 77.2%] | 74.4% | +3.90 [+3.32, +4.48] | +4.33 [+3.90, +4.76] | +3.41 to +4.19 | 0 |
| `selfplay4_balance_s7002` | `rusher` | 180 | 98.3% [96.1%, 100.0%] | 98.3% | +6.97 [+6.57, +7.37] | +7.86 [+7.54, +8.14] | +6.83 to +7.07 | 0 |
| `selfplay4_balance_s7002` | `leaver` | 180 | 72.2% [65.0%, 78.9%] | 73.9% | +3.53 [+2.99, +4.08] | +4.67 [+4.21, +5.11] | +3.06 to +3.86 | 0 |
| `selfplay4_balance_s7003` | `rule` | 180 | 66.1% [59.4%, 72.8%] | 71.1% | +4.38 [+3.72, +5.00] | +5.03 [+4.56, +5.49] | +3.30 to +5.36 | 0 |
| `selfplay4_balance_s7003` | `greedy` | 180 | 71.1% [65.0%, 77.8%] | 73.3% | +3.16 [+2.62, +3.70] | +3.02 [+2.56, +3.49] | +2.73 to +3.49 | 0 |
| `selfplay4_balance_s7003` | `cautious` | 180 | 70.0% [63.3%, 76.1%] | 77.2% | +3.55 [+3.01, +4.12] | +3.60 [+3.12, +4.07] | +3.20 to +3.84 | 0 |
| `selfplay4_balance_s7003` | `reader` | 180 | 73.3% [67.2%, 80.0%] | 76.7% | +4.17 [+3.57, +4.78] | +4.64 [+4.21, +5.11] | +3.79 to +4.44 | 0 |
| `selfplay4_balance_s7003` | `rusher` | 180 | 98.3% [96.1%, 100.0%] | 99.4% | +7.36 [+6.97, +7.71] | +8.14 [+7.86, +8.41] | +7.23 to +7.45 | 0 |
| `selfplay4_balance_s7003` | `leaver` | 180 | 81.1% [75.0%, 86.7%] | 85.0% | +4.11 [+3.62, +4.62] | +5.23 [+4.77, +5.62] | +3.48 to +4.58 | 0 |

### Equal-weight opponent portfolio

All six opponent styles received the same number of attempted games, and no neural benchmark game aborted, so this panel is equally weighted by completed games. It describes the selected bot set, not a universal expected win rate.

| target policy | completed N | win rate [95% CI] | top-score rate | mean margin [95% CI] | mean score [95% CI] | seed margin range |
|:--|--:|:--|--:|:--|:--|:--|
| `selfplay4` | 1080 | 71.7% [69.1%, 74.4%] | 75.6% | +3.92 [+3.68, +4.19] | +4.49 [+4.29, +4.70] | +3.88 to +3.99 |
| `selfplay4_tactical` | 1080 | 74.4% [71.9%, 77.2%] | 78.1% | +4.13 [+3.87, +4.38] | +4.61 [+4.40, +4.82] | +3.89 to +4.26 |
| `selfplay4_balance_s7002` | 1080 | 73.3% [70.7%, 76.0%] | 76.1% | +4.09 [+3.85, +4.35] | +4.63 [+4.43, +4.83] | +4.01 to +4.21 |
| `selfplay4_balance_s7003` | 1080 | 76.7% [74.1%, 79.3%] | 80.5% | +4.46 [+4.21, +4.70] | +4.94 [+4.75, +5.14] | +4.23 to +4.64 |

### Paired margin differences vs shipped `selfplay4`

Positive values mean the named checkpoint had a higher score margin than `selfplay4` on the same evaluation seed and opponent-style cells. The intervals bootstrap paired game-level differences across all 6 styles and 3 seed blocks. These compare particular checkpoints; they are not estimates of training-algorithm effects.

| checkpoint | paired games | margin difference [95% CI] |
|:--|--:|:--|
| `selfplay4_tactical` | 1080 | +0.21 [-0.06, +0.48] |
| `selfplay4_balance_s7002` | 1080 | +0.17 [-0.01, +0.36] |
| `selfplay4_balance_s7003` | 1080 | +0.53 [+0.28, +0.79] |

### Scripted reference policies vs `rule`

These controls make the tested opponent styles easier to interpret; each had 180 attempted games, with engine-aborted games excluded from estimates.

| target policy | completed N | win rate [95% CI] | mean margin [95% CI] | seed margin range | aborted |
|:--|--:|:--|:--|:--|--:|
| `rule` | 179 | 33.5% [26.8%, 40.8%] | -0.09 [-0.56, +0.37] | -0.43 to +0.45 | 1 |
| `rusher` | 178 | 21.3% [15.2%, 27.5%] | -1.83 [-2.31, -1.34] | -2.61 to -1.24 | 2 |
| `leaver` | 179 | 44.1% [36.9%, 51.4%] | +0.48 [-0.11, +1.06] | +0.25 to +0.71 | 1 |

At this player count, **4** scripted-control games aborted. Across the full study, the abort reasons were `step_limit` 51. No neural benchmark or mirror games aborted. The `rusher` control has the highest abort count here; high-abort control estimates are limited diagnostics, not reliable policy estimates.

### Identical-policy mirror matches (luck/seat-spread proxy)

Every seat uses the same checkpoint. The mean within-game score range (highest minus lowest score) captures stochastic outcome and seat effects together; it is not a pure decomposition of luck.

| checkpoint | completed N | mean score range [95% CI] | games with tied top score [95% CI] | mean score by seat |
|:--|--:|:--|:--|:--|
| `selfplay4` | 180 | 4.98 [4.64, 5.32] | 11.1% [6.7%, 15.6%] | +1.71, +1.91, +1.95 |
| `selfplay4_tactical` | 180 | 5.02 [4.67, 5.39] | 10.0% [6.1%, 14.4%] | +1.93, +1.98, +2.16 |
| `selfplay4_balance_s7002` | 180 | 5.01 [4.67, 5.34] | 15.0% [10.0%, 20.6%] | +2.44, +2.29, +2.47 |
| `selfplay4_balance_s7003` | 180 | 4.84 [4.51, 5.16] | 19.4% [13.9%, 25.6%] | +2.41, +2.67, +2.43 |

## 4-player games

Each row has **180 attempted games** (60 per evaluation seed) unless aborted; margins are target score minus the mean score of the other seats. The seed range is the minimum-to-maximum of the 3 seed-block mean margins, not a confidence interval.

| target policy | opponent style | completed N | win rate [95% CI] | top-score rate | mean margin [95% CI] | mean score [95% CI] | seed margin range | aborted |
|:--|:--|--:|:--|--:|:--|:--|:--|--:|
| `selfplay4` | `rule` | 180 | 60.0% [52.8%, 67.2%] | 63.3% | +3.17 [+2.58, +3.77] | +3.54 [+3.07, +4.03] | +2.51 to +3.98 | 0 |
| `selfplay4` | `greedy` | 180 | 52.2% [45.6%, 59.4%] | 60.0% | +1.80 [+1.33, +2.24] | +1.51 [+1.16, +1.88] | +1.35 to +2.37 | 0 |
| `selfplay4` | `cautious` | 180 | 48.3% [41.1%, 55.6%] | 56.7% | +1.60 [+1.18, +2.00] | +1.39 [+1.02, +1.78] | +1.34 to +1.87 | 0 |
| `selfplay4` | `reader` | 180 | 64.4% [57.2%, 71.7%] | 70.6% | +3.48 [+2.95, +4.00] | +3.77 [+3.35, +4.19] | +2.98 to +3.82 | 0 |
| `selfplay4` | `rusher` | 180 | 97.8% [95.6%, 99.4%] | 99.4% | +7.01 [+6.59, +7.37] | +7.89 [+7.61, +8.17] | +6.40 to +7.57 | 0 |
| `selfplay4` | `leaver` | 180 | 75.6% [69.4%, 81.7%] | 78.9% | +4.08 [+3.56, +4.58] | +5.03 [+4.62, +5.43] | +3.48 to +4.83 | 0 |
| `selfplay4_tactical` | `rule` | 180 | 63.9% [56.7%, 70.6%] | 68.9% | +3.97 [+3.35, +4.58] | +4.16 [+3.67, +4.66] | +3.61 to +4.31 | 0 |
| `selfplay4_tactical` | `greedy` | 180 | 50.0% [42.8%, 57.2%] | 61.7% | +1.76 [+1.32, +2.24] | +1.47 [+1.12, +1.86] | +1.44 to +2.29 | 0 |
| `selfplay4_tactical` | `cautious` | 180 | 42.2% [35.0%, 49.4%] | 52.8% | +1.44 [+1.03, +1.88] | +1.37 [+1.01, +1.76] | +1.16 to +1.73 | 0 |
| `selfplay4_tactical` | `reader` | 180 | 55.0% [47.8%, 62.2%] | 60.0% | +2.89 [+2.28, +3.50] | +3.29 [+2.79, +3.80] | +2.39 to +3.44 | 0 |
| `selfplay4_tactical` | `rusher` | 180 | 98.3% [96.1%, 100.0%] | 100.0% | +7.03 [+6.63, +7.44] | +7.89 [+7.59, +8.17] | +6.48 to +7.80 | 0 |
| `selfplay4_tactical` | `leaver` | 180 | 81.7% [75.6%, 87.2%] | 88.3% | +4.52 [+4.04, +5.01] | +5.41 [+4.99, +5.79] | +4.16 to +4.97 | 0 |
| `selfplay4_balance_s7002` | `rule` | 180 | 59.4% [52.2%, 66.7%] | 65.0% | +3.46 [+2.84, +4.07] | +3.85 [+3.34, +4.33] | +3.03 to +4.13 | 0 |
| `selfplay4_balance_s7002` | `greedy` | 180 | 56.7% [50.0%, 63.9%] | 66.1% | +2.29 [+1.87, +2.72] | +1.96 [+1.57, +2.33] | +2.14 to +2.53 | 0 |
| `selfplay4_balance_s7002` | `cautious` | 180 | 50.6% [43.3%, 57.8%] | 56.1% | +1.92 [+1.49, +2.36] | +1.69 [+1.29, +2.12] | +1.66 to +2.15 | 0 |
| `selfplay4_balance_s7002` | `reader` | 180 | 59.4% [52.2%, 66.7%] | 65.0% | +3.29 [+2.71, +3.89] | +3.59 [+3.13, +4.07] | +3.06 to +3.59 | 0 |
| `selfplay4_balance_s7002` | `rusher` | 180 | 97.2% [94.4%, 99.4%] | 98.9% | +6.96 [+6.56, +7.37] | +7.84 [+7.54, +8.14] | +6.15 to +7.43 | 0 |
| `selfplay4_balance_s7002` | `leaver` | 180 | 76.7% [70.0%, 82.2%] | 80.6% | +4.19 [+3.71, +4.70] | +5.01 [+4.58, +5.42] | +3.94 to +4.68 | 0 |
| `selfplay4_balance_s7003` | `rule` | 180 | 72.2% [65.6%, 78.3%] | 76.1% | +4.77 [+4.21, +5.36] | +4.87 [+4.40, +5.33] | +4.38 to +5.21 | 0 |
| `selfplay4_balance_s7003` | `greedy` | 180 | 62.2% [55.0%, 69.4%] | 67.8% | +2.47 [+2.05, +2.92] | +2.11 [+1.74, +2.53] | +2.01 to +3.37 | 0 |
| `selfplay4_balance_s7003` | `cautious` | 180 | 52.2% [45.0%, 59.4%] | 58.9% | +1.93 [+1.50, +2.38] | +1.77 [+1.38, +2.16] | +1.89 to +2.01 | 0 |
| `selfplay4_balance_s7003` | `reader` | 180 | 62.2% [55.0%, 69.4%] | 65.0% | +3.73 [+3.16, +4.31] | +3.99 [+3.51, +4.44] | +3.31 to +4.43 | 0 |
| `selfplay4_balance_s7003` | `rusher` | 180 | 98.9% [97.2%, 100.0%] | 98.9% | +7.62 [+7.22, +8.00] | +8.31 [+8.01, +8.57] | +7.05 to +8.13 | 0 |
| `selfplay4_balance_s7003` | `leaver` | 180 | 83.3% [77.8%, 88.3%] | 87.8% | +4.54 [+4.06, +5.06] | +5.42 [+5.01, +5.82] | +3.83 to +5.05 | 0 |

### Equal-weight opponent portfolio

All six opponent styles received the same number of attempted games, and no neural benchmark game aborted, so this panel is equally weighted by completed games. It describes the selected bot set, not a universal expected win rate.

| target policy | completed N | win rate [95% CI] | top-score rate | mean margin [95% CI] | mean score [95% CI] | seed margin range |
|:--|--:|:--|--:|:--|:--|:--|
| `selfplay4` | 1080 | 66.4% [63.4%, 69.2%] | 71.5% | +3.52 [+3.30, +3.76] | +3.86 [+3.65, +4.08] | +3.14 to +3.83 |
| `selfplay4_tactical` | 1080 | 65.2% [62.3%, 68.0%] | 71.9% | +3.60 [+3.37, +3.84] | +3.93 [+3.72, +4.14] | +3.46 to +3.79 |
| `selfplay4_balance_s7002` | 1080 | 66.7% [64.0%, 69.4%] | 71.9% | +3.68 [+3.46, +3.92] | +3.99 [+3.78, +4.20] | +3.39 to +3.90 |
| `selfplay4_balance_s7003` | 1080 | 71.9% [69.2%, 74.6%] | 75.7% | +4.18 [+3.94, +4.41] | +4.41 [+4.19, +4.63] | +3.80 to +4.40 |

### Paired margin differences vs shipped `selfplay4`

Positive values mean the named checkpoint had a higher score margin than `selfplay4` on the same evaluation seed and opponent-style cells. The intervals bootstrap paired game-level differences across all 6 styles and 3 seed blocks. These compare particular checkpoints; they are not estimates of training-algorithm effects.

| checkpoint | paired games | margin difference [95% CI] |
|:--|--:|:--|
| `selfplay4_tactical` | 1080 | +0.08 [-0.16, +0.31] |
| `selfplay4_balance_s7002` | 1080 | +0.16 [-0.01, +0.34] |
| `selfplay4_balance_s7003` | 1080 | +0.65 [+0.43, +0.87] |

### Scripted reference policies vs `rule`

These controls make the tested opponent styles easier to interpret; each had 180 attempted games, with engine-aborted games excluded from estimates.

| target policy | completed N | win rate [95% CI] | mean margin [95% CI] | seed margin range | aborted |
|:--|--:|:--|:--|:--|--:|
| `rule` | 180 | 16.1% [11.1%, 21.7%] | -0.62 [-0.98, -0.26] | -1.19 to -0.19 | 0 |
| `rusher` | 170 | 10.6% [6.5%, 15.3%] | -1.51 [-1.82, -1.18] | -1.55 to -1.45 | 10 |
| `leaver` | 177 | 54.8% [48.0%, 62.1%] | +2.27 [+1.68, +2.82] | +2.10 to +2.39 | 3 |

At this player count, **13** scripted-control games aborted. Across the full study, the abort reasons were `step_limit` 51. No neural benchmark or mirror games aborted. The `rusher` control has the highest abort count here; high-abort control estimates are limited diagnostics, not reliable policy estimates.

### Identical-policy mirror matches (luck/seat-spread proxy)

Every seat uses the same checkpoint. The mean within-game score range (highest minus lowest score) captures stochastic outcome and seat effects together; it is not a pure decomposition of luck.

| checkpoint | completed N | mean score range [95% CI] | games with tied top score [95% CI] | mean score by seat |
|:--|--:|:--|:--|:--|
| `selfplay4` | 180 | 5.17 [4.89, 5.46] | 11.1% [6.7%, 15.6%] | +0.92, +1.43, +1.36, +1.30 |
| `selfplay4_tactical` | 180 | 5.00 [4.69, 5.31] | 12.8% [7.8%, 17.8%] | +1.33, +1.13, +1.51, +1.36 |
| `selfplay4_balance_s7002` | 180 | 4.87 [4.61, 5.13] | 16.1% [10.6%, 21.7%] | +1.82, +1.57, +2.07, +1.46 |
| `selfplay4_balance_s7003` | 180 | 5.04 [4.79, 5.29] | 15.0% [10.0%, 20.6%] | +1.93, +1.87, +2.24, +1.76 |

## 5-player games

Each row has **180 attempted games** (60 per evaluation seed) unless aborted; margins are target score minus the mean score of the other seats. The seed range is the minimum-to-maximum of the 3 seed-block mean margins, not a confidence interval.

| target policy | opponent style | completed N | win rate [95% CI] | top-score rate | mean margin [95% CI] | mean score [95% CI] | seed margin range | aborted |
|:--|:--|--:|:--|--:|:--|:--|:--|--:|
| `selfplay4` | `rule` | 180 | 48.9% [41.7%, 56.7%] | 53.3% | +2.27 [+1.77, +2.78] | +2.47 [+2.04, +2.90] | +1.92 to +2.46 | 0 |
| `selfplay4` | `greedy` | 180 | 44.4% [37.8%, 51.7%] | 55.0% | +1.33 [+0.97, +1.68] | +0.91 [+0.60, +1.24] | +1.13 to +1.65 | 0 |
| `selfplay4` | `cautious` | 180 | 36.7% [29.4%, 43.9%] | 48.9% | +0.99 [+0.67, +1.34] | +0.74 [+0.43, +1.04] | +0.91 to +1.04 | 0 |
| `selfplay4` | `reader` | 180 | 41.7% [34.4%, 48.9%] | 47.2% | +1.76 [+1.28, +2.27] | +2.03 [+1.64, +2.42] | +1.03 to +2.17 | 0 |
| `selfplay4` | `rusher` | 180 | 97.2% [94.4%, 99.4%] | 98.3% | +6.83 [+6.41, +7.26] | +7.77 [+7.44, +8.06] | +6.25 to +7.38 | 0 |
| `selfplay4` | `leaver` | 180 | 73.9% [67.2%, 80.6%] | 78.9% | +3.72 [+3.23, +4.24] | +4.62 [+4.19, +5.06] | +3.39 to +4.10 | 0 |
| `selfplay4_tactical` | `rule` | 180 | 55.6% [48.3%, 62.8%] | 57.8% | +2.69 [+2.13, +3.21] | +2.85 [+2.41, +3.31] | +2.37 to +3.13 | 0 |
| `selfplay4_tactical` | `greedy` | 180 | 49.4% [42.2%, 57.2%] | 61.7% | +1.66 [+1.29, +2.05] | +1.12 [+0.76, +1.49] | +1.30 to +2.26 | 0 |
| `selfplay4_tactical` | `cautious` | 180 | 36.1% [28.3%, 43.3%] | 47.2% | +0.85 [+0.52, +1.20] | +0.53 [+0.25, +0.84] | +0.62 to +1.16 | 0 |
| `selfplay4_tactical` | `reader` | 180 | 53.3% [46.1%, 60.6%] | 59.4% | +2.39 [+1.92, +2.89] | +2.63 [+2.23, +3.00] | +2.07 to +2.72 | 0 |
| `selfplay4_tactical` | `rusher` | 180 | 96.1% [92.8%, 98.3%] | 99.4% | +7.11 [+6.69, +7.52] | +7.96 [+7.64, +8.24] | +6.72 to +7.53 | 0 |
| `selfplay4_tactical` | `leaver` | 180 | 83.9% [78.3%, 88.9%] | 87.8% | +4.48 [+3.99, +4.96] | +5.24 [+4.80, +5.71] | +4.21 to +4.75 | 0 |
| `selfplay4_balance_s7002` | `rule` | 180 | 55.0% [47.8%, 62.2%] | 61.1% | +3.07 [+2.53, +3.61] | +3.27 [+2.82, +3.74] | +2.18 to +4.12 | 0 |
| `selfplay4_balance_s7002` | `greedy` | 180 | 55.6% [48.3%, 62.8%] | 63.3% | +2.08 [+1.68, +2.46] | +1.62 [+1.27, +1.96] | +1.83 to +2.35 | 0 |
| `selfplay4_balance_s7002` | `cautious` | 180 | 41.1% [33.9%, 48.3%] | 53.9% | +1.35 [+0.99, +1.72] | +1.04 [+0.73, +1.37] | +1.02 to +1.61 | 0 |
| `selfplay4_balance_s7002` | `reader` | 180 | 50.6% [43.3%, 57.8%] | 56.1% | +2.51 [+1.98, +3.01] | +2.74 [+2.31, +3.13] | +1.98 to +3.45 | 0 |
| `selfplay4_balance_s7002` | `rusher` | 180 | 96.7% [93.9%, 98.9%] | 98.3% | +6.73 [+6.29, +7.18] | +7.69 [+7.37, +7.98] | +6.32 to +7.15 | 0 |
| `selfplay4_balance_s7002` | `leaver` | 180 | 77.8% [71.1%, 83.9%] | 82.2% | +4.23 [+3.69, +4.76] | +5.02 [+4.56, +5.49] | +3.70 to +5.13 | 0 |
| `selfplay4_balance_s7003` | `rule` | 180 | 59.4% [52.2%, 66.7%] | 63.3% | +3.53 [+2.95, +4.09] | +3.69 [+3.22, +4.18] | +2.76 to +3.95 | 0 |
| `selfplay4_balance_s7003` | `greedy` | 180 | 62.2% [55.0%, 69.4%] | 70.6% | +2.54 [+2.12, +2.94] | +1.97 [+1.57, +2.37] | +2.22 to +2.97 | 0 |
| `selfplay4_balance_s7003` | `cautious` | 180 | 45.0% [37.8%, 52.2%] | 54.4% | +1.58 [+1.23, +1.96] | +1.24 [+0.90, +1.57] | +1.17 to +1.81 | 0 |
| `selfplay4_balance_s7003` | `reader` | 180 | 63.9% [56.7%, 71.1%] | 68.3% | +3.47 [+2.97, +3.98] | +3.55 [+3.14, +3.95] | +3.26 to +3.65 | 0 |
| `selfplay4_balance_s7003` | `rusher` | 180 | 98.9% [97.2%, 100.0%] | 99.4% | +7.66 [+7.28, +8.01] | +8.36 [+8.08, +8.62] | +7.52 to +7.85 | 0 |
| `selfplay4_balance_s7003` | `leaver` | 180 | 87.8% [82.8%, 92.2%] | 89.4% | +5.08 [+4.53, +5.56] | +5.77 [+5.32, +6.22] | +4.83 to +5.41 | 0 |

### Equal-weight opponent portfolio

All six opponent styles received the same number of attempted games, and no neural benchmark game aborted, so this panel is equally weighted by completed games. It describes the selected bot set, not a universal expected win rate.

| target policy | completed N | win rate [95% CI] | top-score rate | mean margin [95% CI] | mean score [95% CI] | seed margin range |
|:--|--:|:--|--:|:--|:--|:--|
| `selfplay4` | 1080 | 57.1% [54.2%, 60.1%] | 63.6% | +2.82 [+2.60, +3.03] | +3.09 [+2.89, +3.30] | +2.56 to +3.11 |
| `selfplay4_tactical` | 1080 | 62.4% [59.7%, 65.4%] | 68.9% | +3.20 [+2.97, +3.41] | +3.39 [+3.17, +3.60] | +3.04 to +3.41 |
| `selfplay4_balance_s7002` | 1080 | 62.8% [59.9%, 65.6%] | 69.2% | +3.33 [+3.11, +3.56] | +3.56 [+3.36, +3.77] | +3.00 to +3.87 |
| `selfplay4_balance_s7003` | 1080 | 69.5% [66.8%, 72.2%] | 74.3% | +3.98 [+3.75, +4.20] | +4.10 [+3.90, +4.33] | +3.66 to +4.20 |

### Paired margin differences vs shipped `selfplay4`

Positive values mean the named checkpoint had a higher score margin than `selfplay4` on the same evaluation seed and opponent-style cells. The intervals bootstrap paired game-level differences across all 6 styles and 3 seed blocks. These compare particular checkpoints; they are not estimates of training-algorithm effects.

| checkpoint | paired games | margin difference [95% CI] |
|:--|--:|:--|
| `selfplay4_tactical` | 1080 | +0.38 [+0.16, +0.60] |
| `selfplay4_balance_s7002` | 1080 | +0.51 [+0.35, +0.68] |
| `selfplay4_balance_s7003` | 1080 | +1.16 [+0.95, +1.38] |

### Scripted reference policies vs `rule`

These controls make the tested opponent styles easier to interpret; each had 180 attempted games, with engine-aborted games excluded from estimates.

| target policy | completed N | win rate [95% CI] | mean margin [95% CI] | seed margin range | aborted |
|:--|--:|:--|:--|:--|--:|
| `rule` | 178 | 18.0% [12.4%, 23.6%] | -0.04 [-0.36, +0.29] | -0.71 to +0.38 | 2 |
| `rusher` | 151 | 12.6% [7.3%, 17.9%] | -0.74 [-1.06, -0.41] | -0.83 to -0.63 | 29 |
| `leaver` | 177 | 52.5% [45.2%, 59.3%] | +2.37 [+1.84, +2.90] | +1.84 to +2.88 | 3 |

At this player count, **34** scripted-control games aborted. Across the full study, the abort reasons were `step_limit` 51. No neural benchmark or mirror games aborted. The `rusher` control has the highest abort count here; high-abort control estimates are limited diagnostics, not reliable policy estimates.

### Identical-policy mirror matches (luck/seat-spread proxy)

Every seat uses the same checkpoint. The mean within-game score range (highest minus lowest score) captures stochastic outcome and seat effects together; it is not a pure decomposition of luck.

| checkpoint | completed N | mean score range [95% CI] | games with tied top score [95% CI] | mean score by seat |
|:--|--:|:--|:--|:--|
| `selfplay4` | 180 | 4.39 [4.16, 4.63] | 22.8% [16.7%, 28.9%] | +1.07, +1.03, +1.01, +0.92, +0.99 |
| `selfplay4_tactical` | 180 | 4.96 [4.73, 5.18] | 13.3% [8.9%, 18.3%] | +1.15, +1.21, +1.38, +1.27, +1.04 |
| `selfplay4_balance_s7002` | 180 | 4.76 [4.51, 5.03] | 23.9% [17.8%, 30.6%] | +1.46, +1.36, +1.13, +1.32, +1.43 |
| `selfplay4_balance_s7003` | 180 | 4.43 [4.24, 4.62] | 22.2% [16.1%, 28.3%] | +1.63, +1.86, +1.78, +1.73, +1.28 |

## Interpretation and limits

- The primary evidence is the **mean-margin pattern across opponent styles and player counts**, with the seed-block range alongside each pooled interval. A single high win rate against one script is not treated as a balance verdict.
- Mirror ranges show that identical policies can finish at different scores even in the same game; compare these descriptive spreads with policy margins before over-reading small differences.
- The two new continuation checkpoints share the same starting network and are only two training seeds. The tactical checkpoint is a related policy, not an independent replicate. More independent training seeds and held-out human play would be needed for stronger claims.
- These bots are limited, hand-designed or self-play policies. This report **does not establish balance among people**, and the selected opponent mixture is not a model of a human population.
- The zero-gem solo-escape bonus remains enabled throughout; no counterfactual no-bonus rules are included. Blast-conditioned escape analysis is out of scope.

## Reproduction

From the repository root, with dependencies installed:

```sh
python current_rules_balance_study.py
```

The command writes `reports/current_rules_balance_study.md` and the machine-readable `reports/current_rules_balance_study.json`. Options `--games-per-seed`, `--seeds`, `--players`, `--output`, and `--json` can change the run.
