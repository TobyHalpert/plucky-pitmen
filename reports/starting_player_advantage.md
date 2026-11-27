# Starting-player advantage study

Generated **2026-11-10**.

## Method

For each policy and player count, 180 fixed-deal seeds (47000000–47000179) were tested from every possible initial starting seat. Each rotation reused the same shuffled deck seed and each seat's agent RNG seed, then forced the initial starter and planning seat; later haul order followed normal game rules. All seats used the same policy (mirror games).
The primary estimate is starter official win share minus the mean win share of nonstarting seats, averaged over every starter rotation within each fixed-deal seed block. Official ties split one unit among listed winners; score margin is diagnostic only.
The pooled panel weights the 8 policies equally within each seed block. The 95% percentile interval uses 4,000 bootstrap resamples of seed blocks, not individual games.
Rules: [3, 4, 5] players, `n-1` passes, solo escape bonus on, blind blast choice on; neural action temperature 0.70.

## Pooled starter effect

| Players | Attempted games | Aborted | Policies in panel | Common complete seed blocks | Starter win share | Mean nonstarter seat win share | Paired starter Δ [95% CI] | Score-margin Δ (points) |
|--:|--:|--:|--:|--:|--:|--:|:--|--:|
| 3 | 4320 | 0 | 8 | 180 | +32.71% [+30.95%, +34.49%] | +33.65% [+32.73%, +34.54%] | -0.94% [-3.58%, +1.77%] | +0.02 |
| 4 | 5760 | 12 | 8 | 174 | +25.31% [+23.96%, +26.63%] | +24.90% [+24.47%, +25.34%] | +0.41% [-1.32%, +2.20%] | +0.00 |
| 5 | 7200 | 12 | 8 | 175 | +18.99% [+17.91%, +20.02%] | +20.25% [+19.98%, +20.51%] | -1.27% [-2.62%, +0.04%] | -0.02 |

## By policy

| Players | Policy | Complete seed blocks | Aborted games | Starter win share | Nonstarter seat win share | Paired Δ [95% CI] | Starter win rate | Score-margin Δ |
|--:|:--|--:|--:|--:|--:|--:|:--|--:|
| 3 | rule | 180 | 0 | +32.41% [+28.89%, +36.11%] | +33.80% [+32.04%, +35.65%] | -1.39% [-6.94%, +3.89%] | 32.4% | -0.07 |
| 3 | greedy | 180 | 0 | +32.22% [+28.33%, +36.30%] | +33.89% [+31.85%, +35.83%] | -1.67% [-7.50%, +4.44%] | 32.2% | +0.09 |
| 3 | cautious | 180 | 0 | +36.48% [+32.59%, +40.37%] | +31.76% [+29.81%, +33.70%] | +4.72% [-1.39%, +10.56%] | 36.5% | +0.23 |
| 3 | reader | 180 | 0 | +32.41% [+28.70%, +36.11%] | +33.80% [+31.94%, +35.56%] | -1.39% [-6.94%, +4.17%] | 32.4% | -0.07 |
| 3 | rusher | 180 | 0 | +27.96% [+21.67%, +34.07%] | +36.02% [+32.78%, +39.17%] | -8.06% [-16.94%, +1.11%] | 28.0% | +0.00 |
| 3 | leaver | 180 | 0 | +33.70% [+29.26%, +38.33%] | +33.15% [+30.83%, +35.37%] | +0.56% [-6.11%, +7.22%] | 33.7% | -0.03 |
| 3 | selfplay4 | 180 | 0 | +30.19% [+26.30%, +34.07%] | +34.91% [+32.96%, +36.76%] | -4.72% [-10.28%, +0.83%] | 30.2% | -0.25 |
| 3 | selfplay4_winfirst_s8111 | 180 | 0 | +36.30% [+32.41%, +40.37%] | +31.85% [+29.72%, +33.89%] | +4.44% [-1.67%, +10.56%] | 36.3% | +0.23 |
| 4 | rule | 174 | 6 | +24.57% [+21.41%, +27.87%] | +25.14% [+24.04%, +26.25%] | -0.57% [-5.17%, +3.83%] | 24.6% | +0.05 |
| 4 | greedy | 180 | 0 | +25.83% [+22.50%, +29.17%] | +24.72% [+23.66%, +25.79%] | +1.11% [-3.33%, +5.56%] | 25.8% | +0.01 |
| 4 | cautious | 180 | 0 | +27.22% [+23.89%, +30.69%] | +24.26% [+23.10%, +25.37%] | +2.96% [-1.67%, +7.59%] | 27.2% | +0.09 |
| 4 | reader | 174 | 6 | +24.57% [+21.26%, +28.02%] | +25.14% [+23.95%, +26.29%] | -0.57% [-4.98%, +4.02%] | 24.6% | +0.05 |
| 4 | rusher | 180 | 0 | +24.58% [+18.89%, +30.83%] | +25.14% [+23.19%, +27.04%] | -0.56% [-8.15%, +7.59%] | 24.6% | +0.00 |
| 4 | leaver | 180 | 0 | +24.72% [+21.39%, +28.06%] | +25.09% [+23.89%, +26.20%] | -0.37% [-4.63%, +4.26%] | 24.7% | -0.04 |
| 4 | selfplay4 | 180 | 0 | +24.44% [+21.25%, +27.92%] | +25.19% [+24.12%, +26.25%] | -0.74% [-5.19%, +3.70%] | 24.4% | -0.18 |
| 4 | selfplay4_winfirst_s8111 | 180 | 0 | +25.69% [+22.50%, +29.03%] | +24.77% [+23.70%, +25.79%] | +0.93% [-3.15%, +5.19%] | 25.7% | +0.05 |
| 5 | rule | 175 | 6 | +19.66% [+17.37%, +22.06%] | +20.09% [+19.51%, +20.66%] | -0.43% [-3.29%, +2.43%] | 19.7% | -0.07 |
| 5 | greedy | 180 | 0 | +18.22% [+15.67%, +20.83%] | +20.44% [+19.81%, +21.08%] | -2.22% [-5.42%, +0.97%] | 18.3% | -0.04 |
| 5 | cautious | 180 | 0 | +19.22% [+16.56%, +22.00%] | +20.19% [+19.50%, +20.89%] | -0.97% [-4.44%, +2.36%] | 19.2% | -0.01 |
| 5 | reader | 175 | 6 | +19.66% [+17.37%, +22.06%] | +20.09% [+19.51%, +20.66%] | -0.43% [-3.43%, +2.43%] | 19.7% | -0.07 |
| 5 | rusher | 180 | 0 | +14.67% [+10.33%, +19.44%] | +21.33% [+20.19%, +22.47%] | -6.67% [-12.08%, -0.69%] | 14.7% | +0.00 |
| 5 | leaver | 180 | 0 | +20.56% [+17.56%, +23.67%] | +19.86% [+19.08%, +20.61%] | +0.69% [-3.33%, +4.44%] | 20.6% | +0.02 |
| 5 | selfplay4 | 180 | 0 | +18.11% [+15.56%, +20.78%] | +20.47% [+19.81%, +21.11%] | -2.36% [-5.56%, +0.97%] | 18.1% | -0.08 |
| 5 | selfplay4_winfirst_s8111 | 180 | 0 | +21.67% [+19.11%, +24.33%] | +19.58% [+18.92%, +20.28%] | +2.08% [-0.98%, +5.42%] | 21.7% | +0.06 |

## Interpretation and limitations

A positive paired Δ means the seat chosen to start receives more official win share than the average nonstarting seat. The pooled estimates are across equally weighted mirror-policy profiles, not a single universal policy.
All pooled 95% intervals include zero, so this study does not establish a general starter advantage for 3–5 players under the tested policies and rules.
Exploratory subgroup: five-player rusher mirror games show -6.67% [-12.08%, -0.69%] starter Δ. This is one of 24 policy/player comparisons and is not adjusted for multiple testing.
The study uses same-policy mirror games; it does not estimate starting-seat effects in mixed-policy matches or every possible player strategy. The confidence intervals reflect deal-to-deal sampling variability, not uncertainty over which policies were included.
Aborted games are reported above. A seed block with any aborted starter rotation is excluded from that policy's summary; the pooled panel uses only seed blocks complete for every included policy. The JSON records complete seed IDs, seed-block means, and abort counts.
