# Dragon-signal response: forced versus confidence-weighted

Checkpoint: `checkpoints/selfplay4_pushaware.npz`. Full games use one focal neural seat against three fixed opponents; focal seats rotate. Evaluation seeds are paired between arms. The confidence calibration uses a disjoint set of forced-policy four-neural games.

## Confidence calibration

A Beta(2, 8) prior (20% dragon, the base rate within a ten-card backside) smooths the raw truth audit. The soft arm uses the resulting posterior mean for each signal source.

| Signal source | Unique cards | Dragons | Raw precision | Confidence used | Signal decisions |
| :-- | --: | --: | --: | --: | --: |
| Owner walked off own-column card | 8 | 1 | 12.5% | 16.7% | 10 |
| Owner in pit cage, low outside hazard | 2756 | 2496 | 90.6% | 90.3% | 3734 |

## Paired full-game results

Soft response mixes the normal policy distribution with the same distribution conditioned on not passing. The mixture weight is the calibrated dragon confidence; therefore confidence 1 exactly matches the forced dodge, while confidence 0 leaves the policy unchanged.

### Three forced-policy neural controls

| Metric | Forced dodge | Soft response | Soft − forced (paired 95% CI) |
| :-- | --: | --: | --: |
| Win rate | 26.5% | 26.2% | -0.2 [-1.1, 0.6] pp |
| Top-score rate | 31.2% | 31.5% | +0.2 [-0.6, 1.1] pp |
| Mean score | 1.800 | 1.800 | +0.000 [-0.039, 0.039] |
| Mean rank (lower is better) | 2.270 | 2.265 | -0.005 [-0.017, 0.007] |
| Mean margin | 0.013 | 0.021 | +0.008 [-0.036, 0.051] |

Forced signal response: 0/624 signalled decisions passed (0.0%); 0 unfinished games.

Soft signal response: 9/624 signalled decisions passed (1.4%); 0 unfinished games.
### Three rule agents

| Metric | Forced dodge | Soft response | Soft − forced (paired 95% CI) |
| :-- | --: | --: | --: |
| Win rate | 66.5% | 66.2% | -0.2 [-1.1, 0.6] pp |
| Top-score rate | 71.2% | 71.2% | +0.0 [-0.7, 0.7] pp |
| Mean score | 4.235 | 4.237 | +0.003 [-0.008, 0.013] |
| Mean rank (lower is better) | 1.470 | 1.470 | +0.000 [-0.007, 0.007] |
| Mean margin | 3.877 | 3.883 | +0.007 [-0.014, 0.027] |

Forced signal response: 0/419 signalled decisions passed (0.0%); 0 unfinished games.

Soft signal response: 3/416 signalled decisions passed (0.7%); 0 unfinished games.

## Interpretation and limitations

Across these opponent fields the soft arm passed on 12 of 1040 signalled decisions; the forced arm passed on none. The paired full-game intervals include zero for every reported outcome, so this run finds no measurable advantage for either response. That is not proof of equivalence: the treatment changed very few actions.

The owner-in-cage cue calibrated to 90.6% dragon precision over 2756 unique cards in forced self-play. The walk-off cue had only 8 unique cards (1/8 were dragons), so its posterior confidence is prior-dominated and too uncertain to justify a strategic conclusion. The hard rule remains the default; the probe does not change production behavior or engine legality.

The reported intervals are paired normal 95% intervals (mean difference ±1.96 standard errors), bounded to this checkpoint, seed range, and opponent field.
