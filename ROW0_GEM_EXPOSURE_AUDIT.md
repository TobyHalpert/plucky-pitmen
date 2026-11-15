# Exposed row-0 own-column Gems

Checkpoint: `checkpoints/selfplay4_pushaware.npz`. Each game has one focal neural seat against three fixed opponents; focal seat rotates. Default and guard arms share game seeds and neural RNG seeds. The guard masks the Gem whenever an active rival has movement depth greater than zero and another policy action remains. Engine legality is unchanged.

> **Historical implementation note (2026-11-10):** these paired figures evaluate the earlier guard, which could retain the row-0 Gem when other legal alternatives had already been removed by the known-Gem heuristic. The seed-7 regression showed that this happens when the last unplanned neural seat acts after deeper rivals have already planned. The current guard opens deeper foreign alternatives in that case, also opens foreign row-0 cells if no deeper row is reachable, prevents a no-score lorry fallback when a policy-approved dig remains, and is enabled by default for neural seats in the GUI controller. The results below are not a performance estimate of that updated default; rerun `probe_row0_gem_exposure.py` to measure it.

## Event definition and denominators

An **offer** is a neural decision at which its legal row-0 home-column cell is a visible Gem. A **choice** is the neural policy selecting that action. The primary ‘all active rivals on deeper cards’ condition means every rival still in the mine is on a card row greater than 0 at the moment of choice (it is false if no rival is active). ‘Any deeper rival’ instead means at least one active rival has movement depth greater than 0, including a rival in a pit cage at that depth. An **actual displacement** is a successful engine push of the focal seat off row 0 before that round’s reveal. **Gem resolved** means the row-0 Gem was actually collected during that execution; a dragon that ends the reveal first is not counted as resolved. **Pusher took Gem** means the same seat that displaced the focal seat collected that Gem in that reveal. Offers and choices are decision instances (the same Gem can be offered again if planning revisits it); the choice and offer denominators are shown separately.

The candidate is a policy-only mask over the existing neural checkpoint, not a retrain. Performance differences are paired by seed and focal seat; outcome intervals are paired normal 95% intervals. Event-rate intervals are Wilson 95% binomial intervals.

## Three forced-policy neural controls

### Default policy

| Condition at choice | Gem offers | Gem choices / offers | Displaced before reveal / choices | Gem resolved / choices | Displaced + resolved / choices | Same pusher took / displaced + resolved |
| :-- | --: | --: | --: | --: | --: | --: |
| All active rivals on deeper cards | 513 | 149/513 (29.0%) | 116/149 (77.9%; Wilson 95% CI 70.5%–83.8%) | 137/149 (91.9%) | 110/149 (73.8%; Wilson 95% CI 66.2%–80.2%) | 108/110 (98.2%; Wilson 95% CI 93.6%–99.5%) |
| At least one active rival has deeper movement depth (guard condition) | 1760 | 447/1760 (25.4%) | 268/447 (60.0%; Wilson 95% CI 55.3%–64.4%) | 410/447 (91.7%) | 257/447 (57.5%; Wilson 95% CI 52.9%–62.0%) | 254/257 (98.8%; Wilson 95% CI 96.6%–99.6%) |
| At least one unsettled deeper rival (existing exposure signal) | 912 | 108/912 (11.8%) | 99/108 (91.7%; Wilson 95% CI 84.9%–95.6%) | 98/108 (90.7%) | 90/108 (83.3%; Wilson 95% CI 75.2%–89.2%) | 88/90 (97.8%; Wilson 95% CI 92.3%–99.4%) |

All row-0 Gem offers: **1815**; policy chose the row-0 Gem **468** times; successful displacements before reveal: **268**; Gems actually collected at reveal: **431**; same pusher collected: **254**. Among this arm's choices with any deeper rival, the guard would block 384 and retain 63 because no alternative policy action remained. Unfinished games: 0/500.

Examples where the pusher got the Gem:

- seed 8200001, focal P2, haul 2 round 0, pusher P3, reveal occupant P3.
- seed 8200012, focal P1, haul 1 round 0, pusher P3, reveal occupant P3.
- seed 8200015, focal P4, haul 3 round 0, pusher P2, reveal occupant P2.
- seed 8200025, focal P2, haul 2 round 0, pusher P4, reveal occupant P4.
- seed 8200026, focal P3, haul 2 round 0, pusher P4, reveal occupant P4.

### Guard policy

| Condition at choice | Gem offers | Gem choices / offers | Displaced before reveal / choices | Gem resolved / choices | Displaced + resolved / choices | Same pusher took / displaced + resolved |
| :-- | --: | --: | --: | --: | --: | --: |
| All active rivals on deeper cards | 845 | 60/845 (7.1%) | 43/60 (71.7%; Wilson 95% CI 59.2%–81.5%) | 57/60 (95.0%) | 43/60 (71.7%; Wilson 95% CI 59.2%–81.5%) | 43/43 (100.0%; Wilson 95% CI 91.8%–100.0%) |
| At least one active rival has deeper movement depth (guard condition) | 3149 | 89/3149 (2.8%) | 57/89 (64.0%; Wilson 95% CI 53.7%–73.2%) | 82/89 (92.1%) | 57/89 (64.0%; Wilson 95% CI 53.7%–73.2%) | 57/57 (100.0%; Wilson 95% CI 93.7%–100.0%) |
| At least one unsettled deeper rival (existing exposure signal) | 1423 | 4/1423 (0.3%) | 4/4 (100.0%; Wilson 95% CI 51.0%–100.0%) | 4/4 (100.0%) | 4/4 (100.0%; Wilson 95% CI 51.0%–100.0%) | 4/4 (100.0%; Wilson 95% CI 51.0%–100.0%) |

All row-0 Gem offers: **3238**; policy chose the row-0 Gem **141** times; successful displacements before reveal: **57**; Gems actually collected at reveal: **134**; same pusher collected: **57**. Among this arm's choices with any deeper rival, the guard would block 0 and retain 89 because no alternative policy action remained. Unfinished games: 0/500.

Examples where the pusher got the Gem:

- seed 8200001, focal P2, haul 2 round 0, pusher P3, reveal occupant P3.
- seed 8200003, focal P4, haul 3 round 0, pusher P3, reveal occupant P3.
- seed 8200012, focal P1, haul 1 round 0, pusher P3, reveal occupant P3.
- seed 8200015, focal P4, haul 3 round 0, pusher P2, reveal occupant P2.
- seed 8200025, focal P2, haul 2 round 0, pusher P4, reveal occupant P4.

### Paired target-event counts

Per-game counts for choices made while every active rival was on a deeper card; the same seeds and focal seats are paired.

| Measure | Default per game | Guard per game | Guard − default per game (paired 95% CI) |
| :-- | --: | --: | --: |
| Gem offers | 1.026 | 1.690 | +0.664 [+0.547, +0.781] |
| Gem choices | 0.298 | 0.120 | -0.178 [-0.222, -0.134] |
| Displaced before reveal | 0.232 | 0.086 | -0.146 [-0.186, -0.106] |
| Gem resolved | 0.274 | 0.114 | -0.160 [-0.203, -0.117] |
| Exact sequence: displaced + Gem resolved | 0.220 | 0.086 | -0.134 [-0.173, -0.095] |
| Same pusher took Gem | 0.216 | 0.086 | -0.130 [-0.169, -0.091] |

### Full-game outcomes

| Outcome | Default | Guard | Guard − default (paired 95% CI) |
| :-- | --: | --: | --: |
| Win rate | 22.0% | 21.4% | -0.6 [-3.9, 2.7] pp |
| Top-score rate | 26.2% | 25.8% | -0.4 [-3.8, 3.0] pp |
| Mean score | 1.616 | 1.572 | -0.044 [-0.217, 0.129]  |
| Mean rank (lower is better) | 2.334 | 2.322 | -0.012 [-0.097, 0.073]  |
| Mean score margin | -0.193 | -0.277 | -0.085 [-0.305, 0.136]  |

## Three rule agents

### Default policy

| Condition at choice | Gem offers | Gem choices / offers | Displaced before reveal / choices | Gem resolved / choices | Displaced + resolved / choices | Same pusher took / displaced + resolved |
| :-- | --: | --: | --: | --: | --: | --: |
| All active rivals on deeper cards | 587 | 374/587 (63.7%) | 374/374 (100.0%; Wilson 95% CI 99.0%–100.0%) | 370/374 (98.9%) | 370/374 (98.9%; Wilson 95% CI 97.3%–99.6%) | 369/370 (99.7%; Wilson 95% CI 98.5%–100.0%) |
| At least one active rival has deeper movement depth (guard condition) | 1292 | 492/1292 (38.1%) | 491/492 (99.8%; Wilson 95% CI 98.9%–100.0%) | 479/492 (97.4%) | 478/492 (97.2%; Wilson 95% CI 95.3%–98.3%) | 477/478 (99.8%; Wilson 95% CI 98.8%–100.0%) |
| At least one unsettled deeper rival (existing exposure signal) | 490 | 9/490 (1.8%) | 9/9 (100.0%; Wilson 95% CI 70.1%–100.0%) | 9/9 (100.0%) | 9/9 (100.0%; Wilson 95% CI 70.1%–100.0%) | 9/9 (100.0%; Wilson 95% CI 70.1%–100.0%) |

All row-0 Gem offers: **1294**; policy chose the row-0 Gem **493** times; successful displacements before reveal: **491**; Gems actually collected at reveal: **480**; same pusher collected: **477**. Among this arm's choices with any deeper rival, the guard would block 169 and retain 323 because no alternative policy action remained. Unfinished games: 0/500.

Examples where the pusher got the Gem:

- seed 8300000, focal P1, haul 1 round 0, pusher P2, reveal occupant P2.
- seed 8300000, focal P1, haul 3 round 0, pusher P2, reveal occupant P2.
- seed 8300001, focal P2, haul 1 round 0, pusher P3, reveal occupant P3.
- seed 8300001, focal P2, haul 3 round 0, pusher P3, reveal occupant P3.
- seed 8300004, focal P1, haul 1 round 0, pusher P2, reveal occupant P2.

### Guard policy

| Condition at choice | Gem offers | Gem choices / offers | Displaced before reveal / choices | Gem resolved / choices | Displaced + resolved / choices | Same pusher took / displaced + resolved |
| :-- | --: | --: | --: | --: | --: | --: |
| All active rivals on deeper cards | 745 | 327/745 (43.9%) | 327/327 (100.0%; Wilson 95% CI 98.8%–100.0%) | 326/327 (99.7%) | 326/327 (99.7%; Wilson 95% CI 98.3%–99.9%) | 325/326 (99.7%; Wilson 95% CI 98.3%–99.9%) |
| At least one active rival has deeper movement depth (guard condition) | 1639 | 387/1639 (23.6%) | 387/387 (100.0%; Wilson 95% CI 99.0%–100.0%) | 381/387 (98.4%) | 381/387 (98.4%; Wilson 95% CI 96.7%–99.3%) | 380/381 (99.7%; Wilson 95% CI 98.5%–100.0%) |
| At least one unsettled deeper rival (existing exposure signal) | 590 | 0/590 (0.0%) | 0/0 (n/a; Wilson 95% CI n/a) | 0/0 (n/a) | 0/0 (n/a; Wilson 95% CI n/a) | 0/0 (n/a; Wilson 95% CI n/a) |

All row-0 Gem offers: **1641**; policy chose the row-0 Gem **388** times; successful displacements before reveal: **387**; Gems actually collected at reveal: **382**; same pusher collected: **380**. Among this arm's choices with any deeper rival, the guard would block 0 and retain 387 because no alternative policy action remained. Unfinished games: 0/500.

Examples where the pusher got the Gem:

- seed 8300000, focal P1, haul 2 round 0, pusher P2, reveal occupant P2.
- seed 8300000, focal P1, haul 3 round 0, pusher P2, reveal occupant P2.
- seed 8300001, focal P2, haul 1 round 0, pusher P3, reveal occupant P3.
- seed 8300001, focal P2, haul 3 round 0, pusher P3, reveal occupant P3.
- seed 8300004, focal P1, haul 1 round 0, pusher P2, reveal occupant P2.

### Paired target-event counts

Per-game counts for choices made while every active rival was on a deeper card; the same seeds and focal seats are paired.

| Measure | Default per game | Guard per game | Guard − default per game (paired 95% CI) |
| :-- | --: | --: | --: |
| Gem offers | 1.174 | 1.490 | +0.316 [+0.252, +0.380] |
| Gem choices | 0.748 | 0.654 | -0.094 [-0.133, -0.055] |
| Displaced before reveal | 0.748 | 0.654 | -0.094 [-0.133, -0.055] |
| Gem resolved | 0.740 | 0.652 | -0.088 [-0.127, -0.049] |
| Exact sequence: displaced + Gem resolved | 0.740 | 0.652 | -0.088 [-0.127, -0.049] |
| Same pusher took Gem | 0.738 | 0.650 | -0.088 [-0.127, -0.049] |

### Full-game outcomes

| Outcome | Default | Guard | Guard − default (paired 95% CI) |
| :-- | --: | --: | --: |
| Win rate | 64.8% | 64.4% | -0.4 [-2.8, 2.0] pp |
| Top-score rate | 71.6% | 71.0% | -0.6 [-2.9, 1.7] pp |
| Mean score | 4.424 | 4.354 | -0.070 [-0.205, 0.065]  |
| Mean rank (lower is better) | 1.440 | 1.480 | +0.040 [-0.004, 0.084]  |
| Mean score margin | 4.083 | 3.985 | -0.097 [-0.274, 0.080]  |

## Three always-pushers

### Default policy

| Condition at choice | Gem offers | Gem choices / offers | Displaced before reveal / choices | Gem resolved / choices | Displaced + resolved / choices | Same pusher took / displaced + resolved |
| :-- | --: | --: | --: | --: | --: | --: |
| All active rivals on deeper cards | 409 | 79/409 (19.3%) | 79/79 (100.0%; Wilson 95% CI 95.4%–100.0%) | 78/79 (98.7%) | 78/79 (98.7%; Wilson 95% CI 93.2%–99.8%) | 78/78 (100.0%; Wilson 95% CI 95.3%–100.0%) |
| At least one active rival has deeper movement depth (guard condition) | 1687 | 417/1687 (24.7%) | 411/417 (98.6%; Wilson 95% CI 96.9%–99.3%) | 414/417 (99.3%) | 408/417 (97.8%; Wilson 95% CI 95.9%–98.9%) | 408/408 (100.0%; Wilson 95% CI 99.1%–100.0%) |
| At least one unsettled deeper rival (existing exposure signal) | 1081 | 151/1081 (14.0%) | 151/151 (100.0%; Wilson 95% CI 97.5%–100.0%) | 150/151 (99.3%) | 150/151 (99.3%; Wilson 95% CI 96.3%–99.9%) | 150/150 (100.0%; Wilson 95% CI 97.5%–100.0%) |

All row-0 Gem offers: **1797**; policy chose the row-0 Gem **475** times; successful displacements before reveal: **411**; Gems actually collected at reveal: **471**; same pusher collected: **408**. Among this arm's choices with any deeper rival, the guard would block 240 and retain 177 because no alternative policy action remained. Unfinished games: 0/500.

Examples where the pusher got the Gem:

- seed 8400006, focal P3, haul 2 round 0, pusher P4, reveal occupant P4.
- seed 8400008, focal P1, haul 1 round 0, pusher P2, reveal occupant P2.
- seed 8400010, focal P3, haul 1 round 0, pusher P4, reveal occupant P4.
- seed 8400014, focal P3, haul 2 round 0, pusher P4, reveal occupant P4.
- seed 8400017, focal P2, haul 1 round 0, pusher P3, reveal occupant P3.

### Guard policy

| Condition at choice | Gem offers | Gem choices / offers | Displaced before reveal / choices | Gem resolved / choices | Displaced + resolved / choices | Same pusher took / displaced + resolved |
| :-- | --: | --: | --: | --: | --: | --: |
| All active rivals on deeper cards | 545 | 36/545 (6.6%) | 36/36 (100.0%; Wilson 95% CI 90.4%–100.0%) | 35/36 (97.2%) | 35/36 (97.2%; Wilson 95% CI 85.8%–99.5%) | 35/35 (100.0%; Wilson 95% CI 90.1%–100.0%) |
| At least one active rival has deeper movement depth (guard condition) | 2436 | 260/2436 (10.7%) | 256/260 (98.5%; Wilson 95% CI 96.1%–99.4%) | 254/260 (97.7%) | 250/260 (96.2%; Wilson 95% CI 93.1%–97.9%) | 250/250 (100.0%; Wilson 95% CI 98.5%–100.0%) |
| At least one unsettled deeper rival (existing exposure signal) | 1429 | 22/1429 (1.5%) | 22/22 (100.0%; Wilson 95% CI 85.1%–100.0%) | 22/22 (100.0%) | 22/22 (100.0%; Wilson 95% CI 85.1%–100.0%) | 22/22 (100.0%; Wilson 95% CI 85.1%–100.0%) |

All row-0 Gem offers: **2605**; policy chose the row-0 Gem **370** times; successful displacements before reveal: **256**; Gems actually collected at reveal: **361**; same pusher collected: **250**. Among this arm's choices with any deeper rival, the guard would block 0 and retain 260 because no alternative policy action remained. Unfinished games: 0/500.

Examples where the pusher got the Gem:

- seed 8400010, focal P3, haul 1 round 0, pusher P4, reveal occupant P4.
- seed 8400032, focal P1, haul 3 round 0, pusher P2, reveal occupant P2.
- seed 8400065, focal P2, haul 1 round 0, pusher P3, reveal occupant P3.
- seed 8400070, focal P3, haul 2 round 1, pusher P4, reveal occupant P4.
- seed 8400094, focal P3, haul 2 round 1, pusher P4, reveal occupant P4.

### Paired target-event counts

Per-game counts for choices made while every active rival was on a deeper card; the same seeds and focal seats are paired.

| Measure | Default per game | Guard per game | Guard − default per game (paired 95% CI) |
| :-- | --: | --: | --: |
| Gem offers | 0.818 | 1.090 | +0.272 [+0.207, +0.337] |
| Gem choices | 0.158 | 0.072 | -0.086 [-0.120, -0.052] |
| Displaced before reveal | 0.158 | 0.072 | -0.086 [-0.120, -0.052] |
| Gem resolved | 0.156 | 0.070 | -0.086 [-0.120, -0.052] |
| Exact sequence: displaced + Gem resolved | 0.156 | 0.070 | -0.086 [-0.120, -0.052] |
| Same pusher took Gem | 0.156 | 0.070 | -0.086 [-0.120, -0.052] |

### Full-game outcomes

| Outcome | Default | Guard | Guard − default (paired 95% CI) |
| :-- | --: | --: | --: |
| Win rate | 45.6% | 51.4% | +5.8 [2.3, 9.3] pp |
| Top-score rate | 52.2% | 56.0% | +3.8 [0.3, 7.3] pp |
| Mean score | 2.402 | 2.708 | +0.306 [0.125, 0.487]  |
| Mean rank (lower is better) | 1.708 | 1.678 | -0.030 [-0.088, 0.028]  |
| Mean score margin | 1.881 | 2.301 | +0.419 [0.194, 0.644]  |

**Interpretation:** the mask prevents the neural policy from entering a row-0 home Gem while any active deeper rival remains, unless that is the only remaining policy action. This is an experimental policy wrapper; it changes neither game legality nor the trained weights.
