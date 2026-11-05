# Last-haul, score-conditioned tactical training

## Purpose

Teach risk choices in the final haul, especially a trailing player's blast
placement and subsequent keep/cancel decision. No rules or action masks change.
The default net is not replaced automatically. The candidate is
`checkpoints/selfplay4_last_haul.npz` (`neural:selfplay4_last_haul` in the GUI).

## Curriculum

`train_last_haul.py` generates four-seat last-haul scenarios with:

- A unique leading player on 3, 6, 9 or 12 total points; other players trail
  by 1, 3, 5 or 8 points, bounded below at zero.
- Starting retained dynamite counts chosen from 0, 1, 2, 4 and 6 independently
  per player, with varied backsides. Retained dynamite is additional to the
  fresh haul deck, following the engine's next-haul construction.
- Both round 0 and round 3 positions. Round 3 means **three execution rounds
  have already been played**, as in the zero-based viewer counter. Safe source
  moves are used to reach these boards with all players still active; these
  are intentionally survival-conditioned scenarios, not a frequency sample of
  ordinary net play. Safe cards are actually removed and collected, and rows
  are actually dealt; merely changing the round number would be invalid.
- A leading actor in approximately a quarter of generated scenarios and a
  trailing actor otherwise, independent of early/late board selection.
- Initial choices plus follow-up decisions after a policy-legal blast placement
  and the other players' sampled responses. Some phases close before another
  turn, so not every source supplies a follow-up.

Starting scores and retained hands are explicit synthetic conditions, not
claimed replays of two earlier hauls. Source games have disjoint training and
held-out seed ranges. Seed 7 is not used to select training positions or tune
the model.

## Branch teacher and objective

The candidates include current likely choices, legal shelter/pass, extreme
blast placements and their ordinary twins, plus a reachable leader attack.
Branch outcomes use final-score win share, splitting tied tops equally. There
is no reward for risk, announcing, completing a blast, gems, or score margin.

Unlike the earlier tactical pilot, the acting seat's continuation teacher
**passes to retain an announced blast whenever that pass is policy-allowed**.
Thus a blast placement is evaluated as part of a completion plan, rather than
inheriting the frozen net's tendency to cancel it immediately. Root cancellation
choices are still compared explicitly. Opponents remain ordinary frozen nets.
This is a deliberately restricted plan teacher, not an optimal search: it may
miss useful later cancellations. Crucially, the saved net and evaluation do
**not** receive the teacher's pass override. They must learn their own choices.

Hidden board fronts, newly collected opponent loot and the deck are resampled
as in the existing tactical trainer. Known retained start-of-haul dynamite is
kept fixed, not accidentally turned into gems during sampling. Own information,
visible backsides and card inventory stay fixed. This is observation-consistent
sampling, not a full-history posterior over opponents' earlier choices.

Teacher targets use a win-share softmax at temperature 0.1 mixed 40/60 with the
unchanged policy prior. Flat, indistinguishable outcome labels retain the prior
instead of teaching an arbitrary uniform policy. Only policy weights are fit;
value weights stay unchanged. Existing action masks still limit what can be
learned.

## Commands

```sh
python train_last_haul.py --positions 96 --held-out 48 --samples 48 --epochs 30
python mine_gui.py --opponents 3 --policy neural:selfplay4_last_haul --all-seats
```

Training scenarios start at seed 2300000; held-out scenarios at 2400000.
Branch continuations use 48 hidden worlds per position and paired policy seeds
across choices. The report records source scores, hands, board rounds and
pending-blast status. It also compares the two ordinary policies from held-out
positions, eight newly sampled continuations each, with no teacher override.
Win uncertainty is clustered by source scenario rather than pretending all
repeated continuations are independent. That cluster error is approximate when
source scenarios contribute different numbers of positions.

## Results

Completed: **96 training positions / 13,824 branch continuations**, plus
**48 disjoint held-out positions / 7,632 branch continuations**, 48 sampled
worlds per position and 30 fitting epochs. Training contains 54 round-3
positions and 40 pending-blast positions; held-out contains 25 round-3 positions
and 20 pending-blast positions. Labels, source conditions and additional
diagnostics are preserved in the tracked checkpoint history.

| held-out behaviour (same contexts) | unchanged | last-haul candidate |
| :-- | --: | --: |
| trailing actor, initial blast probability (21 positions) | 45.54% | 48.16% |
| trailing actor, pending blast: pass/keep probability (14) | **0.14%** | **8.72%** |
| round-3 initial blast probability (15) | 45.29% | 47.82% |
| leading actor, initial blast probability (7) | 58.39% | 62.85% |

Those groups contain different boards and hands, so the leading/trailing rows
alone do not prove score sensitivity. A separate matched-board diagnostic uses
20 blast-capable held-out initial positions, fixes every rival at 3 points and
changes only the actor's score to 0 (trailing) or 6 (leading):

| initial blast probability, matched boards | trailing | leading |
| :-- | --: | --: |
| unchanged | 68.36% | 68.26% |
| last-haul candidate | 73.62% | 71.68% |

This is a modest increase in score sensitivity, not proof that every resulting
blast is sensible. It measures probabilities, not final wins.

### Unforced continuation outcomes

Each held-out position was played eight more times in newly sampled worlds,
**384 continuations per policy**, using seed range 2800000 onwards (separate
from label sampling), with one evaluated seat against three
unchanged nets. No teacher override is used. All continuations finished.

| ordinary-policy evaluation | unchanged | last-haul candidate |
| :-- | --: | --: |
| outright wins, all positions | 35.94% | 38.54% |
| outright wins, trailing actors (280 continuations) | 22.50% | 24.64% |
| outright wins, round-3 positions (200) | 26.00% | 28.50% |
| mean final score | 5.193 | 5.250 |
| announcements after the fork | 310 | 316 |
| cancellations after the fork | 377 | 373 |
| hauls with own blast actually fired | 81 | 96 |

Cancellations can exceed announcements because some source positions already
contain a pending blast. Do not turn these counts into a cancellation percentage.
The paired win change overall is **+2.60 percentage points, give or take 1.08**
(approximate source-cluster standard error). Trailing players and the round-3
group both improve in this sample. This is a promising result on the targeted
scenario distribution, not evidence of a general-play gain; the held-out set
contains only 28 source scenarios.

The conditional held-out teacher estimate rose 43.38% -> 44.03%, while candidate
probability coverage fell 97.81% -> 96.69%. This estimate assumes the teacher's
blast-completion continuation and is not the ordinary policy's actual win rate.

A general-play check also ran 200 full three-haul games per policy from fresh
seed 2700000, seats rotated, against three unchanged nets. Outright wins were
23.5% -> 20.0% (paired −3.5 percentage points, give or take 3.12). Tied tops were
7.5% -> 11.5%, mean scores 2.185 -> 2.050, and fired-blast hauls 59 -> 64.
Every game finished. This does not justify replacing the default checkpoint.

### Seed 7 specifically

In fixed original haul-3 contexts, P2's pass/keep probability rose:

- Round 0: 0.12% -> 5.95%.
- Round 1: 0.11% -> 8.47%.
- Round 2 (the third planning phase): 0.12% -> 6.57%.

This reduces, but does not cure, its tendency to cancel. Initial announcements
also do not universally increase: P2's first blast probability falls 37.94% ->
34.22%, while its round-1 probability rises 68.27% -> 75.30%.

For an actual replay, play the original net normally through hauls 1 and 2,
then switch all four seats' weights only at the haul-3 boundary, keeping board,
hands, scores [3, 0, 0, 3] and agent RNG states. This avoids comparing two games
that diverged before the final haul:

```sh
python train_last_haul.py --diagnose-seed7
```

The original finishes [3, 0, 4, 3]; the candidate finishes [7, 0, -1, 3]. P2
still does not carry out a blast in this one trajectory. P3 fires in round 2
with the original net and round 1 with the candidate. Earlier risk is not
necessarily a realised win: the candidate's P3 loses here. This single replay
is a demonstration, not a statistical strength comparison. It is inspected
only after fitting; no tuning was done on seed 7.

**Decision:** keep `selfplay4.npz` unchanged and offer
`neural:selfplay4_last_haul` for inspection. The requested curriculum does change
blast retention and modestly changes score sensitivity, and improves wins on
the targeted held-out scenarios in this test. It has not solved P2's seed-7
behaviour or demonstrated a general winning-strength improvement. More blast announcements
alone would not be a valid success criterion.
