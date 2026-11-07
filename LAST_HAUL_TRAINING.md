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

### Displacement

Displacement was not directly rewarded. In the same 384 unforced continuations
per policy, the evaluated seat was **behind with a unique leader available to
displace** at many decisions, and it usually ignored the option:

| measured at trailing decisions | unchanged | last-haul candidate |
| :-- | --: | --: |
| leader-displacement choices available | 236 | 228 |
| leader displacements actually taken | 0 | **7** |
| taken rate | 0% | **3.1%** |

The candidate made 20 tracked displacement moves in total; 18 occurred while it
was behind and 7 displaced the unique leader. Those seven moves appeared in six
continuation branches and came from its round-3 source positions. Baseline did
not make the tracked seat displace anyone, although baseline's other seats made
one non-leader displacement. Across the six candidate leader-displacement
branches, the paired baseline and candidate continuation win rates were both
16.7%; these branches were selected by the candidate's action, so this does not
measure the causal value of pushing. The behaviour is therefore visibly changed,
but taking a leader displacement remains very uncommon.

Reproduce this audit with:

```sh
python probe_last_haul_displacements.py
```

### Two trailers under one policy

The audits above change one evaluated seat at a time. Two further checks put
the last-haul net in **both** of the two lowest-scoring seats at the start of the
final haul.

The isolated-boundary check uses `probe_two_trailing_boundary.py`. For each game,
the baseline plays through hauls 1 and 2; the branch is then cloned at the exact
haul-3 boundary, preserving cards, hands, scores and agent RNG states. Only the
two trailers' networks are swapped in the candidate arm. It ran 1,000 fresh game
seeds from 3100000; 974 had at least two players behind the top.

| both trailers, isolated boundary fork | baseline | last-haul |
| :-- | --: | --: |
| round-0 announcements by the pair | 1,189 | **1,255** |
| cancellations after the fork | 2,288 | **2,168** |
| successful blast resolutions | 222 | **278** |
| explicit pending decisions kept | 0.09% | **5.93%** |
| trailing displacements onto a leader | 0 | **56** |
| leader-push opportunities | 4,436 | 4,008 |
| leader pushes taken | 0% | **1.40%** |
| trailers displacing each other | 1 | **27** |
| mean final score per trailer | 1.006 | **1.049** |
| games with at least one trailer at the top | 247 (25.36%) | **278 (28.54%)** |
| outright win rate per trailer | 10.11% | 10.83% |

The paired score change is +0.043 per trailer, give or take 0.025. The paired
rate that at least one trailer reaches the top is +3.18 percentage points, give
or take 1.34; the per-trailer win change is +0.72 points, give or take 0.61.
Thus the pair visibly completes more blasts and finally takes some leader
pushes. The change is not large enough to call a general strength gain, and
these paired standard errors treat each game as one cluster.

A separate 500-game self-play check ran all four seats on one policy, without a
boundary fork. For the two final-haul trailers, the last-haul net had fewer
total announcements (1,034 -> 860), fewer cancellations (916 -> 806), more
explicit pending keeps (1.66% -> 8.21%), more fired-blast games (171 -> 188),
and 18 leader pushes from 1,835 opportunities versus none. Its mean score per
trailer was lower (0.942 -> 0.866), with essentially unchanged probability that
at least one trailer reached the top (27.05% -> 27.13%). Ordinary whole-game
behaviour is therefore changed, but the outcome result is mixed.

For seed 7 specifically, the isolated two-trailer branch is instructive: both P2
and P3 announce in round 0. The baseline branch ends [3, 0, 4, 3] with one
successful trailer blast. The candidate branch ends [7, 0, -1, 3], has no
successful trailer blast, and neither trailer displaces a leader. This exact
trajectory remains a failure case.

```sh
python probe_two_trailing_boundary.py --games 1000 --seed 3100000
python probe_two_trailing_players.py --games 500 --seed 2900000
```

### Three relentless pushers

`probe_always_pusher_match.py` puts the candidate in one rotating seat and puts
three scripted seats in the other positions. Every time one of those scripted
seats has a legal occupied card it can contest, it takes the displacement,
preferring the current highest total score. Their fallback behaviour is the
deterministic counter policy; they do not receive special knowledge.

1,000 fresh games per arm, four players, three hauls, seed 3300000 onwards:

| one net versus three always-pushers | baseline | last-haul |
| :-- | --: | --: |
| net push opportunities taken | 16 of 5,384 (0.30%) | **126 of 5,390 (2.34%)** |
| pushes taken by the pushers | 20,810 | 20,363 |
| successful pushes against the net | 8,251 | **7,933** |
| outright wins | **61.5%** | 59.9% |
| tied top | 8.7% | 8.9% |
| mean net score | **3.453** | 3.264 |
| mean net margin over rivals | **3.484** | 3.254 |
| mean net gems | 1.521 | **1.544** |

Candidate-vs-baseline win change is −1.6 percentage points, give or take 1.17;
margin change is −0.230 points, give or take 0.081. The last-haul net fights
back more often and suffers slightly fewer incoming pushes, but still takes only
2.34% of its own opportunities and scores below baseline here. It does not yet
have a robust answer to an all-push table.

```sh
python probe_always_pusher_match.py --games 1000 --seed 3300000
```

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
