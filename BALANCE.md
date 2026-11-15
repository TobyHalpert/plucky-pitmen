# Plucky Pitmen balance report

**Updated 2026-11-10.** This is a current, reproducible snapshot—not a claim
that every policy or player count has been exhaustively balanced. Earlier
results are labelled historical where retained; many detailed, stale experiment
notes have been removed.

## Current read

- In the fresh four-player comparison, `selfplay4` has a +2.10-point margin
  against three `leaver` agents; `leaver` has a −1.37-point margin against three
  `selfplay4` agents. Both 95% intervals exclude zero.
- Individual games remain noisy: four identical `selfplay4` seats have a mean
  within-game score spread of **4.875 points**, larger than the measured
  `selfplay4`-versus-`leaver` margin of **2.10 points**. Compare repeated-game
  margins and intervals, not one match's winner.
- The fresh blast intervention separates announcements from completed draws.
  Forcing the focal agent to pass on pending blasts lowers its margin versus
  natural cancellation. Forcing additional announcements on eligible card moves
  did not produce a reliable score gain over that commit-natural arm.
- Both measured policies benefit substantially when rivals fail to bank loot.
  The current run does not evaluate `rusher`, five-player games, or human
  opponents, so it is not a global strategy ranking.

## 1. Empty-handed exits and the withdrawn rusher claim

An earlier version of this report treated `rusher`'s large margin as evidence
that leaving on turn one was dominant. That conclusion was withdrawn: the
measurement depended on a lorry-stand-off/haul-scoring defect that voided other
players' loot while the early leaver still scored. The old `+8.37`
`leaver`-versus-`rusher` result is **not** a current balance measurement. The
fresh evaluation below does not include `rusher`.

Empty-handed escapes and the payoff for rivals who stay in the mine are
opponent-dependent. The current report's per-haul split is in §3; the old
rusher-vs-camper figures should not be used to infer a dominant strategy.

## 2. Fresh four-player evaluation

All results below were generated with `evaluate_balance.py` on 2026-11-10.
Checkpoint `checkpoints/selfplay4.npz` has SHA-256
`009a051ed366c349458bff402815ce04adaa85626dfacc2bd16aad3009617172`.

**Settings:** four players; three hauls per game; temperature 0.7; `n-1`
consecutive passes; solo-escape bonus enabled; printed face-down blast choice;
`rule` benchmark; zero abandoned games. There were 200 games per leaderboard
entry, 200 games per pairwise direction, and 400 mirror games. Leaderboard CIs
are 95% bootstrap intervals over games (2,000 resamples).

### Against three `rule` agents

Margin is the policy's final score minus the mean score of the other three seats.

| policy | games | wins | top score | mean score [95% CI] | margin [95% CI] |
| :-- | --: | --: | --: | :-- | :-- |
| `selfplay4` | 200 | 56.0% | 60.0% | +3.49 [+3.05, +3.94] | **+2.89 [+2.32, +3.47]** |
| `leaver` | 200 | 53.0% | 56.5% | +3.03 [+2.66, +3.40] | **+1.97 [+1.43, +2.51]** |

Across these 400 benchmark games (1,200 hauls), 54.3% of hauls ended on a
**dragon** and 19.2% in a **lorry stand-off**. These rates describe this
particular mix of policies, not a universal game probability.

## 3. Pairwise results and where points come from

Each pairwise row is one focal policy against three copies of the opponent,
with seats rotated. This is more informative than comparing separate benchmark
rows.

| focal policy | three opponents | focal top score | focal margin [95% CI] |
| :-- | :-- | --: | :-- |
| `selfplay4` | `leaver` | 60.0% | **+2.10 [+1.64, +2.58]** |
| `leaver` | `selfplay4` | 17.0% | **−1.37 [−1.74, −0.99]** |

This supports a repeated-game edge for `selfplay4` over this `leaver` policy in
this setup. It is not a complete ranking of all available policies.

Points per haul split by whether at least one rival banked loot. A policy's
benchmark margin can hide how much it depends on opponents' decisions.

| policy | hauls with a rival banking loot | points/haul then | points/haul otherwise |
| :-- | --: | --: | --: |
| `selfplay4` | 61.0% | +0.15 | +2.74 |
| `leaver` | 66.3% | +0.03 | +2.95 |

## 4. Escape, displacement, and tactical-measurement cautions

The long earlier report contained many position-specific probes. The points
below are useful context, not new aggregate measurements from the current
800-game blast study or the current leaderboard.

- **A cage is insurance.** It protects against a dragon; it does not by itself
  make an empty-handed or safe-table escape good. The old report's hazard and
  escape figures came from earlier checkpoint/rule runs. `SEED7.md` preserves
  the seed-specific audit; do not quote its checkpoint comparisons as current.
- **Out of the cage is out; a lorry is not another mine entrance.** The current
  code and tests guard against undoing an escape or re-entering after taking the
  lorry. Their original per-move valuations were measured in older runs.
- **A push is not a universal tactic.** Earlier broad push tests do not settle
  the first-turn case: a first-turn displacement can rationally stop an
  opponent camping. When testing pushes, do not specially spare the starting
  player or its next player, and compare an exposed own-column row-0 entry
  against alternate lanes—it can be a lorry trap. These corrections have
  already been reported as implemented; preserve them in future experiments.
  They are not fresh estimates here.
- The old `probe_push_clock.py` diagnostic was withdrawn; it inferred pushes
  from card moves rather than counting actual displacements. Do not cite it as
  evidence against displacement.
- Focused seed and training studies remain in [`SEED7.md`](SEED7.md),
  [`SEED7_P2.md`](SEED7_P2.md), [`TACTICAL_TRAINING.md`](TACTICAL_TRAINING.md),
  and [`LAST_HAUL_TRAINING.md`](LAST_HAUL_TRAINING.md). They are targeted
  diagnostics or training trials, not the current aggregate leaderboard.

### 4.5. The table hazard and danger-aware escape

The current policy representation includes a table-hazard signal used to
estimate whether a dragon can be woken; an escape should respond to real danger,
not simply to having collected a gem. The previous report's seed-level escape
values and checkpoint lift were measured before the present evaluation and are
not repeated as current results. For the current escape logic, see the tests and
`SEED7.md`.

## 5. Blast commitment and trigger frequency

This is a fresh, paired-seed intervention study on the checkpoint named in §2;
it is not a retest of every blast rule or opponent field. The focal `selfplay4`
seat played against three unchanged `selfplay4` copies. Each arm used the same
800 initial game seeds (`8400000`–`8400799`), rotated the focal seat evenly (200
games per seat), and used matching policy RNG seeds. The bootstrap resampled
whole-game paired differences 10,000 times. No game was abandoned.

**Margin** is focal score minus the mean of its three opponents' scores. In all
commit arms the focal policy still chooses its movement action. A forced trigger
only adds an announcement when the policy has at least two Dynamite, has a
legal card move, and has itself selected a card move. It does not force a card
move when the policy selects Pass, a cage, or the lorry.

The arms are:

1. **`natural`** — preserve the network's own announce and cancel behaviour.
2. **`commit_natural`** — preserve its trigger, but return Pass whenever its own
   blast is pending. The network is still evaluated on that decision; its
   counterfactual non-Pass choices are counted, while Pass is sent to the game.
3. **`extra_25pct`, `extra_50pct`, `force_card_moves`** — on an eligible card move,
   randomly add an announcement with probability 25%, 50%, or 100%; also pass
   while a blast is pending. The forced trigger uses a separate seeded RNG and
   does not change the selected movement action.

Passing prevents the focal agent from voluntarily cancelling its commitment,
but cannot prevent an involuntary cancellation (for example, by displacement
or death). A blast is counted as **completed** only if the engine draws three
cards; an announcement alone is not a fired blast.

### Announcements and completed blasts

“Eligible states” means the focal agent held a dynamite pair and had a legal
card move. A forced announcement additionally requires that the policy chose a
card move. “Policy would cancel” is the number of pending decisions where the
network itself selected a non-Pass action, even in arms that overrode it with
Pass. Engine cancellations also include rule-driven cancellations, so that
count is not the policy's voluntary-cancel rate. Games diverge after
interventions, so later opportunities and totals need not increase linearly
with the requested trigger probability.

| arm | eligible states / game | announcements / game | forced announcements / game | completed blasts / game | blast-dragon hauls / game |
| :-- | --: | --: | --: | --: | --: |
| `natural` | 2.89 | 1.165 | 0.000 | 0.405 | 0.579 |
| `commit_natural` | 2.21 | 0.891 | 0.000 | 0.433 | 0.609 |
| `extra_25pct` | 1.78 | 0.915 | 0.149 | 0.461 | 0.614 |
| `extra_50pct` | 1.51 | 0.934 | 0.253 | 0.468 | 0.618 |
| `force_card_moves` | 1.11 | 0.955 | 0.399 | 0.486 | 0.608 |

For example, the natural arm announced 932 times and completed 324 blasts; the
network chose a non-Pass on 253 pending decisions (0.316/game), and the engine
cancelled 565 pending blasts in total. The engine count also includes
rule-driven cancellations. The remaining announcements did not produce a
three-card draw because the deck could be too short. In `commit_natural`, the
network would have selected a non-Pass on 241 pending decisions, but those were
replaced with Pass; the engine recorded 319 total cancellations, since other
rules can cancel a pending blast as well.

### Paired outcome changes

Each entry is candidate minus control, in points/game except for the volatility
columns and blast/death counts. Brackets are 95% percentile bootstrap CIs over
the 800 matched game-seed/focal-seat pairs. “Margin SD” is the standard
deviation across games of the focal score margin; “score spread” is the
within-game max-minus-min spread across all four seats. Intervals are
unadjusted for the several exploratory comparisons.

| comparison | margin Δ [95% CI] | score-spread Δ [95% CI] | blast-dragon hauls Δ/game [95% CI] | focal blast-death hauls Δ/game [95% CI] | margin-SD Δ [95% CI] |
| :-- | :-- | :-- | :-- | :-- | :-- |
| `commit_natural` vs `natural` | **−0.15 [−0.28, −0.03]** | −0.09 [−0.16, −0.02] | **+0.030 [+0.008, +0.053]** | **+0.025 [+0.008, +0.044]** | −0.02 [−0.10, +0.06] |
| `extra_25pct` vs `commit_natural` | +0.06 [−0.04, +0.17] | +0.02 [−0.04, +0.08] | +0.005 [−0.011, +0.021] | +0.008 [−0.009, +0.023] | **+0.10 [+0.02, +0.20]** |
| `extra_50pct` vs `commit_natural` | +0.00 [−0.12, +0.13] | +0.02 [−0.05, +0.08] | +0.009 [−0.013, +0.031] | +0.013 [−0.008, +0.033] | +0.09 [−0.01, +0.19] |
| `force_card_moves` vs `commit_natural` | −0.04 [−0.18, +0.10] | +0.00 [−0.08, +0.08] | −0.001 [−0.028, +0.025] | +0.009 [−0.013, +0.031] | +0.03 [−0.08, +0.15] |
| `force_card_moves` vs `natural` | **−0.19 [−0.37, −0.02]** | −0.08 [−0.18, +0.01] | +0.029 [−0.003, +0.060] | **+0.034 [+0.009, +0.059]** | +0.01 [−0.12, +0.14] |

### What the blast study says—and does not say

- **Do not force commitment as a default.** Compared with preserving natural
  cancellation, `commit_natural` lowers the focal margin by about 0.15 points
  (95% CI −0.28 to −0.03). It also adds about 0.03 blast-dragon hauls and 0.025
  focal blast-death hauls per game. The observed score-spread change is slightly
  negative, and the margin-SD interval includes zero: this is evidence of a
  mean-score cost and more blast deaths, not a demonstrated increase in overall
  score volatility.
- **Extra triggers did not reliably improve the focal score.** Relative to
  `commit_natural`, every extra-trigger arm's paired margin interval includes
  zero. `force_card_moves` completed about 0.054 more blasts per game than
  `commit_natural`, but its margin change was −0.04 [−0.18, +0.10]. Modest
  effects remain plausible; this is not proof of exact equivalence.
- **There is no clear monotonic volatility result.** The 25% arm's margin-SD
  increase is small but its unadjusted interval is positive; the 50% and 100%
  arms do not reproduce a clear increase, and no arm shows a reliable rise in
  whole-game score spread or blast-dragon outcomes relative to
  `commit_natural`. Treat the 25% signal as exploratory, especially because
  several arms and outcomes were inspected.
- The full `force_card_moves` combination is worse than `natural` by about
  0.19 margin points, but the incremental comparison against `commit_natural`
  is inconclusive. Do not attribute the entire natural-to-forced difference to
  trigger frequency: commitment itself changes decisions and game trajectories.

Older blast sections reported different margins and a “committed blast is worth
zero” result from other checkpoint/rule runs. They are historical, not estimates
for this checkpoint. The earlier `probe_blast_strategy.py` trap-trigger result
also used a separate scripted policy and does not isolate the present neural
trigger intervention.

## 6. Blast threat: an old measurement was superseded

An earlier version claimed that toggling an opponent blast changed the learned
policy's action distribution by only **0.0065** in total variation, and
concluded that the policy did not react. That figure is stale. A later change
to the action mask made a pending blast open a pit-cage option in many of the
paired contexts, so the comparison no longer changed only a state feature.

[`THREAT_LEAGUE.md`](THREAT_LEAGUE.md) records the correction, the split between
mask-changing and mask-identical contexts, and the separate threat-seeded league
experiment. Those runs are not the fresh 800-game blast-frequency study in §5;
use the checkpoint, rule settings, and sample sizes recorded in that document.
Do not repeat 0.0065 as a current behavioural result or conflate a reaction
encoded by a changing legal-action mask with a policy's learned response.

## 7. Engine reliability

The earlier report found and fixed two benchmark-relevant defects: a dead
player's pending blast could prevent a haul from resolving, and the action grid
could omit cards in deeper rows. It also withdrew an alleged tie-break failure
after correcting the card-count arithmetic. Pre-fix results and the withdrawn
bug report are not evidence about current balance.

The fresh leaderboard and all five 800-game blast arms above report **zero
abandoned games**. Any future nonzero abandoned-game count invalidates a score
comparison until the cause is understood; an all-zero aborted game must not be
silently counted as a draw.

## 8. Pass-count rule

The fresh evaluations use the current `n-1` consecutive-pass rule. The earlier
`n`-pass comparisons belong to a different experiment and are not re-estimated
here; do not combine their outcomes with the current tables.

## 9. Skill versus luck

In a mirror match every seat runs the same policy, so within-game differences
come from the game and seat position, not policy strength. In 400 four-player
`selfplay4` mirror games:

- mean max-minus-min score spread: **4.875 points** (maximum 11);
- tied top score: **17.25%** of games;
- mean scores by seat: **[1.845, 1.930, 1.680, 1.6025]**.

The mirror spread is about 2.3 times the +2.10 point paired margin against
`leaver`. That does not make policy skill unmeasurable: repeated-game estimates
can separate an edge whose interval excludes zero. It does mean that a single
match—and win rate from a small sample—can easily give the wrong impression.
These intervals condition on this checkpoint and this run; they do not include
checkpoint-selection or training uncertainty.

## 10. What to test next

1. Keep `rule` and `leaver` as distinct controls. The former measures performance
   against the benchmark; the latter checks whether a margin depends on rivals
   staying in the mine.
2. Re-run promising policy changes with a new base seed and report paired
   margins, top-score share, sample size, checkpoint identity, and abandoned
   games.
3. For displacement, include the first-turn anti-camping case and rotate starts
   without special leniency for the starting player or its successor. Compare
   exposed own-column row-0 choices with alternate lanes.
4. Fix or explicitly test the trainer's game-boundary handling before using new
   self-play runs as evidence (see §12).

### 10.6. Threat-seeded league

The earlier recommendation to seed self-play with `threat` opponents was tested
in [`THREAT_LEAGUE.md`](THREAT_LEAGUE.md) and did not show a reliable gain. That
is a separate, historical training experiment—not a result of the fresh
leaderboard or blast-frequency study. Its tests and confidence intervals are
documented in the companion report.

## 11. Reproduce

Current aggregate evaluation:

```sh
PYTHONPATH=. OMP_NUM_THREADS=1 python evaluate_balance.py \
  --agents checkpoints/selfplay4.npz,leaver --players 4 \
  --games 200 --pair-games 200 --mirror-games 400 --benchmark rule \
  --temperature 0.7 --pass-rule n-1 --solo-escape-bonus \
  --blind-blast-choice --seed 9100000 \
  --out /tmp/balance-current.md --json /tmp/balance-current.json
```

Blast commitment/frequency experiment:

```sh
PYTHONPATH=. OMP_NUM_THREADS=1 python probe_blast_frequency.py \
  --checkpoint checkpoints/selfplay4.npz --games 800 --seed 8400000 \
  --out /tmp/blast_frequency.json
```

The blast script is in [`probe_blast_frequency.py`](probe_blast_frequency.py).
It writes per-game rows and paired summaries as JSON. The pairs share initial
seed and focal seat, not identical later positions: actions can change future
states and randomness. Its CIs are conditional on this checkpoint, code
snapshot, and opponent population.

## 12. Caveats

- **Only four players were freshly evaluated here.** Do not infer three- or
  five-player balance from these results.
- **Opponent dependence:** the benchmark uses `rule`; the direct comparison uses
  `leaver` or `selfplay4`. These results do not rank policies against a broader
  league or human players.
- **Training credit assignment:** `train_selfplay.py` accumulates decisions
  across games, while `compute_returns` groups them by player without a game
  boundary. Returns may therefore leak across episodes. The evaluation above
  remains a valid measurement of this saved policy, but new training comparisons
  should not be treated as clean evidence until returns are computed separately
  per game (or an episode ID is included).
- **Blast experiment scope:** the intervention tests one focal seat against
  three copies of this checkpoint. It does not establish how blasts fare against
  scripted bots, other checkpoints, or human play. The 25%, 50%, and 100% arm
  intervals are unadjusted exploratory comparisons; modest effects remain
  unresolved.
- Re-run with an independent base seed before adopting small differences as a
  policy rule. A bootstrap interval over games does not include model-selection,
  training, or checkpoint uncertainty.
