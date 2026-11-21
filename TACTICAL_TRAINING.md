# Branch-based tactical training

This adds an experimental training path, not a new game rule or an action ban.
The shipped `selfplay4.npz` remains unchanged until the candidate is evaluated.

## What it learns from

`train_tactical.py` generates four-seat, three-haul games without using seed 7.
The collection league mixes neural games with a scripted always-blaster or the
score-aware deterministic pusher. It collects tactical positions when dynamite
pairs or announced blasts are present, seeking both an initial decision and a
later decision with the acting player's own blast pending. At most two
positions come from a source game. Held-out positions use a separate range of
game seeds; they never enter the training targets or optimizer.

For each position the candidate set contains likely policy actions, cage/pass
when policy-allowed, shallow/deep blast placements with their ordinary twins,
and one eligible attack on the highest-scoring reachable rival. Existing masks
remain unchanged: this is learning within the present policy action set.

Each candidate is evaluated through multiple full-game continuations. All
candidates share the same sampled world and per-seat sampling seeds for each
repetition. The reward is final-score **win share** (ties split equally), not
haul loot, escape bonuses as training rewards, or number of blasts. Game scoring
itself is unchanged. Aborted continuations fail the run rather than silently
becoming misleading targets.

## Hidden information

The acting player's own column and private hand stay fixed. Other board fronts,
opponents' collected fronts, and the deck are resampled preserving visible
backsides, total card inventory and the fact that collected cards are safe.
The deck is shuffled, and opponents see their own sampled columns normally.
Pending private blast draws are excluded from collection.

This is **observation-consistent determinization**, not a full-history Bayesian
posterior. Previous opponent actions are not used as likelihood evidence.
Thus it avoids training directly on the actual hidden dragon in a watched
example, but it does not solve inference about all historical tells.

## How labels become policy updates

The continuation policy is frozen for the run. Estimated win shares are turned
into soft targets (temperature 0.15) over compared candidates. These are mixed
50/50 with the frozen policy's full legal-action distribution to limit forgetting.
`PolicyNet.fit_tactical` minimises masked cross-entropy on independent positions;
it does not propagate returns between unrelated positions. Only policy weights
change; value weights remain untouched. Training does not reward announcing,
keeping, cancelling or executing blasts directly.

The held-out metric is expected teacher win share **conditional on the compared
candidate set**, with its policy probability coverage also reported. It is not
a fresh-game win rate, and finite-sample teacher labels remain noisy. A separate
fresh-game comparison tests the candidate against three unchanged nets, with
seats rotated and common game/policy seeds. Blast announcements, cancellations
and hauls with an actually fired blast are counted separately.

## Commands

```sh
python train_tactical.py --positions 64 --held-out 32 --samples 24 --epochs 30 --out checkpoints/selfplay4_tactical.npz
python evaluate_tactical.py --games 200
python mine_gui.py --opponents 3 --policy neural:selfplay4_tactical --all-seats
```

Training seed: 1300000; held-out collection starts at 1400000. Continuation
sampling starts at 1500000 and 1600000 respectively. Fresh-game evaluation
starts at 1900000. Training configuration and before/after held-out measurements
are saved beside the checkpoint as `.history.json`.

This first run is a small tactical-distillation experiment, not an equal-budget
comparison against ordinary self-play, an optimal search teacher, or proof of
convergence. The completed run used 64 training positions (33 source games, including 31
own-blast-pending decisions) and 32 held-out positions (16 source games, including
16 own-blast-pending decisions), with 24 sampled worlds per position and 30
policy-fitting epochs. It evaluated **6,480 training** and **2,688 held-out**
continuations. Training and held-out labels and source seeds are in the tracked
checkpoint history.

## First-run results

| tactical estimate, conditional on compared actions | before | after |
| :-- | --: | --: |
| training estimated win share | 23.03% | 23.37% |
| held-out estimated win share | 22.59% | 22.11% |
| held-out candidate probability coverage | 98.49% | 96.14% |

The held-out estimate slipped, so this is **not** evidence of generalisation.
The finite-sample teacher's best candidates are also noisy; its best-action
estimate is not a proven achievable win rate.

A separate fresh-game test used 200 games per checkpoint, seeds
1900000–1900199, four seats rotated, three hauls, temperature 0.7. One evaluated
seat faces three unchanged `selfplay4` nets in both arms:

| fresh games | unchanged net | tactical candidate |
| :-- | --: | --: |
| outright final-score wins | 20.5% | 19.5% |
| tied for top on final score | 9.0% | 5.5% |
| mean score | 2.205 | 1.745 |
| score margin over mean rival | +0.053 | −0.383 |
| own blast announcements | 420 | 270 |
| own blast cancellations | 361 | 212 |
| hauls in which own blast actually fired | 56 | 55 |
| decisions per game (whole table) | 103.285 | 100.020 |
| unfinished games | 0 | 0 |

Paired outright-win difference: **−1.0 percentage point, give or take 3.47**
(one standard error across paired games). No reliable win improvement is
established. Fewer cancellations alone are not a success: announcements also
fell and actually fired blasts did not increase. Counts of fired blast hauls
are not the same as the number of individual announcements or blasts.

For the fixed seed-7 P2 initial position, which was **not in training**, the
probability of row-0+blast fell from 60.3% to 41.7%; row-2+blast rose from 8.7%
to 16.5%. This shows a local policy change, not proof of better final outcomes.
The seed-7 diagnostic was inspected only after training; it did not select
hyperparameters or a checkpoint.

**Decision:** ship the candidate as an optional experiment, do not replace the
default. The next iteration should improve the teacher and data coverage before
scaling fitting epochs: this run used only 64 positions, 24 sampled worlds each,
and a frozen continuation policy that itself makes poor blast decisions. It
cannot be expected to teach globally coherent multi-turn play from those labels
alone. Refreshing continuation policies across training rounds and testing
higher-confidence labels are hypotheses to test, not improvements already
claimed.
