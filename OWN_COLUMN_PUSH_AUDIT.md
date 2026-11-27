# Own-column Gem exposure against always-pushers

## Finding

The concern was real in the requested stress test. With the original `selfplay4`
policy, Player 1's first action was a Gem in P1's own column in **330 of 500
games (66.0%)**. The other three seats were configured to take every legal
displacement. Every one of those 330 Gem placements was successfully displaced
before execution.

A displacement is not the same as losing the Gem under the current same-card
rule. The displaced player remains on the card, keeps their loot, and gets
another turn (but cannot Pass). In the baseline opening sample, a pusher still
occupied the selected Gem when execution resolved in **85 of 330 cases (25.8%)**.
The probe records both events separately.

## Simulation method

- 500 full games, seeds `3500000` through `3500499`, four seats and three hauls.
- P1 is always seat 0 and the first player to act. This isolates the opening
  decision instead of averaging over a random starting player.
- P1 uses the checkpoint at temperature 0.7. P2–P4 use `AlwaysDisplaceAgent`:
  take a legal displacement whenever one is available, prefer the highest
  scoring victim and then the shallowest row, otherwise use `CounterAgent`.
- The probe reads P1's selected own-column card, counts a displacement only when
  the engine appends a successful `displacement_event` for that exact card, and
  separately checks who occupies the Gem when the execution phase reveals it.
- All arms use the same seeds. The post-change baseline and retrained checkpoint
  use the same probe and pusher implementation. The original-policy baseline
  was measured before the policy changes below; its opening-action result is
  directly comparable, but its all-game totals are not a strict paired ablation.

## Results

| Policy/checkpoint | Opening own-column Gem | Displaced if chosen | Gem on a pusher at resolution | Mean P1 score | Outright wins |
| :-- | --: | --: | --: | --: | --: |
| Original `selfplay4`, original policy features | 330/500 (66.0%) | 330/330 (100%) | 85/330 (25.8%) | 1.408 | 27.4% |
| Push-aware policy, original checkpoint, before retraining | 21/500 (4.2%) | 21/21 (100%) | 5/21 (23.8%) | 2.408 | 43.0% |
| `selfplay4_pushaware`, after retraining | 14/500 (2.8%) | 14/14 (100%) | 3/14 (21.4%) | 2.678 | 45.0% |

The policy change cuts the opening own-column Gem choice rate from 66.0% to
2.8% in this stress test (a 63.2 percentage-point, 96% relative reduction).
It does **not** make a chosen Gem immune to a push: the conditional displacement
rate remains 100% because all three opponents are intentionally push-on-sight.
Nor does every push steal the Gem; the same-card displacement rule gives P1 a
chance to move before execution.

The exact-code paired comparison between the pretraining checkpoint and the
retrained candidate is more modest, and is reported without overclaiming:

| Measure, 500 games | Before retraining | Retrained | Change |
| :-- | --: | --: | --: |
| Opening own-column Gem choices | 21 (4.2%) | 14 (2.8%) | −7 choices |
| All own-column Gem choices | 2,212 | 2,012 | −9.0% |
| Those choices directly displaced | 1,484 (67.1%) | 1,305 (64.9%) | −179 |
| Distinct displaced Gem cards taken by a pusher at execution | 919 | 823 | −96 |
| Mean P1 score | 2.408 | 2.678 | +0.270 |
| Outright win rate | 43.0% | 45.0% | +2.0 pp (paired ±1.9 pp) |
| Mean score margin | 1.913 | 2.247 | +0.334 (paired ±0.124) |

The opening-choice difference between the two push-aware arms is small; it is
not evidence of a large incremental effect from retraining alone. The large
opening reduction comes from correcting the policy's action representation and
mask heuristic. Retraining further reduced opening choices and total
Gem/displacement/theft events, and improved the paired score margin against the
stress-test table.

As a limited ordinary-opponent control, 400 games (seed `4200000`) against
three `rule` seats moved as follows:

| Measure | Baseline | Candidate | Change |
| :-- | --: | --: | --: |
| Win rate | 58.75% | 68.50% | +9.75 pp |
| Top-score rate | 65.00% | 75.00% | +10.00 pp |
| Mean score | 3.940 | 4.565 | +0.625 |
| Mean rank (lower is better) | 1.5675 | 1.3925 | −0.1750 |

No confidence interval was calculated for this single-seed control. It is not a
claim of universal dominance or broad-opponent improvement.

## Policy change and training

`features.py` now marks a legal own-column Gem as exposed when an unsettled,
active opponent can reach its row. The existing action-feature channel 33 gives
that move a learned displacement-risk signal. When **all** legal own-column
Gems are exposed, `digs_past_a_known_gem` no longer treats a foreign face-down
lane as automatically dominated; alternatives remain available to the policy.
This is intentionally a soft policy heuristic, not a hard ban on the Gem and
not a change to the printed game rules. The engine's legality and displacement
rules are unchanged.

The candidate resumed from `checkpoints/selfplay4.npz` for 40 generations of
64 games, four players, seed 72002, with `--pusher-match-prob 0.75` and
`--mirror-prob 0.10`. A pusher-match game places one learner against three
always-pushers. **1,741 of 2,560 training games** were pusher matches. The
resulting checkpoint is `checkpoints/selfplay4_pushaware.npz` (40 additional
generations; the checkpoint's total generation counter is 390).

Reproduction commands:

```sh
.venv/bin/python train_selfplay.py \
  --resume checkpoints/selfplay4.npz \
  --name selfplay4_pushaware \
  --out checkpoints/selfplay4_pushaware.npz \
  --generations 40 --games-per-generation 64 --players 4 \
  --seed 72002 --pusher-match-prob 0.75 --mirror-prob 0.10 \
  --league-prob 0.50 --league-size 8 --snapshot-every 0 --save-every 10 \
  --eval-games 0 --temperature 1.0 --epsilon 0.05 --epochs 1 --batch-size 256

.venv/bin/python probe_always_pusher_match.py \
  --games 500 --seed 3500000 --fixed-p1-first \
  --baseline checkpoints/selfplay4.npz \
  --candidate checkpoints/selfplay4_pushaware.npz \
  --out OWN_COLUMN_PUSH_AUDIT.json

.venv/bin/python - <<'PY'
from learning import PolicyNet
from train_selfplay import evaluate

baseline = PolicyNet.load("checkpoints/selfplay4.npz")
candidate = PolicyNet.load("checkpoints/selfplay4_pushaware.npz")
options = dict(games=400, players=4, seed=4200000,
               opponent="rule", deterministic=False)
print("baseline", evaluate(baseline, **options))
print("candidate", evaluate(candidate, **options))
PY
```
