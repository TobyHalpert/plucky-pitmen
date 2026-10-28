# Is Plucky Pitmen balanced?

Short answer: **no, as printed.** A player who walks straight to the pit cage on
turn one and leaves with zero gems beats the hand-tuned rule policy **54% of the
time** (3-player, 300 games) and out-scores it by **+2.69 points per game**.
A trained self-play agent independently converges on the same behaviour and
does even better (**63%**, **+3.67**). One scoring-rule change removes the
exploit; after that change no policy in this repository beats the cautious rule
baseline at all.

This document collects every measurement behind those sentences, the tools that
produce them, the three engine bugs that were found on the way, and the rule
experiments that should be run next.

All numbers: 3 players, seats rotated every game, 300 games per leaderboard row,
100 games per head-to-head direction, 95% bootstrap confidence intervals
(2000 resamples). Reproduce with the commands at the end.

---

## 1. The dominant strategy: leave on turn one

The printed rules award the haul bonus (+3 points, +4 in the last haul) to a
sole escapee **even when that player has no gems**. Three hauls of that is
+10 points for taking no risk at all, while everyone who stays can lose 1-2
points to the dragon.

`agents.RusherAgent` implements exactly that, in thirty lines: walk to the
nearest pit cage, sit there, pass forever. It never looks at a card front.

### Leaderboard, printed rules (`reports/balance_default_rules.md`)

| policy | win % | top-score % | mean score | margin vs 2 × rule [95% CI] | escapes (0 gems) |
| :-- | --: | --: | --: | :-- | --: |
| selfplay (60 gens) | 63.0% | 69.0% | +4.94 | **+3.67 [+3.09, +4.22]** | 98% (100%) |
| rusher | 54.0% | 58.0% | +4.48 | **+2.69 [+2.13, +3.27]** | 100% (100%) |
| random | 43.7% | 45.3% | +2.10 | +0.97 [+0.46, +1.41] | 57% (69%) |
| rule | 31.7% | 39.3% | +1.79 | -0.23 [-0.62, +0.17] | 67% (21%) |
| cautious | 32.3% | 36.7% | +0.87 | -0.34 [-0.71, +0.06] | 46% (56%) |
| greedy | 21.3% | 28.7% | +0.09 | -1.09 [-1.41, -0.79] | 30% (53%) |

"Margin" is the policy's score minus the mean score of the other two seats. A
33.3% win rate is chance; 54% is a strategy.

### Head-to-head (`reports/balance_default_rules.md`)

| A | B | A alone vs 2 × B | B alone vs 2 × A |
| :-- | :-- | :-- | :-- |
| rusher | rule | **+2.85 [+1.92, +3.78]** | +1.30 [+0.99, +1.65] |
| selfplay | rule | **+4.85 [+3.85, +5.80]** | +1.78 [+0.97, +2.59] |
| selfplay | rusher | **+1.49 [+0.89, +2.06]** | -1.04 [-1.35, -0.72] |
| random | rule | +0.56 [-0.25, +1.40] | +0.24 [-0.46, +0.98] |
| greedy | rule | -0.94 [-1.49, -0.35] | +0.24 [-0.49, +0.95] |

Two things to read out of this table:

* The exploit is **frequency dependent**. One rusher among rule players wins
  +2.85; one rule player among two rushers is *not* crushed (+1.30), because
  two escapees with zero gems split the bonus into +1 each. Rushing pays best
  when you are the only one doing it, which is exactly what makes it a
  temptation in a real group.
* The learned agent beats the pure rusher head-to-head (+1.49 vs -1.04). It
  rushes as well, but it also punishes other rushers.

### Does the learned agent find the exploit on its own?

Yes. The shipped checkpoint (`checkpoints/selfplay.npz`, 60 generations of
league self-play, no hand-coded heuristics) escapes in 98% of hauls with **zero
gems in 100% of them**, exactly like the hand-written rusher. That is the
strongest evidence that this is a discovered strategy and not an artifact of
the `RusherAgent` implementation.

The catch: 0 gems on every escape means the behaviour is pure bonus farming.
The margin is real points, but it is not skill at the game's stated objective
("leaves the mine with the most gems").

---

## 2. The fix: don't pay for an empty escape

`MineEnv(solo_escape_bonus=False)` (CLI: `--no-solo-escape-bonus`) awards nothing
when the only escapee has zero gems. Everything else is unchanged.

### Leaderboard, fixed rules (`reports/balance_fixed_rules.md`)

| policy | win % | mean score | margin [95% CI] | escapes (0 gems) |
| :-- | --: | --: | :-- | --: |
| rule | 32.7% | +1.54 | -0.27 [-0.63, +0.09] | 67% (21%) |
| random | 31.0% | +0.07 | -0.31 [-0.60, -0.05] | 57% (69%) |
| cautious | 23.0% | -0.14 | -0.89 [-1.14, -0.62] | 46% (56%) |
| greedy | 16.7% | -0.47 | -0.99 [-1.22, -0.76] | 30% (53%) |
| selfplay_no_bonus | 24.7% | +0.06 | -1.17 [-1.40, -0.94] | 98% (99%) |
| rusher | 19.3% | +0.16 | **-1.62 [-1.82, -1.42]** | 100% (100%) |

The same rusher that was +2.69 points up is now **1.62 points down**, and the
learned agent trained under the fixed rule is **1.17 points down**. Nobody beats
the rule policy. The exploit is gone.

**But the game is not thereby "solved" into something rich.** The fixed-rules
learner did not learn to mine: it still leaves empty-handed in 99% of hauls and
scores ~0. It is not punished much for that, because leaving scores 0 while
staying risks -1 or -2. The equilibrium moves from "farm the bonus" to "take
the safe zero and hope everyone else wakes the dragon". That is a second, milder
degenerate strategy, and it is the reason the rule change alone is not the
whole answer (see §5).

---

## 3. Skill vs luck

Mirror matches: every seat runs the *same* policy, so the spread is luck and
seat position only.

| rules | mean within-game spread | games with a tied top score |
| :-- | --: | --: |
| printed | **4.92 points** | 11% |
| fixed | **4.20 points** | 18% |
| `n` passes | 4.50 points | 10% |
| `n-1` passes | 4.68 points | 15% |

The largest real skill margin in this document is the self-play agent's +3.67,
which is *smaller* than the 4.9-point spread of identical agents playing each
other. Conclusion: **a single game is mostly luck**. The skill signal is real
(selfplay's confidence interval is far from zero and it wins in both
directions against every policy), but you need tens of games to see it, and
`win %` alone is a poor metric because ties are broken by drawing cards.

Seat position is not a big factor: mean score by seat is 1.83 / 1.70 / 1.71.

---

## 4. Is blasting too weak? No - the engine was peeking

The printed blast rule: draw three cards **face down**, choose two, *then* look
at them. `MineEnv` used to hand the chooser all three fronts, which turns the
gamble into a card-selection engine. `--blind-blast-choice` restores the printed
rule.

`rule:always` = announce a blast whenever two dynamite are in hand.

| engine | leaderboard margin [95% CI] | always alone vs 2 × rule | rule alone vs 2 × always |
| :-- | :-- | :-- | :-- |
| peek (old default) | **+1.65 [+1.10, +2.20]** | +0.70 [-0.11, +1.48] | **-1.23 [-1.88, -0.49]** |
| blind (printed) | +0.62 [+0.11, +1.14] | -0.23 [-1.07, +0.70] | -0.27 [-1.02, +0.57] |

With the peek, blasting is one of the strongest things you can do. Without it,
blasting is statistically neutral. So "blasting is too weak" is **not
supported**; the observed 0.00-0.22 blasts per game from trained agents is a
consequence of the escape exploit (a player who leaves on turn one never
collects the two dynamite a blast needs), not of the blast payoff.

---

## 5. Open question: n-1 passes or n passes?

The printed rule is "as many players pass in a row as there are players in the
mine, minus one". `--pass-rule n` plays the alternative.

| rule | rusher win % | rusher margin | hauls ending in a stall | dragon endings | decisions / game | mirror spread |
| :-- | --: | --: | --: | --: | --: | --: |
| **n-1 (printed)** | 48.0% | +2.07 [+1.40, +2.75] | **4.8%** | 58.2% | **77.2** | 4.68 |
| n | 84.5% | +5.74 [+5.20, +6.24] | **27.6%** | 39.2% | 143.4 | 4.50 |

`n-1` is clearly better: the same games run in half the decisions, deadlocks
drop by a factor of six, and the dominant strategy is roughly halved in power.
It does not remove the exploit (that is the bonus rule), but it removes a whole
class of "everybody sits still" endings.

While measuring this, a bug surfaced: `step_one_player()` - the API the
self-play driver uses - compared the pass count against `len(active_players)`
instead of `self._pass_threshold()`, so `--pass-rule` was silently ignored
outside the single-learner `step()` API. That is fixed, and `test_selfplay.py`
now covers the multi-agent path. Every number in this document uses the
corrected code.

### The lorry stall

Even with `n-1`, a haul can deadlock. Sitting on your lorry is safe from the
dragon, and the printed rules give the haul no ending condition for "everybody
is parked". `stall_rounds=8` ends such a haul and gives campers zero, which is a
band-aid, not a rule.

The valve is not the problem: in rule-only games the stall rate is **2.0% of
hauls at 8 rounds and 2.0% at 40 rounds**. What raises it is a policy that
actually parks - 4.8% across the mixed benchmark and 13.6% when a rusher is at
the table. So the deadlock is a genuinely absorbing state, not a threshold set
too tight, and raising `--stall-rounds` only makes those games longer.

Suggested real fix: a haul also ends when a full round passes without any card
being collected, or when every remaining player has been on their lorry for a
round. Then score normally.

---

## 6. What the learner did and did not learn

* League self-play (`train_selfplay.py`, 60 generations x 64 games, one hidden
  layer, REINFORCE + value baseline) **discovered the escape exploit**
  (98% of hauls, zero gems, 63% wins, +3.67 margin). Training is unaided: no
  rule heuristics, no demonstrations, and the encoder only sees the cheat-safe
  `PlayerActionContext`.
* Trained under the fixed rule (`--no-solo-escape-bonus --escape-shaping 0`) it
  **does not** learn a mining policy; it converges to leaving empty-handed and
  ends up 1.17 points *behind* the scripted rule agent.
* Re-running the same training under the corrected pass rule (60 generations)
  did not re-find the exploit either: best evaluation was 0.425 wins at
  generation 50, with a flat 0.25-0.43 curve. The earlier high score came from
  the exploit, not from general strength.

That is a limitation of the learner, not a proof that the game cannot be played
well. A human who counts card backs should beat all of these agents. If you want
the agents themselves to become the balance oracle, the next step is a stronger
learner (longer training, deeper network, PPO or search), not more evaluation.

---

## 7. Recommended changes, in order

1. **Do not pay the haul bonus for an empty escape.** Either require at least
   one gem, or award the bonus to the gem leader whether or not they left the
   mine. Measured effect: rusher 54% -> 19% wins, margin +2.69 -> -1.62.
2. **Fix the blast information leak.** Draw three, choose two, then look.
   Measured effect: blasting goes from dominant (+1.65) to neutral (+0.62).
3. **Keep `n-1` passes and add an anti-camping end condition.** Measured effect:
   halved game length, 4.8% instead of 27.6% stalled hauls.
4. **Consider scoring gems directly** (e.g. +1 per gem brought out). Today gems
   only decide who gets the 3-point bonus, so risking a card has no upside
   beyond that bonus and the safe zero is structurally attractive. This is the
   most promising untested lever; `_score_haul` is the only place to change.
5. Tune victory points last. The 3/4-point bonus is ~60% of a 10-point game, so
   it dominates any tuning of the -1/-2 penalties. Change the bonus rule before
   changing its size.

---

## 8. Reproduce

```powershell
python -m pip install -r requirements.txt

# 300-game leaderboards and pairwise table
python evaluate_balance.py --agents rule,greedy,random,cautious,rusher,checkpoints/selfplay.npz --games 300 --out reports/balance_default_rules.md
python evaluate_balance.py --agents rule,greedy,random,cautious,rusher,checkpoints/selfplay_no_bonus.npz --no-solo-escape-bonus --games 300 --out reports/balance_fixed_rules.md

# blast: peek vs printed rule
python evaluate_balance.py --agents rule,rule:rich,rule:always --games 200 --skip-mirror --out reports/balance_blast.md
python evaluate_balance.py --agents rule,rule:rich,rule:always --games 200 --skip-mirror --blind-blast-choice --out reports/balance_blast_blind.md

# pass rule
python evaluate_balance.py --agents rule,greedy,random,rusher --games 200 --pass-rule n --out reports/pass_n.md

# train from scratch (3 minutes on a laptop CPU)
python train_selfplay.py --generations 60 --games-per-generation 64
```

`reports/` is git-ignored; it holds the generated leaderboards and the JSON
behind every table above.

## 9. Caveats

* Everything here is 3-player. The code supports 3-5; player count changes the
  bonus maths (see the frequency dependence in §1) and should be measured.
* The "rule" baseline is hand-tuned, not optimal. Under the printed rules it is
  statistically indistinguishable from random mining, so beating it is a low
  bar; losing to the rusher by 1.6 points is a strong signal. Under the fixed
  rules it beats every other scripted policy in both directions.
* The learner is small (48-64 hidden units, REINFORCE). "Self-play did not find
  a better policy" only means "at this budget".
* Bootstrap intervals assume the games are independent draws. They are (fresh
  seeds), but 300 games still leaves about +/-0.25 points of uncertainty on a
  margin.
