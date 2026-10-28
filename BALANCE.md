# Is Plucky Pitmen balanced?

**Correction (this revision).** An earlier version of this document claimed
that "leave on turn one" is a dominant strategy. That claim was wrong, and the
error is worth keeping on the record because it is the classic benchmarking
trap. Against opponents who collect loot and then leave — which the printed
rules let them do — an empty-handed escape is worth **exactly zero**. The
rusher's big margins came from opponents who *stayed in the mine and died*, and
from a stall valve that voided those opponents' loot while still paying the
escaping player.

The supportable conclusions are:

* There is **no dominant strategy**. Leaving early is a *parasite*: it converts
  other players' mistakes into points, and it is worthless against a player who
  simply takes a gem and walks out.
* The best policy in this repository is the opposite of rushing: **collect loot,
  then leave**. It beats the rusher **+8.72 points, 100% top score** and beats
  the trained self-play agent **+5.08 points**.
* A leaderboard against fixed bots cannot detect this on its own, because the
  ranking depends on what the bots do. Head-to-head against a leaver is the test
  that matters.

All numbers: 3 players, seats rotated every game, 300 games per leaderboard row,
60 games per head-to-head direction, 95% bootstrap CIs (2000 resamples).

---

## 1. The parasite: what "leave on turn one" is actually worth

`agents.RusherAgent` walks to the nearest pit cage and never looks at a card.
Against the shipped benchmarks it looks strong:

| policy | win % | mean score | margin vs 2 × rule [95% CI] | escapes (0 gems) |
| :-- | --: | --: | :-- | --: |
| selfplay (60 gens) | 68.0% | +5.26 | **+4.14 [+3.54, +4.70]** | 99% (99%) |
| rusher | 54.0% | +4.48 | **+2.69 [+2.13, +3.27]** | 100% (100%) |
| leaver (collect + leave) | 51.3% | +3.95 | +2.17 [+1.60, +2.73] | 99% (94%) |
| random | 43.7% | +2.10 | +0.97 [+0.46, +1.41] | 57% (69%) |
| rule | 31.7% | +1.79 | -0.23 [-0.62, +0.17] | 67% (21%) |
| cautious | 32.3% | +0.87 | -0.34 [-0.71, +0.06] | 46% (56%) |
| greedy | 21.3% | +0.09 | -1.09 [-1.41, -0.79] | 30% (53%) |

Now split the rusher's points by what its opponents did (200 games, rusher vs
2 × rule, per haul):

| what the other two did | share of hauls | rusher points/haul |
| :-- | --: | --: |
| left the mine **with gems** | **50.7%** | **0.00** |
| woke the dragon, still inside | 29.0% | +3.29 |
| stalled on their lorries, still inside | 13.2% | +3.24 |

**Every single point the rusher earns comes from hauls where nobody else
banked loot.** When the other players leave with gems, the rules compare gems
among everybody who left and the rusher's empty escape is worth nothing. This
is the mechanism the first version of this report missed.

The same split for every policy in the pool is in the report's
"Where the points come from" section, which is generated for exactly this
reason.

## 2. The counter: collect loot, then leave

`agents.LeaverAgent` is the cheapest policy that exploits the race: take a gem,
then walk to the pit cage. It is not clever — it picks a card, holds position,
and leaves once it holds loot.

| A | B | A top-score % | A margin [95% CI] | B top-score % | B margin [95% CI] |
| :-- | :-- | --: | :-- | --: | :-- |
| **leaver** | **rusher** | **100.0%** | **+8.72 [+8.20, +9.20]** | 41.7% | -0.41 [-1.28, +0.48] |
| **leaver** | **selfplay** | 81.7% | **+5.08 [+4.03, +6.06]** | 13.3% | -3.00 [-3.64, -2.28] |
| selfplay | rusher | 73.3% | +0.62 [-0.12, +1.38] | 51.7% | -0.60 [-0.89, -0.32] |
| rusher | rule | 60.0% | +2.34 [+1.23, +3.50] | 100.0% | +1.08 [+0.70, +1.53] |

* A single rusher against two leavers tops the scoreboard 41.7% of the time
  with a **negative** margin: it is a losing seat.
* A single leaver against two rushers wins **99.5%** of games (measured over 200
  games) with a **+8.68** margin, because the rushers leave, no further rows are
  dealt, and the leaver then picks up loot unopposed before walking out with
  more gems than the empty-handed pair.
* The trained self-play agent loses to the leaver as well (−3.00 as the lone
  seat, +5.08 for the leaver). Self-play converged on the exploit because its
  opponent pool — scripted bots and its own snapshots — did not punish it.

Under the fixed scoring (`--no-solo-escape-bonus`) the ordering is the same
where it matters: leaver vs rusher **+8.72**, leaver vs the retrained agent
**+5.29** at 93.3% top score.

## 3. Why the leaderboard lied

Two things made the rusher look dominant, and neither is about the game's
quality:

1. **The benchmark population decides the ranking.** Against the scripted bots
   the order is selfplay > rusher > leaver. Against a leaver it inverts
   completely. A margin measured against a fixed, exploitable opponent says
   nothing about whether a strategy is strong — only that it is strong *there*.
2. **The stall valve pays the parasite.** Once somebody leaves, no new rows are
   dealt, and parked players can keep a haul alive forever. `stall_rounds=8`
   ends such a haul and gives the campers nothing — while still paying the
   escapee the +3 they had already earned. So the valve converts "the campers
   might still have left and won" into "the camper's loot is void, the escapee
   banks the bonus".

The methodological fix that is now built in: always include a `rusher` **and** a
`leaver` in `--agents`, and read the "Where the points come from" table. A
policy whose points vanish whenever a rival banks loot is farming mistakes.

## 4. Is blasting too weak? No — the engine was peeking

The printed blast rule: draw three cards **face down**, choose two, *then* look
at them. `MineEnv` handed the chooser all three fronts, turning a gamble into a
card-selection engine. `--blind-blast-choice` restores the printed rule.

`rule:always` = announce a blast whenever two dynamite are in hand.

| engine | leaderboard margin [95% CI] | always alone vs 2 × rule | rule alone vs 2 × always |
| :-- | :-- | :-- | :-- |
| peek (old default) | **+1.65 [+1.10, +2.20]** | +0.70 [-0.11, +1.48] | **-1.23 [-1.88, -0.49]** |
| blind (printed) | +0.62 [+0.11, +1.14] | -0.23 [-1.07, +0.70] | -0.27 [-1.02, +0.57] |

With the peek, blasting is one of the strongest things you can do; without it,
blasting is statistically neutral. So "blasting is too weak" is **not
supported**, and the near-zero blast rate of the trained agents is a consequence
of the escape behaviour (a player who leaves on turn one never collects the two
dynamite a blast needs), not of the blast payoff.

## 5. n-1 passes or n passes?

The printed rule is "as many players pass in a row as there are players in the
mine, minus one". `--pass-rule n` plays the alternative.

| rule | rusher win % | rusher margin | hauls ending in a stall | dragon endings | decisions / game | mirror spread |
| :-- | --: | --: | --: | --: | --: | --: |
| **n-1 (printed)** | 48.0% | +2.07 [+1.40, +2.75] | **4.8%** | 58.2% | **77.2** | 4.68 |
| n | 84.5% | +5.74 [+5.20, +6.24] | **27.6%** | 39.2% | 143.4 | 4.50 |

`n-1` is better on every operational measure: half the decisions per game, a
sixth of the deadlocks. (Both columns still show the rusher beating the scripted
bots, for the reasons in §1.)

While measuring this, a bug surfaced: `step_one_player()` — the API the
self-play driver uses — compared the pass count against `len(active_players)`
instead of `self._pass_threshold()`, so `--pass-rule` was silently ignored
everywhere except the single-learner `step()` API. Fixed, with a test.

### The lorry stall

In rule-only games the stall rate is **2.0% of hauls at a threshold of 8 rounds
and 2.0% at 40 rounds** — raising `--stall-rounds` only lengthens games. With
parked policies at the table it rises to 4.8%. It is a genuine absorbing state
of the printed rules, not a threshold set too tight. Suggested real fix: end a
haul when a full round passes with no card collected, and score it like any
other ending.

## 6. Skill vs luck

Mirror matches: every seat runs the *same* policy, so the spread is luck plus
seat position only.

| rules | mean within-game spread | games with a tied top score |
| :-- | --: | --: |
| printed | **4.92 points** | 11% |
| fixed | 4.20 points | 18% |
| `n` passes | 4.50 points | 10% |

The largest skill margin measured here is the leaver's +8.72 head-to-head, which
dwarfs the luck band. But the between-policy margins that a leaderboard reports
(0.2-4 points) are the same order as pure luck between identical agents, so
`win %` alone is not evidence of anything. Seat position is not a big factor
(mean score by seat 1.83 / 1.70 / 1.71).

## 7. What is actually worth changing

1. **Ship the controls, not just the leaderboard.** `rusher` and `leaver` are
   now first-class policies (`--agents ...,rusher,leaver`) and the report always
   prints the "where the points come from" split. Without them, any future
   tuning can be gamed the same way this report was.
2. **Fix the stall scoring.** A stalled haul currently pays the escapee and
   voids the campers. Score it like a normal ending (or pay nobody).
3. **Then reconsider the empty-escape bonus.** It is a free-ride payoff in weak
   fields, and removing it costs the game little — but it is no longer the
   headline problem it looked like.
4. **Only then tune points.** The 3/4-point haul bonus is ~60% of a typical
   10-point game, so it dominates any tuning of the -1/-2 penalties.
5. **Retrain against a pool that includes leavers.** Self-play found a
   best response to a weak population, which is exactly what it is supposed to
   do — the population, not the learner, was the problem.

## 8. Reproduce

```powershell
python -m pip install -r requirements.txt

# the two controls that make the leaderboard honest
python evaluate_balance.py --agents rule,greedy,random,cautious,rusher,leaver,checkpoints/selfplay.npz --games 300 --out reports/balance_default_rules.md
python evaluate_balance.py --agents rule,greedy,random,cautious,rusher,leaver,checkpoints/selfplay_no_bonus.npz --no-solo-escape-bonus --games 300 --out reports/balance_fixed_rules.md

# blast: peek vs the printed face-down rule
python evaluate_balance.py --agents rule,rule:rich,rule:always --games 200 --skip-mirror --out reports/balance_blast.md
python evaluate_balance.py --agents rule,rule:rich,rule:always --games 200 --skip-mirror --blind-blast-choice --out reports/balance_blast_blind.md

# pass rule
python evaluate_balance.py --agents rule,greedy,random,rusher --games 200 --pass-rule n --out reports/pass_n.md

# train from scratch (3 minutes on a laptop CPU)
python train_selfplay.py --generations 60 --games-per-generation 64
```

`reports/` is git-ignored and holds the generated leaderboards plus the JSON
behind every table.

## 9. Caveats

* Everything here is 3-player. The scoring is a race for a single +3 bonus among
  everyone who left, so player count changes the incentives and should be
  measured separately.
* `leaver` and `rusher` are crude scripted controls, not strong players. They
  are strong enough to separate "wins by playing" from "wins by being left
  alone", which is the job. A human who counts card backs should beat all of
  them.
* The learner is small (48-64 hidden units, REINFORCE). "Self-play did not find
  a better policy" only means "at this budget".
* Bootstrap intervals assume independent games. They are (fresh seeds), but 300
  games still leaves about ±0.25 points of uncertainty on a margin.
