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
  the trained self-play agent **+3.36 points**.
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
| rusher | 59.0% | +4.93 | **+3.29 [+2.75, +3.86]** | 100% (100%) |
| leaver (collect + leave) | 52.3% | +4.08 | +2.27 [+1.72, +2.82] | 99% (94%) |
| random | 43.7% | +2.17 | +1.04 [+0.55, +1.51] | 57% (69%) |
| selfplay (60 gens) | 42.3% | +2.29 | +0.80 [+0.36, +1.24] | 70% (78%) |
| rule | 33.7% | +1.88 | -0.16 [-0.57, +0.25] | 61% (21%) |
| cautious | 32.0% | +0.86 | -0.37 [-0.74, +0.00] | 45% (56%) |
| greedy | 23.3% | +0.16 | -1.00 [-1.34, -0.68] | 30% (54%) |

Now split the rusher's points by what its opponents did (200 games, rusher vs
2 × rule, per haul):

| what the other two did | share of hauls | rusher points/haul |
| :-- | --: | --: |
| left the mine **with gems** | **50.7%** | **0.00** |
| woke the dragon, still inside | 29.0% | +3.29 |
| parked on their lorries, still inside | 13.2% | +3.24 |

(Measured before the lorry rule replaced the stall valve; the shares shift
slightly under the new rule but the mechanism is unchanged — in the report's
"Where the points come from" table the rusher still scores 0.00 in every haul
where a rival banks loot.)

**Every single point the rusher earns comes from hauls where nobody else
banked loot.** When the other players leave with gems, the rules compare gems
among everybody who left and the rusher's empty escape is worth nothing. This
is the mechanism the first version of this report missed.

The same split for every policy in the pool is in the report's
"Where the points come from" section, which is generated for exactly this
reason.

### The valve was part of the problem

The first version of this repository patched the lorry deadlock with a
`stall_rounds` valve: a haul in which nobody collected a card for eight rounds
was scored as "stalled", campers got nothing, and **the player who had already
escaped kept the bonus**. That is not a rule, it is a subsidy: it converts "the
campers might still have left and beaten me" into "their loot is void and I keep
the +3".

It is now replaced by an actual rule, contributed in review:

> If all players still in the mine stand on their lorry at the end of a planning
> phase, the haul is over.

This is the natural reading of the game — a lorry collects nothing, so such a
haul can never progress — and it changes the incentives rather than papering
over them. Measured effect on self-play, same budget, same seeds:

| metric | old valve | lorry rule |
| :-- | --: | --: |
| trained agent's escapes carrying gems | 0.01 | **0.28** |
| trained agent's escapes with zero gems | 99% | 78% |
| trained agent's margin vs 2 × rule | +4.14 | **+0.80** |

The learner stops farming campers and starts escorting loot out of the mine.
The valve had been paying it to keep opponents parked.

## 2. The counter: collect loot, then leave

`agents.LeaverAgent` is the cheapest policy that exploits the race: take a gem,
then walk to the pit cage. It is not clever — it picks a card, holds position,
and leaves once it holds loot.

| A | B | A top-score % | A margin [95% CI] | B top-score % | B margin [95% CI] |
| :-- | :-- | --: | :-- | --: | :-- |
| **leaver** | **rusher** | **100.0%** | **+8.72 [+8.20, +9.20]** | 41.7% | -0.41 [-1.28, +0.48] |
| **leaver** | **selfplay** | 70.0% | **+3.36 [+2.48, +4.29]** | 10.0% | -2.35 [-2.85, -1.82] |
| rusher | selfplay | 68.3% | +1.68 [+0.93, +2.47] | 61.7% | +0.70 [-0.07, +1.52] |
| rusher | rule | 68.3% | +2.90 [+1.81, +4.06] | 31.7% | -1.13 [-1.57, -0.65] |

* A single rusher against two leavers tops the scoreboard 41.7% of the time
  with a **negative** margin: it is a losing seat.
* A single leaver against two rushers wins **99.5%** of games (measured over 200
  games) with a **+8.68** margin, because the rushers leave, no further rows are
  dealt, and the leaver then picks up loot unopposed before walking out with
  more gems than the empty-handed pair.
* The trained self-play agent loses to the leaver as well (−3.00 as the lone
  seat, +3.36 for the leaver). Self-play converged on the exploit because its
  opponent pool — scripted bots and its own snapshots — did not punish it.

Under the fixed scoring (`--no-solo-escape-bonus`) the ordering is the same
where it matters: leaver vs rusher **+8.72**, leaver vs the retrained agent
**+2.71** at 78.3% top score.

## 3. Why the leaderboard lied

Two things made the rusher look dominant, and neither is about the game's
quality:

1. **The benchmark population decides the ranking.** Against the scripted bots
   the order is selfplay > rusher > leaver. Against a leaver it inverts
   completely. A margin measured against a fixed, exploitable opponent says
   nothing about whether a strategy is strong — only that it is strong *there*.
2. **The old stall valve paid the parasite** — see above. It voided the
   campers' loot while the escapee kept the bonus, so a policy that made other
   players park was rewarded twice over. The lorry stand-off rule removes that
   subsidy; the rusher's remaining +3.29 against the scripted bots comes from
   those bots' own camping, which is their mistake to make.

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

| rule | rusher win % | rusher margin | dragon endings | decisions / game | mirror spread |
| :-- | --: | --: | --: | --: | --: |
| **n-1 (printed)** | 56.0% | +3.10 [+2.44, +3.77] | 55.0% | **66.9** | 4.97 |
| n | 88.5% | +6.75 [+6.21, +7.26] | 38.1% | 79.9 | 4.77 |

`n-1` is still the better rule, but less dramatically than before the lorry
rule: the 27.6% deadlock rate that used to make `n` unusable is gone, because a
haul now ends as soon as everybody inside is parked. What remains is that `n`
lets the early-leaver run away with the game far more easily (88.5% vs 56%
wins), because requiring an extra pass gives the miners more chances to waste.

While measuring this, a bug surfaced: `step_one_player()` — the API the
self-play driver uses — compared the pass count against `len(active_players)`
instead of `self._pass_threshold()`, so `--pass-rule` was silently ignored
everywhere except the single-learner `step()` API. Fixed, with a test.

### The lorry stand-off

There is no stall valve any more. In rule-only games a haul ends in a lorry
stand-off **2.0% of the time**; across the mixed benchmark it is 20.8%, because
several of the scripted baselines genuinely park. Raising a round limit never
changed those rates, which is why the limit was the wrong instrument: the state
was absorbing, and the fix is to end the haul, not to wait longer.

## 6. Skill vs luck

Mirror matches: every seat runs the *same* policy, so the spread is luck plus
seat position only.

| rules | mean within-game spread | games with a tied top score |
| :-- | --: | --: |
| printed | **5.08 points** | 8% |
| fixed | 4.24 points | 16% |
| `n` passes | 4.77 points | 12% |

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
2. **Adopt the lorry stand-off rule** (already implemented here): if everyone
   left in the mine is on a lorry at the end of a planning phase, the haul is
   over. It removes the "wait it out" option without inventing a subsidy.
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
