# Is Plucky Pitmen balanced?

**Correction (this revision).** An earlier version of this document claimed
that "leave on turn one" is a dominant strategy. That claim was wrong, and the
error is worth keeping on the record because it is the classic benchmarking
trap. Against opponents who collect loot and then leave — which the printed
rules let them do — an empty-handed escape is worth **exactly zero**. The
rusher's big margins came from opponents who *stayed in the mine and died*, and
from a stall valve that voided those opponents' loot while still paying the
escaping player. With that valve replaced by a real rule (a lorry stand-off ends
the haul and the players still inside leave and are scored), the rusher is worth
**+0.26 points [-0.23, +0.77]** against the scripted bots — noise.

The supportable conclusions are:

* There is **no dominant strategy**. Leaving early is a *parasite*: it converts
  other players' mistakes into points, and it is worthless against a player who
  simply takes a gem and walks out.
* The best policy in this repository is the opposite of rushing: **collect loot,
  then leave**. It beats the rusher **+8.37 points, 100% top score** and beats
  the trained self-play agent **+3.10 points**.
* A leaderboard against fixed bots cannot detect this on its own, because the
  ranking depends on what the bots do. Head-to-head against a leaver is the test
  that matters.
* **Blasting is not too weak, but it is not a free win either.** It is a strong
  move on the old peeking engine (**+1.00** at four players) and a coin-flip
  under the printed face-down rule (**+0.38**, interval touching zero); against
  a field that banks loot it *loses* points (**-0.52**). That is a coherent
  design; and the fact that the self-play learners barely blast is mostly an
  artefact of the learner, not a verdict on the rule — it seldom holds two
  dynamite and cancels most of the blasts it announces (§5).
* **The threat of a blast is real — worth about 0.4 points to a non-blaster —
  and the AI never learns to react to it.** A one-line "walk to the pit cage
  when someone's blast is pending" policy recovers the whole loss; the trained
  network moves its action distribution by 0.65% (§6).
* Answering the 4-player question: the game does **not** get more random or
  less skill-driven at four players, the leaver is still the best single seat,
  and the learner trained on four players (**+1.11**, 40.7% wins) is genuinely
  better than the same learner trained on three (**-0.31**) (§4).
* **Two engine bugs** were found, and both charged the agents for them. A
  player killed by a dragon while its blast was pending left the haul unable to
  finish, so the game was scored all-zero (36 of 60 always-blaster games). And
  the action grid was sized by the *opening* board — five rows — while the mine
  keeps growing a row every round: rows 5 and deeper could not be named as an
  action and were cut out of the observation, so the agents could not dig the
  front of a deep mine (10-13% of dealt cards were stranded there). Both fixed,
  every checkpoint retrained, every number in this document regenerated, and
  every report now prints its abandoned-game count (§7).
* A third "bug" I initially reported in the same area — an unwinnable
  tie-breaker — does not exist: a haul spends one dragon and the deck is
  rebuilt every haul, so five remain against the four a tie-break can need. §7
  records the withdrawal, including the bad arithmetic in my first correction.

All numbers were regenerated after the action-grid fix and a full retrain of
every checkpoint (§7), because the grid changed what the agents could do.
4-player results are labelled as such; everything else is 3 players. Seats
rotated every game, 300 games per leaderboard row, 60 games per head-to-head
direction, 240 games per blast measurement, 95% bootstrap CIs (2000 resamples).
Every report states how many games the engine had to abandon (always zero now —
see §7, because it used to be a silent source of fake data).

---

## 1. The parasite: what "leave on turn one" is actually worth

`agents.RusherAgent` walks to the nearest pit cage and never looks at a card.
Against the shipped benchmarks it looks strong:

| policy | win % | mean score | margin vs 2 × rule [95% CI] | escapes (0 gems) |
| :-- | --: | --: | :-- | --: |
| **leaver (collect + leave)** | 46.7% | +3.26 | **+1.19 [+0.68, +1.75]** | 99% (94%) |
| random | 37.3% | +1.70 | +0.35 [-0.14, +0.80] | 58% (67%) |
| rusher | 35.7% | +2.79 | +0.26 [-0.23, +0.77] | 100% (100%) |
| rule | 31.3% | +2.33 | +0.08 [-0.36, +0.48] | 82% (24%) |
| selfplay (60 gens) | 35.3% | +1.32 | -0.31 [-0.75, +0.10] | 67% (79%) |
| cautious | 30.7% | +0.76 | -0.56 [-0.91, -0.18] | 48% (52%) |
| greedy | 24.3% | +0.18 | -1.00 [-1.30, -0.66] | 31% (51%) |

Now split the rusher's points by what its opponents did (300 games, rusher vs
2 × rule, per haul):

| what the other two did | share of hauls | rusher points/haul |
| :-- | --: | --: |
| at least one left the mine **with gems** | **65.9%** | **0.00** |
| nobody banked loot (dragon or lorry stand-off) | 34.1% | +2.72 |

(The mechanism is unchanged by the lorry rule and is still visible in every
report: the rusher scores **0.00** in every haul where a rival banks loot. What
changed is that campers now get to bank theirs.)

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

It is replaced by an actual rule, contributed in review:

> If all players still in the mine stand on their lorry at the end of a planning
> phase, the haul is over.

A lorry collects nothing, so such a haul can never progress. Crucially, ending
the haul means the remaining players **leave the mine automatically and take
part in the scoring** with the loot they are holding — the same treatment the
rules already give "only one player remains" and "a full row cannot be formed".
Standing on a lorry costs a turn; it does not cost you the haul.

That second half is what closes the exploit. Measured over the same benchmark:

| metric | old valve (campers voided) | lorry rule (campers scored) |
| :-- | --: | --: |
| rusher margin vs 2 × rule | **+2.69 [+2.13, +3.27]** | **+0.26 [-0.23, +0.77]** |
| trained agent margin vs 2 × rule | +4.14 [+3.54, +4.70] | -0.31 [-0.75, +0.10] |
| trained agent's escape rate | 99% | 67% |
| hauls ended by a lorry stand-off | 20.8% | 19.2% |

The escapee's empty +3 now competes with campers who hold two or three gems, so
the free ride disappears. Self-play notices immediately: the retrained agent is
at parity with the scripted bots instead of far ahead.

## 2. The counter: collect loot, then leave

`agents.LeaverAgent` is the cheapest policy that exploits the race: take a gem,
then walk to the pit cage. It is not clever — it picks a card, holds position,
and leaves once it holds loot.

| A | B | A top-score % | A margin [95% CI] | B top-score % | B margin [95% CI] |
| :-- | :-- | --: | :-- | --: | :-- |
| **leaver** | **rusher** | **100.0%** | **+8.37 [+7.92, +8.79]** | 35.0% | -0.92 [-1.54, -0.30] |
| **leaver** | **selfplay** | 71.0% | **+3.10 [+2.38, +3.83]** | 21.0% | -1.68 [-2.17, -1.17] |
| rule | selfplay | 57.0% | +1.68 [+0.89, +2.46] | 43.0% | +0.24 [-0.42, +0.95] |
| rusher | rule | 49.0% | +0.67 [-0.25, +1.53] | 100.0% | +1.30 [+0.99, +1.65] |
| rule | leaver | 43.0% | +0.75 [+0.06, +1.49] | 52.0% | +1.20 [+0.17, +2.22] |

* A single rusher against two leavers tops the scoreboard 35.0% of the time
  with a **negative** margin: it is a losing seat.
* A single leaver against two rushers wins **99.5%** of games (measured over 200
  games) with a **+7.98** margin, because the rushers leave, no further rows are
  dealt, and the leaver then picks up loot unopposed before walking out with
  more gems than the empty-handed pair. At four players it is the same story:
  one leaver against three rushers wins **99.5%** with **+7.94**.
* The trained self-play agent loses to the leaver as well (−1.68 as the lone
  seat, +3.10 for the leaver). Self-play converged on an empty-handed escape
  because its opponent pool — scripted bots and its own snapshots — kept giving
  it away.

Under the fixed scoring (`--no-solo-escape-bonus`) the ordering flips exactly
where the bonus is doing the work: leaver vs rusher is still **+8.37**, leaver
vs the retrained agent **+2.19** at 69.0% top score, and **every** policy now
loses to `rule` (leaver **-1.79**, rusher **-2.30**, the retrained agent
**-1.23**, `rule` itself **-0.01**). The empty-escape bonus is what makes the
parasite pay; without it, collecting and leaving is the only thing that works.

## 3. Why the leaderboard lied

Two things made the rusher look dominant, and neither is about the game's
quality:

1. **The benchmark population decides the ranking.** Against the scripted bots
   the order used to be selfplay > rusher > leaver. Against a leaver it inverts
   completely — the leaver is now the top row. A margin measured against a fixed, exploitable opponent says
   nothing about whether a strategy is strong — only that it is strong *there*.
2. **The old stall valve paid the parasite** — see above. It voided the
   campers' loot while the escapee kept the bonus, so a policy that made other
   players park was rewarded twice over. With campers scored instead, the rusher
   falls from +2.69 to +0.26 and the trained agent from +4.14 to -0.31.

The methodological fix that is now built in: always include a `rusher` **and** a
`leaver` in `--agents`, and read the "Where the points come from" table. A
policy whose points vanish whenever a rival banks loot is farming mistakes.

## 4. Three players vs four players

Everything above is three players, because that is what the original question
was measured on. The printed game seats up to five, and player count changes the
incentives: the haul bonus goes to the single escapee holding the most gems, so
a fourth player is one more rival to out-collect *and* one more body left in the
mine to die. So the same benchmark was re-run with four players and a learner
trained on four players (`checkpoints/selfplay4.npz`).

| policy | win % | top-score % | margin vs 3 × rule [95% CI] | escapes (0 gems) |
| :-- | --: | --: | :-- | --: |
| **leaver (collect + leave)** | 39.0% | 46.3% | **+1.30 [+0.84, +1.78]** | 99% (99%) |
| **selfplay4 (4-player learner)** | 40.7% | 45.7% | **+1.11 [+0.73, +1.49]** | 75% (82%) |
| random | 27.7% | 33.7% | +0.22 [-0.10, +0.58] | 46% (64%) |
| rule | 23.3% | 28.0% | +0.03 [-0.29, +0.37] | 64% (21%) |
| rusher | 28.3% | 32.3% | -0.11 [-0.46, +0.26] | 100% (100%) |
| cautious | 23.3% | 31.7% | -0.30 [-0.56, -0.03] | 39% (59%) |
| greedy | 18.3% | 26.3% | -0.63 [-0.85, -0.41] | 24% (65%) |

What changes with a fourth player:

* **The mine is more lethal, not more random.** Hauls end on a dragon 62.1% of
  the time (52.8% at three players), lorry stand-offs rise to 25.8% (19.2%),
  and a game takes 101 planning decisions instead of 64. The luck band barely
  moves: mirror-match spread 4.63 points and 15% tied top scores, against 5.01
  and 13% at three players. Four players is *more* mine, not more coin-flip.
* **The cheap escapes get worse, playing gets better.** The rusher's margin goes
  from +0.26 to -0.11 and a lone rusher tops the board only 7% of the time
  against three leavers (+8.05 for the leaver and -2.02 for the rusher, both
  directions, both intervals excluding zero). With one more rival left inside,
  there is one more player to out-collect, and an empty-handed escape competes
  with three loot bags instead of two.
* **The learner is genuinely better at four players.** The 4-player network has
  a **+1.11** margin and 40.7% wins while the 3-player network managed -0.31 in
  its own setting — and it is now the second row of the table, ahead of every
  scripted policy. It is not a different algorithm; it is the same REINFORCE
  net over 60 generations, and the 4-player league gave it opponents worth
  copying. Head-to-head it still loses to the leaver: **+2.99** for the leaver,
  **-1.04** for the learner.
* **The leaver survives the extra seat.** Its margin is +1.30 against +1.19 at
  three players, it still beats every scripted policy in both directions
  (+1.82 head-to-head against `rule`), and it is still the top row.

So the answer to "is this balanced" does not flip with player count: collect and
leave is the best single-seat policy at both counts, rushing is a parasite at
both counts, and no policy is dominant. What player count changes is how much
the *trained* agent gets out of self-play.

## 5. Is blasting too weak? No — but it is not a free win either

The printed blast rule: draw three cards **face down**, choose two, *then* look
at them. `MineEnv` handed the chooser all three fronts, turning a gamble into a
card-selection engine. `--blind-blast-choice` restores the printed rule.

`rule:always` = announce a blast whenever two dynamite are in hand. The blaster
seat runs the *same* movement policy as its opponents and differs only in that
decision, so the margin is the price of blasting and nothing else.

**Three players** (200 games per row):

| engine | leaderboard margin [95% CI] | always alone vs 2 × rule | rule alone vs 2 × always |
| :-- | :-- | :-- | :-- |
| peek (old default) | **+1.71 [+1.17, +2.20]** | +1.25 [+0.53, +1.95] | **-1.53 [-2.12, -0.95]** |
| blind (printed) | +0.39 [-0.14, +0.90] | +0.10 [-0.57, +0.84] | -0.12 [-0.81, +0.57] |

**Four players** (240 games per row, `analyze_blasts.py`):

| engine | blaster win % | blaster margin vs 3 × rule [95% CI] | blasts / game | hauls ending on a dragon | points per blast haul | points otherwise |
| :-- | --: | :-- | --: | --: | --: | --: |
| peek | 37.5% | **+1.00 [+0.59, +1.43]** | 0.78 | 65% | +1.42 | +0.34 |
| blind | 30.0% | +0.38 [-0.00, +0.76] | 0.77 | 71% | +0.41 | +0.40 |

Against a field that collects and leaves instead of parking (4 players, same
seats and seeds):

| engine | blaster margin vs 3 × leaver [95% CI] | the same policy, never blasting |
| :-- | :-- | :-- |
| peek | -0.12 [-0.50, +0.23] | -0.26 [-0.61, +0.08] |
| blind | **-0.52 [-0.88, -0.19]** | -0.26 [-0.61, +0.08] |

Read the three tables together:

1. **On the peeking engine blasting is one of the strongest things you can do.**
   That engine was wrong, and the leak is worth +1.7 points at three players.
2. **Under the printed rule it is a coin-flip against other miners** (+0.38,
   interval touching zero, and `points per blast haul` +0.41 versus +0.40
   otherwise — a blast haul is worth the same as a normal one), and a
   **losing** move against a field that banks loot (-0.52).
3. Its risk is real and measurable: across 443 hauls with a fired blast, **58%
   ended on a dragon**, killing everyone still inside, and the blaster was
   holding 2.14 gems when it happened.

So "blasting is too weak" is **not supported**: under the printed rule it is
roughly neutral in a passive field and negative in a competent one, which is a
normal risk-reward trade rather than a dead rule. §6 is about what the learner
leaves on the table by ignoring the *threat* of a blast.

### Why the learner blasts 0.03 times a game

The obvious reading — "blasting is a coin-flip, so the learner declines it" —
turns out to be only a small part of the story. `probe_blast_value.py` forks
the game at every position where a seat holds two dynamite and can move to a
card, announces the blast in one copy and not in the other, and plays both out:

| engine | blast worth from that position | when behind on gems | level or ahead |
| :-- | :-- | :-- | :-- |
| peek (blaster sees the three fronts) | **+0.49** (se 0.27), 140 forks | +0.28 (se 0.29) | +1.33 (se 0.63) |
| printed rule (choose blind) | +0.06 (se 0.28), 140 forks | -0.06 (se 0.30) | +0.59 (se 0.74) |

Note first what a blast *is*: the rules give you the card under your figure as
usual and then **additionally** three off the draw pile, of which you keep two.
It is not a substitute for a dangerous dig — you still take the card you are
standing on — so blasting cannot be used to dodge a dragon you know about, and
two dynamite are pure cost only until the moment you cash them in.

Does choosing the two keepers by backside knowledge reduce the risk? Not
measurably. Same trigger, same movement, only the choice differs, 150 games
each, under the printed rule:

| who chooses | blasts / game | blasts that woke the dragon | seat margin [se] |
| :-- | --: | --: | :-- |
| informed (backside risk) | 0.69 | **47/103 (46%)** | +0.01 (0.22) |
| at random | 0.67 | 40/100 (40%) | -0.12 (0.21) |
| never blasts (control) | 0.00 | — | -0.43 (0.21) |

Backside bookkeeping is worth nothing here: each backside carries exactly two
dragons of ten, the cards you can identify (your own column, the backsides of
collected cards) barely move the posterior for three cards off the top of the
pile, and you choose **before** looking. The ~40–50% risk of ending the haul is
what the blast costs, and it is the same either way.

### Testing "blast while a dragon is waiting for me"

The most natural reading of a blast — *two dynamite are dead cards, a known
dragon in my own column makes staying put a losing move anyway, so cash them in*
— was implemented and measured rather than argued about.
`probe_blast_strategy.py` keeps the rule bot's movement in every seat and varies
only the blast trigger (plus, in the second table, the keeper choice), 600 games
per arm, paired seeds, printed rule, committed blasts so nothing is cancelled:

| trigger | blasts / game | gems behind at the blast | woke the dragon | margin vs 3 × rule | paired vs never blasting |
| :-- | --: | --: | --: | :-- | :-- |
| never blasting | 0.00 | — | — | +0.21 (0.13) | — |
| **known dragon ahead in my own column** | 0.64 | -0.8 (40% behind) | 164/385 (43%) | -0.08 (0.12) | **-0.29 (0.14)** |
| behind on gems | 0.43 | -3.0 (100% behind) | 123/258 (48%) | +0.20 (0.13) | -0.02 (0.10) |
| always (any dynamite pair) | 0.91 | -0.7 (39% behind) | 252/545 (46%) | +0.01 (0.13) | -0.21 (0.15) |

The premise about the *cost* of holding dynamite is right, and it is in the
table: a seat that is about to blast is on average **0.8 gems behind** the
leader, and when the trigger is "somebody is ahead" it is 3.0 gems behind and
behind in 100% of the cases. Two dynamite really are dead weight. What the
numbers do not support is that a blast converts that deficit into points: from
the same behind-on-gems positions it is worth **-0.02** in expectation.

Waiting for a known dragon makes blasting **worse**, not better. A known dragon
is a free move you were about to make anyway: you walk around it, and a blast
does not spare you the card you are standing on, while the 43% chance of ending
the haul is paid in full. Blasting *because you are behind* is a wash — an
earlier 200-game sample put it at +0.36 and it collapsed to -0.02 at 600 games,
which is why these arms are paired and re-run before anything is written down.

The second half of the idea, that keeping cards of a *different* backside cuts
the risk, does not just fail to help — it is 48% against 45% for deliberately
keeping the *same* backside, i.e. noise around the same coin (trap trigger, same
600-game setup, 376-385 blasts per row):

| keepers | blasts / game | woke the dragon | margin vs 3 × rule |
| :-- | --: | --: | :-- |
| backside risk (default) | 0.64 | 164/385 (43%) | -0.08 (0.12) |
| **different backside than the known dragon** | 0.63 | 180/376 (**48%**) | -0.18 (0.12) |
| same backside as the known dragon | 0.63 | 171/380 (45%) | -0.09 (0.12) |
| at random | 0.63 | 174/380 (46%) | -0.16 (0.12) |

A dragon you can see is one card you know about; the other 29 in the pile do not
care which backside you name. Each backside carries two dragons in ten, so the
backside of a card tells you nothing about whether it *is* a dragon — the only
information a blast-keeper can use, and it is not there to use.

And the premise underneath the whole idea — *if a dragon is waiting in my column
I am losing anyway* — is not true at these tables. In **34,137** decisions where
a seat could see a dragon still ahead of it in its own column (13,986 of them
while holding a dynamite pair), the same player also had, on those very states:

| a way out | share of those decisions |
| :-- | --: |
| leave on the lorry | 100.0% |
| dig another column | 99.8% |
| retreat to a pit cage | 81.5% |
| pass | 53.8% |

A known dragon is a card to walk around, not a wall. That is why paying a 43%
chance of ending the haul to "use up" two dynamite is a loss rather than a
rescue — and it is also the honest answer to *why does the AI not blast*: there
is nothing there to rescue it from.

So why 0.03? Two measured reasons, and neither is a judgement about that
trade-off.

1. **It rarely can.** Blasting needs two dynamite in hand and a card move, and
   the learner leaves the mine early with thin loot (75% escapes, 0.21 gems per
   escape): **0.17 blastable decisions per game** when sampling, 0.00 under
   argmax, against **11.2** for a rule bot in the same seats. "Rarely blasts"
   is mostly "rarely has two dynamite".
2. **Its blasts are cancelled by its own next move.** The rules abort a blast
   if the player later picks anything but "Pass" in that planning phase. Over
   60 four-player games the learner announced **0.17 blasts per game and fired
   0.02** — 10% survive. A `rule` bot told to blast whenever it can keeps 31%
   (9.23 announced, 2.85 fired). The network can see `own_blast_pending` in its
   features, but it has not learned the commitment, so most of its blast
   announcements are silently thrown away.

And the sharpest evidence that this is under-learning rather than strategy: the
nets were trained on the **peeking** engine, where the same blast was worth
**+0.49** per decision and the always-blasting seat beat the field by +1.00 —
and they still learned to blast essentially never. A rare branch with a
33%-of-the-action-space twin is hard for a 60-generation REINFORCE run to find.
Two things would change it, neither of them a rules change: train under the
printed rule (`--blind-blast-choice`), which is the game we actually play, and
make the commitment visible to the learner by masking everything but "Pass"
while its blast is pending.

## 6. The threat of a blast: real, worth ~0.4 points, and never learned

A pending blast is a *public* threat. The blaster keeps two of three drawn cards,
and if either is a dragon the haul ends and every player still inside the mine
dies. That is a promise to end the haul with a 58% chance of a dragon — so
blasting should also work as a way to scare rivals out of the mine, and a strong
player should react to it.

The trained agents do not, and asking the network directly shows it is not a
measurement artefact. Each real decision is scored twice — once with a credible
opponent blast pending, once without, every other feature identical — and the
action distributions are compared by total variation distance over the 53
primitive actions:

| checkpoint | blasts fired / game | paired decisions | TV distance [95% CI] | shift towards leaving | shift towards the cage |
| :-- | --: | --: | :-- | --: | --: |
| `selfplay4.npz` (vs rule league) | 0.03 | 309 | **0.0065 [0.0061, 0.0068]** | -0.0015 | -0.0024 |
| `selfplay4_blast.npz` (60% of the league blasts) | 0.05 | 296 | **0.0062 [0.0060, 0.0065]** | -0.0024 | -0.0026 |

A 0.65% probability shift is nothing. Training the learner for 60 generations
against a league in which 60% of the scripted opponents blast *on sight* does
not make it threaten (0.05 blasts/game) or flinch (same 0.6% shift). Facing
blasters costs it margin — **-0.34 [-0.73, +0.06]** paired, `selfplay4.npz`, and
**-0.66 [-1.03, -0.31]**, `selfplay4_blast.npz` — and it never adapts.

That on its own does not prove the threat is *worth* reacting to, so the threat
was measured directly with scripted players. Four conditions, all on the same
seeds (so the intervals are on paired per-game differences):

* `calm` — a plain `rule` seat, nobody blasts.
* `threat` — the same `rule` seat, one opponent blasts on sight.
* `reacting` — the same seat plays `threat`: rule movement, except it walks to
  the pit cage whenever an opponent's blast is pending, then waits out the haul.
* `react_calm` — `threat` against a field that never blasts (the control that
  proves the reaction only fires on threats).

| seat | win % | margin | points | threatened decisions / game | answered with a cage move |
| :-- | --: | --: | --: | --: | --: |
| rule, calm field | 25.8% | -0.15 | +1.11 | 0.0 | 0% |
| **rule, blasting field** | **15.0%** | **-0.58** | **+0.69** | 0.0 | 0% |
| **threat, blasting field** | **21.2%** | **-0.22** | **+0.95** | 3.3 | 77% |
| threat, calm field | 25.8% | -0.15 | +1.11 | 0.0 | 0% |

* Cost of the threat to a non-blaster: **-0.43 points [-0.81, -0.06]** and 10.8
  points of win rate. The threat is real, and it is not small.
* Value of reacting: **+0.37 points [+0.12, +0.63]**, 6.2 points of win rate.
* What is left once you react: **-0.06 [-0.51, +0.37]** — statistically
  indistinguishable from a calm field. The one-line reaction neutralises the
  threat.
* Control: the threat agent against a calm field is **+0.00** different from
  plain `rule` on identical seeds, so the gain comes from the reaction and not
  from a different movement policy.

So the design question has a clean answer: **a pending blast is worth about 0.4
points/game to the blaster's rivals, and roughly that much is recoverable by the
cheapest possible counter.** The trained agent captures none of it — the
reaction is 0.65%, and the cost it pays for ignoring the threat (-0.34) is the
same order as the value on the table (+0.37). The threat is a genuine part of
the game that self-play at this budget simply does not discover, because it is
rare (0.8 blasts/game), announced early, and punishes the blaster's own haul 58%
of the time — so it never earns enough games to show up in the gradient.

## 7. Two engine bugs — and one bug I had to withdraw

Asking the AI to blast surfaced a real engine bug. It produced a silent failure:
the game could not finish, so `play_game` returned `aborted=True` with **every
score at zero**, and a benchmark that quietly averages zeros looks exactly like
a hard game.

**The bug: a dead player holding the round open.** A dragon that kills the
players still inside left `blast_pending` set on the corpse. The round-finish
guard waited for that player's blast to resolve, no player could act, and the
haul was abandoned. With four always-blasting `rule` players this hit **36 of
60 games**; every one of them was being scored 0-0-0-0 and averaged in as a
draw. Fixed twice over: the dragon cancels a dead player's blast, and only a
player who can still act is allowed to defer the end of the round.

**The bug in the action grid: a five-row mine that keeps growing.** The rules
deal a new row at the bottom of the mine every round, so a haul can reach ten
rows at three players, seven at four and six at five. `MAX_ROWS` was pinned at
5 — the three-row opening board plus two — and a card action is
`row * MAX_COLUMNS + column`, so every card in row 5 or deeper simply had no
action of its own. Worse, `PIT_CAGE` sat exactly where row 5 would have
started, so the deepest row an agent could *name* was row 4. The observations
had the same ceiling (`rows[:MAX_ROWS]`), so those rows were not even visible.

Measured over 60 rule-bot games: **23% of three-player hauls and 19% of
four-player hauls grew past five rows**, and **13% / 10% of all dealt cards**
ended up in rows that could not be played. The visible symptom was an agent
that walked *backwards*: standing at the front of a six-row mine, its only
legal card actions pointed at rows 0-4. That is the report that started this
fix — a reviewer watching the GUI saw the AI "give up space" and refuse to dig
in row 5. It was not a policy failure; the action space could not express it.

Fixed by sizing the grid to the deck instead of the opening board:
`MAX_ROWS = DECK_SIZE // MIN_PLAYERS = 10`, giving 50 card actions plus pit
cage, lorry and pass. Tests assert that every row of a grown mine has a
reachable card action, that the front row is visible through both the
observation and `player_view`, and that no card action can collide with the
pit cage. Every checkpoint was retrained and every number in this document
regenerated. The front row is now chosen in **27%** of the decisions where a
free card sits there, against **64%** for the scripted `rule` seat — the
learner is more reluctant to be deepest in the mine, but it is no longer
impossible.

**The bug I withdrew: an "unwinnable tie-breaker".** An earlier draft of this
document claimed the tie-break could run out of dragons — that after three hauls
every dragon might be held by a player, leaving contenders drawing safe cards
forever. That is wrong, and the reviewer was right twice over: a haul spends
**one** dragon, not several, because drawing a dragon ends the haul; and the
deck is rebuilt to all 30 cards at the start of every haul, so the final
tie-break sees **five** dragons (six, if the last haul ended on a lorry or a
full board) regardless of what the first two hauls did. Four is the most a
tie-break can ever need, for five contenders.

Measured over 1260 games at 4 and 5 players, 249 tie-breaks: five dragons
available in 219 of them, six in 26, and four in just 4 — those four being games
where the short-deck branch of `_prepare_blasts` had pushed leftover blast
cards, a dragon included, into a player's hand. The tightest state observed was
five dragons for five contenders. Availability never fell below four, and never
below `contenders - 1`.

My first correction of this passage was also wrong: it said "at most three
dragons are consumed", which is the arithmetic that made the guard look
necessary in the first place. What I had actually observed was the dead-blast
abort above, with the state printed at the wrong moment; I wrote it up as a
second bug before finding the first one's cause.

The engine keeps a top-up anyway — if fewer dragons remain than `contenders - 1`
it rebuilds the deck — but it is a guard, not a fix for an observed failure: it
has never fired, the flag it sets is called `tie_breaker_shortage`, and a test
asserts the invariant on real games so that a future rules change (more players,
fewer dragons per backside, a new way to lose cards) fails loudly instead of
silently.

What both fixes did change is the reporting: every report now prints `games
the engine abandoned` in its header and every measurement counts it, so an
aborted game can never be invisible again. Every number in this document was
regenerated afterwards, and the ones that moved most were the scripted
blast rows: the 3-player `rule:always` seat reads **+1.71** on the peeking
engine and **+0.39** under the printed rule, where the version of this document
written before the fixes had +1.43 and +0.38. An earlier draft also claimed
that a blasting field compressed the 4-player field into 22% tied top scores;
that was aborted games being scored as ties. The real effect is 9% either way.
Withdrawn.

## 8. n-1 passes or n passes?

The printed rule is "as many players pass in a row as there are players in the
mine, minus one". `--pass-rule n` plays the alternative.

| players | rule | rusher win % | rusher margin [95% CI] | lorry endings | decisions / game | mirror spread |
| --: | :-- | --: | :-- | --: | --: | --: |
| 3 | **n-1 (printed)** | 29.0% | -0.39 [-1.05, +0.26] | 20.5% | **64.9** | 5.01 |
| 3 | n | 4.0% | **-3.15 [-3.36, -2.92]** | 44.1% | 79.0 | 4.83 |
| 4 | **n-1 (printed)** | 34.0% | +0.33 [-0.12, +0.82] | 24.4% | **102.8** | 4.63 |
| 4 | n | 4.0% | **-2.15 [-2.35, -1.94]** | 41.6% | 115.9 | 4.53 |

`n-1` is the better rule at both counts: fewer decisions per game, roughly half
as many lorry stand-offs, and a dragon ending rather than a stalemate. It is
also gentler — under `n` the early leaver is punished far harder, because the
extra pass gives the miners time to fill their lorries with loot before the haul
is closed.

While measuring this, a bug surfaced: `step_one_player()` — the API the
self-play driver uses — compared the pass count against `len(active_players)`
instead of `self._pass_threshold()`, so `--pass-rule` was silently ignored
everywhere except the single-learner `step()` API. Fixed, with a test.

### The lorry stand-off

There is no stall valve any more. In rule-only games a haul ends in a lorry
stand-off **12.1% of the time** at three players and 14.8% at four; across the
mixed benchmark it is 19.2% and 25.8%, because several of the scripted
baselines genuinely park. Raising a round limit never changed those rates, which is why
the limit was the wrong instrument: the state was absorbing, and the fix is to
end the haul, not to wait longer.

## 9. Skill vs luck

Mirror matches: every seat runs the *same* policy, so the spread is luck plus
seat position only.

| rules | players | mean within-game spread | games with a tied top score |
| :-- | --: | --: | --: |
| printed | 3 | **5.01 points** | 13% |
| printed | 4 | 4.63 points | 15% |
| fixed | 3 | 4.59 points | 15% |
| `n` passes | 3 | 4.83 points | 18% |
| `n` passes | 4 | 4.53 points | 20% |

The largest skill margin measured here is the leaver's +8.37 head-to-head, which
dwarfs the luck band. But the leaderboard margins (+1.30 down to -1.00) are the
same order as pure luck between identical agents, so `win %` alone is not
evidence of anything. Seat position is not a big factor (3-player seats score
+1.78 / +1.81 / +1.69; 4-player +0.91 / +1.01 / +0.99 / +0.98).

Which leaderboard margins are actually distinguishable from noise? At three
players: the **leaver** (+1.19), and the two worst seats, `cautious` (-0.56) and
`greedy` (-1.00). At four: the **leaver** (+1.30), the **4-player learner**
(+1.11), `cautious` (-0.30) and `greedy` (-0.63). Everything else — `random`,
`rule`, `rusher`, and the 3-player learner — is inside the noise, which is the
honest way to read a table like this.

## 10. What is actually worth changing

1. **Ship the controls, not just the leaderboard.** `rusher` and `leaver` are
   now first-class policies (`--agents ...,rusher,leaver`) and the report always
   prints the "where the points come from" split. Without them, any future
   tuning can be gamed the same way this report was.
2. **Adopt the lorry stand-off rule** (already implemented here): if everyone
   left in the mine is on a lorry at the end of a planning phase, the haul is
   over and those players leave automatically and are scored. Ending the haul
   *without* scoring them (the first version of this patch) is just the old
   subsidy in a new coat: it took the rusher from +2.69 to +0.26 only once the
   campers' loot counted too.
3. **Then reconsider the empty-escape bonus.** It is a free-ride payoff in weak
   fields, and removing it costs the game little — but it is no longer the
   headline problem it looked like.
4. **Only then tune points.** The 3/4-point haul bonus is ~60% of a typical
   10-point game, so it dominates any tuning of the -1/-2 penalties.
5. **Retrain against a pool that includes leavers.** Self-play found a
   best response to a weak population, which is exactly what it is supposed to
   do — the population, not the learner, was the problem. The 4-player run
   (+1.11 instead of -0.31) is the same lesson again: the league decides what
   the learner becomes.
6. **Do not nerf or buff the blast.** Under the printed rule a blast is worth
   about the same as a normal haul (+0.41 vs +0.40 points) and costs the
   blaster's rivals ~0.4 points when it is pending. If anything is worth
   touching it is *visibility*: the engine already exposes a pending blast in
   the feature vector and the network ignores it (§6), which is a training
   problem, not a rules problem. A cheap fix is to seed the league with
   `threat` agents — the reaction policy measured above — so that the value of
   respecting a pending blast shows up in the games the learner plays.
7. **Fix silent aborts before trusting any benchmark.** An engine bug scored
   whole games as all-zero (§7). Any head-to-head table produced before it is
   suspect, and the reports now print the abandoned-game count for exactly this
   reason.

## 11. Reproduce

```powershell
python -m pip install -r requirements.txt

# what a single blast is worth, and whether backside knowledge helps the choice
python probe_blast_value.py --games 150

# is "blast while a dragon is waiting for me" a good rule? (triggers, paired seeds)
python probe_blast_strategy.py --games 600

# the two controls that make the leaderboard honest
python evaluate_balance.py --agents rule,greedy,random,cautious,rusher,leaver,checkpoints/selfplay.npz --games 300 --out reports/balance_default_rules.md
python evaluate_balance.py --agents rule,greedy,random,cautious,rusher,leaver,checkpoints/selfplay_no_bonus.npz --no-solo-escape-bonus --games 300 --out reports/balance_fixed_rules.md

# blast: peek vs the printed face-down rule
python evaluate_balance.py --agents rule,rule:rich,rule:always --games 200 --skip-mirror --out reports/balance_blast.md
python evaluate_balance.py --agents rule,rule:rich,rule:always --games 200 --skip-mirror --blind-blast-choice --out reports/balance_blast_blind.md

# pass rule, both player counts, both variants
python evaluate_balance.py --agents rule,greedy,random,rusher --games 200 --out reports/pass_n-1.md
python evaluate_balance.py --agents rule,greedy,random,rusher --games 200 --pass-rule n --out reports/pass_n.md
python evaluate_balance.py --players 4 --agents rule,greedy,random,rusher --games 200 --out reports/pass4_n-1.md
python evaluate_balance.py --players 4 --agents rule,greedy,random,rusher --games 200 --pass-rule n --out reports/pass4_n.md

# four players: the leaderboard, and the learner trained on four players
python evaluate_balance.py --players 4 --agents rule,greedy,random,cautious,rusher,leaver,checkpoints/selfplay4.npz --games 300 --out reports/balance4_default_rules.md

# is blasting worth it, does the AI threaten, does it react, is the threat real
python analyze_blasts.py --checkpoint checkpoints/selfplay4.npz --players 4 --games 240 --out reports/blasts4_noblastleague.md
python analyze_blasts.py --checkpoint checkpoints/selfplay4_blast.npz --players 4 --games 240 --out reports/blasts4_blastleague.md

# train from scratch (3 minutes on a laptop CPU, ~4 for four players)
python train_selfplay.py --generations 60 --games-per-generation 64
python train_selfplay.py --players 4 --generations 60 --games-per-generation 64 --out checkpoints/selfplay4.npz
# ... and the same, but with 60% of the league blasting on sight
python train_selfplay.py --players 4 --generations 60 --games-per-generation 64 --league-blast-mode always --league-blast-prob 0.6 --out checkpoints/selfplay4_blast.npz
```

**A checkpoint is only valid for the action grid it was trained on.** The row
features are normalised by `MAX_ROWS` and the observation carries one cell per
row, so any change to the board geometry means retraining every checkpoint
before re-running the measurements — the numbers above were all produced after
the §7 grid fix, on nets retrained from scratch. `reports/` is git-ignored and
holds the generated leaderboards plus the JSON behind every table.

## 12. Caveats

* Three and four players are measured; five are not, though the code supports
  it (`--players 5`). The haul bonus goes to the escapee with the most gems, so
  a fifth seat is another rival to out-collect and another body in the mine —
  extrapolate with care, do not assume.
* The blast/threat measurements use `rule` seats as the field, because they are
  the only scripted policy that collects dynamite reliably. A league of strong
  human-like players could make threats more valuable than the ~0.4 points
  measured here.
* `leaver` and `rusher` are crude scripted controls, not strong players. They
  are strong enough to separate "wins by playing" from "wins by being left
  alone", which is the job. A human who counts card backs should beat all of
  them.
* The learner is small (48-64 hidden units, REINFORCE). "Self-play did not find
  a better policy" only means "at this budget".
* Bootstrap intervals assume independent games. They are (fresh seeds), but 300
  games still leaves about ±0.25 points of uncertainty on a margin.
* The action grid is sized for the deepest board any deck can build at the
  smallest supported table (10 rows at three players). At four or five players
  the bottom rows of that grid are simply never legal; the mask costs a little
  compute and nothing else. Anything that changes the deck (card count, dragons
  per backside) or the minimum table size must change `MAX_ROWS` with it, and a
  test (`test_action_grid_covers_the_deepest_board_any_deck_can_produce`) fails
  if it does not.
