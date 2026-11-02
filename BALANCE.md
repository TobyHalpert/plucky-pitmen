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
| **selfplay4 (4-player learner)** | **44.3%** | 50.7% | **+1.65 [+1.24, +2.07]** | 85% (88%) |
| **leaver (collect + leave)** | 39.0% | 46.3% | **+1.30 [+0.84, +1.78]** | 99% (99%) |
| random | 27.7% | 33.7% | +0.22 [-0.10, +0.58] | 46% (64%) |
| rule | 23.3% | 28.0% | +0.03 [-0.29, +0.37] | 64% (21%) |
| rusher | 28.3% | 32.3% | -0.11 [-0.46, +0.26] | 100% (100%) |
| cautious | 23.3% | 31.7% | -0.30 [-0.56, -0.03] | 39% (59%) |
| greedy | 18.3% | 26.3% | -0.63 [-0.85, -0.41] | 24% (65%) |

What changes with a fourth player:

* **The mine is more lethal, not more random.** Hauls end on a dragon 61.3% of
  the time (52.8% at three players), lorry stand-offs rise to 26.8% (19.2%),
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
  a **+1.65** margin and 44.3% wins while the 3-player network managed -0.31 in
  its own setting — and it is now the top row of the table, ahead of every
  scripted policy including the leaver. It is not a different algorithm; it is
  the same REINFORCE net, and the 4-player league gave it opponents worth
  copying. Head-to-head it still loses to the leaver: **+2.63** for the leaver,
  **-1.57** for the learner — the leaderboard is one seat against three `rule`
  bots, and against a field that banks loot the leaver is still king.
* **The leaver survives the extra seat.** Its margin is +1.30 against +1.19 at
  three players, and it still beats every scripted policy in both directions
  (+1.82 head-to-head against `rule`). It is now the second row, because the
  learner passed it on this benchmark — but not in a direct duel, below.

So the answer to "is this balanced" does not flip with player count: collect and
leave is the best *scripted* policy at both counts, rushing is a parasite at
both counts, and no scripted policy is dominant. What player count changes is
how much the trained agent gets out of self-play.

### Training on: 120 more generations

The 4-player learner was resumed from its 60-generation checkpoint and trained
for **120 more generations** (7,680 more games, 11,520 in total), saved as
`checkpoints/selfplay4_long.npz` and promoted to `selfplay4.npz` (the checkpoint
this section benchmarks; the old one is kept as `selfplay4_g060.npz`).

Head-to-head, both networks in the same games (200 games, seats rotated, two
`rule` bots filling), the new network is **+1.13 points per game** better
(se 0.25) and ahead in 76% of games. The benchmark win rate against three `rule`
bots went from 35% to **45%**, the margin from +0.72 to **+1.52**. So the extra
training helped — but *how* it helped is the interesting part
(`profile_agent.py`, 600 games each, one seat, seats rotated):

| habit | 60 generations | 180 generations |
| :-- | --: | --: |
| win % / margin / score vs 3 × rule | 35% / +0.72 / +1.56 | **45% / +1.52 / +2.44** |
| planning decisions per game | 14 | 12 |
| digs per game | 7.1 | **5.3** |
| ...in its own column (fronts visible) | 42% | **50%** |
| ...in the front row | 14% | 12% |
| ...mean row dug (0 = nearest the lorry) | 0.89 | 0.72 |
| ...into a face-up dragon in its own column | 0.54 | 0.46 |
| passes per game | 2.61 | 2.92 |
| **pit-cage retreats per game** | 2.32 | **2.96** |
| **lorry exits per game** | 1.97 | **0.75** |
| decisions with a dynamite pair in hand | 0.32 | 0.22 |
| blasts announced / fired per game | 0.12 / 0.03 | 0.07 / 0.02 |
| hauls ending on a dragon | 56% | 51% |
| counted as escaped (with 0 gems) | 76% (83%) | **88% (89%)** |
| gems carried out per escape | 0.21 | 0.13 |
| points/haul when a rival banks loot | -0.01 | +0.02 |
| points/haul when no rival does | +1.17 | **+1.96** |

Three things changed, and they are one thing:

1. **It stopped leaving and started camping.** Lorry exits fell from 1.97 to
   0.75 per game while pit-cage retreats rose from 2.32 to 2.96 and passes from
   2.61 to 2.92. A seat standing in a pit cage is immune to the dragon revealed
   in that round — it is marked escaped before the cards are turned — so
   "be in a cage when the mine kills everyone else" is a real line, and the
   learner found it.
2. **It digs less, shallower, and nearer home.** Digs per game fell 7.1 → 5.3,
   the mean dug row moved from 0.89 to 0.72, and the share of digs in its own
   (fully visible) column rose 42% → 50%. Consequently fewer hauls end on a
   dragon (56% → 51%) and its escape count rises to 88% — while gems carried
   out per escape fall 0.21 → 0.13. It is buying survival, not loot.
3. **The points now come from opponents' graves.** When a rival banks loot it
   still scores nothing (+0.02); when nobody does, it scores +1.96 per haul, up
   from +1.17. Read together with the escape lines, the extra training taught
   it the same lesson the leaver exploits, one row earlier in the game: let the
   others die in the mine, and be the one who is not in it.

Two honest caveats. The training curve itself is noisy and *not* monotone — the
benchmark win rate at 20-generation checkpoints during the extra run went
41.7 / 43.3 / 40.0 / 36.7 / 35.0 / 40.0% — so the headline is the paired
head-to-head, not the curve. And the crown is benchmark-specific: it tops the
leaderboard against three `rule` bots, but in a direct duel the leaver still
beats it **+2.63 to -1.57**, because the leaver banks loot before the dragon
arrives and the camper does not.

One habit did not improve and is worth watching in the viewer: it still steps
onto a **face-up dragon in its own column 0.46 times per game** (down from
0.54), where the `rule` bot does it 0.00 times. In a 100-game sample of that
habit, all 41 cases had at least one other legal action available, so it is a
policy choice rather than a forced move. What that choice is worth is the next
question, and the answer is not the one the count suggests.

### The dragon in your own column is bait, not a blunder

The habit looks suicidal and is not. The rule that decides it is displacement:

* only the **owner** of a column knows the front of a card in it — everyone else
  sees a backside, so nobody else can know the card is a dragon;
* anyone standing **deeper in the mine** may displace an opponent who occupies a
  card in that opponent's own column, and the displaced figure goes to the
  nearest pit cage *towards the lorries*;
* a figure in a pit cage **leaves the mine at the start of the execution phase**,
  before the cards are turned. It is safe from the dragon and it banks the gems
  it is carrying. The rival who pushed it is now standing on the card.

So a known dragon in your own column is a lever: stand on it, and the first
rival who gambles on that backside pushes you into a free, safe exit and then
eats the dragon himself. Note also that the blunder audit behind this section's
predecessor counted every *intermediate* planning decision, not the committed
one — a plan can be changed up to the last turn, so a figure seen on the card in
the viewer is not yet a figure that will take it. Ask the engine instead: 600
games, learner in seat 0 against three `rule` bots, temperature 0.7.

| measurement | value |
| :-- | --: |
| hauls where the learner named its own face-up dragon | 257 of 1,800 (0.43/game) |
| ...of those, hauls where some rival stood deep enough to push | 251 |
| times a rival did push it off a dragon card | 248 |
| in the 242 hauls holding such a push: a rival took the dragon | 228 (94%) |
| in those hauls: the learner escaped (in the cage) | 173 (71%) |
| times the learner ended up eating the dragon itself | 3 of 251 (1.2%) |
| hauls where **no** rival was deep enough to push | 6 |
| ...of those, the learner took the card and died | 3 of 6 |

The last two rows are the only real mistake in the whole pattern: with nobody
deep enough, there is nothing to bait, so stepping on the card means eating it.
That variant is now filtered out (`features.doomed_dragon_digs`, applied in
`action_mask`), and it costs nothing measurable — 400 games, guard on vs off:

| | win % | score | doomed digs | ate the dragon |
| :-- | --: | --: | --: | --: |
| guard on | 43.2% | +2.08 | 0 | 0 |
| guard off | 43.5% | +2.08 | 10 | 2 |

What is *not* filtered is the bait itself, because it is not a mistake. An
earlier attempt did mask every dig onto a visible dragon; it cost the 4-player
learner **8.7 points of win rate** (29.8% with, 38.5% without, 400 games) by
forbidding a legal move that pays. That mask is gone.

Two honest limits on the table above. First, it measures the `rule` bot, which
picks uniformly among the cards it can reach and therefore pushes without
looking at the backside at all — precisely the opponent a bluff is designed for.
Against three copies of the same network the same plan appears (86 of 600 hauls)
but a rival takes the dragon in only 16 of them and the learner eats it itself
13 times. Second, this is why "the points come from opponents' graves" is a
statement about this field, not a law: the trap works while the field is naive,
and a field that respects backside odds walks away from it.

What changed, in one line: the learner was not talked out of the move, the
measurement around it was fixed — the "blunder" counted every intermediate
planning decision as a commitment — and the one genuinely losing variant, the
dig nobody can punish, is now removed from the action space instead of being
argued about.

### Displacement is a tax on the seat that parks, not a weapon

The bait above only works because a push is cheap for the pusher, so the obvious
next question is whether it is cheap for the *victim*. In the viewer's seed 7 it
was not: P3 parked on its own front-row gem with three rivals deeper than it,
P4 pushed it off four decisions later at a 13% sample, and P3 — holding nothing
— left the mine with nothing. That is the whole haul gone, and it looks like the
kind of mistake a policy should not make.

Over twenty four-net games it is not a rare accident: **1.15 pushes a game**, and
**18 of the 23 victims were holding no gems** (a push always parks the victim in
a pit cage, i.e. safely out of the mine, so it only *banks* anything for a seat
that has already collected). The network does not price any of it:

| measurement, twenty four-net games | value |
| :-- | --: |
| probability on its own-column gem, while a rival could still push it | 18% (615 options) |
| ...with no rival deep enough to push it | 19% (85 options) |
| times the net took a displacement when one was on offer | 2% (0 of 495 first choices) |

So both halves are blind: the net neither pushes nor steps around the risk. The
question is whether either blindness costs points. 1,000 seed-paired games per
arm, four seats, temperature 0.7, one arm in seat 1 (`probe_displacement.py`):

| arm | win | score | pushes/game | pushed off/game | paired difference |
| :-- | --: | --: | --: | --: | :-- |
| four networks | 24.1% | +1.09 | 0.20 | 0.21 | — |
| **pusher**: takes every displacement the hand rule would | 24.3% | +0.98 | 1.62 | 0.10 | +0.2pp (se 1.9), −0.11 pts (se 0.10) |
| **cautious**: never ends a phase on a pushable own-column card, empty-handed | 20.2% | +0.83 | 0.31 | **0.04** | **−3.9pp (se 1.8), −0.26 pts (se 0.09)** |
| **front row only** | 21.7% | +0.90 | 0.28 | 0.14 | **−2.4pp (se 1.8), −0.19 pts (se 0.09)** |

Eight times the pushes buys nothing; cutting its own exposure by five buys
*worse* results. Both ends of the mechanic are a wash at best, because a push
hands the victim a safe exit with whatever it holds — and if it holds nothing it
was going to lose the haul anyway — while the pusher spends a turn to take a card
it could usually take by waiting. Parking on your own front-row card is not a
blunder: it is what collecting your own column costs.

Two caveats, both about the field rather than the rules. The hand `rule` bot
never even offers the opportunity — it digs its own column on 2.6% of its
decisions, so in ten rule-vs-rule games **not one push was available** — which is
why no table in this file had shown the mechanic's price until now. And the nets
push at 0.2 a game, so a league that pushes ten times as often might teach the
cautious side to be worth something; the arms above say it is not worth anything
against the population that exists.

### Why a push is not the sharpest move in the game

A push reads like the strongest thing on the board: you take a card out of a
rival's column, out of a rival's hand, and send it to the cage. Asked why the nets
use it so little, the answer is in three parts — what the rules actually do, what
the nets can see, and what the numbers say.

**What the rules do.** A push drops the victim into the cage at the row it stood
on, and a figure in a cage may only reach rows *shallower* than that
(``MineEnv._displace``, ``MineEnv._movement_depth``):

| pushed from | cage index | what the victim can still do |
| :-- | --: | :-- |
| row 0 | 0 | no dig at all — lorry (lose the round) or leave with what it holds |
| row 1 | 1 | rows 0 only |
| row 3 | 3 | rows 0–2, i.e. its position was all it lost |

So a push is only a *weapon* at row 0. Everywhere else it is a swap of positions.
Three things hammer the value home even at row 0:

- the pusher spends its own card of the round, and the card it takes is one it
  cannot read (only the owner sees a column's fronts) — usually the *safer* of the
  two options is a card in its own column;
- the victim's remedy is cheap: park on the lorry and it is back at the front of
  the mine next round, with the same gems;
- a victim holding cards *banks* them by being pushed, so the push saves its loot,
  and an empty-handed victim was worth nothing in the gem race to begin with. The
  push denies it a card; the pusher was going to take a card anyway.

**What the nets can see.** They cannot see the row interaction. Over 20 seeds
there were **1,798 decisions with a push on offer** (11 a game in the reasoned
case: a row-0, empty-handed victim) and the net's probability on the push barely
moved with the row:

| victim's row | 0 | 1 | 2 | 3 | 4 | 5 |
| :-- | --: | --: | --: | --: | --: | --: |
| net's probability on the push | 2.0% | 1.8% | 1.4% | 1.2% | 0.8% | 0.5% |
| decided to push | 1.4% | 1.1% | 1.9% | 2.1% | 2.0% | 0% |

The action block carried "a rival on its own column" and the row separately, so the
product was representable but never learned. Two action features spelling out the
consequence — *this push leaves the victim no card* and *this push banks the
victim's loot* — were added and a network was fine-tuned from the shipped
checkpoint for 24 generations. It pushed **less**: 0.1% of the offers, entropy down
to 0.44. So the encoder was not the binding constraint; exploration was.

**What the numbers say.** Pushing is priced against the model's own best non-push
move at the same decision, same re-roll seeds, 20 positions per bucket, 60
re-rolls per arm:

| bucket | push | push − best other move |
| :-- | --: | :-- |
| row 0, empty-handed victim | +1.60 | **−0.01 (se 0.23)** |
| row ≥ 1, empty-handed victim | +1.05 | **+0.00 (se 0.06)** |
| victim carrying cards | +1.50 | **+0.30 (se 0.25)** |

and forcing the reasoned push on one seat, 800 paired four-net games, is
**+1.6pp wins (se 1.8), +0.11 points (se 0.09)** — positive, underpowered, and
inside the noise of the earlier `pusher` arm (+0.2pp, −0.11).

None of that is the *rate* the mechanics suggest, and the honest conclusion is
the one already in "Displacement is a tax on the seat that parks": a push is worth
about what the card you would otherwise take is worth. What the analysis adds is
*where* it is not: at row 0 it is a genuine weapon, and the nets are blind to that
distinction rather than indifferent to it. Teaching it is a training question — a
scripted pusher in the league, so the population both exercises the push and
punishes the seats that park in front of it — not a mask rule.

#### The lure: the trap the rules actually build

**"Raider" means one thing here**: a figure that turned a card in a column its own
seat does not own (`column != player_index`; the engine's `home_column_dragon` is
False for it). Every figure is reset to the front of its *own* column at the start
of a round and may then step into any column, so being a raider is always a choice
made that round — never a state a seat is pushed into (a displaced figure goes to
the cage with no column at all, and phase 1 removes it before any reveal). It is
not the seat that *pushes*; that is the pusher.

Two consequences of that definition are worth having on the record. First, a raider
is **the one figure a push cannot touch**, because `_can_displace` requires the
occupant to be standing in its own column: raiding buys immunity from displacement
and pays for it with the information you cannot have. Second, over 20 four-net
games, of 319 cards actually turned:

| the figure stood in | cards turned | gem | dynamite | **dragon** |
| :-- | --: | --: | --: | --: |
| its own column | 259 (81%) | 126 (49%) | 120 (46%) | **13 (5%)** |
| somebody else's | 60 (19%) | 21 (35%) | 19 (32%) | **20 (33%)** |

A blind choice should hit a dragon at the deck's rate, one card in five, so 33% is
not the deck being cruel — it is selection: seats raid *into columns whose owners
have left*, and owners leave columns that hold dragons. The trap is visible from
this side of the table too, and a net that cannot read fronts cannot see it coming.

Your last two scenarios are the reason the push is not the sharpest weapon -- a
push gets a rival out of a card, but the *trap* gets it out of a haul. Two rules
do the work, and they interlock:

1. Within a round, **phase 1 comes before phase 2**: everybody standing in the pit
   cage leaves, and is then immune to anything revealed that round; only after
   that does every remaining figure reveal the card it is standing on.
2. A column's fronts are readable only by its owner. Everybody else sees a
   backside.

So the owner of a column can see a dragon under a rival's figure and get out of
the mine **on the very round that dragon is turned** -- collecting the haul bonus
while the raider takes −2 and every other figure still inside takes −1. Nothing
else in the game gives a seat that pairing of information and immunity.

Measured over 30 four-net games (`lure3`-style engine instrumentation, every round
recorded at the reveal):

| dragons turned by a raider standing in somebody else's column | 31 (≈1 per game) |
| :-- | --: |
| ...and the column's owner was **already out** of the mine | **30** |
| ...of those, the owner left in the *same* round as the reveal | 24 |
| ...of those, the owner had left a round or more earlier (trap left set) | 6 |
| ...and the owner was still inside and ate the −1 with everybody else | 1 |

What it pays, per haul, in the hauls where it happened:

| | same-round (n=24) | left earlier (n=6) |
| :-- | --: | --: |
| the column owner | **+2.42** | +0.17 |
| the raider that turned it | **−0.83** | −0.17 |
| everybody else | +0.71 | +2.08 |

Two readings, and the second is the interesting one:

* **The trap fires about once a game and the owner is out for 30 of the 31.** In
  10 of the 24 same-round cases the owner planned *after* the raider, so it could
  see the figure on its dragon and choose the cage knowing why: that is your
  "player A goes to the pit cage" line, and the nets already find it.
* The other 14 are the same trap without the intent: the owner was leaving anyway
  and the raider walked into a column whose owner was publicly in the cage. That
  is a misread the raider cannot fix -- it cannot see the front, and the only
  warning is "the owner is leaving", which it has no rule for.

The cost of *not* leaving is on the same table: the one owner that stayed inside
took the −1 with the rest.

#### Banking dynamite across a haul

The second scenario is real in the engine and has an exact rule: **dynamite is the
only card that survives a haul boundary**, and it survives only if you are not
dead (`_start_next_haul` keeps `DYNAMITE` and nothing else, and only for live
seats). So a seat holding two dynamite that leaves the mine keeps them and starts
the next haul able to blast; a seat that dies holding them loses them, and pays −1
as well.

It is already happening a lot. Per seat-haul at the start of a haul, over 30
four-net games:

| dynamite in hand | 0 | 1 | 2 | 3 | 4 |
| :-- | --: | --: | --: | --: | --: |
| seat-hauls (n=360) | 273 | 49 | **32** | 5 | 1 |

So **38 of 360 seat-hauls (10.6%) start with enough dynamite to blast**, without
anybody planning it. What is that worth? Paired from the same haul-start position,
seat 0, same agent dice in every arm, 42 haul starts × 8 re-rolls:

| arm | seat 0's haul points | vs plain |
| :-- | --: | --: |
| plain (no dynamite) | +0.62 | — |
| **banked two dynamite, net's own play** | +0.61 | **−0.01 (se 0.10)** |
| **banked two and spent them at the first opportunity** | +0.81 | **+0.20 (se 0.11)** |

So the bank is a real edge of about **a fifth of a point per haul** -- and only if
the blast is actually made. Held and never spent it is worth nothing, which is
also true for gems in hand. It is not a "massive" bonus: BALANCE.md §5 has a blast
haul scoring +0.41 against +0.40 for an ordinary one under the printed rule.

Which gives the answer to "should A leave to protect its dynamite?". Leaving
protects the dynamite only because death loses it, and death also costs −1 and the
haul's loot, so **leaving to dodge a likelier dragon is right and leaving purely
to bank dynamite is not**: a zero-gem exit pays nothing (the sole-escapee bonus is
withheld at zero gems) and costs every card left in the mine. The dynamite bank is
a reason to *prefer an exit you were taking anyway* over a risky last dig, not a
reason to leave early.

#### The abandoned column: the deadliest cell, and a weak threat

Once a column's owner has left the mine, nobody can read that column and nobody can
be signalled about it -- there is no owner left to read a front or say a word. If
that is true anywhere on the board it is true there, and the numbers agree. Over 25
four-net games, cards actually turned, by whose column the figure stood in:

| the figure stood in | cards turned | gem | dynamite | **dragon** |
| :-- | --: | --: | --: | --: |
| its own column | 322 | 157 (49%) | 151 (47%) | **14 (4%)** |
| a live rival's column | 68 | 22 (32%) | 23 (34%) | **23 (34%)** |
| **a column whose owner had left** | 10 | 2 (20%) | 2 (20%) | **6 (60%)** |

Directional (ten cards), but the order is the order the information predicts: own
fronts are read, a live rival is a blind draw, and an abandoned column is a blind
draw in a mine that keeps the dragons nobody has been willing to touch.

The window is real, too. In 25 games, 23 of 75 hauls had an escape, and after the
first escape the haul still played a mean of **3.4 more rounds**. So abandoned
columns are on the board for about three rounds per escaped haul, and a figure
standing in one is a live threat to everybody still inside: a dragon kills every
figure left in the mine (-1 each) and ends the haul.

The pressure you describe -- *someone raids the dead column, so the leader should
get out* -- fires almost never in the current meta: **10 cells in 25 games**, and
only **2 rounds in 25 games** had a non-leader standing in an abandoned column while
the score leader was still inside. In the one of those where the leader had a
choice it left. Outside that, leaders leave in 12% of rounds and non-leaders in 10%:
there is no learned "the leader gets cautious" behaviour at all.

And when it does fire, the pressure is worth less than it looks, for a reason that
is in the scoring and not in the dragon:

* The bonus goes to the seat with the most gems **among the escaped** (`_score_haul`).
  A leader that leaves alone takes the +3 (+4 on the last haul) *whatever* happens
  in the mine, and nobody can overtake it once it is out. So compliance is nearly
  free for it: it was going to be paid for those gems anyway.
* A wipe costs **every figure still inside the same -1** and the thrower **-2**, so
  it barely moves the standings between the survivors and moves them one step
  *against* the seat that caused it.

Put together: the abandoned column is a minefield, not a lever. It hurts whoever
*cannot* afford to leave -- the seats still chasing gems -- and it asks the leader
for something it was going to do anyway. That is also the honest reading of what
made the lure pay earlier: not the dragon's damage to other people (which is
uniform), but the **bonus secured by being the one who is already out**.

#### What a push actually takes, and when it is right

A push is legal only onto a figure standing in **its own** column, so the victim
chose the cell with the fronts in front of it. Its choice is therefore a signal, and
that changes the arithmetic from "a card for a card". Over 20 four-net games, at
every one of the **1,069 decisions where a push was legal**:

| | gem | dynamite | dragon |
| :-- | --: | --: | --: |
| card under the victim (unreadable to me, but chosen by the victim) | **52%** | 20% | 28% |
| best card left in my own column (readable) | **13%** | 70% | 17% |

| shape of the decision | count | share |
| :-- | --: | --: |
| my column has no reachable gem, the victim's card is a gem | 417 | **39%** |
| both are gems | 300 | 28% |
| my column has a gem, the victim's card is not | 254 | 24% |
| my own column has no reachable card at all | 98 | 9% |

So the "is a push worth it" question is not "is my column a coin flip" — at the
moments a push is even available, **my own column offers a gem only 13% of the time,
because it is already dug out**. The push is a gem upgrade in 39% of those moments
and an equal card plus a denial in another 28%; it is a downgrade in 24%. Among the
27 pushes the nets actually made, the card they pushed onto was a gem 41% of the
time, a dragon 30%.

The price is the dragon, and it is asymmetric in a way worth naming: **the dragon
in my own column is readable — I only ever stand on one when nothing better is
legal — while the dragon under a victim is a lottery I cannot see.** A push that
lands on one is a −2 for me, ends the haul for everyone inside, and sends the
victim to the cage where phase 1 turns it into an **escape**. The mirror image of
the lure is real: a figure standing on a dragon in its own column is *inviting* a
push, and the pusher pays for it.

Which gives the rule this section keeps arriving at, from a third direction: a push
is worth taking when **my own column has nothing to take** and the victim's cell is
probably a gem (39% + 28% of decisions), and it is a mistake when I have a readable
gem of my own (24%). That is a policy, not a mask rule — the engine already offers
the push in all those states (1,798 of 1,798 in the mask), and the agent has to
learn that its own column being empty *raises* the value of displacement instead of
lowering it.

#### Measured as denial: the victim's side of a push

The price above is measured on the **pusher's** score, which is the wrong lens if a
push is really about "not giving the rival what it wants". So it was measured again
from the victim's side: same paired design, same re-roll seeds, push against the
model's own best non-push move, this time reporting both seats and the gap to the
best rival.

Row-0 pushes (the only ones that cost the victim a dig at all), **40 positions, 50
paired re-rolls each**:

| delta (push − best other move) | value |
| :-- | --: |
| my score | +0.175 (se 0.177) |
| **the victim's score** | **−0.140 (se 0.188)** |
| my score − best rival | +0.192 (se 0.194) |
| win share | +4.1pp (se 2.7) |

and restricted to pushes on **the current score leader**, 28 positions: my score
+0.152 (se 0.229), **the leader's score +0.146 (se 0.230)** — pushing the seat you
have to beat does not cost it a point at all — win share +4.7pp (se 3.3).

So the denial is real at row 0 but small, about the same size as the gain, and it
disappears against the leader. The mechanism explains why, and it is the same fact
as the price above seen from the other side: **a push transfers one card, it does
not destroy one.** The card the victim stood on goes to the pusher; everything else
in the victim's column is untouched; the figures reset to the front every round; so
the victim arrives at the next round with a fresh choice and the same column. And
if it was holding gems, being pushed *banks* them, which is why its score does not
fall. What the push costs the victim is a **round**, and a round is cheap while the
gem it wanted is still in the mine.

The intuition would be right in a game that scored the race — "reach N gems before
anyone else" — where one card taken from the leader is worth a tempo. This one
scores *hauls*: the bonus goes to the escapee with the most gems at the end of each
haul, so a single card only matters when it changes who that is. That is why the
win share is the metric that moves (+4pp, ~1.5σ) and the victim's score is not.

### The information the table cannot spend

The one genuinely cooperative act this game would reward is a word in a rival's
ear: *"the card you are standing on is a dragon, I can see it, get off."* The
printed rules give the table no channel for it, and the cost of that is
measurable.

A column's fronts are readable only by its owner, so the case is common and one
sided: a rival raids into somebody else's column, lands on a card it cannot read,
and the owner is looking straight at it. Over 20 four-net games there were **40
such positions — two a game**. The owner's side of it is handled correctly: the
hazard reads **1.00 in 49 of 49** decisions taken in that position, and the owner
gets out (18 cages, 29 stays that mostly left later; it died exactly once), which
is right, because the dragon is certain for it and only the cage removes a figure
from the haul.

What cannot happen is the warning. In **15 of those 40 positions the raider turned
the dragon itself** — −2 for it, −1 for every other figure still in the mine, the
haul's loot gone. Roughly **0.75 hauls a game** are lost to a fact somebody at the
table could see and could not say.

The rules do offer one implicit channel: a seat can **act** its information — a
column owner leaving the mine reads as "there is something under that card" — and
the rule bot grew a reader for exactly that (``_is_signaled_danger``: the owner of
my column fled, and was not displaced, so it left on purpose). It was measured, and
it is the weaker of the two inferences: a seat that flees is equally likely to be
banking loot or waiting out a blast, so reading the *behaviour* is what
``trust_sitter`` does, and the backside-honest `reader` policy beats it (see
"Backsides have to be read"). Signalling by action is real but noisy; signalling by
declaration does not exist.

Two ways to give the table the channel, both rules changes and neither made here:

1. **A guard move.** Today a raider standing in your column cannot be displaced:
   ``MineEnv._can_displace`` requires the occupant to be standing on its *own*
   column, so the owner can evict nobody. Letting the owner push a raider off a
   card it can see is a dragon converts private information into an action the
   engine already understands — and it would make raiding a column genuinely
   dangerous, which is a real change to the game's core economy.
2. **A declaration.** Once per haul, a seat may reveal one card. That is table
   talk: a new mechanic, and the only one that would let the informed seat spend
   what it knows without spending its turn.

### Backsides have to be read

The trap above pays only against a table that watches the *player* instead of
the *backside*. The baseline `rule` policy did exactly that: `_find_displacement_raid`
treated a rival sitting quietly on its own column as proof that the card was
safe (`risk × 0.1`), which is an inference about behaviour, not a reading of the
card, and against a sitter on a visible dragon it is wrong 100% of the time.
Two things were changed.

1. **`reader`, a backside-honest opponent.** Same policy, with
   `trust_sitter=False`: the only evidence is the backside and the cards already
   known off it, so a fresh backside (2 dragons in 10) is not raided. It is not
   a weakened bot — against three `rule` bots it is indistinguishable from
   `rule` (32.0% wins, margin +0.21 both ways, 600 games; with no sitter to
   misread, the two policies take the same actions). It is now 2 of the 7
   scripted opponents in the training pool, which previously was three parts
   `rule` to one part everything else.
2. **The encoding stopped calling a visible dragon safe.** In the action
   features, a card whose front the player can see had the risk channel at 0 and
   the safety channel at 1 — so a *visible dragon* was encoded exactly like a
   *visible gem*, with only a one-hot to tell them apart. A visible front is now
   scored as the answer it is (risk 1, safety 0 for a dragon; 0 and 1 for a gem),
   and an unknown front keeps the diluted backside risk.

The 4-player net was then resumed for 60 more generations (4,800 games) with
that pool and that encoding, and evaluated on both fields, 400 games each,
temperature 0.7, learner in seat 0:

| checkpoint | vs 3 × `rule` | vs 3 × `reader` | card-risk rank | ate its own bait |
| :-- | :-- | :-- | --: | --: |
| `selfplay4_blind.npz` (before) | 42.8% · +1.08 margin | 36.2% · +0.37 | 0.382 | 0.17/game vs readers |
| `selfplay4_reader.npz` (after) | **47.0% · +1.40** | **43.5% · +0.68** | **0.270** | 0.13/game vs readers |

Card-risk rank is the mean percentile of the chosen card's dragon risk among the
cards on offer, over decisions where the risks differ: 0 is "always the safest
card", 0.5 is "blind". Reading backsides is not a detail — it is the whole
difference between a fair trade and a free one. Two caveats keep this honest:
the retrained net is *not* stronger in general (paired head-to-head against the
blind checkpoint is +0.02, se 0.05, over 800 seat-games), the gain is
field-specific; and the bluff survives against the blind field (0.30 pushes per
game, it eats its own bait 0.003 times per game), so what changed is that the
table can now punish it, not that the move disappeared.

Shipped then: `selfplay4_reader.npz` was promoted to `checkpoints/selfplay4.npz`
(superseded by the loot-gated snapshot in the next subsection; the file is kept
under its own name), the predecessor is `checkpoints/selfplay4_blind.npz`, and
`reader` is available everywhere a scripted policy is
(`make_scripted_agent("reader")`).

### Pricing the bluff, per table

"The field matters" is a claim until it is a number per decision, so
`probe_bait_value.py` forks it. Every time the learner names a move onto a
dragon it can see, one copy of the game plays that move and the other copy makes
the same policy choose again with face-up-dragon digs removed; a dragon in a
column is visible only to its owner, and the fork diverges from that one
decision alone, so the difference in the seat's final score is what the move was
worth there.

| field (3 opponents) | forked decisions | worth of the bait | forks it helped / hurt |
| :-- | --: | :-- | --: |
| `rule` | 122 | **+1.16** (se 0.24) | 42% / 14% |
| `reader` | 140 | **+0.64** (se 0.31) | 43% / 31% |
| the same net in all four seats | 156 | **-0.78** (se 0.16) | 15% / 40% |

Every fork here has a rival deep enough to push, because the variant without one
is already masked (`doomed_dragon_digs`). So the move is worth more than a point
against a bot that raids on confidence, half that against a bot that reads the
backside, and it is a loss against its own peers. The flow of the same decision
says why: over 200 games the learner is pushed off a dragon card in
**0.00** cases per game against peers — nobody takes the bait — and instead eats
the dragon itself **0.45** times a game, against **0.01** against `rule` bots.

What the learner cannot do is tell the two tables apart: nothing in its input
says whether the field punishes a bluff, and until now nothing in the engine
recorded the evidence either. That is the change. Displacements are made in the
open, so every seat can see them; the engine now counts them per player, carries
the count across hauls, and the state encoding carries each tracked opponent's
count (capped at three). "Has this table ever taken a bait?" is now answerable
from `PlayerActionContext` alone. The counter adds one feature per opponent
(82 state features, 103 inputs), so checkpoints trained before it need
`migrate_checkpoint.py`, which inserts zero columns — the migrated net behaves
exactly like the original until it is retrained.

Two interventions were then measured and **not** shipped:

| intervention | vs 3 × `rule` | vs 3 × `reader` | vs 3 × itself |
| :-- | :-- | :-- | :-- |
| hide the bait while no rival has ever displaced (rule) | -2.5pp, -0.25 | -0.2pp, -0.08 | -9.5pp, -0.74 |
| 60 more generations with the count in the input | 40.9% (was 54.9%) | 44.0% (was 44.1%) | 32.0% (was 43.0%) |

The mask turns a read into a rule and pays for it everywhere — including the
peer field, where switching it off for *all four* seats is worth 9.5 points:
at a table that bluffs back, the bluffs are load-bearing, and taking them away
from everybody helps nobody. The retrain did learn something real (it names a
bait four times less often, 0.09 against 0.41 per game, and each one is worth
+2.81 instead of +1.16) but it paid 14 points of win rate against `rule` bots
for it, so the promoted checkpoint stays `selfplay4.npz` and the run is kept as
`checkpoints/league/selfplay4_dispfeat-gen0060.npz`. The information is now in
the input; 4,800 games were not enough to make the network use it, and the
honest reading is that a rule cannot stand in for the read.

### The training signal was paying for empty-handed exits

A viewer reported seats walking into a pit cage "with no present danger
whatsoever". The frequencies are real, but the asymmetry is the news. Over 400
games each, counting only the plan that stands when the phase executes:

| seat | cage commitments / game | ...empty-handed | ...empty-handed with nothing pressing |
| :-- | --: | --: | --: |
| `rule` bot | 0.66 | 24% | 11% |
| learner (`selfplay4`) | 0.21 | **86%** | 57% |

The hand-written bot leaves more often but almost always with loot in hand; the
learner leaves less often and usually with nothing. The cause was in the reward,
not in the rules: `_reward_vector` paid `ESCAPE_SHAPING = 0.30` for *any* escape,
loot or not, and a gem is only worth `GEM_SHAPING = 0.20`. Walking out
empty-handed therefore scored like one and a half gems in training, whatever the
board said.

So the bonus now requires at least one gem banked (`test_an_empty_handed_escape_is_not_paid_for`).
Fine-tuning the shipped net for 30 generations under the corrected signal keeps
its strength and halves the habit it never should have learned:

| checkpoint | vs 3 × `rule` (800 games) | empty-handed cage walks | of those, unthreatened |
| :-- | :-- | --: | --: |
| `selfplay4_reader.npz` (before) | 47.4% · +2.67 | 0.18/game | 0.12/game |
| `selfplay4` (after, gen 30) | 48.1% · +2.61 | 0.12/game | 0.07/game |

Paired head-to-head over 800 seat-games is +0.04 (se 0.05): the same strength,
less of the habit. Reproduce with

```bash
python train_selfplay.py --players 4 --resume checkpoints/selfplay4_reader.npz \
  --generations 30 --games-per-generation 80 --seed 6161 --name selfplay4 \
  --out checkpoints/selfplay4.npz
```

Two things this does not fix, stated plainly. Running the fine-tune for 60
generations instead of 30, or dropping the escape bonus entirely, both *lost*
strength (42.5% and 39.5% against three `rule` bots, with the policy entropy
collapsing from 1.27 to 0.93), so the 30-generation snapshot is what shipped.
And an empty-handed exit is not always wrong: it collects the printed
sole-escapee bonus of +3, which the learner actually won in 11 of its 53
empty-handed exits. That is why the move stays legal and why no mask forbids it;
only the shaping that paid for it unconditionally is gone.

### "It climbed past two free cards"

That is the next thing a viewer notices, and the first question is whether it is
even a decision. It is not: planning is sequential, so what the viewer shows as
a step is a *plan*, and the plan that counts is the one standing when the phase
executes. In 300 four-player games with a network in every seat:

| observation | count |
| :-- | --: |
| steps onto a card shallower than free cards still in the mine | 3,061 |
| ...where the target card was in the mover's own column (its front visible to that seat alone) | 2,079 (68%) |
| ...a raid on a rival standing on their own card | 79 (2.6%) |
| ...no visible reason at all | 897 (29%) |
| planning phases that actually **ended** on a hidden card while a free known gem sat deeper | 19 in 250 games (0.08/game) |

The viewer draws nothing but backsides, so the 68% are invisible knowledge: the
seat is digging a card it can see the front of, which is the one thing the rules
keep private. For the rest, the depth of the card buys nothing — every figure is
put back at the front of the mine at the start of each planning phase
(`MineEnv._set_players_to_starting_positions`), and a player takes exactly one
card per round either way, and a skipped card stays in the mine rather than
being consumed — though it usually does not come back either: of 955 skips of a
free known-safe card, 111 (12%) were taken later in the same haul. What the skip
does *not* decide is the score: committing to the skipped gem instead is worth
**+0.00 points (se 0.59)** over 19 forced forks, because one gem rarely decides a
haul and the hidden card pays a gem or a dynamite often enough to cancel it.

So the move is legal, informed, and roughly free — but it *looks* arbitrary, and
that part was a real defect. The viewer now annotates every candidate with what
the deciding seat knows, and whether the standing plan is locked or tentative
(`features.action_reason`, `GameController.reason_for` / `plan_summary`). This is
the same trap as the "92 blunders" audit in the previous section: everyone,
including me, reads an intermediate planning step as a commitment.

The same question came back sharper — *"that is irrational"* — so the moves were
classified instead of explained, 250 four-player games with a network in every
seat, looking at the card the figure walks past on the next row up:

| why the shallower card was taken | share |
| :-- | --: |
| a card in the mover's own column, front visible to that seat alone | 91.5% |
| the shallower card is **dearer** than the card it passed | 4.1% |
| the two cost the same | 3.3% |
| the shallower card is **cheaper**: it is reading the backsides | 1.1% |

The first row is invisible knowledge — the viewer draws backsides, so a seat
digging its own column looks like it is guessing. The last row is the behaviour
that was asked for. The middle two rows are the ones with no defence: if two
face-down cards cost the same, walking past one to take the other buys nothing,
because **the backside is the price** and the position you end the round on does
not matter (every figure is put back at the front of the mine each round, and a
player takes one card per round either way).

So that is now a rule (`features.walks_past_a_free_card`, applied in
`action_mask`): a face-down card is dominated when a deeper free card is at
least as safe, and the dominated ones are removed unless that would empty the
action space. Measured on the promoted checkpoint, 4 players, seed-paired
800-game cells:

| field | rule off | rule on | delta |
| :-- | :-- | :-- | :-- |
| 3 × `rule` | 54.9% · +3.12 | **56.4% · +3.18** | +1.5pp |
| 3 × `reader` | 44.1% · +2.34 | **46.1% · +2.51** | +2.0pp |
| 3 × `leaver` | 22.5% · +1.22 | 21.2% · +1.21 | -1.2pp (noise) |

With the rule in, the same audit finds the "dearer" and "same price" rows down to
1.0% and 0.2%, i.e. what a spectator still sees is almost entirely a card whose
front only that player is allowed to look at.

### Four nets parked on their lorries

Seed 7, network in every seat: all four figures stood on their lorries at the end
of the first planning phase and the haul ended there with nobody having taken a
card. That is not a rare accident of one seed — over 200 games with the shipped
checkpoint in all four seats, **93 lost the first haul that way**, and it was a
sampling artefact in the strict sense: under the argmax (the viewer's "Greedy"
box) it happened **0 times in 200**.

What the move is worth, read off the rules rather than argued about: standing on
a lorry collects nothing, and it does not protect the figure either. A woken
dragon kills **every** player still in the mine, the ones on their lorry
included — only a pit cage takes a figure out of the haul. So the lorry has one
use, refusing the card under your own figure, and that is worth a round *only*
when that card is a dragon the seat can see. Everywhere else it trades a card
(eight in ten are a gem or a dynamite) for nothing.

It is also a coordination trap, which is what makes the whole table do it: each
seat dodges for itself, so four seats on lorry is not four mistakes in isolation
but one outcome. And the seat-level numbers say where the habit came from — over
300 games the learner parked **2.49 times a game** against `rule` bots (0.46 of
them in the first planning phase of a haul), *and won 48.1% doing it*. Parking
was not costing it points against a field that also leaves loot behind, so
nothing in the reward was pushing back.

`features.lorry_skips_a_card_worth_taking` (applied in `action_mask`) hides the
lorry unless the seat is standing on a dragon it can see, and never empties the
action space. A/B, 4 players, learner in seat 0, seed-paired:

| field | rule off | rule on | first haul lost to a stand-off | any haul |
| :-- | :-- | :-- | :-- | :-- |
| 3 × `rule` (800) | 48.1% · +1.82 | **56.2% · +2.81** | 22.5% → 32.9% | 20.5% → 29.9% |
| 3 × `reader` (800) | 37.5% · +1.02 | **44.1% · +1.77** | 17.9% → 23.2% | 18.1% → 22.2% |
| 3 × `leaver` (400) | 19.5% · +0.73 | **24.5% · +0.81** | 7.8% → 0.0% | 7.7% → 0.1% |
| the same net in all four seats (400) | 37.8% · +1.81 | 37.8% · +1.86 | **29.8% → 1.2%** | 29.2% → 0.8% |

Three things in that table are worth reading carefully.

1. **The habit was the field's, and fixing it pays: +8.1 points against `rule`
   and +6.6 against `reader`.** The seat stops parking, takes its card, and the
   points follow because the card is usually worth something.
2. **The stand-off rate against scripted bots rises, and that is the control
   being the control.** In every one of those stand-offs the learner had already
   left; the parked seats were bots, whose policy this repo does not mask. What
   changes is that the learner no longer joins them.
3. **In a field of its own kind the stand-off disappears (29.8% → 1.2%) and the
   win rate does not move** (37.8% both ways, +1.86 against +1.81). A table of
   copies splits a fixed pot, so removing a symmetric mistake helps everybody
   equally — it is the spectator-facing defect that goes away, not a new edge.
   Seed 7 under the rule: haul 1 ends normally with 4 gems on the table instead
   of nobody collecting anything.

Closing the lorry was the right fix for the lorry and the wrong fix for the
habit: the next seed the viewer opened had all four figures in **pit cages**
instead, which is the subject of the next subsection.

### An escape has to bank something

The lorry rule closes one way of throwing a round away, and the viewer found the
next one immediately: **seed 7, network in every seat, every figure in a pit cage
at the end of the first planning phase, not one gem collected** — and the same
thing in the next haul, so two of the three hauls banked nothing at all and the
game (5–4–1–1) was decided by the last haul alone, which is exactly the "skill
becomes luck" outcome the design is supposed to avoid.

The chain is worth reading in the log. Seat 3 was **displaced** into a cage by a
raid in the first planning phase — ordinary, aggressive play — and the escape
rule of the previous section counted "somebody is already in a cage" as a
*threat*: seat 2 read it as danger and left, seat 1 read two caged seats, seat 0
read three. Four figures in cages, no cards turned, +1 each. It is the same class
of mistake as the lorry: a rule that was written about danger, applied to a table
state that is not dangerous.

What an escape is actually worth: the cage is the only exit that takes the figure
out of the haul *before* the cards are turned and banks the gems in hand. So it
is worth a round for exactly two reasons — there is loot to bank, or there is
something to escape from (a blast is charged, or the seat can see a dragon about
to be turned: the card under its own figure, or a card in its column with a rival
standing on it). Everything else is a wasted round, and "somebody is in a cage"
is not on the list. That is `features.cage_banks_nothing`, applied in
`action_mask`.

Measured on the viewer's own harness (`gui_controller.GameController`, four
neural seats, temperature 0.7, 150 games = 450 hauls; `probe_escape.py`):

| escape rule | hauls thrown away | games with one | gems collected per haul | seed 7 |
| :-- | --: | --: | --: | :-- |
| as first written (a cage is a reason to flee) | **40.0%** | 76.0% | 0.89 | two hauls in cages with nothing banked, 5–4–1–1 |
| without the cage/herd clause | 4.9% | 13.3% | **1.91** | haul 1 collects 7 gems, 4–6–0–0 |
| only a rival who has *left* counts | 5.8% | 16.7% | 1.52 | 7 gems over two hauls, 0–2–−1–−1 |
| **an escape must bank a gem (shipped)** | **2.9%** | **8.0%** | 1.63 | 7 gems over two hauls, 3–4–0–−1 |

"Thrown away" is a haul where at least three seats escaped and the table
collected no gems at all. A third of all hauls were being thrown away before
this, and three quarters of games contained one.

The price, against scripted fields (4 players, learner in seat 0, 1,600 games per
cell, seed-paired):

| field | as first written | shipped rule | paired difference |
| :-- | :-- | :-- | :-- |
| 3 × `rule` | 56.0% · +2.89 | 52.3% · +2.68 | -0.21 points (se 0.06), -3.7pp wins |
| 3 × `reader` | 44.1% · +1.79 | 40.3% · +1.70 | -0.09 (se 0.06), -3.8pp |
| 3 × `leaver` (800) | 22.6% · +0.77 | 25.6% · +0.81 | wins more |

So the old rule was **worth about a fifth of a point per game**, and it earned
them by joining herds of empty-handed escapes: the printed scoring gives a sole
escapee +3 even with no gems, and a tie for most gems +1 each, so being the
fourth seat into a cage is cheap insurance against a haul other seats were going
to end anyway. That is not skill and it is not what the game is about — it is a
way to skip a third of the hauls — so the points are the cost of the fix, and
they are small. In exchange, seed 7 plays a real haul 1 (2, 1, 0, 1 gems), loses
haul 2 to a dragon in round 0, and lets haul 3 finish 1–1–1–0: every haul is
played rather than skipped.

That game is read move by move — what each seat knew, the probability the network
put on the move, and what one move was worth from that position — in
**[SEED7.md](SEED7.md)**, which is also where the sample noise behind the viewer
(47% of moves are not the network's first choice) is quantified.

### A cage is insurance, and the table's hazard was not an input

The rule above leaves one case open, and the viewer found it too: **seed 7, a
seat holding a gem walks to the cage while nobody is standing on a card that can
be a dragon.** Nothing on the table can blow, so the escape buys nothing. The
seat can leave later in the same planning phase, once a rival has moved onto a
card, and it can leave later in the haul with more gems in hand — and the haul
bonus compares gem *counts*, so a gem banked in round 0 is the weakest claim a
seat can make.

What a cage actually buys is insurance against the one thing that destroys loot:
a dragon, which ends the haul and scores every figure still in the mine as dead.
Priced that way the escape is right when the table is dangerous (three rivals
each on a 20% card is ~50% that the mine blows this round) and wrong when it is
not. Seed 7, every escape priced with 60 re-rolled continuations of the same
position (`analyze_game.py --forks`):

| table hazard at the escape | what the cage was worth | escapes |
| :-- | :-- | --: |
| 0.00 – 0.22, nothing on the board can blow | −0.35 points on average (+0.27, −0.12, −0.18, −1.38) | 4 |
| 0.44 – 1.00, a card or a blast that can blow | +0.53 points on average (+0.88, +0.00, +0.72) | 3 |

The network could not make that distinction, and not because it is hard: **the
hazard was not in its input.** The state vector carries the seat's own column
and a twelve-number block per rival — position, column, score, card count, blast
flag, displacement count — but never the card *under* a rival, and the board
enters the policy only through the candidate-action block. There was no way to
represent "three of us are standing on cards that can be a dragon". Two seats,
one at a table hazard of 0.00 and one at 0.56, looked identical to the encoding
on the escape decision. Among 471 chosen escapes in 60 sampled four-net games,
**11% were taken at a table where nothing could blow** — and 49 of those 54 were
a single gem banked.

The fix is in three parts, and each part is measured:

1. **`features.round_dragon_hazard(ctx)`** — one minus the product, over the
   surviving figures, of the risk of the card each is standing on, and a flat
   1.0 whenever a blast is charged. It is appended to the state vector
   (`HAZARD_FEATURE` = index 82, `N_INPUT_FEATURES` 103 → 104) and
   `migrate_checkpoint.py --append-hazard` widens checkpoints that predate it.
   It is deliberately a *lower* bound: a rival is free to walk onto a card
   mid-round, so a table where nobody stands on a dragon-capable card is more
   dangerous than zero. (What it counts as dangerous was wrong at first — see
   the correction at the end of this section.)
2. **`cage_banks_nothing` reads it.** One gem in hand is no longer enough on its
   own: the cage needs a hazard above 0.15, or a charged blast, or a dragon the
   seat can see about to be turned. The reason this is right and not just
   tidy — leaving is a decision that *repeats*, so waiting one move gives the
   same information and keeps the gems. Worth, 1,000 paired four-net games with
   the tested seat's escapes masked against the same seat unmasked:

   | checkpoint | paired difference | escapes blocked |
   | :-- | :-- | --: |
   | the previous one (`selfplay4_prehazard.npz`) | +0.02 points (se 0.04), −0.6pp wins | 0.71/game |
   | retrained with the hazard as an input | **+0.10 points (se 0.04), +1.5pp wins** | 0.79/game |

   The rule is worth something only once the net can *use* the round it buys,
   which is the whole point of the feature: it waits, watches, and leaves in the
   same phase once the danger is on the board.
3. **The checkpoint is retrained** from the shipped one — the new input starts at
   zero weight, so this is learning what to do with it, not starting over:
   30 generations with the feature in place, then 20 more under the rule above.
   Against the previous checkpoint, 800 paired four-net games with the tested net
   in seat 0: **+4.5pp wins (se 2.0), +0.45 points (se 0.10)**. Against scripted
   fields: 3 × `rule` **51.8% · +3.27** (was 48.2% · +2.60), 3 × `reader`
   **41.0% · +2.22** (was 33.6% · +1.41).

Seed 7 now plays a different game — 82 decisions, 13 escapes and not one of them
at a table where nothing could blow, scores 5–4–0–1. The checkpoint that played
the game analysed in [SEED7.md](SEED7.md) is kept as
`checkpoints/selfplay4_prehazard.npz`, so that analysis stays reproducible.

The residual is honest: about five escapes a game are still empty-handed. The
rule allows them whenever a card or a blast can blow, and an empty escape still
takes +3 when it is the only exit — that is the game's own bargain, insurance
paid for with a round, not a defect in the model.

#### Correction: the hazard was priced on the wrong cards

The viewer came back with one case from seed 7: **haul 1, planning round 1**,
P1 holding one gem and standing on the gem it could see in its own column, no
blast charged, no dragon face up — and it walked to the cage at 89%. The rule
above allowed it, because the table hazard read **0.61**: three rivals, each
standing on a face-down card.

That reading was wrong, and the game says why. A rival standing in **its own
column** can see the card under it, and stepping away from a dragon is always
legal — the pit cage is reachable from any card — so it will not turn one. The
figures that can wake a dragon are the ones standing on a card *nobody at the
table can see*: a raid into somebody else's column, plus a charged blast.
Re-rolled from that exact position 200 times, the haul ended on a dragon **1%**
of the time, against a believed 61%. The escape itself was worth −0.21 ± 0.24
points against staying to collect: not a blunder, but insurance bought at a price
eight times too pessimistic.

`figure_dragon_risk` now charges only the raiders. That makes the same clause
*safe*, which the earlier mask tests could not show — they were run with the
miscalibrated hazard, and raising the bar then cost points (−0.19 at 0.45,
−0.34 at 0.60) because those thresholds were forbidding escapes at tables that
merely looked dangerous. With the raider-only hazard, 1,000 paired four-net games
per row:

| escape clause (hazard ≤ 0.15 masks the cage) | score | wins |
| :-- | :-- | :-- |
| the checkpoint before this change, off → on | −0.04 (se 0.07) | −0.3pp |
| retrained under the corrected feature, off → on | −0.05 (se 0.08) | +2.9pp (se 1.7) |

The checkpoint was retrained again, 20 generations under the corrected feature:
**+0.03 points (se 0.09)** against the one it replaces in 800 paired four-net
games — parity, because the point of the change is a truthful input rather than a
stronger player — and 54.2% · +3.50 against three `rule` bots (1,000 games),
38.5% · +2.21 against three readers (the predecessor's 41.0% is inside the noise,
se 1.6). Seed 7 no longer contains an escape at a table where nothing visible can
blow.

#### Correction: an empty escape with a gem one step away

The next report from the same game: **P3 leaves through the cage in the first
planning phase, with no danger on the table.** There *was* a danger — P3 was
standing on a dragon in its own column, and only the owner of a column sees its
fronts, so the viewer draws a plain card — but the escape was wrong anyway. P3
held nothing, and a gem it could see sat one row away. The rule treated "a dragon
I can see is under me" as licence to leave the haul; it now also needs the seat to
have **no card it can see is safe** to step onto instead. Priced from that
position: the gem **+2.59 ± 0.12**, the empty escape **+2.23 ± 0.10** — and P3
took the gem at its next decision in the same phase, so the cage was a plan it
abandoned. Worth, 1,000 paired four-net games with the clause on one seat against
the same seat without it: **+0.00 points (se 0.03)**, i.e. free.

The rest of the class stays legal, and that is measured too: forbidding every
empty escape taken while *any* card is reachable — including a card the seat
cannot see — costs **−0.11 points (se 0.04)**, so stepping onto an unknown card
instead of leaving is a real trade rather than a wasted round.

#### Whose escape you are watching

Three reports in a row (this section, its predecessor, and #15) said the trained
AI leaves the mine through the cage with no danger on the table, and each time the
all-neural game of that seed had no such move. The reason is the viewer's table.
It is mixed unless "AI in every seat" is ticked: `--policy neural:selfplay4` puts
the network on P1 and leaves P2–P4 on the built-in `simple` bot, and with no
`--policy` every seat is that bot. The scripted bot escapes far more often, and
now that every escape prints what it banks and what it runs from, it says why:
over 60 default-viewer games it takes **1,362 escapes, 444 of them with no gem in
hand, 158 of those in a haul's first planning phase**, and every one of them now
carries its reason — *`pit cage, banks 0g (P2 is on my dragon)`*, a dragon in its
own column with a rival standing on it, a card only that seat can see.

Those gem-less escapes are not a bug either. Removing them from the bot (an
escape banks nothing, so stay and dig) costs it **0.76 points a seat and wins 32.8%
of tables against a 50% null** over 400 paired games, because fleeing before a
dragon it can see turns a sure −1 into 0 or, as sole escapee, +3. What looked arbitrary
is a private read. The all-neural table escapes far less often: seed 7 plays 94
decisions with fourteen, five of them with nothing in hand.

Both the player list and `analyze_game.py` now name the policy behind every seat,
and `--viewer` replays the mixed table, so a report can be matched to the game it
came from.

#### A pass in the cage is the escape

The mask that refuses an escape with nothing to bank only ever gated the *cage
move*. A seat already standing in a cage could take the same departure by
passing, and a second rule made that the only move it had: the lorry is masked
unless it skips a card the seat can see is a dragon, and in a cage there is no
card at all. So the mask removed the lorry and left one legal action — a pass that
ends the haul.

Seed 7, haul 1, planning phase 1 is exactly that position. P3 takes the gem it can
see at r0c2 (45%, its second choice), P1 raids that slot off a 1% sample and
displaces P3 into cage index 0, the hazard reads **0.00**, nothing is banked, and
the network takes the pass at 100% while the lorry was the better move: **lorry
+0.93 ± 0.11 against pass +0.55 ± 0.10** over 200 re-rolls from that position.

The two moves are now decided together: from a cage, an escape that banks nothing
is refused and the lorry keeps the haul alive; an escape that banks something or
answers a charged blast drops the lorry and keeps the pass. Over 1,200 paired
four-net games that is 25.1% / +1.70 before and 25.8% / +1.62 after — **+0.8pp
wins (se 1.7), −0.09 points (se 0.08)**, a wash, and the reason to ship it is that
the old rule did not *let* the seat choose. The lorry standoff does not come back
with it: **0 wasted hauls in 900** in both arms (BALANCE.md, "Four nets parked on
their lorries").

#### Correction: the row you take a card at decides what a raid can do to you

An earlier version of this paragraph argued the opposite of what follows, and it
was wrong about the mechanism. It priced taking the front gem of your own column
against the bottom one — **+0.13 ± 0.16**, a wash — and concluded the front gem is
fine because "a displacement banks the loot; the figure leaves with the gems it
carries". That part is true and beside the point: what a displacement does is end
the *phase*, and the row the card was taken at is the cage index it leaves the
figure with. From a cage only rows **shallower** than the cage index are legal
again (``MineEnv._movement_depth``), so

- displaced from **row 0** → cage index 0 → **no card is legal at all**; the only
  moves are the lorry and the pass, and a pass in a cage is a departure;
- displaced from **row 3** → cage index 3 → rows 0–2 are open, so the seat acts
  again and can still take the shallow gem it passed over.

Seed 7, haul 1, planning phase 1 is the same table twice: P3 is pushed into cage
index 0 and has exactly `{lorry, pass}`; P1, on the same board, has gems at r0, r1
and r3 in its column and takes **r0c0 at 55%** while the bottom gem is legal. Over
200 four-net games, **114 of 209 mid-phase raids landed on a figure standing in
row 0** (a deep figure can only be raided by one standing at the front of the
mine) and **136 of those 209 — 65% — left the victim with no card move at all**.

So the two orders are not the trade the earlier section claimed. Taking the deep
gem first is the same gem, raidable by fewer seats, and if a raid does land the
seat can still take the shallow one *in the same phase*: deep-first can bank two
cards in a phase, shallow-first at most one. The rest of the codebase already
prefers deeper for face-down cards (`walks_past_a_free_card`, which masks a
shallower dig behind an equally-priced deeper one, and justified it with "ending
the round on a deeper card buys nothing" — true *across* rounds, since every round
resets the figures to the front, false *within* one). This is the same rule for
cards the seat can see: ``strands_the_figure``.

**What it costs, measured.** 2,000 paired four-net games with the rule on seat 1
and three identical opponents: 25.6% / +1.59 with it against 25.7% / +1.60
without — **−0.1pp wins (se 0.9), −0.01 points (se 0.046)**. Mechanism check over
400 games: raids on that seat's cell 127 → 115, and of those, phases it could not
recover from 77 → 75. The frequency is the honest weakness of this section: a seat
is raided mid-phase about 0.3 times a game, so the option this restores is worth
little on average and the rule cannot be justified by its score. It is shipped on
the argument instead: the mask should not hand the network a move that is
dominated inside the position it is in, and a network that has not learned the
engine's cage-index rule cannot be relied on to see the difference itself.

#### A face-down card is not a trade against a gem you can see

Seed 7, haul 1, planning phase 1: P1 stands at row 2 of its own column, whose
fronts read gem (r0), gem (r1), dynamite (r2). It takes **r1c1** — an empty cell
in P2's column, face down, backside "~", **25% dragon** — at 4%, while the gem at
r1c0 is legal and was the model's own first choice at 59%.

Two facts decide this, and neither needs a simulation.

The first is what the move *buys*: nothing. Displacement only touches a figure
standing on its own column, and P2 was standing in **P3's** column at the time, so
r1c1 was an empty cell — no raid, no push, no sabotage. The move is a card and
nothing else.

The second is the branch structure. A front in the seat's own column is
information nobody else has: it is a gem or a dynamite, never a dragon. A card
still face down may be a dragon — two per backside — and the seat that turns it is
the dragon player. Take the gem and walk the branches of the alternative:

| the face-down card turns out | against taking the known gem |
| :-- | :-- |
| a gem (~40%) | equal — one card either way |
| dynamite (~35–40%) | worse: unscored, and worth only blast material (≈0.2 points, §5) |
| a dragon (25% here) | far worse: −2 for the turner and the haul ends |

Weak dominance in every branch, so it holds at every table: the rule needs **no
threshold**, unlike everything else in this file. It fires only when the seat has
a gem it can see in its own column, and only against face-down digs that displace
nobody — a raid on a rival sitting on its own column stays legal, because pushing
that rival is worth something the card is not (``strands_the_figure`` and the
displacement section cover that). It is the same argument as
"walking past a free card", with the one card whose price cannot go wrong:
the card the seat can see the front of.

The rule that resulted is ``digs_past_a_known_gem``. What it changes about the
*policy* is nothing — the model already preferred the gem 59% to 4% — and that is
the point: the six rules above exist because a sampler at temperature 0.7 turns a
4% tail into a visible move several times a game, and a dominated action left in
the mask is a dominated action the viewer will eventually watch.

**What this costs on screen, honestly.** The seed-7 game reshuffles: 146 decisions
against 106, scored 0–6–0–1, and haul 2 becomes a four-way escape at zero gems
(a rival's blast puts three seats in cages, they park on their lorries, the haul
ends as a standoff, and the four-way gem tie pays **+1 each**). Without the rule
that haul is an instant dragon instead, and the same four-way tie is paid from a
different table. Both versions squander the haul; the scoring that pays a
zero-gem mass escape +1 apiece is a separate defect, listed here rather than
patched quietly.

#### A gem you can see beats a dynamite you can see

Seed 7, haul 1, planning phase 2, four rows on the board. Three of the four seats
opened the phase by walking onto a **dynamite** in their own column while a card
they could see, one row deeper, was a gem:

| seat | own column, top to bottom | first move | its own first choice |
| :-- | :-- | :-- | :-- |
| P1 | gem, gem, dynamite, **gem** | dynamite at r2 (44%) | gem at r3 (56%) |
| P2 | —, gem, dynamite, **gem** | dynamite at r2 (70%) | dynamite at r2 |
| P4 | dynamite, —, **gem**, dragon | gem at r2 (correct: its r3 is a dragon) | — |
| P3 | —, dynamite, dynamite, dragon | face-down card in a rival's column | — |

Two face-up cards in one column are the only choice in this game with no
uncertainty in it, and they are not a trade. A gem is scored: the escapee with the
most gems takes the haul bonus (3, or 4 on the last haul). A dynamite is not
scored at all, and one on its own does nothing — a blast needs two, and the blast
deals three cards off the deck of which the seat keeps two, ending the haul if one
is a dragon (46% of blasts in §5 woke one). Leaving the dynamite costs nothing
either: it is face up in the seat's own column and can be taken in a later round,
whereas the gem below it becomes unreachable the moment a figure stands above it,
because movement is one-way.

The same one-wayness decides ties. Two cards of the **same front** are also not a
trade: the deeper one is raidable by fewer seats (a rival can only reach a card it
stands strictly above), it leaves more of the column reachable afterwards, and a
raid on it strands the figure less far up. `strands_the_figure` therefore ranks by
depth inside a front, and `takes_the_gem_over_the_dynamite` ranks gem above
dynamite, with dragons excluded from both — a dragon the seat can see is decided by
the bait rule.

**What a first move looks like now, in the same phase:** P1 and P2 open on the
deepest row, both on a gem they can see (100% and 99%); P3, whose column has
dynamite at r1 and r2 and a dragon at r3, opens on a face-down card in a rival's
column (it holds nothing, and 40% of face-down cards are gems against a certain
dynamite); P4 banks a gem into the cage at a 22% hazard — the insurance rule of
"A cage is insurance", not a card choice. Not one of them opens on a dynamite it
could see while a gem sat in the same column.

On this rule, "always go to the deepest row first" is *nearly* right, and the one
place it was wrong is worth spelling out. Movement is one-way, so a deep position
is the reversible one while a shallow one is the commitment; the deepest card is
also the only one nobody can raid you out of. What looked like the exception —
P3's and P4's bottom card is a dragon **they can see** — is not an exception at
all, and the earlier version of this paragraph was wrong to call it one:

#### A visible dragon is a waiting square you can step off

A card is turned only when the round resolves with a figure still standing on it,
so a visible dragon is not a trap by itself. Standing on it costs a decision and
buys the safest square in the mine, and the seat walks off at its next decision.

The question is whether a next decision is guaranteed, and the engine answers it
exactly. A round resolves on ``n - 1`` consecutive passes, and a pass is legal
only *after* the seat has planned in that round. So if anybody who acts after you
is still unplanned, it cannot pass, the streak cannot complete, and your next
decision arrives:

- **first decision of the round, not last in the order** — everything behind you
  is unplanned and must make a planning move, so the round cannot end before your
  turn. Safe: ``can_wait_on_a_dragon``.
- **any later decision** — every seat has planned, so the other ``n - 1`` can pass
  in a row and the round ends with you still on the dragon: −2 to you, the haul
  over, and everybody else in the mine dead.
- **first decision, last in the order** — the other seats have already planned and
  can pass in a row without ever coming back to you.

The middle and last cases are why the sketch version of this argument ("somebody
else still has to move, so I always get another turn") is not enough: three
consecutive passes cover three of the four seats. Traced on seed 7's first round,
where the order is P4, P1, P2, P3 — P4, P1 and P2 plan and then pass in a row, and
**P3 gets one decision in that round, not two**. A seat in that position sitting
down on a visible dragon never gets to stand up.

So ``doomed_dragon_digs`` no longer masks every unpushed dragon dig; it masks the
ones taken when the seat has no waiting square, which is the exact condition
above. On seed 7 nothing changes — it plays the same 94 decisions and scores
0 / −1 / −1 / 5 — but the mask now permits the waiting square at **258 of 1,598
decisions** over seeds 0–11, and the network takes a dragon it can see in **136**
of them (most are still the bait case, where a rival deep enough is invited to
push the figure into a cage and eat the dragon itself).

#### The escape the hazard rightly authorises

Seed 7, haul 1, planning phase 2, P4 (its first decision of the round): it has a
gem in hand and takes the cage at 70%. The board shows no dragon anywhere — every
front but P4's own column is face down — and the hazard reads **0.22**. Where it
comes from, seat by seat:

| figure | standing on | its own column? | risk |
| :-- | :-- | :-- | --: |
| P4 (self) | the front of the mine | — | 0.00 |
| P1 | row 3, column 1 — its own, and it can read it | yes | 0.00 |
| P2 | row 3, column 2 — its own, and it can read it | yes | 0.00 |
| P3 | **row 2, column 1 — P1's column** | no | **0.22** |

That is the one danger the rule is for, and it is the inverse of the mistake fixed
in "the hazard was priced on the wrong cards": P1 and P2 stand on their own cards,
so they can see them and will step away from a dragon rather than turn one — priced
at zero. P3 **raided into P1's column**, so it is standing on a card nobody
standing there can read, and the deck still leaves 22% of that backside a dragon.
If it turns, every figure still in the mine dies.

The card was, in this deal, a gem — 78% of the time it is — so no dragon turned in
200 re-rolls of the rest of that round. That is a fact about the deal, not about
the price: the seat's 22% is the honest number before the card is turned.

What the numbers say about the move itself, 200 paired re-rolls from that
position: cage **+1.84 ± 0.15**, the gem P4 can see in its own column **+1.88 ±
0.16**, a face-down card in a rival's column **+1.98 ± 0.16**. The escape is the
worst of the three, and inside the noise of all of them. There is no rule to write
here, and the reason is worth stating: taking the certain gem first and escaping at
the next decision is a real trade, not a free improvement, because P4's dig resets
the pass count and the other three seats can then pass in a row and resolve the
round with P4 still in the mine — the same mechanics that make a dragon a waiting
square for a seat that acts early in the order.

So this is the one configuration where a cage with no visible danger is not a
modelling defect: the danger is a rival's raid on an unreadable card, it is priced
from the seat's own information, and escaping at 22% with a gem in hand is worth
what staying is worth.

The viewer also stops being the only one who cannot see the danger: `action_reason`
now spells out what an escape banks and what it is running from —
`pit cage, banks 0g (the dragon under me)`, `pit cage, banks 1g (a rival's blast)`,
`pit cage, banks 1g (25% the mine blows this round)` — so the move on screen
explains itself instead of reading as an arbitrary walk out of a quiet mine.

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

The blast and threat numbers in this section and in §6 were measured on the
60-generation 4-player checkpoint (`checkpoints/selfplay4_g060.npz`). The
180-generation network in §4 blasts even less (0.02 fired per game) and has
fewer chances to (0.22 decisions with a dynamite pair, against 0.32), so the
conclusion is unchanged by the extra training.

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

That first cut was too blunt, and the sharper measurement below replaces it.
Backsides *do* carry information — a backside whose dragon is already face up
has fewer dragons left in it — but the effect is a few points rather than an
order of magnitude, and the right keeper rule is the opposite of "avoid that
backside". What is *not* in doubt: the ~40–50% risk of ending the haul is what
the blast costs.

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

The keeper choice is a real effect, and it runs the opposite way round to
"avoid the backside whose dragon you can see". A backside with one dragon
already face up is *diluted*: on these very states the engine's count puts a
drawn card of that backside at **27.3%** dragon risk against **39.1%** for a
backside where no dragon has been seen yet. So in an A/B/B draw with a known
A-dragon, the A is the card to keep — which is exactly what `preferred_blast_choice`
already does, and why the "informed" row below is the best of the four. Keeping
two of a *different* backside than the known dragon, as first proposed, is the
worst of them (trap trigger, 600 games, 376-385 blasts per row):

| keepers | blasts / game | woke the dragon | margin vs 3 × rule |
| :-- | --: | --: | :-- |
| backside risk (default: keep the diluted one) | 0.64 | **164/385 (43%)** | -0.08 (0.12) |
| *different* backside than the known dragon | 0.63 | 180/376 (48%) | -0.18 (0.12) |
| same backside as the known dragon | 0.63 | 171/380 (45%) | -0.09 (0.12) |
| at random | 0.63 | 174/380 (46%) | -0.16 (0.12) |

How much is there to win at most? Over the 429 blasts this arm takes, keeping
the two cards the count likes best fails **50.4%** of the time; keeping the two
it likes least, **60.3%**. So a perfect backside rule is worth about ten points
of failure probability — worth capturing (and captured), but not what decides
whether to blast.

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

## 7. Three engine bugs — and one bug I had to withdraw

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

**The bug found while pricing the lure: a table with nothing to reach cannot
close its round.** A round closes when `n−1` consecutive seats pass, and a seat may
only pass after it has planned. Neither the pit cage nor the lorry is a pass, so
two seats caged at index 0 (cage index 0 leaves no dig legal) and two parked on
their lorries (no dig legal either) bounce cage-to-lorry forever: every choice is
a legal no-op, `consecutive_passes` never reaches the threshold, the round never
executes, and the haul neither progresses nor ends. Seen live in a four-net game,
where the runaway `planning_turns` reached 128 and the `int8` observation raised
`OverflowError` mid-game.

The breaker is a last resort, not a rule: at `HANG_BREAKER_TURNS = 64` decisions
with no card reachable, the round is closed. No round in 40 four-net games needs
more than **45** decisions (busiest legitimately-seen round: 28), so nothing that
could already finish a round changes — **seed 7 is identical (94 decisions,
0 / −1 / −1 / 5)** and the whole suite is green.

What I deliberately did **not** do is the honest fix, because it is a rules
decision and it moves the balance: "a figure is placed once per round, so a second
cage/lorry choice is that seat's pass" — or its equivalent, closing the round as
soon as nothing is reachable. Both make the bounce end at the moment it starts,
and both change finished games: seed 7 becomes **101 decisions, 0 / +1 / −1 / 3**,
and the empty-handed escape starts paying again (**rusher +1.19** against its
`< 1.0` guard). That is the same defect as the haul-scoring parasite, dressed as a
tie-break: *who escapes this round* would be decided by a no-op loop. Worth fixing,
but as a rule, with the scoring change that goes with it.

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
7. **Do not chase displacement.** It reads like the sharpest move in the game —
   eject a rival off its own card and take it — and it is worth nothing: eight
   times the pushes (1.62 vs 0.20 a game) move the win rate by +0.2pp, and
   refusing to stand where you can be pushed costs 2.4–3.9pp (see "Displacement
   is a tax on the seat that parks").
8. **Fix silent aborts before trusting any benchmark.** An engine bug scored
   whole games as all-zero (§7). Any head-to-head table produced before it is
   suspect, and the reports now print the abandoned-game count for exactly this
   reason.
9. **Price the escape with the table's hazard, and put the hazard in the
   input.** A cage is insurance against a dragon and nothing else, so the only
   reason to spend the round is a round that can blow: one gem in hand is not a
   reason (the seat can leave later, in the same phase or later in the haul).
   Measured, the danger-free escapes are worth −0.35 points and the dangerous
   ones +0.53 (§4.5). The deeper defect was informational — the encoding could
   not represent "somebody is standing on a card that can be a dragon" — and
   fixing that is worth +0.45 points against the previous checkpoint, a fifth of
   which is the escape rule itself.

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

# how often a seat parks on its lorry, and how often a haul is lost to that
python probe_lorry.py --games 200

# escape rules: how often a haul is thrown away by a herd of empty escapes,
# and what each variant is worth against scripted fields
python probe_escape.py --games 150
python probe_escape.py --field rule --strength-games 1600

# what a bluff onto a dragon you can see is worth, per field (forked rollouts)
python probe_bait_value.py --games 300 --fields rule,reader,selfplay4

# does the learner read a backside when it picks among unknown cards?
python probe_backside_reading.py checkpoints/selfplay4.npz --games 600

# widen a checkpoint saved before the per-opponent displacement feature
python migrate_checkpoint.py checkpoints/selfplay4_reader.npz

# widen one saved before the table-hazard feature (the append case)
python migrate_checkpoint.py old.npz new.npz --append-hazard

# what one move was worth, from the identical position, and why it looked right
python analyze_game.py --seed 7
python analyze_game.py --seed 7 --forks --fork-samples 60 --forks-max 40

# the hazard fix: 30 generations with the feature, then 20 under the escape rule
python train_selfplay.py --players 4 --resume checkpoints/selfplay4_prehazard.npz --generations 30 --games-per-generation 80 --seed 6161 --name selfplay4_hazard --out checkpoints/selfplay4_hazard.npz
python train_selfplay.py --players 4 --resume checkpoints/selfplay4_hazard.npz --generations 20 --games-per-generation 80 --seed 6162 --out checkpoints/selfplay4.npz

# is blasting worth it, does the AI threaten, does it react, is the threat real
python analyze_blasts.py --checkpoint checkpoints/selfplay4.npz --players 4 --games 240 --out reports/blasts4_noblastleague.md
python analyze_blasts.py --checkpoint checkpoints/selfplay4_blast.npz --players 4 --games 240 --out reports/blasts4_blastleague.md

# train from scratch (3 minutes on a laptop CPU, ~4 for four players)
python train_selfplay.py --generations 60 --games-per-generation 64
python train_selfplay.py --players 4 --generations 60 --games-per-generation 64 --out checkpoints/selfplay4.npz
# the extra 120 generations in section 4 (resume, then compare the habits)
python train_selfplay.py --players 4 --resume checkpoints/selfplay4.npz --generations 120 --games-per-generation 64 --seed 7001 --out checkpoints/selfplay4_long.npz
python profile_agent.py --checkpoints checkpoints/selfplay4_g060.npz,checkpoints/selfplay4_long.npz --games 600
# ... and the same, but with 60% of the league blasting on sight
python train_selfplay.py --players 4 --generations 60 --games-per-generation 64 --league-blast-mode always --league-blast-prob 0.6 --out checkpoints/selfplay4_blast.npz
```

**Which network is in the repository.** `checkpoints/selfplay4.npz` is tracked
(everything else under `checkpoints/` is ignored, apart from its predecessor) so
the viewer has something to play out of the box. It is the checkpoint from the escape work in §4.5 — the
table-hazard input, the escape rule, and 50 resumed generations on top of the
previous run — and it wins 51.8% against three `rule` bots and 41.0% against
three `reader` bots (500 games each, seat 0, 0.7 temperature). The checkpoint it
replaced, which is the one every table before §4.5 was measured on, is kept as
`checkpoints/selfplay4_prehazard.npz` and tracked for the same reason.

**A checkpoint is only valid for the feature layout it was trained on.** The row
features are normalised by `MAX_ROWS` and the observation carries one cell per
row, so any change to the board geometry means retraining every checkpoint; a
change to the *encoding* (as with the per-opponent displacement count) can be
bridged once with `migrate_checkpoint.py`, which zero-pads the added inputs and
leaves the behaviour bit-identical until the net is trained again —
`PolicyNet.load` names that tool in its error rather than failing on a shape
mismatch. The numbers above were all produced after
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
