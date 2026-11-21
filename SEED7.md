# Seed 7, move by move

One game, four neural seats (`neural:selfplay4`), the seed the viewer plays:

```powershell
python analyze_game.py --seed 7                    # every decision, what the seat knew
python analyze_game.py --seed 7 --forks            # what each flagged move was worth
python analyze_game.py --seed 7 --greedy           # the same seed with the sampler off
```

> **Update: the cage bar moved after this read.** The escape rule used to allow a
> seat holding a gem to leave whenever the round's hazard was above 0.15 — a bar
> below the 0.2-0.3 that a four-seat mine reads every round, so it was no bar at
> all. It is now 0.45 (BALANCE.md, "One gem in hand is not a reason to leave
> either"): the checkpoint that shipped the bar change played seed 7 as
> **−1 / 0 / 3 / 2**, with haul 1
> collecting 3-4-0-2 gems instead of a 1-1-1-1 bail-out. The move-by-move read
> below is of the earlier game, which is why it is quoted with the checkpoint that
> played it.

> **Update: the weights were retrained after this read too.** The escape rule change
> is in the mask, so it needed new weights behind it; the promoted checkpoint (20
> more generations, 600-game head-to-head at +0.34 a seat) plays seed 7 as
> **−1 / 1 / 5 / −1**, haul gems `[[3,3,1,1],[1,2,2,1],[0,0,0,1]]`, 53 decisions
> flagged for a look instead of 109. Both of the first two hauls end with all four
> seats out and every one of those exits banks at least one gem; the only empty
> exit in the game is P4 stepping off a rival's charged blast in haul 3. The
> details are in BALANCE.md, "The nets were retrained under the new bar".

> **Which game this is.** The checkpoint that plays `--seed 7` today is not the
> one that played the game read below: while this analysis was being made, two of
> the escapes in it were taken at a table where nothing could blow, which turned
> out to be a hole in the *encoder* (the hazard a cage insures against had no
> input) rather than in the search. Fixing that — a `round_dragon_hazard` state
> feature and an escape rule that waits for it — retrained the checkpoint
> (BALANCE.md, "A cage is insurance, and the table's hazard was not an input").
> The shipped checkpoint now plays seed 7 differently: **94 decisions, fourteen
> escapes, nine of them banking a gem** — including a
> correction to the cage itself: a figure in a cage leaves the mine when the
> cards are turned, so a pass taken there is a departure, and the mask used to
> remove the lorry and leave nothing but that pass (BALANCE.md, "A pass in the
> cage is the escape"). The move the viewer flagged (haul 1, round 1: P1 holding a gem and
> standing on a gem it could see, caging at 89%) is gone twice over — the escape
> rule now recognises that table as calm, and the hazard it reads was corrected
> to price only the cards nobody at the table can see (0.61 believed, 1% realised
> over 200 re-rolls; BALANCE.md, "A cage is insurance"). The
> game read here is the one the previous checkpoint played, kept as
> `checkpoints/selfplay4_prehazard.npz`, and every command above reproduces it
> with `--policy neural:selfplay4_prehazard`. What the file is for does not
> change: it prices individual moves and separates policy from sample noise.

The numbers below come from those three commands. Everything the tool prints is
read from the position *before* the step, so a decision that ends a haul is not
filed under the next one, and the replay of the recorded decisions reproduces
the game exactly (`recorded decisions replayed: [3, 4, 0, -1]`).

Until the viewer fix in the same change the viewer handed the seats that open a
haul ahead of P1 (P2 opening haul 2, for instance) to the environment's built-in
`simple` bot instead of their networks, so a seed opened in the viewer before
then played slightly differently; `GameController.reset(..., advance=False)` is
what lets the tool own the opening move as well.

## The game

| haul | outcome | gems P1 P2 P3 P4 | left the mine | points |
| :-- | :-- | :-- | :-- | :-- |
| 1 | continued after 2 rounds | 2 1 0 1 | all four | P1 +3 |
| 2 | **dragon in round 0** | 0 0 0 0 | P2 only (no gems) | P2 +3, P3 −2, P1/P4 −1 |
| 3 | escaped after 2 rounds | 1 1 1 0 | all four | P1/P2/P3 +1 each |

Final scores **3, 4, 0, −1** — P2 wins. Three moves decided that:

* **Haul 1, P4 displaces P3 (13%).** P3 has put itself on its own gem in the
  front row and then passed twice. P4 wanders through two other columns
  and then steps onto P3's card, which is legal only because P3 is standing on a
  card in P3's own column: P3 is pushed into a cage and out of the haul with
  nothing, P4 takes the gem. The haul ends with P1 on two gems to P2's and P4's
  one.
* **Haul 2, P3 takes a 25%-dragon card (36%).** Its own top choice was *pass*
  (64%) on a 22%-dragon card. The sample drew the dragon. The haul is worth
  nothing to anybody: P3 −2, P1 and P4 −1, and P2 — who left empty-handed after
  reading the dragon under the rival standing in its own column — collects the
  sole escapee's +3.
* **Haul 3, P3 displaces P4 (4%).** Same pattern as the first: P4 sits on its
  own gem, P3 ejects it, takes the gem. P4 finishes the haul with nothing.

## Is the behaviour rational?

Mostly yes, and in one place no. Each of the five points below is a measurement,
not an impression.

### 1. What you watched is a *sample* of the policy, not the policy

72 decisions, **34 of them the network's own first choice (47%)**. The rest:
9 second choices, 15 third, 5 fourth, and 9 outside its own top four. Over
twenty seeds it is 49% (1374 decisions). The sampler runs at temperature 0.7,
so a seat's top choice is typically only 20–30% — a 10–15% action is one flip of
four or five, not a preference.

Sampling matters because the argmax game is a different game:

| | sampler at 0.7 | argmax |
| :-- | :-- | :-- |
| final scores | 3, 4, 0, −1 | **5, 2, 2, 2** |
| points scored in the game | 6 | 11 |
| dragon | haul 2, round 0 | none |
| hauls nobody gets anything out of | haul 2 (dragon) | haul 3 (four cages, 0 gems) |

Neither game is clean, and the argmax one is not automatically the better game —
the gems banked actually fall from 7 to 5. What it does show is what the policy
believes this seed should look like: no dragon at all, and a winner on five
points. What the sampler produced instead is a game in which a third of the game
evaporates on one draw.

The argmax game's own defect is a different one: haul 3 ends with all four seats
in cages and nothing banked. That is not the empty-handed herd of `8cd3265` — P2
announces a blast and the other three leave rather than sit through it — but it
is still a haul nobody profits from, and it deserves a look on its own.

### 2. "It walked past its own visible gem" is *not* the irrationality it looks like

This is the pattern that jumps out of the transcript: turn after turn a seat
with a certain gem in its own column takes a face-down card in someone else's.
It happens 26 times in this game.

It is a trade, not a blunder, and both arms of the trade are real. Taking a
rival's card is worth up to a two-point swing — they lose a gem, you gain one,
and the +3 goes to whoever has the most — while taking your own is worth exactly
one. Against that, a face-down card is 40% gem, 40% dynamite, 20% dragon before
the backsides are read, and the dragon takes everything.

Evidence that it is a trade and not an error:

* Over twenty seeds, in the 524 positions where a known safe gem was legal, the
  network made it its first choice 19% of the time and put 18% mass on it. The
  hand-written `rule` policy — the baseline every table in BALANCE.md is
  measured against — takes the gem **3%** of the time. The net is the *more*
  human-intuitive of the two.
* Paired counterfactuals (same position, 60 re-rolled continuations, one move
  changed) split both ways. Ten decisions in this game took something other than
  a gem it could see and legally take: the gem would have been worth more in four
  of them, less in three, and the same in three. **Net +0.37 points for the
  moves that were made** — on this evidence the gem is worth no more than what
  the seat did instead.
* The worst gem decision in the game is P4 in haul 2, taking its own column's
  face-down card instead of the gem two rows nearer the lorry (−0.63). The best
  is the same seat's cage in haul 1, which beat the known gem by +0.88. The rest
  are inside the noise.

What *is* worth criticising is the near-indifference: 20% on a certain gem and
20% on a face-down card is a coin flip between a sure gem and a gamble on a
dragon, and at temperature 0.7 near-indifference becomes exactly that.

Across all 31 counterfactuals at the end of this file the sampled moves come out
at **−0.07 points** in total — statistically nothing. The moves the sampler made
were worth, on average, exactly what the moves it passed over were worth. What
wrecked this game was not a bias but a variance: one draw out of 72.

### 3. Displacement looks like the blind spot — and is not worth exploiting

The sharpest move in this game is not a dig. It is a move onto a card an
opponent is standing on in that opponent's own column: the rival goes to a pit
cage, and everyone in a cage leaves at the end of the planning phase, so the
rival is out of the haul — keeping what it already holds, collecting nothing
more — and the card it was standing on is yours. That is why the front row is a
dangerous place to park: the rival you eject there has usually collected nothing
yet, and the gem you take is the one you were both after.

* In twenty seeds there were **495 positions where a displacement was legal**.
  The best one was the network's first choice **zero times**; mean probability
  **2%**. The hand `rule` policy takes it far more eagerly — though in
  rule-vs-rule games the chance never arises, because `rule` barely digs its own
  column (2.6% of its decisions), so nobody is ever standing where it can be
  pushed.
* In seed 7 the two times it happened, both off the sampler's tail (13% and 4%),
  it was the best move on the board by re-rolled value: **+1.07** over pass and
  +0.52 over the cage for P4; **+1.57** over pass and +1.37 over the cage for
  P3. Together they decided who won two of the three hauls.
* The other half of the blind spot is that the network is easy to displace. In
  twenty seeds the probability it puts on a gem in its own column is **18% when
  a rival can still push it and 19% when it cannot**, i.e. it does not price the
  risk at all. And the risk is not rare: **1.15 pushes a game**, with **18 of 23
  victims holding no gems**.

So P4's push and P3's parking both look like the two ends of the sharpest
mechanic in the game. `probe_displacement.py` prices them over 1,000 seed-paired
games per arm (four seats, one arm in seat 1):

| arm | win | score | pushes/game | pushed off/game |
| :-- | --: | --: | --: | --: |
| four networks | 24.1% | +1.09 | 0.20 | 0.21 |
| **pusher**: takes every displacement the hand rule would | 24.3% | +0.98 | 1.62 | 0.10 |
| **cautious**: never parks on a pushable own-column card, empty-handed | 20.2% | +0.83 | 0.31 | **0.04** |
| **front row only** | 21.7% | +0.90 | 0.28 | 0.14 |

Eight times the pushes buys **+0.2pp** of win rate (se 1.9). Cutting its own
exposure by five costs **−3.9pp** (se 1.8) — and even the narrow "do not park in
the front row empty-handed" costs 2.4pp. The mechanic is a tax on whoever parks,
and parking is what collecting your own column costs; the network's 2% is a
correctly low price, not an unexploited edge. **Nothing to fix here.** What *is*
worth doing is keeping the caveat alive: the league pushes 0.2 times a game, and
against a field that pushed ten times as often the answer could change.

### 4. The way it actually lost the game was one 36% sample

P3 in haul 2, on a 22%-dragon card in a rival's column, was offered a 25%-dragon
card in another rival's column at 36%. Its own first choice was pass (64%). The
sample took the card, the card was a dragon, and the haul was gone:

| option | value for P3 (60 re-rolls) |
| :-- | :-- |
| the card it took (`r0c1`) | −0.08 ± 0.16 |
| pass, keeping the card it already stood on | **+1.98 ± 0.27** |
| the cage | +0.90 ± 0.22 |

The policy was right and the sampler was wrong. Nothing in the network's
preferences needs fixing here — but every move you watch has that 30–50% chance
of not being the move the network wanted.

### 5. The exits look right

Seven cage moves, **six with a gem in hand** — the `cage_banks_nothing` rule
shipped in `8cd3265` allows all seven, and two of them (`h1r1 P1` and `h3r1 P1`,
one gem in hand, table hazard 0.00: nobody standing on a card that could be a
dragon) are the ones the rule as it stands today forbids — that is the change
described at the top of this file. The seventh escape is the best read
in the game: P2 leaves in haul 2 with no gems, which the rule allows because a
rival is standing on a card in P2's own column and P2 can see that the card is a
**dragon**. P2 left; P3 turned it; the sole escapee's +3 went to P2. That is
exactly the "backsides should be read" behaviour the rule was written to permit.

No blast in 72 decisions, and that is not a policy statement either: no seat ever
held more than one dynamite in this game.

## What I would change, and what I would not

1. **The viewer.** 47% of the moves in this game are not the ones the network
   would defend, so an **argmax / temperature** switch in the viewer seems worth
   having before the next round of feedback: `analyze_game.py --greedy` already
   replays any seed that way, and the same seed then plays a visibly different
   game.
2. **Nothing on displacement.** It reads like the sharpest thing in the game and
   it is worth nothing either way (`probe_displacement.py`, 1,000 paired games
   per arm): a seat that takes every push the hand rule offers wins +0.2pp, and a
   seat that refuses to stand where it can be pushed loses 2.4–3.9pp. The
   network's 2% is not an unexploited edge.
3. **The gem-versus-raid trade.** I would not touch it. The net already takes its
   own gem six times more often than the hand rule does, and the paired numbers
   are a wash.

## Appendix: the 31 counterfactuals

Sixty re-rolled continuations per branch, same position and same dice; positive
means the move that was made beat the alternative. `own top` is the seat's own
first choice; `safe gem` is a card in its own column it can see is a gem.

```text
h1r0 P4  card r2c0 +1.73  vs own top  card r1c3 +1.47   +0.27
h1r0 P1  card r2c1 +1.17  vs own top  card r0c0 +1.47   -0.30
h1r0 P2  card r2c3 +1.78  vs own top  card r0c1 +1.60   +0.18
h1r0 P3  card r1c0 +1.82  vs own top  card r0c2 +1.65   +0.17
h1r0 P2  card r1c3 +2.80  vs own top  pass      +2.63   +0.17
h1r0 P2  card r1c3 +2.80  vs safe gem card r0c1 +2.70   +0.10
h1r0 P4  card r1c2 +2.00  vs own top  pass      +1.70   +0.30
h1r0 P2  card r0c0 +2.07  vs own top  pass      +2.03   +0.03
h1r0 P2  card r0c0 +2.07  vs safe gem card r0c1 +1.73   +0.33
h1r0 P4  card r0c2 +1.72  vs own top  pass      +0.65   +1.07
h1r0 P1  card r0c1 +1.22  vs own top  pass      +1.92   -0.70
h1r1 P1  cage      +1.13  vs safe gem card r1c0 +0.73   +0.40
h1r1 P4  card r2c1 +1.00  vs own top  cage      +1.13   -0.13
h1r1 P4  card r2c1 +1.00  vs safe gem card r1c3 +0.93   +0.07
h1r1 P4  cage      +1.37  vs safe gem card r1c3 +0.48   +0.88
h2r0 P1  card r2c2 +3.27  vs own top  card r0c0 +3.93   -0.67
h2r0 P2  card r1c2 +1.25  vs own top  pass      +1.32   -0.07
h2r0 P3  card r1c3 +1.05  vs own top  pass      +0.35   +0.70
h2r0 P4  card r0c0 +0.52  vs own top  pass      +0.27   +0.25
h2r0 P4  card r0c0 +0.52  vs safe gem card r0c3 +1.15   -0.63
h2r0 P2  card r0c3 +1.95  vs own top  pass      +0.52   +1.43
h2r0 P1  card r0c2 +1.77  vs own top  pass      +3.08   -1.32
h2r0 P3  card r0c1 -0.08  vs own top  pass      +1.98   -2.07
h3r0 P3  card r2c2 -0.03  vs own top  card r1c2 -0.37   +0.33
h3r0 P3  card r2c2 -0.03  vs safe gem card r0c2 +0.13   -0.17
h3r0 P1  card r0c2 +3.35  vs own top  card r0c0 +2.97   +0.38
h3r1 P1  cage      +2.80  vs safe gem card r0c0 +2.75   +0.05
h3r1 P2  card r2c0 +3.20  vs own top  card r1c1 +3.48   -0.28
h3r1 P1  card r1c3 +3.03  vs own top  card r1c0 +3.22   -0.18
h3r1 P1  card r1c3 +3.03  vs safe gem card r0c0 +3.37   -0.33
h3r1 P1  cage      +2.77  vs safe gem card r0c0 +3.10   -0.33
```

Totals: **−0.07 points** for the moves that were made — against the seat's own
first choice, −0.44 over 21 comparisons (11 better, 7 worse, 3 level); against
the safe gem, +0.37 over 10 comparisons (3 better, 4 worse, 3 level). Standard
errors are 0.14–0.34, so only the rows of ±0.6 or more are individually solid:
P4's displacement (+1.07), its cage in haul 1 (+0.88), P2's raid in haul 2
(+1.43), P1's −1.32 in the same round, P1's raid in haul 1 (−0.70), P4's gem
refusal (−0.63), P2's first haul-2 dive (−0.67) and P3's dragon (−2.07).
