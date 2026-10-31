# Plucky Pitmen: Gymnasium Q-learning model

This project turns **Plucky Pitmen** into a Gymnasium environment for testing strategy hypotheses.

**Feel free to contribute!** I appreciate your feedback.

**Is the game balanced?** Mostly yes, and the interesting part is *why* it is
hard to measure: leaving the mine early looks dominant against fixed bots, but
is worth exactly zero against a player who collects loot and then leaves. The
measurements, the confidence intervals and the reproduction commands are in
**[BALANCE.md](BALANCE.md)**; the agents that produced them are in this
repository and can play any of the rule variants.

## Rules

The winner of a haul is the player who **leaves the mine with the most gems**.
Everyone inside the mine when the dragon is awakened loses everything.
The players must **cooperate** to avoid waking the dragon and **compete** to win the game.

During a haul, **planning** and **execution phases** alternate.
The table shows the consequences in the execution phase, depending on where a player's figure stands at the end of the planning phase:

| **Position** | **Loot** | **Dragon Awakened** | **Dragon Stays Asleep** |
| :--- | :--- | :--- | :--- |
| Card | **Receives the card** | Loses loot | Participates in the rest of the haul |
| Lorry | - | Loses loot | Participates in the rest of the haul |
| Pit cage | - | Leaves mine / <br>**Keeps loot** | Leaves mine / <br>Sits out for the rest of the haul |

Pit cages are in between each row of cards.

### Setup

There are 3 sets of 10 cards, each set sharing the same card back. Each set contains 4 gems, 4 dynamite, and 2 dragons. They are shuffled into one pile. Every player gets 3 cards, looks at them and places them into a column under their lorry.

### Planning Phase

Beginning with the starting player, players take turns until exactly **as many players pass in a row as there are players remaining** in the mine (number of players in the mine minus one). 

On their turn, a player chooses **one** of the following options:

*   **Move to a free card:** The player places their figure on an unoccupied card.
*   **Move to the nearest pit cage:** A pit cage can accommodate any number of players.
*   **Pass:** The player does not change their announced plan. If it is their turn again later, all options are open to them once more. Passing is not allowed during the first round of the planning phase.
*   **Displace an opponent:** This is only possible if the opponent is standing on a card within that opponent's own column (the color of the player figure is the same as the color of the lorry). To do this, the player moves the opponent to the nearest pit cage in the direction of the lorries and places their own figure on that opponent's card.
*   **Sit on their lorry:** Anyone who has occupied their lorry at any point during a planning phase can only switch between the last pit cage and their lorry for the remainder of that planning phase.

For the options "free card", "pit cage", and "displace", the following rule applies: The card or pit cage **must be closer to the lorries** than the row the player is currently in.

### Execution Phase

All players who are in a **pit cage** at the beginning of the execution phase leave the mine and **are safe from the dragon**. They sit out for the rest of the haul, meaning they do not participate in any further planning or execution phases and only become active again for the final scoring.

Beginning with the starting player, **everyone** who is still in the mine and not standing on a lorry **takes the card** beneath their figure and looks at it. If it is a **dragon**, the haul ends immediately—the dragon has been awakened. Otherwise, the player places the card face down in front of them. The player remains in the mine. The card backs of collected cards remain visible to everyone.

If no one has left the mine so far: Beginning with the starting player, each player draws (if possible) a **card from the draw pile**, looks at it, and places it face down in their column of the newly created row at the bottom end of the mine (see Figure 2). If someone has left the mine: No cards are added.

The **figures** of the remaining players are placed **in front of the newly created row of cards**. The starting player marker is passed clockwise. (The starting player does not necessarily have to be inside the mine.) The next planning phase begins.

### Blasting

If a player has collected two dynamites, they can announce a blast during their turn in the **planning phase**. To do this, they flip **two dynamite cards** face up and place their figure on a free card as described.

If it becomes the player's turn again during the planning phase, choosing any option other than "Pass" aborts the blast. In this case, they flip their two dynamite cards back face down and choose a different option instead. They may announce a blast on a different space. Used dynamite is consumed at the end of the planning phase.

In the **execution phase**, the player **receives the card as usual**. Additionally, the player draws **three cards from the draw pile** and places them face down in front of themselves. They must choose **two of them**. They look at the two cards. If a dragon is among them, the round ends immediately—the dragon has been awakened. Otherwise, they place the treasures face down in front of themselves. The third card is placed at the bottom of the draw pile. If there are fewer than three cards left in the draw pile, the choice is skipped and the player takes all remaining cards.

### End of a Haul

A haul **ends**
*   immediately if the dragon is awakened.
*   if only one player remains in the mine after the execution phase. This player leaves the mine automatically.
*   if a complete row cannot be formed after an execution phase. The remaining players leave the mine automatically.
*   if there are no treasures left in the mine.
*   if **all players still in the mine stand on their lorry** at the end of a planning phase. Standing on a lorry collects nothing, so such a haul can never progress: the haul ends and the players still inside leave the mine automatically, taking part in the scoring with the loot they are holding (just like "only one player remains" or "a full row cannot be formed").

Now, the **scoring** takes place:
*   Players who left the mine reveal their collected cards. The player with the **most gems** receives **+3 points** (+4 points in the final haul). If only one player has left the mine, they receive these points even without having any gems. In the event of a **tie** for gems, each player with the highest number of gems receives **+1 point**.
*   The player who **awakened the dragon** receives **-2 points**.
*   All other players who **died in the mine** receive **-1 point**.
*   A player's score cannot fall below -1 point.
*   If a player awakens the dragon (-2 points) and nobody has left the mine, the other players receive +2 points instead of -1 point. The other players are allowed to keep their dynamite.

A full game consists of **3 hauls**. If there is another haul to be played, all cards are collected, shuffled, and a new mine is set up. Players who successfully left the mine are allowed to keep their dynamite.


### End of the Game

The game ends after the third haul. The player with the most points wins.

In the event of a **tie**, the contenders for victory take turns - starting with the starting player - drawing a card either from the mine or from the draw pile. Players who draw a dragon are eliminated. The last survivor wins.

## Open Questions

* How is the balance between skill and luck?
* Is there a dominant strategy? Is blasting too weak? Should the victory points be tweaked?
* Would the game be better with n-1 passes instead of n passes?

These are not rhetorical any more: `train_selfplay.py` and
`evaluate_balance.py` turn them into experiments (see
[Self-improving AI players](#self-improving-ai-players-self-play)).

## Setup

Use a Python 3.11+ interpreter in VS Code, then run:

```powershell
python -m pip install -r requirements.txt
python train_q_learning.py --episodes 50000 --policy random
```

To watch the players move through the mine in a graphical interface:

```powershell
python mine_gui.py --policy random
```

Use **Take action** to choose the learner's destination, or **Autoplay** to run the simulation automatically. The viewer shows card backs only; opponent gems and dynamite remain hidden.

### Watching the trained agents

Any checkpoint in `checkpoints/` shows up in the **Policy** dropdown as
`neural:<name>`, and the autoplay driver then hands every decision to the
network — including whether to announce a blast and which of the three blast
cards to keep (the scripted policies never blast, so before this the viewer
could not show that part of the game at all).

```powershell
# the 4-player self-play learner against three rule bots
python mine_gui.py --opponents 3 --policy neural:selfplay4

# ... or the same network in every seat, which is the clearest view of its habits
python mine_gui.py --opponents 3 --policy neural:selfplay4 --all-seats

# play the argmax instead of sampling at temperature 0.7 (what the benchmarks do)
python mine_gui.py --opponents 3 --policy neural:selfplay4 --deterministic
```

The **AI POLICY** panel lists the network's top actions for Player 1 with their
probabilities, marking the action it actually took, so a surprising move can be
traced back to the distribution that produced it. It is empty for scripted
policies, during a tie-break, and while a blast is waiting on a card.

What to look for, given the measurements in [BALANCE.md](BALANCE.md): it should
rarely blast (0.03-0.05 per game), it should walk to a pit cage as soon as it
holds loot, and it should *not* visibly change its mind when an opponent
announces a blast (§6) — that missing reaction is the finding, not a bug in the
viewer.

Try each opponent model:

```powershell
python train_q_learning.py --policy random
python train_q_learning.py --policy cautious
python train_q_learning.py --policy greedy
```

A strategy is evidence of dominance only if it remains strong across opponent policies, player counts, and random seeds. Compare the printed `mean_reward`, `escape_rate`, and `dragon_rate`; repeat with several `--seed` values.

---

## Self-improving AI players (self-play)

`train_q_learning.py` learns **one seat** against a **fixed** opponent policy.
That is a good smoke test, but it cannot answer the questions above: the
opponents never improve, the observation is a coarse 65-number vector, and a
single `mean_reward` has no confidence interval.

This repository also ships a league-based self-play stack. It is numpy-only,
uses the existing cheat-safe `PlayerActionContext`, and puts an agent in
**every seat**:

| module | purpose |
| :-- | :-- |
| `features.py` | fixed-size, seat-invariant state + action features; no hidden information |
| `agents.py` | `Agent` interface, scripted baselines (incl. `rusher`, `leaver`, and `threat` — a policy that retreats when a rival's blast is pending), `NeuralAgent` |
| `learning.py` | masked-softmax MLP policy + value baseline, REINFORCE, Adam, save/load |
| `selfplay.py` | full-game driver: every seat controlled, blasts, tie-breaks, per-player returns |
| `train_selfplay.py` | league self-play trainer with checkpoints and a learning curve |
| `evaluate_balance.py` | benchmarks, pairwise dominance, luck analysis, markdown report |
| `analyze_blasts.py` | is a blast worth it, does the learner threaten, does it react, and is the threat real |

### Train

```powershell
python -m pip install -r requirements.txt
python train_selfplay.py --generations 60 --games-per-generation 64
```

This writes `checkpoints/selfplay.npz`, older snapshots to
`checkpoints/league/`, and the learning curve to
`checkpoints/selfplay.history.json`.

Useful flags:

| flag | meaning |
| :-- | :-- |
| `--players 3\|4\|5` | number of seats |
| `--hidden 0` / `--hidden 64` | linear policy / one hidden layer |
| `--epochs`, `--batch-size`, `--lr` | optimisation budget |
| `--mirror-prob` | fraction of games in which every seat is the current learner |
| `--league-prob`, `--league-size` | how often opponents are older snapshots |
| `--eval-opponent rule` | benchmark used during training |
| `--no-solo-escape-bonus` | play the rules variant without the free bonus for a sole escapee with zero gems |
| `--blind-blast-choice` | choose blast cards face down, as the printed rules say |
| `--pass-rule {n-1,n}` | play the "n passes" variant instead of the printed "n-1" |
| `--league-blast-mode {never,rich,always}` | how the scripted league blasts (default `rich`) |
| `--league-blast-prob` | fraction of that league that uses the blast mode in the chosen proportion (default 0.15; set `1.0 --league-blast-mode always` for a league that blasts on sight) |

Each game gives the learner one rotating seat while the other seats are filled
from a league of the hand-tuned rule policy, simple bots and older snapshots of
the learner itself. A quarter of games are pure mirror self-play. Decisions are
scored with REINFORCE against a learned value baseline, with the entropy bonus
keeping some exploration alive.

### Evaluate

```powershell
python evaluate_balance.py --agents rule,greedy,random,cautious,checkpoints/selfplay.npz --games 300
```

The report (`reports/balance.md`, plus a JSON file) contains:

1. **Leaderboard** – each policy in one rotating seat against `n-1` benchmark
   agents, with bootstrap confidence intervals for the score and the margin.
2. **Pairwise head-to-head** – A alone against `n-1` copies of B and the mirror
   image, which is what "dominance" actually requires.
3. **Game dynamics** – dragon rate, stall rate, blast usage, game length.
4. **Seat effects** – mean score per seat; column ownership is not symmetric.
5. **Luck vs skill** – mirror matches where every seat runs the same policy, so
   any spread is pure luck plus seat position.

Seats are rotated every game on purpose: a policy that only wins from seat 0 is
not a dominant strategy.

### What the agents see

The encoder reads only `PlayerActionContext`, which already hides opponent card
fronts, opponent collections and private blast draws. The state block covers the
public situation (scores, seats, deck size, visible own-column fronts, inferred
dragon risk per backside, per-opponent public facts). Each of the 53 primitive
actions gets an action block (row/column, backside, visible front, risk, whether
the occupant can be displaced). The grid covers every row a deck can produce —
ten at three players, seven at four — because the mine keeps growing rows and
an action space sized by the opening board cannot name the front of a deep
mine. Blast announcements are a separate half of the
action space, so the learner also decides **when to blast** instead of receiving
a hand-coded rule.

### Findings

The full write-up, with confidence intervals and reproduction commands, is in
**[BALANCE.md](BALANCE.md)**. The short version:

* **There is no dominant strategy — and early leaves are a trap.** An
  empty-handed escape only earns the +3 haul bonus while the other players fail
  to leave; the moment someone leaves with gems, the rules compare gems and the
  rusher scores **0.00**. With the lorry stand-off rule in place, a `rusher` that
  walks to the pit cage on turn one is now worth **+0.26 points [-0.23, +0.77]**
  against the scripted bots — indistinguishable from noise — and it loses its
  seat the moment it meets players who collect loot.
* **Collect and leave is the strongest policy in the pool.** `leaver` takes one
  gem and walks out: it is the best row on the leaderboard (**+1.19 [+0.68,
  +1.75]** at three players, **+1.30 [+0.84, +1.78]** at four), it beats the
  rusher **+8.37 points with a 100% top score**, and it beats the trained agent
  **+3.10** (whose own score drops to -1.68). A lone rusher against two leavers
  tops the scoreboard only 35.0% of the time with a *negative* margin.
* **Four players is more mine, not more coin-flip.** The same benchmark with
  four seats makes the mine more lethal (62% of hauls end on a dragon, against
  53% at three) while the luck band stays put (mirror spread 4.63 points, 15%
  tied top scores). The leaver is still the best seat, the rusher still a
  parasite, and the learner **trained on four players is genuinely better**
  (+1.11 [+0.73, +1.49], second row of the four-player table) than the same
  learner trained on three (-0.31).
* **A leaderboard against fixed bots cannot tell these apart.** Against the
  scripted bots the ranking is selfplay > rusher > leaver; against a leaver it
  inverts. The report therefore always includes `rusher` and `leaver`, plus a
  "where the points come from" split, so a policy that farms opponents' mistakes
  is visible instead of looking skilful.
* **Blasting is not too weak, but the engine was cheating.** The blast rule says
  to draw three cards face down and choose two before looking. `MineEnv` showed
  the chooser all three fronts, which makes blasting dominant (+1.71 points at
  three players, +1.00 at four). With the printed rule (`--blind-blast-choice`)
  it is neutral (+0.39 at three players, +0.38 at four), a blast haul is worth
  the same as a normal one (+0.41 vs +0.40 points), and against a field that
  banks loot it is a losing move (-0.52).
* **The blast threat is real; the AI ignores it.** A pending blast costs the
  blaster's rivals **-0.43 points [-0.81, -0.06]** and 11 points of win rate, and
  the cheapest counter — walk to the pit cage while the blast is pending —
  recovers **+0.37 [+0.12, +0.63]** of it. The trained network moves its action
  distribution by **0.65%** when a blast is pending and never learns to blast
  itself (0.03–0.05 per game), even after 60 generations against a league in
  which 60% of opponents blast on sight. That is a training-budget finding, not
  a rules finding.
* **`n-1` passes is better than `n`.** Shortens games (65 vs 79 decisions per
  game) and punishes the early leaver harder under `n` than `n-1` once the lorry
  rule is in place (-3.15 vs -0.39 points for the rusher). This also uncovered a
  real bug:
  `step_one_player()` ignored `--pass-rule` entirely, so the variant only worked
  in the single-learner API.
* **The lorry stand-off rule is what closes the exploit.** Replacing the old
  "no collection for 8 rounds" valve with "everyone still inside leaves when the
  haul ends" does two things: it stops paying the escapee while voiding the
  campers' loot, and it hands the campers' gems into the scoring. The trained
  agent's margin against the scripted bots fell from +4.14 to -0.31 (parity),
  the rusher's from +2.69 to +0.26, and the games that used to be farmed now
  end with the campers banking what they collected.
* **Luck dominates a single game.** Identical agents playing each other spread
  **4.5–5.0 points** within a game, far more than any between-policy margin here.
  Use the confidence intervals.
* **Two engine bugs were found by watching the agents, not by testing the
  rules.** A dead blaster could hold a haul open until the game was abandoned
  and scored all-zero, and the action grid was sized by the opening board, so
  rows 5 and deeper of a grown mine could neither be seen nor played (10-13% of
  dealt cards). Both are fixed, both have regression tests, every checkpoint was
  retrained, and every report now prints its abandoned-game count.
* **An engine bug was silently scoring whole games as zero.** A player killed by
  a dragon while a blast was pending left the haul unable to finish, so the game
  was recorded `aborted` with every score at 0 (36 of 60 always-blaster games).
  Fixed, and every report now prints how many games the engine abandoned. A
  second bug reported alongside it — a tie-break with no dragons left to draw —
  does not exist: the card economy always leaves enough, and
  [BALANCE.md](BALANCE.md) §7 records the withdrawal.

### Testing the rules variants

```powershell
# remove the free bonus for a sole, empty-handed escapee and retrain
python train_selfplay.py --no-solo-escape-bonus --escape-shaping 0 --out checkpoints/selfplay_no_bonus.npz

# the printed rules, and the fixed ones, with both hand-written controls
python evaluate_balance.py --agents rule,greedy,random,cautious,rusher,leaver,checkpoints/selfplay.npz --games 300 --out reports/balance_default_rules.md
python evaluate_balance.py --agents rule,greedy,random,cautious,rusher,leaver,checkpoints/selfplay_no_bonus.npz --no-solo-escape-bonus --games 300 --out reports/balance_fixed_rules.md

# blast: peek vs the printed face-down rule
python evaluate_balance.py --agents rule,rule:rich,rule:always --games 200 --skip-mirror --out reports/balance_blast.md
python evaluate_balance.py --agents rule,rule:rich,rule:always --games 200 --skip-mirror --blind-blast-choice --out reports/balance_blast_blind.md

# n-1 vs n passes on otherwise identical runs
python evaluate_balance.py --agents rule,greedy,random,rusher --games 200 --pass-rule n --out reports/pass_n.md

# four players: leaderboard, and the learner trained on four seats
python evaluate_balance.py --players 4 --agents rule,greedy,random,cautious,rusher,leaver,checkpoints/selfplay4.npz --games 300 --out reports/balance4_default_rules.md
python train_selfplay.py --players 4 --generations 60 --games-per-generation 64 --out checkpoints/selfplay4.npz

# blast value, threat learning, threat reaction, and whether the threat is real
python analyze_blasts.py --checkpoint checkpoints/selfplay4.npz --players 4 --games 240 --out reports/blasts4_noblastleague.md
```

Two rule gaps surfaced while building this, plus one bug. The gaps are
switchable so you can measure them instead of guessing:

| gap | behaviour | switch |
| :-- | :-- | :-- |
| free solo-escape bonus | +3/+4 for leaving alone with zero gems | `--no-solo-escape-bonus` |
| blast peek | the blaster sees all three draws before choosing | `--blind-blast-choice` (the printed rule) |
| pass count | `step_one_player()` ignored `--pass-rule` and always used `n` | fixed; both variants now work |

The lorry deadlock used to be patched with a `stall_rounds` valve, which paid the
player who had escaped while voiding the campers' loot. That valve is gone: the
haul now simply ends when everybody still in the mine stands on their lorry,
which is the rule the game was missing (see the rules section above).

The defaults keep the historical environment behaviour, so old scripts
reproduce. For balance work, pass `--blind-blast-choice` and decide explicitly
what to do about the other two; [BALANCE.md](BALANCE.md) has the measured effect
of each.

The open questions map directly onto the tooling:

| question | experiment |
| :-- | :-- |
| skill vs luck | mirror matches (always in the report) |
| dominant strategy? | pairwise head-to-head with the `rusher` **and** `leaver` controls |
| is blasting too weak? | `rule:rich` / `rule:always`, with and without `--blind-blast-choice`; `analyze_blasts.py` for the 4-player money question |
| does the AI learn to threaten, and does it react? | `analyze_blasts.py` (paired counterfactual on the network, plus the scripted `threat` reaction in a four-condition experiment) |
| n-1 vs n passes | `--pass-rule n` on otherwise identical runs, at 3 and 4 players |
| victory-point tweaks | `--solo-escape-bonus` / `--no-solo-escape-bonus`, or edit `_score_haul` |

### Development

The original test suite still runs, plus tests for the self-play stack:

```powershell
python -m pip install pytest
python -m pytest -q
```
