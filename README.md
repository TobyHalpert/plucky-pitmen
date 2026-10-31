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
*   if **all players still in the mine stand on their lorry** at the end of a planning phase. Standing on a lorry collects nothing, so such a haul can never progress; it ends and only players who actually left the mine score.

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
| `agents.py` | `Agent` interface, scripted baselines (incl. the `rusher` and `leaver` controls), `NeuralAgent` |
| `learning.py` | masked-softmax MLP policy + value baseline, REINFORCE, Adam, save/load |
| `selfplay.py` | full-game driver: every seat controlled, blasts, tie-breaks, per-player returns |
| `train_selfplay.py` | league self-play trainer with checkpoints and a learning curve |
| `evaluate_balance.py` | benchmarks, pairwise dominance, luck analysis, markdown report |

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
dragon risk per backside, per-opponent public facts). Each of the 28 primitive
actions gets an action block (row/column, backside, visible front, risk, whether
the occupant can be displaced). Blast announcements are a separate half of the
action space, so the learner also decides **when to blast** instead of receiving
a hand-coded rule.

### Findings

The full write-up, with confidence intervals and reproduction commands, is in
**[BALANCE.md](BALANCE.md)**. The short version:

* **There is no dominant strategy — but there is a parasite.** An empty-handed
  escape earns the full +3 haul bonus whenever the other players fail to leave.
  A `rusher` that walks to the pit cage on turn one beats the scripted bots 54%
  of the time for +2.69 points, and a self-play agent finds the same policy by
  itself. But split those points by what the opponents did: **every point comes
  from hauls where nobody else banked loot.** In the 50.7% of hauls where the
  others left with gems, the rusher scores **0.00**.
* **The counter is to collect and leave.** `leaver` takes one gem and walks out:
  it beats the rusher **+8.72 points with a 100% top score**, and beats the
  trained agent **+3.36** (the agent's own score goes to -2.35). A lone rusher
  against two leavers tops the scoreboard only 41.7% of the time with a
  *negative* margin.
* **A leaderboard against fixed bots cannot tell these apart.** Against the
  scripted bots the ranking is selfplay > rusher > leaver; against a leaver it
  inverts. The report therefore always includes `rusher` and `leaver`, plus a
  "where the points come from" split, so a policy that farms opponents' mistakes
  is visible instead of looking skilful.
* **Blasting is not too weak, but the engine was cheating.** The blast rule says
  to draw three cards face down and choose two before looking. `MineEnv` showed
  the chooser all three fronts, which makes blasting dominant (+1.65 points).
  With the printed rule (`--blind-blast-choice`) it is neutral (+0.62).
* **`n-1` passes is better than `n`.** Shortens games (67 vs 80 decisions per
  game) and keeps the early-leaver's payoff down (56% vs 88% wins for the
  rusher). Most of the old difference was deadlock, which the lorry rule now
  resolves on its own. This also uncovered a real bug:
  `step_one_player()` ignored `--pass-rule` entirely, so the variant only worked
  in the single-learner API.
* **The lorry stand-off rule changes what self-play learns.** Merely ending the
  haul when everyone left in the mine is parked (instead of the old
  "no collection for 8 rounds" valve) took the trained agent from "escapes 99%
  of hauls empty-handed" to "escapes 70% of hauls carrying 0.28 gems", and its
  margin against the scripted bots fell from +4.14 to +0.80. The valve had been
  paying it to keep opponents camping.
* **Luck dominates a single game.** Identical agents playing each other spread
  **5.1 points** within a game, more than most between-policy margins here. Use
  the confidence intervals.

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
| is blasting too weak? | `rule:rich` / `rule:always`, with and without `--blind-blast-choice` |
| n-1 vs n passes | `--pass-rule n` on otherwise identical runs |
| victory-point tweaks | `--solo-escape-bonus` / `--no-solo-escape-bonus`, or edit `_score_haul` |

### Development

The original test suite still runs, plus tests for the self-play stack:

```powershell
python -m pip install pytest
python -m pytest -q
```
