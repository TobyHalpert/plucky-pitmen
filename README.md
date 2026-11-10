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
*   **Displace an opponent:** This is only possible if the opponent is standing on a card within that opponent's own column (the color of the player figure is the same as the color of the lorry). The player places their own figure on the opponent's card, while the displaced opponent stays on that same card until their next turn. When it is the displaced opponent's turn, they have all regular options from that card's row (a free card, displacing an opponent, the nearest pit cage, or their lorry) except **Pass**.
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

Use **Take action** to choose the learner's destination, or **Autoplay** to run the simulation automatically. The viewer shows card backs only; opponent gems and dynamite remain hidden. The action menu is rebuilt from the current legal-action mask, so an occupied card appears only when it is a legal displacement, labelled **Displace Player N**; selecting it takes the move and any stale action is rejected before it can reach the engine.

### Watching the trained agents

Any checkpoint in `checkpoints/` shows up in the **Policy** dropdown as
`neural:<name>`, and the autoplay driver then hands every decision to the
network — including whether to announce a blast and which of the three blast
cards to keep.

`checkpoints/selfplay4.npz` ships with the repository, so `neural:selfplay4`
works in a fresh clone. It now uses **same-card displacement** and **face-down
blast card selection**, retrained for 120 generations of 64 games (four seats,
seed 7002, 81.7% win rate vs three `rule` bots), with the downstream tactical
and curriculum nets (`selfplay4_tactical`, `selfplay4_last_haul`, and
`selfplay4_row0_blast_sequence`) retrained on top of it as well. The historical
pit-cage predecessor is preserved as `neural:selfplay4_pitcage`; all checkpoints
play the current rule when selected.

To watch the retrained net:

```sh
python mine_gui.py --opponents 3 --policy neural:selfplay4 --all-seats
```

Other generated checkpoints are local to whichever machine trained them, which is why
the dropdown can be empty. Then the window says so in its message pane, and
training one (about four minutes on a laptop CPU) adds it on the next **New
game** — the list is rebuilt every time a game starts, so no restart is needed:

```powershell
python train_selfplay.py --players 4 --generations 60 --games-per-generation 64 --out checkpoints/selfplay4.npz
python mine_gui.py --opponents 3 --policy neural:selfplay4 --all-seats
```

`--policy` also takes a path (`--policy checkpoints/selfplay4.npz`) and a
checkpoint that was trained **after** the viewer was installed, and it names the
training command instead of dying on `invalid choice` when the network is not
there.

A checkpoint saved before a feature was added is refused rather than listed
(it has no place in today's encoder); the message pane names it and
`python migrate_checkpoint.py <file>` widens it with zero-weight columns, which
leaves its play bit-identical until it is trained again.

The `blaster*` policies are the rule bot's movement with a fixed
blast trigger, which is the only way to see the mechanic on screen: the trained
learner fires a blast about once every thirty games.

| policy | blasts when |
| :-- | :-- |
| `blaster` | every dynamite pair it can cash in |
| `blaster:trap` | it can see a dragon in its own column, still ahead of it |
| `blaster:behind` | another player is ahead on gems |

With `--all-seats`, **Autoplay** hands every decision to the network at
temperature 0.7, so roughly half of the moves on screen are *not* the network's
first choice — it drew them, and some of those draws are the interesting ones. `analyze_game.py`
prints the same game with the probability behind every decision, and
`--greedy` replays the seed with the sampler off so you can see the move the
network actually preferred:

Every escape in the log names what it banks and what it is running from
(`pit cage, banks 0g (the dragon under me)`), which matters because a dragon in a
seat's own column is a card only that seat can see.

The seat list matters as much as the log: the viewer is a **mixed table** unless
you ask for otherwise. `--policy` drives P1 only — P2–P4 stay on the environment's
built-in `simple` bot — and with no `--policy` at all every seat is that bot. Each
seat's policy now shows in the player list (`P2 [simple]`), the analysis header
names all four, and `--viewer` replays exactly the table the viewer draws:

```powershell
python analyze_game.py --seed 7 --viewer     # P1 neural, P2-P4 the built-in bot
python analyze_game.py --seed 7 --viewer --policy simple   # the all-bot default table
```

```powershell
python analyze_game.py --seed 7              # every decision of the game you just watched
python analyze_game.py --seed 7 --forks      # what each flagged move was worth, re-rolled
python analyze_game.py --seed 7 --greedy     # the same seed, argmax only
```

One such game — seed 7, four neural seats, scored **0–−1–−1–5** with the shipped
checkpoint — is read in full in **[SEED7.md](SEED7.md)**: which moves were the
policy's own first choice, which were 10% samples, where the network is genuinely
blind (displacement), and what the argmax game of the same seed looks like
instead. The file analyses the 3–4–0–−1 game that the previous checkpoint played,
which is kept as `checkpoints/selfplay4_prehazard.npz` and still runs:

```powershell
python analyze_game.py --seed 7 --policy neural:selfplay4_prehazard
```

```powershell
# the 4-player self-play learner against three rule bots
python mine_gui.py --opponents 3 --policy neural:selfplay4

# ... or the same network in every seat, which is the clearest view of its habits
python mine_gui.py --opponents 3 --policy neural:selfplay4 --all-seats

# play the argmax instead of sampling at temperature 0.7 (what the benchmarks do)
python mine_gui.py --opponents 3 --policy neural:selfplay4 --deterministic

# ... and baselines that really do blast, for comparison
python mine_gui.py --opponents 3 --policy blaster
python mine_gui.py --opponents 3 --policy blaster --all-seats
# the "a dragon is waiting for me anyway" rule, which measures *worse* (§5)
python mine_gui.py --opponents 3 --policy blaster:trap --all-seats
```

The **AI POLICY** panel lists the network's top actions for the seat on turn
with their probabilities, marking the action it actually took, so a surprising
move can be traced back to the distribution that produced it. Every candidate
carries the reason the *deciding seat* had for it ("own column, gem", "back v
(20% dragon)", "raid P2, back o (11% dragon)"), because the board only ever
draws backsides: only the owner of a column sees the fronts in it. A line under
the list says whether the plan currently standing is **locked** or still
**tentative** — planning is sequential, so a figure parked on a card can move
again before the phase ends. The panel is empty for scripted policies, during a
tie-break, and while a blast is waiting on a card.

What to look for, given the measurements in [BALANCE.md](BALANCE.md): it should
rarely blast (0.02-0.05 per game *fired*, for two reasons — it seldom holds two
dynamite, and most blasts it announces are cancelled by its own next move, §5),
it should walk to a pit cage as soon as it holds loot, and it *does* leave when
an opponent announces a blast — but not because it learned to: the action mask
treats a pending blast as certain danger, which is what makes the empty-handed
cage legal, and the seat then takes the escape it is offered (69% of threatened
decisions against a 17% control, `probe_threat_reaction.py`). The correction to
§6, and the leftovers, are in
[THREAT_LEAGUE.md](THREAT_LEAGUE.md). Watch `blaster` once to see what blasting
looks like, including the dragon it wakes roughly half the time.

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
| `agents.py` | `Agent` interface, scripted baselines (incl. `rusher`, `leaver`, `threat`, `reader` — `rule` without the "a quiet rival means a safe card" inference — `bluffer`, which announces a blast it always cancels, and `counter`, the deterministic seat that farms the trained net), `NeuralAgent` |
| `learning.py` | masked-softmax MLP policy + value baseline, REINFORCE, Adam, save/load |
| `selfplay.py` | full-game driver: every seat controlled, blasts, tie-breaks, per-player returns |
| `train_tactical.py` | branch-based tactical distillation from sampled hidden worlds and full-game win-share comparisons |
| `train_row0_blast.py` | targeted Row-0 blast/cancellation training from real announced-blast states |
| `train_blast_trigger.py` | distils the blast trigger itself: real blast-legal positions, plain and announcing branches from the same hidden world, both continued by a seat that *keeps* the blast so the label prices a committed one |
| `evaluate_row0_candidates.py` | paired Row-0 candidate matches and actual announcement/cancel audits |
| `audit_row0_choice.py` | counterfactual Row-0/Row-2/pass probabilities in unchanged collected states |
| `evaluate_tactical.py` | fresh-game candidate comparison against three unchanged nets, with actual blast behaviour |
| `train_selfplay.py` | league self-play trainer with checkpoints and a learning curve |
| `evaluate_balance.py` | benchmarks, pairwise dominance, luck analysis, markdown report |
| `analyze_blasts.py` | is a blast worth it, does the learner threaten, does it react, and is the threat real |
| `probe_threat_reaction.py` | does a neural seat flinch in play? cage/lorry/pass on decisions taken while a rival's blast is pending, against a no-threat control |
| `probe_threat_mask.py` | where the flinch lives: the paired threat test split by whether the blast changes the legal set or only the state |
| `probe_blast_trigger.py` | does a checkpoint actually fire blasts, under the printed rule and under the engine's peek, and does the retrained one play any better |
| `probe_dragon_step.py` | does a seat step onto a dragon it can see and get off again before the reveal? digs per game, the share followed by another decision, and where the figure is when the card is turned |
| `probe_vacated_dragons.py` | is a rival walking off a card in its own column a dragon tell? precision against ground truth, the risk it moves, and whether the table plays differently |
| `probe_blast_colour.py` | does the net use the colour of a dragon it has seen? the keeper rule under the printed vs peeking blast draw, and a paired counterfactual on the announcement |
| `probe_bluff.py` | what an announced blast nobody fires is worth: a bluffing seat against three unchanged nets, with the same movement policy and no announcements as the control |
| `probe_blast_value.py` | what a single blast is worth from a real position (forked rollouts), and whether backside knowledge improves the blind choice |
| `probe_blast_strategy.py` | is "blast while a dragon is waiting for me" a good rule? paired arms over the blast trigger and the keeper choice |
| `probe_bait_value.py` | what a move onto a dragon the player can see is worth, forked per opponent (rule bots vs readers vs its own peers) |
| `probe_push_value.py` | what a single displacement is worth from a real position, forked, split by the card it took |
| `probe_push_clock.py` | deprecated historical clock diagnostic; its conclusions are withdrawn (see BALANCE.md) |
| `probe_seed7_p2.py` | exact-position forks of seed-7 P2 push, blast and cancellation; limitations and results in SEED7_P2.md |
| `probe_pusher_match.py` | deterministic pusher versus three retrained nets, rotated seats, no-push control and actual displacement counts |
| `probe_always_pusher_match.py` | one net versus three seats that take every displacement opportunity, with incoming and outgoing pushes and paired scores |
| `probe_nextcard_training.py` | old versus retrained nets under the current rule: actual pushes, own-column card moves, scores and completed games |
| `probe_backside_reading.py` | does the learner read a backside? risk of the card it takes, against the risk it was offered |
| `probe_lorry.py` | how often a seat parks on its lorry, and how often a haul is lost to a stand-off because every seat did |
| `probe_escape.py` | escape rules side by side: how often a haul is thrown away by a herd of empty-handed cage moves, and what each variant is worth |
| `analyze_game.py` | one game move by move: what each seat knew, the probability the network put on the move, and what the move was worth from there |
| `probe_displacement.py` | is a push worth anything — to the seat that pushes, or to the one that parks where it can be pushed? paired arms over a 1,000-game league |
| `migrate_checkpoint.py` | widen a checkpoint saved before a feature was added (zero columns or a zero row, behaviour unchanged; `--append-hazard` for the table-hazard feature, `--add-phase-clock` for the planning-clock inputs) |
| `profile_agent.py` | what a checkpoint actually does: digs, front-row digs, passes, retreats, blasts, escapes — side by side for two checkpoints |
| `gui_controller.py` | Tk-free viewer logic: moves, blasts, tie-breaks, the AI POLICY snapshot |
| `mine_gui.py` | the Tk viewer itself |

### Train

```powershell
python -m pip install -r requirements.txt
python train_selfplay.py --generations 60 --games-per-generation 64
```

This writes `checkpoints/selfplay.npz`, older snapshots to
`checkpoints/league/`, and the learning curve to
`checkpoints/selfplay.history.json`.

To train an existing network further, resume it and keep the old one for the
comparison:

```powershell
python train_selfplay.py --players 4 --resume checkpoints/selfplay4.npz `
    --generations 120 --games-per-generation 64 --seed 7001 --out checkpoints/selfplay4_long.npz

# what changed, in habits rather than points
python profile_agent.py --checkpoints checkpoints/selfplay4.npz,checkpoints/selfplay4_long.npz --games 300
```

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
| `--blind-blast-choice` / `--no-blind-blast-choice` | choose blast cards face down, as the printed rules say (default on) |
| `--pass-rule {n-1,n}` | play the "n passes" variant instead of the printed "n-1" |
| `--league-blast-mode {never,rich,always}` | how the scripted league blasts (default `rich`) |
| `--league-blast-prob` | fraction of that league that uses the blast mode in the chosen proportion (default 0.15; set `1.0 --league-blast-mode always` for a league that blasts on sight) |
| `--league-threat-prob` | share of scripted league seats replaced by `threat`, the policy that leaves when a rival's blast is pending (default 0, which keeps the historical league bit-identical); measured: no gain, see [THREAT_LEAGUE.md](THREAT_LEAGUE.md) |

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
dragon risk per backside, the table's dragon hazard for this round — one minus
the chance that nobody turns a dragon-capable card before the haul ends — and
per-opponent public facts including how often each
opponent has displaced somebody this game — displacement is made in the open,
and it is the one read on whether a table takes bait). Each of the 53 primitive
actions gets an action block (row/column, backside, visible front, risk, whether
the occupant can be displaced). The grid covers every row a deck can produce —
ten at three players, seven at four — because the mine keeps growing rows and
an action space sized by the opening board cannot name the front of a deep
mine. Blast announcements are a separate half of the
action space, so the learner also decides **when to blast** instead of receiving
a hand-coded rule. A checkpoint stores its own weight shapes; if a feature is
added, an older file is widened once with `python migrate_checkpoint.py` rather
than retrained from scratch.

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
  four seats makes the mine more lethal (61% of hauls end on a dragon, against
  53% at three) while the luck band stays put (mirror spread 4.63 points, 15%
  tied top scores). The leaver is still the best scripted seat, the rusher still
  a parasite, and the learner **trained on four players is genuinely better**
  (+1.65 [+1.24, +2.07], top of the four-player table) than the same learner
  trained on three (-0.31). Training it another 120 generations took it from
  35% to 45% wins against three rule bots and taught it to camp a pit cage
  instead of leaving (BALANCE.md §4).
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
* **Nobody throws a haul away any more.** The three exits that cost a round and
  buy nothing — parking on a lorry, fleeing into a cage with no loot and nothing
  to flee from, and paying more for a face-down card than one you walk past — are
  masked, with the measurement behind each in BALANCE.md §4. Before them, 40% of
  hauls in four-net games ended with three or more seats in cages and no gems
  collected; now 2.9% do.
* **An escape is insurance, so the policy is told what can blow.** A cage buys
  protection against a dragon and nothing else, which makes it right when the
  table is dangerous and wrong when nobody is standing on a dragon-capable card —
  the seat can leave later, in the same phase or later in the haul, with more gems
  to its name. The encoder had no way to say "three of us are standing on cards
  that can be a dragon": the board reached the policy only through the
  candidate-action block, never the card under a rival. `round_dragon_hazard` is
  now a state feature (input 103 → 104) and the escape rule reads it. Seed 7's
  danger-free escapes were worth **−0.35** points and its dangerous ones **+0.53**;
  the retrained checkpoint is **+0.45 points** better than the one before it and
  takes no escape at a table where nothing can blow (BALANCE.md §4.5). The first
  version of that feature charged a rival the risk of the card under it even when
  the rival was standing in its own column — where it can see the card and will
  step away from a dragon rather than turn one. Only figures on a card *nobody
  can see* (a raid) can wake a dragon, plus blasts. Priced the old way, the
  table the viewer flagged in seed 7 read 0.61; priced on the raiders it read
  0.00, and re-rolled 200 times it ended on a dragon 1% of the time. The
  correction is strength-neutral (+0.03 points, se 0.09, against the checkpoint
  it replaces) and the escape rule costs nothing measurable with it.
* **A forty-line deterministic seat beats the trained net by 2.8 points a game.**
  `agents.CounterAgent` — never pushes, never takes a card it has not seen the
  front of, banks two gems — takes **+3.94 a seat against the nets' +1.16** and wins
  57% of games outright, in all four seats. The two tells it was built around are
  worth less than the habit that does the work: not displacing is worth **+0.10**,
  the dragon read **+0.20** in the variant that raids and **0.00** in the shipped
  one, and plain early banking is the whole margin — `leaver` gets +2.96 without
  reading anything. The field it farms dies 0.41 times a seat-haul and loses 60% of
  hauls to dragons; the cage bar that stopped the table leaving for no reason also
  made it **0.44 points easier to farm**. BALANCE.md, "The counter".
* **A seat holds the card it chose, and the drift is gone.** A figure scores the
  card under it when the planning phase closes and movement is one-way, so a second
  move in the same phase gives up that card for a shallower one. Rule 9
  (`climbs_off_the_card_it_chose`) holds unless the seat is stepping off a dragon it
  can see or onto a gem it can see (and not when it is standing on one already):
  **+0.441 a seat (se 0.065)** over 200 paired games with the weights unchanged.
  Phases taking two or more moves fell 66% → 23%; on the current mask **75.6% of
  1,985 seat-phases take exactly one card move**, and every remaining second dig is
  a re-entry from a cage or a step off a visible dragon. BALANCE.md, "The card you
  are standing on".
* **A seat that has walked into a cage leaves it.** A figure in a cage keeps its
  loot and is out of the haul at the execution phase, so a card move taken from
  there undoes the escape. The shipped net did it **604 times in 60 games**, 313
  of them while holding a gem the escape would have banked, and it is worth
  **−0.72 (se 0.12)** a move. Rule 10 refuses it when there is a bank to forfeit
  (`+0.029`, se 0.032 — it removes a move with no upside). The harder version,
  holding the seat to its departure so the figure cannot walk in and out of the
  cage, is priced and **left off**: +0.145 a seat and a third fewer decisions, at
  the cost of 28% of the gems in the mine (`ESCAPE_STANDS_ONCE_TAKEN`). The
  viewer's default opponent, `rule_policy`, had the same habit — 520 steps out of a
  cage in 40 games — and now holds a banked escape too. BALANCE.md, "Out of the
  cage is out".
* **A seat that sits on its lorry stays out of the mine.** The printed rule —
  *"anyone who has occupied their lorry at any point during a planning phase can
  only switch between the last pit cage and their lorry for the remainder of that
  phase"* — was not enforced: from the lorry the cage was legal, and from a cage
  at index 1 or deeper the mine opened again, so `card -> lorry -> cage -> card`
  was a legal way back into a mine a figure had already left. `LORRY_COMMITS_THE_SEAT`
  now refuses the card move for the rest of the phase. It changes nothing and it
  cannot: sitting on the lorry is the seat's decision for that phase, so no seat
  that has done it is asked again before the round ends (0 of 30 games), and a
  figure still on the lorry when a round starts cannot step onto a card either.
  Paired over 200 games the difference is **+0.000** — the line was missing, not
  harmful. The check is kept as a guard, with a test pinning the fact that makes it
  inert. BALANCE.md, "Sit on their lorry".
* **Same-card displacement is now the default.** A push contests a rival's
  chosen spot without giving that rival free pit-cage shelter or moving them
  automatically: the displaced figure stays on the same card and on its next
  turn has all regular options from that position except `Pass`. The displaced
  figure keeps its loot and its next decision; a dragon can still make it choose
  shelter. The aim is competition for spots, not profitable indiscriminate pushing.
  Earlier pass-count comparisons did not distinguish passing on a card from
  passing in shelter and do not test that aim. See BALANCE.md for retraining.
* **Historical pit-cage-rule results: indiscriminate pushing did not pay.** A net that has
  decided its round holds the card it chose (rule 9), and it is usually a gem:
  of the seat-phases that took exactly one card move and then held, **82%** ended
  on a gem. But a displacement is only legal against a figure standing in **its
  own column**, and only that owner sees the fronts — so **every legal push takes
  a card the pusher has never seen** (4 gems, 4 dynamite, 2 dragons in 10).
  Priced from the identical position over 109 forked positions: a push is worth
  **−0.41 (se 0.18)**, because the 59% that take a **gem** are worth +0.69 and the
  **27% that take a dragon are worth −2.48**, and a net never stays on a dragon it
  can see — the ones a pusher meets are the bait. Nor does a push take the
  victim's loot: a displaced figure leaves through the pit cage with every card
  it collected (0.2 cards on average, and 83% of victims had nothing yet). The
  more it pushes the worse it gets: −0.55 at 522 pushes, −2.75 at 1,933.
  `agents.PusherAgent` is the control, `probe_push_value.py` prices one push.
  BALANCE.md, "The parked seat".
* **A plain deterministic seat beats the trained nets: +2.19 a game.** One
  `CounterAgent` against three `selfplay4` seats, seats rotated, 300 games:
  +3.60 a seat, best score in 56% of games. What does the work is digging,
  banking two gems and leaving — the push adds −0.003. Against the viewer's
  default opposition (three `rule` bots) the same seat scores +0.88 and a margin
  of −1.37: the field matters more than the tactic.
* **The "keeps moving forward" is the rules, not the seat.** `MineEnv` puts every
  surviving figure back at the mouth of the mine when a round is dealt, so the
  first dig of every planning phase is mandatory. Of 1,836 seat-phases in 60
  four-net games, 1,291 digs held the card, 249 (**13.6%**) were abandoned before
  the cards turned — and of those, 124 left through a cage or lorry already taken
  and 111 stepped off a dragon they could see under themselves (the printed bait
  that pays when a rival pushes them off it). Holding a seat to the card it just
  stepped onto was priced at **−0.06** a seat and dropped, and banning the bait
  costs **−0.23**: `BAIT_IS_ALLOWED` stays `True`. BALANCE.md, same section.
* **A seat holds the card it chose.** A figure scores the card under it when the
  planning phase closes, and movement is one-way, so a second move in the same
  phase gives up that card for a shallower one — and at the bottom of the column
  the only moves left are the lorry and the cage. The shipped net made two or more
  moves in **66% of its seat-phases**, every extra one with a pass available.
  Rule 9 (`climbs_off_the_card_it_chose`) holds unless the seat is stepping off a
  dragon it can see or onto a gem it can see (and not when it is standing on one
  already): **+0.441 a seat (se 0.065)** over 200 paired games with the weights
  unchanged, gems per haul 2.90 → 4.33, thrown-away hauls 22% → 12%, two-move
  phases 66% → 23%, phases ending on the lorry 11% → 8.7%. Retraining under the
  earlier, looser version of the rule changed nothing (−0.06, se 0.17), and it
  drops the `counter` bot's edge from +3.21 to +2.02 a game. BALANCE.md,
  "The card you are standing on".
* **The planning clock is in the input, and the push is still declined.** A push
  only puts a rival out of the haul if the phase closes before that rival's next
  turn, and whether it can is decided by how many passes are left and which
  rivals have already planned. Both are now inputs (a 13th per-opponent flag and
  two state features, input 104 → 110), and the retrained net takes **1 push in
  1,953 offers** — the same habit it had before — while the pushes it declines
  price at **−0.30 (se 0.06)**. The blind spot was real; closing it did not make
  the move good. BALANCE.md, "The planning clock went into the input".
* **The blast threat is real; the AI ignores it.** A pending blast costs the
  blaster's rivals **-0.43 points [-0.81, -0.06]** and 11 points of win rate, and
  the cheapest counter — walk to the pit cage while the blast is pending —
  recovers **+0.37 [+0.12, +0.63]** of it. The trained network moves its action
  distribution by **0.65%** when a blast is pending and never learns to blast
  itself (0.02–0.05 per game), even after 180 generations against a league in
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
| blast peek | historical variant where the blaster sees all three draws before choosing | fixed to face-down by default (`--no-blind-blast-choice` reproduces historical peeking) |
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

### Score-aware deterministic pushers

`pusher-leader` prefers the reachable rival with the most total points.
`pusher-leader-only` attacks only a rival tied for the table lead. Both retain
the pusher's banking and risk safeguards. They are selectable scripted policies;
`pusher` remains the original held-tell version. On 400 fresh games each against
three retrained nets, outright wins were 33.5% and 33.0%, versus 33.8% for the
old any-tell pusher and 32.8% without pushing: no clear improvement yet.
See BALANCE.md and rerun with:

```sh
python probe_pusher_match.py --games 400 --seed 910000 --score-aware
```

### Branch-based tactical training

The experimental `neural:selfplay4_tactical` checkpoint is trained on generated
blast, cancellation and attack positions using branch continuations, not seed-7
imitation. The ordinary `neural:selfplay4` remains unchanged. See
[TACTICAL_TRAINING.md](TACTICAL_TRAINING.md) for the method, hidden-information
limitations, reproducible commands and evaluation.

### Last-haul tactical curriculum

`neural:selfplay4_last_haul` is an optional candidate trained from final-haul
positions with varied scores, 0/1/2/4/6 starting dynamite, and both initial and
round-3 boards. It more often retains a pending blast and has a promising
targeted-scenario result, but **no demonstrated general-play gain**; the default
checkpoint is unchanged. See
[LAST_HAUL_TRAINING.md](LAST_HAUL_TRAINING.md) for results and limitations.

```sh
python mine_gui.py --opponents 3 --policy neural:selfplay4_last_haul --all-seats
python train_last_haul.py --diagnose-seed7
```

The second command switches weights only at the watched game's haul-3 boundary,
so earlier hauls cannot explain the difference.

### Row-0 blast-sequence curriculum

`neural:selfplay4_row0_blast_sequence` targets one reported sequence: announce
a blast while moving to a Row-0 card, then leave by pit cage or lorry on the next
decision. Positions are harvested from ordinary games. The policy keeps every
action legal; nearby announcements and pass/leave choices are scored by complete
sampled continuations with hidden cards resampled honestly. At a Row-0
announcement, probability may move to Row 2 or, if Row 2 is unavailable, Row 1.
At the following cancellation decision, it may move to `pass`. A change is only
encouraged when the replacement's measured value is close enough to justify it.

On 1200 fresh paired games, where one seat used the candidate and the other three
remained `selfplay4_last_haul`, the candidate's outright win rate was 19.75%
against the baseline's 20.42%: a paired change of −0.67 pp, give or take 1.54 pp.
The candidate did not gain overall wins, but its announced blasts survived much
longer: 449 hauls contained a fired candidate blast instead of 301, while
cancellations fell from 1895 to 973.

The 150-game all-candidate audit measured the complaint directly: a Row-0 move
made while announcing fell from 1.313 to 0.267 per game, and the immediate
row-0 exit rate after such a move fell from 71.1% to 35.0%. The counterfactual
audit `audit_row0_choice.py` is the check for the replacement option: at the
52 unchanged baseline Row-0 announcement states, probability on a Row-2 blast
rose from 3.41% to 8.45%; at the 76 unchanged follow-up states, probability on
pass rose from 5.17% to 43.25%. The candidate still plays fewer Row-2 moves in
fresh games than the baseline, so this should not be sold as “the net now prefers
Row 2” everywhere. It is best described as no longer rewarding the announce-there-
then-leave sequence, while keeping a nearby blast as a stronger, legal alternative.
The default checkpoint remains `selfplay4_last_haul`.

```sh
python mine_gui.py --opponents 3 --policy neural:selfplay4_row0_blast_sequence --all-seats

python train_row0_blast.py --positions 64 --held-out 32 \
  --samples 32 --epochs 10 --lr 0.0002 --teacher-mode policy --target-mode sequence \
  --bad-reduction 0.7 --evidence-scale 0.06 --transfer-margin 0.06 \
  --teacher-temperature 0.12 --save-every 1 \
  --out checkpoints/selfplay4_row0_blast_sequence.npz

python evaluate_row0_candidates.py --games 600 --audit-games 150 --seed 3600000 \
  --baseline checkpoints/selfplay4_last_haul.npz \
  --candidates checkpoints/selfplay4_row0_blast_sequence.npz
python evaluate_tactical.py --baseline checkpoints/selfplay4_last_haul.npz \
  --candidate checkpoints/selfplay4_row0_blast_sequence.npz --games 1200 \
  --seed 3600000 --out /tmp/row0_blast_sequence_1200.json
python audit_row0_choice.py --positions 128 --seed 3600000 \
  --candidates checkpoints/selfplay4_last_haul.npz checkpoints/selfplay4_row0_blast_sequence.npz
```

Earlier experiments are useful warnings. Broad distillation over Row-0, Row-2
and non-row candidates suppressed announcements widely and made held-out value
worse. Training only Row-0 placement, or only the cancellation decision, did not
combine the desired effects as well as training both decisions together. A
separate Row-0 exit mask was rejected: the exit remains legal when it genuinely
banks something.

### A threat-seeded league, and where the flinch actually lives

`--league-threat-prob` puts the `threat` reaction policy into the scripted
league, which is what BALANCE.md §10.6 asked for: a league that answers blasts
so the value of respecting one shows up in the learner's games. Measured against
a league that is otherwise identical (same blasting opponents, same resume
point, same seed), it does nothing: **−0.38 ± 1.19 pp** paired over 800 fresh
games. The reason is a measurement problem that turned into a finding.

The §6 "the network ignores a pending blast (TV 0.0065)" was generated one
commit before `round_dragon_hazard` started treating a pending blast as certain
danger — which is what makes an *empty-handed* cage move legal. From then on the
counterfactual changes the legal set, not just the state, and the same probe
reads **0.7766** on the shipped checkpoint. `probe_threat_mask.py` splits it:
78% of the pairs are ones where the toggle opens the cage, and 94% of the
probability moves onto actions that were illegal a moment ago; on the 22% where
the legal set is unchanged the shift is 0.0006 — the policy itself still reads a
pending blast as nothing. In play the seat answers 69% of its threatened
decisions with a cage, against 17% when no blast is pending.

So the reaction is delegated to the mask, and a league cannot teach a policy the
mask never asks to choose. The residue worth attacking is the masked-identical
case (a threatened seat that already has the cage legal and still passes 25% of
the time), which is a distillation job on `train_tactical.py`'s machinery.
[THREAT_LEAGUE.md](THREAT_LEAGUE.md) has the arms, the tables and the limits.

```sh
python mine_gui.py --opponents 3 --policy neural:selfplay4_threatleague --all-seats
python probe_threat_reaction.py --games 300 \
  --checkpoints checkpoints/selfplay4.npz,checkpoints/selfplay4_threatleague.npz
python probe_threat_mask.py --checkpoint checkpoints/selfplay4.npz --games 200
python analyze_blasts.py --checkpoint checkpoints/selfplay4.npz --players 4 --games 200
```
