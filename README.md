# Plucky Pitmen: Gymnasium Q-learning model

This project turns **Plucky Pitmen** into a Gymnasium environment for testing strategy hypotheses.

**Feel free to contribute!** I appreciate your feedback.

## Rules

The winner of a round is the player who **leaves the mine with the most gems**.
Everyone inside the mine when the dragon is awakened loses everything.
The players must **cooperate** to avoid waking the dragon and **compete** to win the game.

During a round, **planning** and **execution phases** alternate.
The table shows the consequences in the execution phase, depending on where a player's figure stands at the end of the planning phase:

| **Position** | **Loot** | **Dragon Awakened** | **Dragon Stays Asleep** |
| :--- | :--- | :--- | :--- |
| Card | **Receives the card** | Loses loot | Participates in the rest of the round |
| Lorry | - | Loses loot | Participates in the rest of the round |
| Pit cage | - | Leaves mine / <br>**Keeps loot** | Leaves mine / <br>Sits out for the rest of the round |

Pit cages are in between each row of cards.

### Setup

There are 3 sets of 10 cards, each set sharing the same card back. Each set contains 4 gems, 4 dynamite, and 2 dragons. They are shuffled into one pile. Every player gets 3 cards, looks at them and places them into a column under their lorry.

### Planning Phase

Beginning with the starting player, players take turns until exactly **as many players pass in a row as there are players remaining** in the mine (number of players in the mine minus one). 

On their turn, a player chooses **one** of the following options:

*   **Move to a free card:** The player places their figure on an unoccupied card.
*   **Move to the nearest pit cage:** A pit cage can accommodate any number of players.
*   **Pass:** The player does not change their announced plan. If it is their turn again later, all options are open to them once more. Passing is not allowed during the first round of the planning phase.
*   **Displace an opponent:** This is only possible if the opponent is standing on a card within that opponent's own column (the color of the player figure matches the color of the lorry). To do this, the player moves the opponent to the nearest pit cage in the direction of the lorries and places their own figure on that opponent's card.
*   **Sit on their lorry:** Anyone who has occupied their lorry at any point during a planning phase can only switch between the last pit cage and their lorry for the remainder of that planning phase.

For the options "free card", "pit cage", and "displace", the following rule applies: The card or pit cage **must be closer to the lorries** than the row the player is currently in.

### Execution Phase

All players who are in a **pit cage** at the beginning of the execution phase leave the mine and **are safe from the dragon**. They sit out for the rest of the round, meaning they do not participate in any further planning or execution phases and only become active again for the final scoring.

Beginning with the starting player, **everyone** who is still in the mine and not standing on a lorry **takes the card** beneath their figure and looks at it. If it is a **dragon**, the round ends immediately—the dragon has been awakened. Otherwise, the player places the card face down in front of them. The player remains in the mine. The card backs of collected cards remain visible to everyone.

If no one has left the mine so far: Each player draws (if possible) a **card from the draw pile**, looks at it, and places it face down in their column of the newly created row at the bottom end of the mine (see Figure 2). If someone has left the mine: No cards are added.

The **figures** of the remaining players are placed **in front of the newly created row of cards**. The starting player marker is passed clockwise. (The starting player does not necessarily have to be inside the mine.) The next planning phase begins.

### Blasting (not yet implemented)

If a player has collected two dynamites, they can announce a blast during their turn in the **planning phase**. To do this, they flip **two dynamite cards** face up and place their figure on a free card as described.

If it becomes the player's turn again during the planning phase, choosing any option other than "Pass" aborts the blast. In this case, they flip their two dynamite cards back face down and choose a different option instead. They may announce a blast on a different space. Used dynamite is consumed at the end of the planning phase.

In the **execution phase**, the player **receives the card as usual**. Additionally, the player draws **three cards from the draw pile** and places them face down in front of themselves. They must choose **two of them**. They look at the two cards. If a dragon is among them, the round ends immediately—the dragon has been awakened. Otherwise, they place the treasures face down in front of themselves. The third card is placed at the bottom of the draw pile. If there are fewer than three cards left in the draw pile, the choice is skipped and the player takes all remaining cards.

### End of a Round

A round **ends**
*   immediately if the dragon is awakened.
*   if only one player remains in the mine after the execution phase. This player leaves the mine automatically.
*   if a complete row cannot be formed after an execution phase. The remaining players leave the mine automatically.

Now, the **scoring** takes place:
*   Players who left the mine reveal their collected cards. The player with the **most gems** receives **+3 points** (+4 points in the final round). If only one player has left the mine, they receive these points even without having any gems. In the event of a **tie** for gems, each player with the highest number of gems receives **+1 point**.
*   The player who **awakened the dragon** receives **-2 points**.
*   All other players who **died in the mine** receive **-1 point**.
*   A player's score cannot fall below -1 point.
*   If a player awakens the dragon (-2 points) and nobody has left the mine, the other players receive +2 points instead of -1 point. The other players are allowed to keep their dynamite.

A full game consists of **3 rounds**. If there is another round to be played, all cards are collected, shuffled, and a new mine is set up. Players who successfully left the mine are allowed to keep their dynamite.


### End of the Game

The game ends after the third round. The player with the most points wins.

Not yet implemented: In the event of a **tie**, the contenders for victory take turns—starting with the starting player—drawing a card either from the mine or from the draw pile. Players who draw a dragon are eliminated. The last survivor wins.

## Open Questions

* How is the balance between skill and luck?
* Is there a dominant strategy? Is blasting too weak? Should the victory points be tweaked?
* Would the game be better with n-1 passes instead of n passes?

## Scope of the first model

`MineEnv` models the core decision pressure: move toward a card, leave via the shared pit cage or your lorry, or remain exposed while card identities stay hidden. Opponents follow a configurable `random`, `cautious`, or `greedy` policy. The environment runs three decision rounds and gives the learner reward for gems, escaping, and avoiding the dragon.

This is not yet a complete rules engine.

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