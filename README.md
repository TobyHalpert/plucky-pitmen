# Plucky Pitmen: Gymnasium Q-learning model

This project turns the supplied German rules for **Plucky Pitmen** into a small Gymnasium environment for testing strategy hypotheses.

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