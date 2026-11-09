# Seeding the league with `threat` seats, and where the flinch actually lives

This is the experiment BALANCE.md §10 recommended as the cheap fix for a learner
that "never reacts to a pending blast" (§6): put the reaction policy `threat`
into the scripted league, next to a league that blasts, so that staying inside a
threatened mine costs the learner more and the value of leaving shows up in the
games it plays.

It does not work — and the reason is worth more than the result. **The reaction
was never missing from behaviour.** It lives in the action mask, which the §6
counterfactual could not see once the mask started moving. What the §6 table
measured was a network that ignored the threat as a *feature*; what a seat
actually does when a rival announces a blast is leave.

## 1. The correction first: §6's number is stale

BALANCE.md §6 reports that toggling a credible opponent blast on, in an otherwise
identical context, shifts the action distribution by a total variation of
**0.0065**, and reads that as "the trained agent captures none of it". The table
was generated at commit `bd7d0ad`. One commit later, `91810d0` ("Give the escape
its hazard") added `round_dragon_hazard`, which returns **1.0 for any pending
blast** — and that value is what `cage_banks_nothing` reads. From then on a
pending blast makes an *empty-handed* pit-cage move legal, which it never was
before.

So the toggle stopped being a pure change of state. It now changes the **legal
set**. Re-running the identical probe (`analyze_blasts.py --games 200
--seed 424242`, four players) on the shipped checkpoint gives:

| checkpoint | paired decisions | mean TV [95% CI] | shift onto the cage |
| :-- | --: | :-- | --: |
| `selfplay4.npz` | 268 | **0.7766 [0.7310, 0.8189]** | +0.7467 |
| `selfplay4_blastleague.npz` | 224 | 0.7304 [0.6809, 0.7797] | +0.6991 |
| `selfplay4_threatleague.npz` | 228 | 0.7782 [0.7286, 0.8249] | +0.7430 |

`probe_threat_mask.py` splits those pairs by whether the toggle actually moved
the mask (200 games, seed 424242):

| checkpoint | mask moved | mean TV there | mask identical | mean TV there | mass on newly legal actions |
| :-- | --: | --: | --: | --: | --: |
| `selfplay4.npz` | 78% | 0.9406 | 22% | **0.0006** | 94% |
| `selfplay4_blastleague.npz` | 78% | 0.8921 | 22% | 0.0013 | 89% |
| `selfplay4_threatleague.npz` | 78% | 0.9514 | 22% | 0.0006 | 95% |

The split is the whole story:

* Where the toggle **opens the cage** (78% of pairs), 94% of the probability
  moves onto actions that were not legal before. That is a huge reaction, and it
  is the mask plus the policy taking the new option.
* Where the **legal set is unchanged** (22%), TV is 0.0006 — the network reads
  the pending blast as nothing at all. That is exactly §6's finding, and it is
  still true; it is just not what happens in most threatened decisions.

The behavioural check agrees with the mask. One tracked seat against three `rule`
opponents that blast on sight, 300 games (`probe_threat_reaction.py`):

| checkpoint | threat decisions/game | answered with a cage | cage rate when *no* threat was pending |
| :-- | --: | --: | --: |
| `selfplay4.npz` | 2.18 | **69%** | 17% |
| `selfplay4_blastleague.npz` | 2.12 | 68% | 18% |
| `selfplay4_threatleague.npz` | 2.13 | 70% | 19% |

A 52-point gap between threatened and unthreatened decisions is a seat that
flinches. It is not one that has *learned* to flinch: it is one whose legal
moves change when the threat appears, and which then takes the escape on offer.
The 25% pass rate on threatened decisions is the residue §6 was pointing at —
staying in the mine while an opponent's blast is pending, when the cage is right
there and legal.

**Consequence for the tests in this repository:** any claim of the form "the net
ignores the threat" has to say *which* distribution it is scoring. Over masked
distributions the answer is now dominated by the mask. `test_threat_mask.py`
pins the mechanism: a pending blast is read as certain danger, and it is what
makes an empty-handed cage legal.

## 2. The experiment: does a threat-seeded league teach anything left over?

`train_selfplay.py` gained `--league-threat-prob`, the share of scripted league
seats replaced by `threat`. With the flag absent the league's random stream is
untouched, so old runs still reproduce (asserted in `test_training_league.py`).

Two arms, identical in everything but that flag. Both resume `selfplay4.npz`,
play four seats, 60 generations × 64 games, seed 7001, and face a league that
blasts as often as its trigger allows (`--league-blast-mode rich
--league-blast-prob 1.0`) — because a threat responder is only meaningful in a
field that threatens:

| arm | league | checkpoint |
| :-- | :-- | :-- |
| A (control) | 100% blasting scripted seats | `selfplay4_blastleague.npz` |
| B | same, with 60% of scripted seats replaced by `threat` | `selfplay4_threatleague.npz` |

```sh
python train_selfplay.py --players 4 --generations 60 --games-per-generation 64 \
  --seed 7001 --resume checkpoints/selfplay4.npz --name selfplay4_blastleague \
  --league-blast-mode rich --league-blast-prob 1.0 --league-threat-prob 0.0 \
  --out checkpoints/selfplay4_blastleague.npz

python train_selfplay.py --players 4 --generations 60 --games-per-generation 64 \
  --seed 7001 --resume checkpoints/selfplay4.npz --name selfplay4_threatleague \
  --league-blast-mode rich --league-blast-prob 1.0 --league-threat-prob 0.6 \
  --out checkpoints/selfplay4_threatleague.npz
```

### Head-to-head, one candidate seat against three unchanged nets

800 fresh games per comparison, seats rotated, three hauls, temperature 0.7
(`evaluate_tactical.py`, seed 7100000):

| candidate seat | unchanged nets | candidate win % | paired change | blasts fired (hauls) |
| :-- | :-- | --: | :-- | --: |
| `blastleague` | 3 × `selfplay4` | 20.2% | **−1.62 ± 1.54 pp** | 219 |
| `threatleague` | 3 × `blastleague` | 21.6% | **−0.38 ± 1.19 pp** | 229 |
| `threatleague` | 3 × `selfplay4` | 22.0% | **+0.12 ± 1.58 pp** | 222 |

Nothing separates the arms. Replacing 60% of the league with a reaction policy
moved the paired win rate by −0.38 pp, give or take 1.19. Both arms are also
indistinguishable from the checkpoint they were resumed from.

### Behaviour and the field

`analyze_blasts.py`, 200 games per checkpoint, seed 424242. `Δ margin` is the
paired per-game difference between facing a field that blasts and the same field
that never does, so a seat that handles threats loses less:

| checkpoint | blasts fired/game | win vs blasting field | margin, calm field | margin, blasting field | Δ margin [95% CI] |
| :-- | --: | --: | --: | --: | :-- |
| `selfplay4` | 0.005 | 72.5% | +5.34 | +4.66 | **−0.68 [−1.22, −0.15]** |
| `blastleague` | 0.010 | 76.0% | +5.75 | +5.38 | −0.36 [−0.82, +0.10] |
| `threatleague` | 0.005 | 75.5% | +5.47 | +5.13 | −0.33 [−0.81, +0.13] |

The two trained arms lose about half as much to a blasting field as the shipped
net, and their intervals include zero — but the flag that was supposed to cause
that is not the difference: the control arm has the same improvement, and the
head-to-head says the arms are equal. Firing blasts stayed where it was: these
nets still almost never announce one (0.005–0.010 per game), so 60 generations
against a league that blasts permanently did not teach threatening either.

## 3. What this means for the open problem

BALANCE.md §10.6 asked for `threat` agents in the league so the value of
respecting a pending blast would show up in the learner's games. **That
recommendation is now executed, and it is superseded by the mask**: the learner's
seat already leaves, because `action_mask` hands it the cage. A league cannot
teach a policy it never asks to choose.

What is still unexplained is the *mask-identical* case — a threatened seat that
already has the cage on offer (it holds loot, or the board hazard was high) and
still passes 25% of the time. That is a policy question, not a league question,
and the cheap way to attack it is the branch-based distillation already in
`train_tactical.py` / `train_row0_blast.py`: harvest only decisions where a rival
blast is pending **and** the cage is already legal, price `pass` against the
cage with full-game continuations, and distil that preference. It should be
measured on `probe_threat_mask.py`'s split (the 22% group), not on the mixed TV,
which the mask dominates.

## 4. Limitations

* Two arms, one seed, one resume point. The head-to-head interval (±1.2 pp) rules
  out a large effect, not a small one.
* The mask change is what makes the §6 correction necessary, but §6's *other*
  numbers — the −0.43 point cost of a threat and the +0.37 recoverable by
  reacting — were measured with scripted seats, which have no action mask, and
  are unaffected.
* `probe_threat_reaction.py` counts only decisions where an opponent's blast is
  pending at the moment of the decision. A blast announced later in the same
  planning phase is not counted for decisions already taken.
* `--league-threat-prob 0.6` is a guess, not a tuned value. A share closer to the
  natural frequency of blasts (one per haul at most) might be less distorting,
  but the point of the arm was to make the threat common enough to learn from.

## 5. Is the pattern rational? The announce-and-cancel cycle, priced

The viewer shows a net collect dynamite, announce a blast, cancel it — and every
other seat answer the announcement by walking to a pit cage, giving up the card
it had planned. It reads as two mistakes. `probe_bluff.py` prices both halves,
one seat against three unchanged `selfplay4` nets, 400 games, seats rotated,
seed 811000. Every arm faces the same field; the arms differ only in what the
tracked seat does with a dynamite pair.

| tracked seat | own score | mean rival score | announcements/game | share that fired | rivals' threatened decisions/game | answered with a cage |
| :-- | --: | --: | --: | --: | --: | --: |
| `net` (unchanged) | **+2.15** | +2.12 | 2.11 | 14% | 5.39 | 69% |
| `net`, announcements muted | +1.81 | +2.18 | 0 | — | 0 | — |
| `quiet` (rule movement, no announcements) | +1.34 | +1.75 | 0 | — | 0 | — |
| `bluff, free` (announce only when already moving to a card) | +1.24 | +1.83 | 2.45 | 5% | 6.65 | 77% |
| `bluff, forced` (announce on every opportunity) | +0.91 | +1.88 | 2.98 | 1% | 7.62 | 81% |

Paired, same seeds (`bootstrap` on per-game differences):

| comparison | paired change |
| :-- | :-- |
| `net` without its announcements − `net` | **−0.41 [−0.67, −0.16]** |
| `bluff, free` − `quiet` | −0.18 [−0.46, +0.11] |
| `bluff, forced` − `quiet` | **−0.56 [−0.84, −0.27]** |

Three things fall out.

**The announce-and-cancel turns are not lost.** Muting the unchanged net's
announcements costs it **0.41 points**, the same order as the ~0.4 points §6 says
a pending blast is worth to the blaster's rivals. The announcement is the
weapon; the explosion is optional. Cancelling costs the announcer nothing but a
move it was going to make anyway, and the dynamite is only consumed by blasts
that actually stand at the end of the planning phase.

**Firing is the part that does not pay.** Under the printed (blind) blast rule a
fired blast is worth +0.70 against +0.61 for an ordinary haul (§1) — a wash for
a 38–58% chance of waking the dragon. So a seat that collects dynamite, threatens
and declines to fire is running the mechanic at its best terms, not wasting it.

**The flinch is cheap for the fleer, which is why it cannot be farmed.** Across
the five arms the *rivals'* mean score barely moves (1.75 → 2.18), even though
they answer up to 81% of 7.6 threatened decisions a game with a cage. That is not
a contradiction: a pit cage is the only exit that **banks the loot**, so a seat
that flees converts a provisional position into points. It gives up the card, not
the score. And an agent that pays for announcements — forcing a card move it did
not want — loses 0.56 points doing it. The flinch is real, and it is not worth
buying.

**What is actually irrational, then,** is narrower than it looks, and it is not a
choice any seat makes:

* the reaction is a **mask rule** (`round_dragon_hazard` = 1.0 for *any* pending
  blast), so it cannot tell a seat that fires 14% of its announcements from one
  that fires every time. Every seat applies the same prior, which is why all four
  flinch together;
* because no seat can ever call a bluff, bluffing can never be profitable, so the
  equilibrium cannot move in either direction. The table pays a small deadweight
  cost (cages taken for nothing) and no one can exploit it;
* the residue §6 pointed at survives: on threatened decisions where the cage is
  *already* legal, the policy reads nothing (TV 0.0006) and still passes 25% of
  the time.

Limits: the two bluff arms run *rule* movement, so they are compared with their
own `quiet` control, not with the net; 400 games per arm on one checkpoint and
one seed; and "share that fired" counts blasts that stood when the planning phase
closed, which is the only way a bluff becomes a real blast.
