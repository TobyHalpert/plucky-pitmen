# Seed 7, haul 2: P2's push, blast and cancellation

Measured using the shipped `selfplay4.npz`, current next-card rule and the
known-dragon safeguard added in `c77b961`. Four neural seats, three hauls.

## Exact watched position

P1 leads 3–0–0–0 and first steps onto r0c0. P2 has no gems and two dynamite;
it can legally push P1 onto P1's lorry, or take its own known gem at r0c1.
The actual r0c0 is a dragon, but **P2 cannot see that**. No mask blocks the push.
The policy assigns about 60% to gem+blast and 27% to gem without a blast.
It samples gem+blast. P3 also announces a blast, P4 takes a cage and P1 takes
a cage. P2 then cancels its own blast by taking a cage. Later, with both blasts
cancelled, P2 moves to the lorry.

## Method and limits

The pre-fork history is replayed exactly from the viewer. Each action is forced
once and the unchanged neural policies then play to the end of the game. 400
continuations per arm, seeds 920000–920399 mapped to independent per-seat policy
RNGs, identically across arms. Board and deck remain the actual watched ones.
Only future policy sampling is rerolled. All 3,200 continuations completed.

**This is not a belief-state comparison.** It holds the hidden dragon fixed;
it does not average over cards consistent with P2's knowledge. A poor push here
cannot prove that P2 should have rejected that push before seeing the card.
Results also depend on these imperfect continuation policies. Outright wins
mean a strictly higher final score than all rivals; tied tops are separate, not
resolved with the game's tie breaker.

```sh
python probe_seed7_p2.py --samples 400
```

## P2's initial choice (same pre-action position)

| choice | outright wins | tied top | mean P2 final score | mean P1 final score |
| :-- | --: | --: | --: | --: |
| own gem, no blast | 9.50% | 3.25% | 1.670 | 3.923 |
| own gem + blast (recorded) | 8.25% | 1.75% | 1.083 | 3.755 |
| push P1, no blast | 4.25% | 1.75% | 1.515 | 4.820 |
| push P1 + blast | 9.00% | 1.75% | 1.113 | 3.750 |

Paired win changes versus gem without blast: −1.25 percentage points (give or
take 1.82), −5.25 (1.17), and −0.50 (1.87). Give-or-take numbers are one
standard error over paired continuation samples, not confidence bounds.
The no-blast push is worse on this actual hidden board. The blast announcement
does not show a clear win advantage over taking the gem without a blast.
Pushing P1 off row 0 denies its card **this round**, not its entire haul:
its lorry does not prevent collecting again next round. Nor does pushing take
any of its existing loot.

## P2's next turn: keep the blast or cancel it?

This forks a different position: all earlier recorded actions, including P3's
blast and P1/P4's cage choices, have happened. Pass is legal and policy-allowed.

| choice | outright wins | tied top | mean P2 score | mean P3 score |
| :-- | --: | --: | --: | --: |
| cage, cancel blast (recorded) | 9.25% | 0.75% | 1.490 | 3.060 |
| pass, retain blast | 9.25% | 1.50% | 0.873 | 4.663 |

No outright-win difference, give or take 2.00 percentage points. Neither this
result nor the initial-choice result establishes a coherent benefit to the
announce-then-cancel sequence. It also does not prove cancellation alone loses
wins: continuing the blast was not better here. Retaining it does not force it
to fire immediately; later decisions remain the ordinary policy's choices.

## Why the subsequent lorry move happens

Once P2 is in the cage and both blasts are cancelled, its hand has zero gems.
`cage_banks_nothing` reports no loot and no live threat; `action_mask` suppresses
PASS from the cage and enables LORRY. Thus the lorry is not evidence that the net
intended a useful feint. The mask re-evaluates shelter after the threat vanishes
and refuses the empty departure. Earlier cancellation was a learned decision;
the subsequent departure from shelter is constrained by the hand-written mask.

One more fork at that exact cage position:

| choice | outright wins | tied top | mean P2 score |
| :-- | --: | --: | --: |
| lorry (recorded) | 10.50% | 0.75% | 1.495 |
| pass, leave empty (engine-legal, policy-filtered) | 12.00% | 0.75% | 1.565 |

Difference +1.50 percentage points for leaving empty, give or take 1.27. This
small result does not settle the general mask rule. Blindly committing every
cage arrival or banning blast cancellations would be new restrictions, not a
proven repair. The concrete problem exposed is a policy that frequently
announces a blast, then cancels it, interacting with a shelter mask that forces
reconsideration. No new rule, mask, reward or checkpoint change was made for
this diagnostic. In particular, legitimate shelter from a charged blast remains
available.
