<!-- _class: caveat -->

### Iteration 4 — a caution we keep

## Run A shipped from the edge of its own grid

- κ = 1 won run A — the **smallest κ it measured**
- Run B, two decades wider: the optimum is interior, **κ = 0.3**, 6/6
- And κ\* keeps falling as votes accumulate: 3 → 0.1 over the horizon

<!-- This is a deliberate honest-lessons slide; play it straight, not as a
     confession. Run A swept κ over 1, 3, 10, 30, 100, and κ = 1 won — the
     smallest value on the grid, which should have been read as "the optimum
     is off the edge", not "the optimum is 1". Instead the recommendation
     merged to production about an hour into run B's cell array. Run B, with
     the grid extended two decades down, found the true optimum interior at
     κ = 0.3, winning in all six environments — so the shipped setting was
     beaten in every environment measured, within a day of shipping.

     Two lessons to state explicitly. Process: a winner on the edge of its own
     grid is a prompt to extend the grid, and shipping before the wider run
     finishes is how you ship from the edge. Substance: κ* is not even a
     constant — it keeps falling as votes accumulate, from around 3 early to
     around 0.1 deep, so any fixed κ is a compromise across regimes. A
     schedule κ ∝ 1/n is the obvious one-line follow-up arm, and it reappears
     on the roadmap slide. -->
