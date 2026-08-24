![bg right:56% fit](figs/ui-three-panel.webp)

### What it is

## Search, judge, repeat

- Seed with **text** or an **example item**
- Judge what comes back: **Good** or **Bad**
- The ranking updates in seconds — then it asks again

<!-- Walk the three panels left to right, slowly; this is the audience's
     first sight of the tool.

     Say what is on the screen first, because it is a real session and it
     helps: a few hundred photographs, and someone about ten votes into
     looking for the cats in them. Nothing was labelled when they started.

     Left: the corpus, ordered by how well it currently matches. Nothing here
     is labelled — this is just everything the user has. Middle: one item,
     large, with two buttons under it. That is the entire interaction the user
     performs — Good or Bad, one item at a time. Right: the votes cast so far,
     kept as two piles.

     The loop starts from a seed, because a model with no votes has nothing to
     rank by: type a phrase, or point at one example item you already have.
     From then on it is vote, retrain, re-rank. A retrain is a fraction of a
     second, because the thing being trained is a small linear head on top of
     frozen embeddings — the heavy model never moves.

     Worth pointing at, because it is where the next slides go: the "Select"
     control on the left panel. The user is not scrolling a result list
     choosing what to judge. The system is choosing what to put in front of
     them. -->
