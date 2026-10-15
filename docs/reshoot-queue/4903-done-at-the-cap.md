# #4903: Autopilot stops every session by 200 votes

Done now also comes once a session holds 200 votes (after the check before Done), and the panel and Toasty say
"You have cast 200 votes, as many as Autopilot asks for" when the cap, not the lights or the level, said Done. A
session still looking for its first Goods at 200 votes gets an **Autopilot has stopped** note above its steps and
the **Update Sort Example?** prompt once more. Shots of a short session are unchanged; a staged long one may land
on the new wording.

- `autopilot-progress`: the phase panel's Refine Boundary and Explore Diversity tooltips add "or at 200 votes"
- `unstick-prompt`: unchanged; the how-to's text now says the prompt comes again at the cap
