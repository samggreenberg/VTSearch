<!-- _class: full -->

![bg fit](figs/test-draws.png)

## Luck of<br>the Draw

<!-- build: figs/test-draws.build1.png -->

<!-- build: figs/test-draws.build2.png -->

<!-- **a** — Every band has its own curve now. How do we get one range for the
     whole line? Simulate. One draw picks a possible match count for every band
     at the same time: the three bands above the line get a column each, and
     everything below the line is summed into one. Then do it again, 4,000
     times. -->

<!-- **b** — Each draw is a whole possible pile, so it has its own Right, Found
     and F1. Draw one keeps 17 matches and leaves 12 below: 17 of 32 is 53%
     right, 17 of 29 is 59% found, F1 0.56. -->

<!-- **c** — Take all 4,000 and keep the middle 95% of each column: Right
     likely 44 to 66%, Found 26 to 70%, F1 0.35 to 0.67. The truth we planted
     is 56%, 49% and 0.52, inside all three. And since every number comes from
     the same draws, they can never contradict each other. -->
