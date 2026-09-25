# Literal examples (#3945): the same 300 Autopilot votes at C = 1 and C = 0.1

Each block refits the shipped head at C = 1 and at C = 0.1 on the vote set the shipped arm's Autopilot collected by click 300, and scores the harness's held-out half. The sessions are the ones where C = 0.1 beats C = 1 by the most oracle cost at that click, one per environment. Ranks are 1 = top of the haystack. `labels` is the image's full annotation, so a 'loss' that plainly contains the category is an annotation gap, not a model error.

## visual_genome_m × siglip: `nose`, seed 0

Votes: 44 Good / 256 Bad. Held-out: 77 positives in 2097. AP: C = 1 0.27, C = 0.1 0.32. Oracle cost gap: +0.16.

| kind | file | labels | rank at C = 1 | rank at C = 0.1 |
|---|---|---|---|---|
| C = 0.1 wins (positive lifted) | `2657.jpg` | bird, eye, grass, leg, nose | 1238 | 442 |
| C = 0.1 wins (positive lifted) | `285681.jpg` | arm, cap, giraffe, glass, hand, hat, head, jacket | 953 | 245 |
| C = 0.1 wins (positive lifted) | `3984.jpg` | arm, bench, chair, ear, glass, hand, head, man | 1066 | 371 |
| C = 0.1 wins (positive lifted) | `285985.jpg` | head, mountain, nose, rock, roof, tree | 1106 | 489 |
| C = 0.1 wins (positive lifted) | `285998.jpg` | bed, boy, eye, hair, nose | 775 | 163 |
| C = 0.1 losses (negative lifted) | `497927.jpg` | table | 1886 | 361 |
| C = 0.1 losses (negative lifted) | `3531.jpg` | bowl, box, cup, plate | 1782 | 396 |
| C = 0.1 losses (negative lifted) | `150531.jpg` | book, building, chair, floor, wall | 1910 | 545 |
| C = 0.1 losses (negative lifted) | `712966.jpg` | glass, laptop | 1662 | 325 |
| C = 0.1 losses (negative lifted) | `3522.jpg` | box, floor, wall, window | 1712 | 449 |

## visual_genome_m × siglip2_l: `nose`, seed 4

Votes: 50 Good / 250 Bad. Held-out: 68 positives in 2097. AP: C = 1 0.26, C = 0.1 0.29. Oracle cost gap: +0.10.

| kind | file | labels | rank at C = 1 | rank at C = 0.1 |
|---|---|---|---|---|
| C = 0.1 wins (positive lifted) | `498142.jpg` | hand, head, laptop, man, nose, pants, person, shirt | 1621 | 813 |
| C = 0.1 wins (positive lifted) | `285812.jpg` | ear, face, glass, hair, hand, leaf, man, nose | 804 | 321 |
| C = 0.1 wins (positive lifted) | `712998.jpg` | ball, boy, ear, fence, hair, hand, nose, shirt | 882 | 418 |
| C = 0.1 wins (positive lifted) | `285998.jpg` | bed, boy, eye, hair, nose | 767 | 324 |
| C = 0.1 wins (positive lifted) | `3598.jpg` | bowl, box, door, man, nose, shirt, table | 550 | 163 |
| C = 0.1 losses (negative lifted) | `4847.jpg` | glass, light, table | 1775 | 845 |
| C = 0.1 losses (negative lifted) | `3875.jpg` | plate, table | 1669 | 784 |
| C = 0.1 losses (negative lifted) | `4969.jpg` | boy, cup, glass, hair, plate, shirt, table | 1686 | 819 |
| C = 0.1 losses (negative lifted) | `150262.jpg` | table | 1555 | 705 |
| C = 0.1 losses (negative lifted) | `3871.jpg` | table | 1842 | 997 |

## coco_val × siglip: `scissors`, seed 0

Votes: 9 Good / 291 Bad. Held-out: 16 positives in 2476. AP: C = 1 0.55, C = 0.1 0.52. Oracle cost gap: +0.06.

| kind | file | labels | rank at C = 1 | rank at C = 0.1 |
|---|---|---|---|---|
| C = 0.1 wins (positive lifted) | `000000173008.jpg` | banana, cake, cup, microwave, orange, oven, person, scissors | 432 | 202 |
| C = 0.1 wins (positive lifted) | `000000245576.jpg` | cat, cell phone, cup, keyboard, scissors, tv | 568 | 409 |
| C = 0.1 wins (positive lifted) | `000000298396.jpg` | banana, bowl, chair, clock, dining table, knife, oven, scissors | 405 | 280 |
| C = 0.1 wins (positive lifted) | `000000231831.jpg` | book, cat, cell phone, chair, remote, scissors | 25 | 21 |
| C = 0.1 wins (positive lifted) | `000000099024.jpg` | kite, person, scissors | 3 | 2 |
| C = 0.1 losses (negative lifted) | `000000346703.jpg` | cake, couch, cup, person, spoon, wine glass | 2231 | 1062 |
| C = 0.1 losses (negative lifted) | `000000541773.jpg` | bottle, couch, dining table, person, wine glass | 2061 | 947 |
| C = 0.1 losses (negative lifted) | `000000213255.jpg` | bottle, car, hot dog, person, sandwich, traffic light | 1928 | 814 |
| C = 0.1 losses (negative lifted) | `000000302536.jpg` | couch, person, remote, sports ball | 1490 | 540 |
| C = 0.1 losses (negative lifted) | `000000533206.jpg` | bottle, bowl, fork, knife, sandwich, wine glass | 1830 | 910 |

## coco_val × siglip2_l: `cell phone`, seed 1

Votes: 66 Good / 234 Bad. Held-out: 122 positives in 2476. AP: C = 1 0.75, C = 0.1 0.75. Oracle cost gap: +0.04.

| kind | file | labels | rank at C = 1 | rank at C = 0.1 |
|---|---|---|---|---|
| C = 0.1 wins (positive lifted) | `000000245576.jpg` | cat, cell phone, cup, keyboard, scissors, tv | 1157 | 428 |
| C = 0.1 wins (positive lifted) | `000000366225.jpg` | cell phone, keyboard, mouse, tv | 1145 | 431 |
| C = 0.1 wins (positive lifted) | `000000176799.jpg` | cell phone, person, skateboard | 1441 | 782 |
| C = 0.1 wins (positive lifted) | `000000119233.jpg` | bird, cat, cell phone, cup, laptop, spoon | 1292 | 689 |
| C = 0.1 wins (positive lifted) | `000000569059.jpg` | cell phone, chair, keyboard, mouse, person, tv | 1363 | 773 |
| C = 0.1 losses (negative lifted) | `000000122046.jpg` | chair, person, umbrella | 1790 | 510 |
| C = 0.1 losses (negative lifted) | `000000001503.jpg` | keyboard, laptop, mouse, tv | 2274 | 994 |
| C = 0.1 losses (negative lifted) | `000000423944.jpg` | tie | 1725 | 462 |
| C = 0.1 losses (negative lifted) | `000000339823.jpg` | person, umbrella | 1991 | 803 |
| C = 0.1 losses (negative lifted) | `000000378099.jpg` | keyboard, mouse | 2102 | 925 |
