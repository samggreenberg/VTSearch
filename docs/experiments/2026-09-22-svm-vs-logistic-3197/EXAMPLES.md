# Literal examples (#3197): same 40 Autopilot votes, two heads

Each block refits the shipped SVM and the shipped logistic head on the vote set the **SVM arm's** Autopilot collected by click 40, and scores the harness's held-out test half. The cells are the ones where the two in-loop arms' oracle costs differed most in the SVM's favour (one per environment). Ranks are 1 = top of the haystack; `in-loop gap` is the logistic arm's oracle cost minus the SVM arm's at that click in Stage B.

## visual_genome_m × siglip2_l — `neck`, seed 0

Votes: 5 Good / 35 Bad. Held-out: 20 positives in 2097. AP on these votes: SVM 0.09, logistic 0.08. In-loop gap at click 40: +0.37.

| kind | file | label | SVM rank | logistic rank |
|---|---|---|---|---|
| SVM wins (positive ranked higher) | `285669.jpg` | arm, boy, ear, eye, girl, hair | 262 | 917 |
| SVM wins (positive ranked higher) | `498019.jpg` | dog, floor, glass, hand, head, light | 408 | 1031 |
| SVM wins (positive ranked higher) | `498002.jpg` | bottle, bowl, face, flower, glass, hair | 340 | 875 |
| SVM wins (positive ranked higher) | `4951.jpg` | dog, eye, head, neck, table | 341 | 669 |
| SVM wins (positive ranked higher) | `713031.jpg` | building, man, neck, wheel | 312 | 529 |
| SVM losses (negative ranked higher) | `150418.jpg` | bench, eye, horse, pants, tail, woman | 396 | 1311 |
| SVM losses (negative ranked higher) | `2363.jpg` | girl, hair, jacket, man, pants, people | 325 | 1200 |
| SVM losses (negative ranked higher) | `2865.jpg` | building, wall | 898 | 1735 |
| SVM losses (negative ranked higher) | `285681.jpg` | arm, cap, giraffe, glass, hand, hat | 562 | 1364 |
| SVM losses (negative ranked higher) | `285655.jpg` | boat, line, pants, snow | 606 | 1407 |

## visual_genome_m × siglip — `hand`, seed 3

Votes: 9 Good / 31 Bad. Held-out: 117 positives in 2097. AP on these votes: SVM 0.22, logistic 0.20. In-loop gap at click 40: +0.34.

| kind | file | label | SVM rank | logistic rank |
|---|---|---|---|---|
| SVM wins (positive ranked higher) | `4206.jpg` | ear, eye, face, hair, hand, head | 457 | 1345 |
| SVM wins (positive ranked higher) | `61580.jpg` | building, ear, glass, hand, hat, head | 941 | 1636 |
| SVM wins (positive ranked higher) | `3521.jpg` | arm, bag, chair, door, face, hand | 472 | 1139 |
| SVM wins (positive ranked higher) | `285894.jpg` | boy, ear, face, glass, hair, hand | 566 | 1186 |
| SVM wins (positive ranked higher) | `2422.jpg` | boy, building, clock, face, hand, jacket | 1451 | 2054 |
| SVM losses (negative ranked higher) | `150456.jpg` | bag, building, cap, clock, hat, letter | 732 | 1793 |
| SVM losses (negative ranked higher) | `2440.jpg` | building, sign, sky, window | 467 | 1455 |
| SVM losses (negative ranked higher) | `2417.jpg` | building, car, jacket, light, line, man | 426 | 1383 |
| SVM losses (negative ranked higher) | `2340.jpg` | building, jacket, man, person, window, woman | 1007 | 1941 |
| SVM losses (negative ranked higher) | `4208.jpg` | building, bush, fence, letter, sign, sky | 820 | 1727 |

## coco_val × siglip2_l — `microwave`, seed 1

Votes: 3 Good / 37 Bad. Held-out: 28 positives in 2476. AP on these votes: SVM 0.34, logistic 0.32. In-loop gap at click 40: +0.25.

| kind | file | label | SVM rank | logistic rank |
|---|---|---|---|---|
| SVM wins (positive ranked higher) | `000000441247.jpg` | chair, vase, couch, dining table, orange, oven | 407 | 1367 |
| SVM wins (positive ranked higher) | `000000000139.jpg` | chair, vase, book, person, tv, clock | 564 | 1406 |
| SVM wins (positive ranked higher) | `000000216497.jpg` | chair, couch, dining table, microwave, oven, sink | 400 | 1221 |
| SVM wins (positive ranked higher) | `000000540502.jpg` | orange, apple, bowl, chair, dining table, vase | 301 | 639 |
| SVM wins (positive ranked higher) | `000000571313.jpg` | bed, chair, dining table, keyboard, microwave, mouse | 112 | 289 |
| SVM losses (negative ranked higher) | `000000195842.jpg` | couch, remote, bowl, chair, clock, person | 387 | 2011 |
| SVM losses (negative ranked higher) | `000000022192.jpg` | bed, dog, handbag | 633 | 2162 |
| SVM losses (negative ranked higher) | `000000413247.jpg` | backpack, book, bottle, laptop, mouse | 346 | 1871 |
| SVM losses (negative ranked higher) | `000000465718.jpg` | person, keyboard, tv, cell phone, laptop, mouse | 680 | 2088 |
| SVM losses (negative ranked higher) | `000000559543.jpg` | chair, couch, person, remote, vase | 1009 | 2358 |

## coco_val × siglip — `dining table`, seed 3

Votes: 14 Good / 26 Bad. Held-out: 255 positives in 2476. AP on these votes: SVM 0.63, logistic 0.56. In-loop gap at click 40: +0.15.

| kind | file | label | SVM rank | logistic rank |
|---|---|---|---|---|
| SVM wins (positive ranked higher) | `000000549167.jpg` | broccoli, dining table | 496 | 1591 |
| SVM wins (positive ranked higher) | `000000242060.jpg` | cake, spoon, cup, dining table | 497 | 1575 |
| SVM wins (positive ranked higher) | `000000132116.jpg` | broccoli, bowl, dining table | 534 | 1596 |
| SVM wins (positive ranked higher) | `000000529568.jpg` | wine glass, cup, bowl, book, vase, dining table | 564 | 1601 |
| SVM wins (positive ranked higher) | `000000308430.jpg` | carrot, book, bowl, dining table, oven | 844 | 1871 |
| SVM losses (negative ranked higher) | `000000161397.jpg` | vase, person | 730 | 2091 |
| SVM losses (negative ranked higher) | `000000458768.jpg` | oven, bench, couch, sink | 1016 | 2335 |
| SVM losses (negative ranked higher) | `000000300233.jpg` | spoon, bottle, bowl | 840 | 2131 |
| SVM losses (negative ranked higher) | `000000356347.jpg` | bowl, spoon | 932 | 2214 |
| SVM losses (negative ranked higher) | `000000354072.jpg` | bottle, potted plant, sink | 824 | 2088 |

## caltech101_m × siglip — `airplanes`, seed 0

Votes: 15 Good / 25 Bad. Held-out: 117 positives in 419. AP on these votes: SVM 1.00, logistic 1.00. In-loop gap at click 40: +0.00.

| kind | file | label | SVM rank | logistic rank |
|---|---|---|---|---|
| SVM wins (positive ranked higher) | `airplanes/image_0173.jpg` | airplanes | 40 | 100 |
| SVM wins (positive ranked higher) | `airplanes/image_0135.jpg` | airplanes | 34 | 92 |
| SVM wins (positive ranked higher) | `airplanes/image_0214.jpg` | airplanes | 9 | 66 |
| SVM wins (positive ranked higher) | `airplanes/image_0234.jpg` | airplanes | 21 | 75 |
| SVM wins (positive ranked higher) | `airplanes/image_0126.jpg` | airplanes | 23 | 70 |
| SVM losses (negative ranked higher) | `trilobite/image_0027.jpg` | trilobite | 172 | 271 |
| SVM losses (negative ranked higher) | `cougar_face/image_0024.jpg` | cougar_face | 207 | 302 |
| SVM losses (negative ranked higher) | `cougar_face/image_0022.jpg` | cougar_face | 273 | 363 |
| SVM losses (negative ranked higher) | `kangaroo/image_0020.jpg` | kangaroo | 220 | 309 |
| SVM losses (negative ranked higher) | `cougar_face/image_0017.jpg` | cougar_face | 253 | 342 |

## caltech101_m × siglip2_l — `car_side`, seed 0

Votes: 3 Good / 37 Bad. Held-out: 22 positives in 419. AP on these votes: SVM 1.00, logistic 1.00. In-loop gap at click 40: +0.00.

| kind | file | label | SVM rank | logistic rank |
|---|---|---|---|---|
| SVM wins (positive ranked higher) | `car_side/image_0036.jpg` | car_side | 10 | 14 |
| SVM wins (positive ranked higher) | `car_side/image_0023.jpg` | car_side | 12 | 15 |
| SVM wins (positive ranked higher) | `car_side/image_0029.jpg` | car_side | 7 | 9 |
| SVM wins (positive ranked higher) | `car_side/image_0045.jpg` | car_side | 6 | 8 |
| SVM wins (positive ranked higher) | `car_side/image_0031.jpg` | car_side | 15 | 17 |
| SVM losses (negative ranked higher) | `airplanes/image_0148.jpg` | airplanes | 90 | 348 |
| SVM losses (negative ranked higher) | `airplanes/image_0201.jpg` | airplanes | 138 | 382 |
| SVM losses (negative ranked higher) | `airplanes/image_0128.jpg` | airplanes | 61 | 303 |
| SVM losses (negative ranked higher) | `airplanes/image_0153.jpg` | airplanes | 96 | 337 |
| SVM losses (negative ranked higher) | `airplanes/image_0213.jpg` | airplanes | 99 | 332 |
