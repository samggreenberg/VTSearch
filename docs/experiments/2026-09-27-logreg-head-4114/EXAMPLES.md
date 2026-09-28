# Literal examples: same 40 Autopilot votes, the SVM and `lrconv`

Each block refits the shipped SVM and the `lrconv` head (logistic regression fitted to convergence, C = 1) on the vote set the **SVM arm's** Autopilot collected by click 40, and scores the harness's held-out test half. The cells are the ones where the two in-loop arms' oracle costs differed most in the SVM's favour (one per environment, then one per environment in the other arm's favour). Ranks are 1 = top of the haystack; `in-loop gap` is the `lrconv` arm's oracle cost minus the SVM arm's at that click in Stage B.

## coco_better × siglip — `knife@large`, seed 2

Votes: 4 Good / 36 Bad. Held-out: 51 positives in 12252. AP on these votes: SVM 0.16, lrconv 0.16. In-loop gap at click 40: +0.25.

| kind | file | label | SVM rank | lrconv rank |
|---|---|---|---|---|
| SVM wins (positive ranked higher) | `000000185236.jpg` | knife@large | 187 | 257 |
| SVM wins (positive ranked higher) | `000000561517.jpg` | knife@large | 318 | 384 |
| SVM wins (positive ranked higher) | `000000502529.jpg` | knife@large | 266 | 298 |
| SVM losses (negative ranked higher) | `000000554800.jpg` | skateboard@large | 490 | 737 |
| SVM losses (negative ranked higher) | `000000223598.jpg` | banana@large | 445 | 669 |
| SVM losses (negative ranked higher) | `000000530146.jpg` | single serving drinking vessel@medium | 349 | 530 |
| lrconv wins (positive ranked higher) | `000000232627.jpg` | knife@large | 345 | 260 |
| lrconv wins (positive ranked higher) | `000000476934.jpg` | knife@large | 297 | 218 |
| lrconv wins (positive ranked higher) | `000000022340.jpg` | knife@large | 514 | 480 |
| lrconv losses (negative ranked higher) | `000000057739.jpg` | scissors@large | 743 | 388 |
| lrconv losses (negative ranked higher) | `000000400606.jpg` | scissors@medium | 633 | 499 |
| lrconv losses (negative ranked higher) | `000000040716.jpg` | spoon@small | 575 | 447 |

## coco_better × siglip — `cell phone@medium`, seed 3

Votes: 5 Good / 35 Bad. Held-out: 38 positives in 12252. AP on these votes: SVM 0.18, lrconv 0.20. In-loop gap at click 40: +0.14.

| kind | file | label | SVM rank | lrconv rank |
|---|---|---|---|---|
| SVM wins (positive ranked higher) | `000000472602.jpg` | cell phone@medium | 164 | 259 |
| SVM wins (positive ranked higher) | `000000555755.jpg` | cell phone@medium | 490 | 571 |
| SVM wins (positive ranked higher) | `000000150821.jpg` | cell phone@medium | 58 | 114 |
| SVM losses (negative ranked higher) | `000000257497.jpg` | scissors@large | 459 | 1355 |
| SVM losses (negative ranked higher) | `000000147000.jpg` | kite@large | 421 | 1095 |
| SVM losses (negative ranked higher) | `000000427870.jpg` | skateboard@medium | 491 | 1038 |
| lrconv wins (positive ranked higher) | `000000315393.jpg` | cell phone@medium | 542 | 265 |
| lrconv wins (positive ranked higher) | `000000158540.jpg` | cell phone@medium | 299 | 173 |
| lrconv wins (positive ranked higher) | `000000573815.jpg` | cell phone@medium | 297 | 192 |
| lrconv losses (negative ranked higher) | `000000503557.jpg` | dining table@medium | 1292 | 462 |
| lrconv losses (negative ranked higher) | `000000041311.jpg` | tie@large | 1224 | 414 |
| lrconv losses (negative ranked higher) | `000000400528.jpg` | scissors@small | 1095 | 441 |

## coco_better × siglip — `traffic light@small`, seed 1

Votes: 3 Good / 37 Bad. Held-out: 50 positives in 12252. AP on these votes: SVM 0.12, lrconv 0.12. In-loop gap at click 40: +0.04.

| kind | file | label | SVM rank | lrconv rank |
|---|---|---|---|---|
| SVM wins (positive ranked higher) | `000000165202.jpg` | traffic light@small | 384 | 436 |
| SVM wins (positive ranked higher) | `000000371114.jpg` | traffic light@small | 135 | 172 |
| SVM wins (positive ranked higher) | `000000441433.jpg` | traffic light@small | 428 | 448 |
| SVM losses (negative ranked higher) | `000000540404.jpg` | enclosed road vehicle@large | 416 | 702 |
| SVM losses (negative ranked higher) | `000000066182.jpg` | airplane@medium | 438 | 652 |
| SVM losses (negative ranked higher) | `000000296265.jpg` | airplane@small | 399 | 580 |
| lrconv wins (positive ranked higher) | `000000039321.jpg` | traffic light@small | 377 | 285 |
| lrconv wins (positive ranked higher) | `000000079230.jpg` | traffic light@small | 425 | 336 |
| lrconv wins (positive ranked higher) | `000000239509.jpg` | traffic light@small | 259 | 188 |
| lrconv losses (negative ranked higher) | `000000236182.jpg` | enclosed road vehicle@small | 618 | 424 |
| lrconv losses (negative ranked higher) | `000000503957.jpg` | motorcycle@medium | 658 | 469 |
| lrconv losses (negative ranked higher) | `000000406152.jpg` | bag or luggage@small | 656 | 476 |

## coco_better × siglip — `toothbrush@small`, seed 0

Votes: 3 Good / 37 Bad. Held-out: 48 positives in 12252. AP on these votes: SVM 0.08, lrconv 0.08. In-loop gap at click 40: -0.28.

| kind | file | label | SVM rank | lrconv rank |
|---|---|---|---|---|
| SVM wins (positive ranked higher) | `000000478115.jpg` | toothbrush@small | 113 | 150 |
| SVM wins (positive ranked higher) | `000000030983.jpg` | toothbrush@small | 63 | 83 |
| SVM wins (positive ranked higher) | `000000563574.jpg` | toothbrush@small | 321 | 340 |
| SVM losses (negative ranked higher) | `000000055567.jpg` |  | 458 | 800 |
| SVM losses (negative ranked higher) | `000000314758.jpg` | frisbee@medium | 489 | 782 |
| SVM losses (negative ranked higher) | `000000426261.jpg` | enclosed road vehicle@small | 496 | 721 |
| lrconv wins (positive ranked higher) | `000000238688.jpg` | toothbrush@small | 478 | 322 |
| lrconv wins (positive ranked higher) | `000000551036.jpg` | toothbrush@small | 407 | 319 |
| lrconv wins (positive ranked higher) | `000000230735.jpg` | toothbrush@small | 392 | 316 |
| lrconv losses (negative ranked higher) | `000000233264.jpg` |  | 903 | 494 |
| lrconv losses (negative ranked higher) | `000000390061.jpg` | scissors@small | 653 | 428 |
| lrconv losses (negative ranked higher) | `000000538380.jpg` | toothbrush@large | 647 | 448 |

## coco_better × siglip — `bowl@large`, seed 0

Votes: 3 Good / 37 Bad. Held-out: 57 positives in 12252. AP on these votes: SVM 0.06, lrconv 0.06. In-loop gap at click 40: -0.27.

| kind | file | label | SVM rank | lrconv rank |
|---|---|---|---|---|
| SVM wins (positive ranked higher) | `000000412036.jpg` | bowl@large | 198 | 336 |
| SVM wins (positive ranked higher) | `000000135628.jpg` | bowl@large | 444 | 571 |
| SVM wins (positive ranked higher) | `000000215587.jpg` | bowl@large | 422 | 505 |
| SVM losses (negative ranked higher) | `000000123764.jpg` |  | 273 | 676 |
| SVM losses (negative ranked higher) | `000000247624.jpg` |  | 378 | 707 |
| SVM losses (negative ranked higher) | `000000370857.jpg` | bird@large | 413 | 737 |
| lrconv wins (positive ranked higher) | `000000215249.jpg` | bowl@large | 309 | 206 |
| lrconv wins (positive ranked higher) | `000000423815.jpg` | bowl@large, knife@large | 203 | 164 |
| lrconv wins (positive ranked higher) | `000000231087.jpg` | bowl@large | 154 | 125 |
| lrconv losses (negative ranked higher) | `000000056212.jpg` |  | 806 | 393 |
| lrconv losses (negative ranked higher) | `000000286892.jpg` | orange@large | 802 | 401 |
| lrconv losses (negative ranked higher) | `000000279780.jpg` | sink@large | 856 | 475 |

## coco_better × siglip — `remote@medium`, seed 2

Votes: 3 Good / 37 Bad. Held-out: 51 positives in 12252. AP on these votes: SVM 0.05, lrconv 0.05. In-loop gap at click 40: -0.27.

| kind | file | label | SVM rank | lrconv rank |
|---|---|---|---|---|
| SVM wins (positive ranked higher) | `000000399054.jpg` | remote@medium | 184 | 237 |
| SVM wins (positive ranked higher) | `000000188824.jpg` | remote@medium | 497 | 517 |
| SVM wins (positive ranked higher) | `000000051993.jpg` | remote@medium | 46 | 59 |
| SVM losses (negative ranked higher) | `000000286285.jpg` | scissors@medium | 326 | 544 |
| SVM losses (negative ranked higher) | `000000103018.jpg` | chair@large | 454 | 663 |
| SVM losses (negative ranked higher) | `000000328084.jpg` |  | 495 | 673 |
| lrconv wins (positive ranked higher) | `000000212420.jpg` | chair@large, remote@medium | 435 | 403 |
| lrconv wins (positive ranked higher) | `000000495692.jpg` | remote@medium | 327 | 304 |
| lrconv wins (positive ranked higher) | `000000053800.jpg` | remote@medium | 490 | 472 |
| lrconv losses (negative ranked higher) | `000000118199.jpg` | scissors@medium | 660 | 479 |
| lrconv losses (negative ranked higher) | `000000040569.jpg` | tie@medium | 640 | 476 |
| lrconv losses (negative ranked higher) | `000000251723.jpg` | mouse@small, tv@medium | 478 | 337 |
