# #3825 gate tables

## gate_by_arm

| arm       | cases | rate_reportable | thr_moved_pct | admitted_moved_pct | d_thr_median | d_thr_p90 | d_adm_frac_median | d_adm_frac_p90 | d_adm_frac_max | d_adm_median_of_changed | prov_changed | cut_speedup_median |
|-----------|-------|-----------------|---------------|--------------------|--------------|-----------|-------------------|----------------|----------------|-------------------------|--------------|--------------------|
| ll1e-3    | 2258  | yes             | 80.5          | 80.4               | 0.00751      | 0.0346    | 0.033             | 0.203          | 0.468          | 122                     | 825          | 8.08               |
| ll1e-4    | 2258  | yes             | 79.6          | 79.5               | 0.00401      | 0.0144    | 0.0146            | 0.066          | 0.302          | 53                      | 821          | 4.53               |
| ll1e-5    | 2258  | yes             | 77.1          | 77.1               | 0.00143      | 0.00514   | 0.00527           | 0.0243         | 0.0882         | 18                      | 813          | 3.08               |
| ll1e-6    | 2258  | yes             | 72.1          | 72.1               | 0.000394     | 0.00164   | 0.0015            | 0.00699        | 0.0315         | 6                       | 796          | 2.28               |
| ll1e-7    | 2258  | yes             | 58.2          | 58.2               | 6.55e-05     | 0.000562  | 0.00042           | 0.00197        | 0.00539        | 2                       | 743          | 1.85               |
| ll1e-8    | 2258  | yes             | 34.8          | 34.8               | 0            | 0.000192  | 0                 | 0.000499       | 0.00244        | 1                       | 677          | 1.58               |
| ll1e-9    | 2258  | yes             | 11.6          | 11.6               | 0            | 2.14e-05  | 0                 | 0.00041        | 0.00244        | 1                       | 603          | 1.4                |
| free1e-3  | 2258  | yes             | 82            | 81.9               | 0.00848      | 0.0444    | 0.0403            | 0.288          | 0.549          | 137                     | 826          | 9.43               |
| free1e-4  | 2258  | yes             | 80.2          | 80.2               | 0.00534      | 0.0235    | 0.0224            | 0.125          | 0.475          | 76                      | 821          | 5.6                |
| free1e-5  | 2258  | yes             | 75.7          | 75.7               | 0.00211      | 0.0113    | 0.00864           | 0.0514         | 0.465          | 34                      | 817          | 3.58               |
| free1e-6  | 2258  | yes             | 62.7          | 62.7               | 0.000311     | 0.00524   | 0.00144           | 0.0222         | 0.432          | 16                      | 802          | 2.38               |
| param1e-6 | 2258  | yes             | 1.51          | 1.51               | 0            | 0         | 0                 | 0              | 0.000499       | 1                       | 432          | 1.27               |
| iter400   | 2258  | yes             | 13.5          | 13.5               | 0            | 0.000127  | 0                 | 0.000959       | 0.356          | 7                       | 585          | 0.993              |
| iter25    | 2258  | yes             | 70.7          | 70.6               | 0.00133      | 0.0237    | 0.00585           | 0.145          | 0.559          | 42                      | 1503         | 3.56               |

## gate_by_inclusion

| arm       | inclusion | cases | admitted_moved_pct | d_adm_frac_median | d_adm_frac_p90 |
|-----------|-----------|-------|--------------------|-------------------|----------------|
| free1e-3  | -3        | 2258  | 78.7               | 0.0235            | 0.14           |
| free1e-3  | 0         | 2258  | 81.9               | 0.0403            | 0.288          |
| free1e-3  | 3         | 2258  | 82                 | 0.0862            | 0.463          |
| free1e-4  | -3        | 2258  | 76.1               | 0.013             | 0.0748         |
| free1e-4  | 0         | 2258  | 80.2               | 0.0224            | 0.125          |
| free1e-4  | 3         | 2258  | 79.7               | 0.0524            | 0.291          |
| free1e-5  | -3        | 2258  | 71                 | 0.00452           | 0.031          |
| free1e-5  | 0         | 2258  | 75.7               | 0.00864           | 0.0514         |
| free1e-5  | 3         | 2258  | 76.5               | 0.0192            | 0.129          |
| free1e-6  | -3        | 2258  | 57.6               | 0.000499          | 0.0132         |
| free1e-6  | 0         | 2258  | 62.7               | 0.00144           | 0.0222         |
| free1e-6  | 3         | 2258  | 68.9               | 0.00337           | 0.0581         |
| iter25    | -3        | 2258  | 65.6               | 0.0025            | 0.0961         |
| iter25    | 0         | 2258  | 70.6               | 0.00585           | 0.145          |
| iter25    | 3         | 2258  | 73                 | 0.0125            | 0.312          |
| iter400   | -3        | 2258  | 12.2               | 0                 | 0.000484       |
| iter400   | 0         | 2258  | 13.5               | 0                 | 0.000959       |
| iter400   | 3         | 2258  | 14.8               | 0                 | 0.000995       |
| ll1e-3    | -3        | 2258  | 76.7               | 0.0197            | 0.121          |
| ll1e-3    | 0         | 2258  | 80.4               | 0.033             | 0.203          |
| ll1e-3    | 3         | 2258  | 81                 | 0.0779            | 0.39           |
| ll1e-4    | -3        | 2258  | 73.6               | 0.00835           | 0.0431         |
| ll1e-4    | 0         | 2258  | 79.5               | 0.0146            | 0.066          |
| ll1e-4    | 3         | 2258  | 78.4               | 0.0402            | 0.164          |
| ll1e-5    | -3        | 2258  | 69.6               | 0.00271           | 0.014          |
| ll1e-5    | 0         | 2258  | 77.1               | 0.00527           | 0.0243         |
| ll1e-5    | 3         | 2258  | 77.1               | 0.0147            | 0.0635         |
| ll1e-6    | -3        | 2258  | 63.1               | 0.000824          | 0.00435        |
| ll1e-6    | 0         | 2258  | 72.1               | 0.0015            | 0.00699        |
| ll1e-6    | 3         | 2258  | 75.2               | 0.00479           | 0.0194         |
| ll1e-7    | -3        | 2258  | 50.5               | 0.000406          | 0.00146        |
| ll1e-7    | 0         | 2258  | 58.2               | 0.00042           | 0.00197        |
| ll1e-7    | 3         | 2258  | 70.2               | 0.00145           | 0.00541        |
| ll1e-8    | -3        | 2258  | 31.8               | 0                 | 0.000495       |
| ll1e-8    | 0         | 2258  | 34.8               | 0                 | 0.000499       |
| ll1e-8    | 3         | 2258  | 56.6               | 0.000414          | 0.00166        |
| ll1e-9    | -3        | 2258  | 13.5               | 0                 | 0.000414       |
| ll1e-9    | 0         | 2258  | 11.6               | 0                 | 0.00041        |
| ll1e-9    | 3         | 2258  | 31.7               | 0                 | 0.000496       |
| param1e-6 | -3        | 2258  | 2.21               | 0                 | 0              |
| param1e-6 | 0         | 2258  | 1.51               | 0                 | 0              |
| param1e-6 | 3         | 2258  | 5.89               | 0                 | 0              |

## gate_by_env

| arm       | dataset         | embedder     | style       | cases | changed | d_adm_frac_median | d_adm_frac_p90 | admitted_moved_pct |
|-----------|-----------------|--------------|-------------|-------|---------|-------------------|----------------|--------------------|
| ll1e-3    | visual_genome_m | siglip       | whole_image | 319   | 317     | 0.0573            | 0.159          | 99.4               |
| ll1e-4    | visual_genome_m | siglip       | whole_image | 319   | 318     | 0.0315            | 0.0705         | 99.7               |
| ll1e-5    | visual_genome_m | siglip       | whole_image | 319   | 316     | 0.0116            | 0.0307         | 99.1               |
| ll1e-6    | visual_genome_m | siglip       | whole_image | 319   | 309     | 0.00351           | 0.00961        | 96.9               |
| ll1e-7    | visual_genome_m | siglip       | whole_image | 319   | 260     | 0.000985          | 0.00246        | 81.5               |
| ll1e-8    | visual_genome_m | siglip       | whole_image | 319   | 146     | 0                 | 0.000501       | 45.8               |
| ll1e-9    | visual_genome_m | siglip       | whole_image | 319   | 40      | 0                 | 0.000482       | 12.5               |
| free1e-3  | visual_genome_m | siglip       | whole_image | 319   | 317     | 0.0579            | 0.177          | 99.4               |
| free1e-4  | visual_genome_m | siglip       | whole_image | 319   | 316     | 0.0377            | 0.0844         | 99.1               |
| free1e-5  | visual_genome_m | siglip       | whole_image | 319   | 317     | 0.0195            | 0.0448         | 99.4               |
| free1e-6  | visual_genome_m | siglip       | whole_image | 319   | 297     | 0.0089            | 0.0218         | 93.1               |
| param1e-6 | visual_genome_m | siglip       | whole_image | 319   | 2       | 0                 | 0              | 0.627              |
| iter400   | visual_genome_m | siglip       | whole_image | 319   | 107     | 0                 | 0.0106         | 33.5               |
| iter25    | visual_genome_m | siglip       | whole_image | 319   | 310     | 0.0183            | 0.0863         | 97.2               |
| ll1e-3    | visual_genome_m | dinov3_patch | max_patch   | 317   | 312     | 0.0165            | 0.119          | 98.4               |
| ll1e-4    | visual_genome_m | dinov3_patch | max_patch   | 317   | 307     | 0.00721           | 0.0392         | 96.8               |
| ll1e-5    | visual_genome_m | dinov3_patch | max_patch   | 317   | 293     | 0.0029            | 0.0127         | 92.4               |
| ll1e-6    | visual_genome_m | dinov3_patch | max_patch   | 317   | 256     | 0.000985          | 0.00394        | 80.8               |
| ll1e-7    | visual_genome_m | dinov3_patch | max_patch   | 317   | 180     | 0.000483          | 0.00145        | 56.8               |
| ll1e-8    | visual_genome_m | dinov3_patch | max_patch   | 317   | 93      | 0                 | 0.000496       | 29.3               |
| ll1e-9    | visual_genome_m | dinov3_patch | max_patch   | 317   | 34      | 0                 | 0.000479       | 10.7               |
| free1e-3  | visual_genome_m | dinov3_patch | max_patch   | 317   | 313     | 0.0184            | 0.311          | 98.7               |
| free1e-4  | visual_genome_m | dinov3_patch | max_patch   | 317   | 303     | 0.0102            | 0.105          | 95.6               |
| free1e-5  | visual_genome_m | dinov3_patch | max_patch   | 317   | 254     | 0.00244           | 0.0323         | 80.1               |
| free1e-6  | visual_genome_m | dinov3_patch | max_patch   | 317   | 151     | 0                 | 0.00803        | 47.6               |
| param1e-6 | visual_genome_m | dinov3_patch | max_patch   | 317   | 6       | 0                 | 0              | 1.89               |
| iter400   | visual_genome_m | dinov3_patch | max_patch   | 317   | 14      | 0                 | 0              | 4.42               |
| iter25    | visual_genome_m | dinov3_patch | max_patch   | 317   | 222     | 0.00148           | 0.0672         | 70                 |
| ll1e-3    | visual_genome_m | dinov3_patch | whole_image | 317   | 317     | 0.129             | 0.316          | 100                |
| ll1e-4    | visual_genome_m | dinov3_patch | whole_image | 317   | 316     | 0.0469            | 0.115          | 99.7               |
| ll1e-5    | visual_genome_m | dinov3_patch | whole_image | 317   | 312     | 0.0152            | 0.0406         | 98.4               |
| ll1e-6    | visual_genome_m | dinov3_patch | whole_image | 317   | 300     | 0.00401           | 0.0104         | 94.6               |
| ll1e-7    | visual_genome_m | dinov3_patch | whole_image | 317   | 254     | 0.000975          | 0.00245        | 80.1               |
| ll1e-8    | visual_genome_m | dinov3_patch | whole_image | 317   | 159     | 0.000478          | 0.000965       | 50.2               |
| ll1e-9    | visual_genome_m | dinov3_patch | whole_image | 317   | 47      | 0                 | 0.000485       | 14.8               |
| free1e-3  | visual_genome_m | dinov3_patch | whole_image | 317   | 317     | 0.157             | 0.389          | 100                |
| free1e-4  | visual_genome_m | dinov3_patch | whole_image | 317   | 317     | 0.07              | 0.308          | 100                |
| free1e-5  | visual_genome_m | dinov3_patch | whole_image | 317   | 315     | 0.0298            | 0.22           | 99.4               |
| free1e-6  | visual_genome_m | dinov3_patch | whole_image | 317   | 290     | 0.0088            | 0.0674         | 91.5               |
| param1e-6 | visual_genome_m | dinov3_patch | whole_image | 317   | 5       | 0                 | 0              | 1.58               |
| iter400   | visual_genome_m | dinov3_patch | whole_image | 317   | 104     | 0                 | 0.00898        | 32.8               |
| iter25    | visual_genome_m | dinov3_patch | whole_image | 317   | 305     | 0.0705            | 0.328          | 96.2               |
| ll1e-3    | caltech101_m    | siglip       | whole_image | 239   | 27      | 0                 | 0.00297        | 11.3               |
| ll1e-4    | caltech101_m    | siglip       | whole_image | 239   | 23      | 0                 | 0              | 9.62               |
| ll1e-5    | caltech101_m    | siglip       | whole_image | 239   | 19      | 0                 | 0              | 7.95               |
| ll1e-6    | caltech101_m    | siglip       | whole_image | 239   | 10      | 0                 | 0              | 4.18               |
| ll1e-7    | caltech101_m    | siglip       | whole_image | 239   | 4       | 0                 | 0              | 1.67               |
| ll1e-8    | caltech101_m    | siglip       | whole_image | 239   | 2       | 0                 | 0              | 0.837              |
| ll1e-9    | caltech101_m    | siglip       | whole_image | 239   | 2       | 0                 | 0              | 0.837              |
| free1e-3  | caltech101_m    | siglip       | whole_image | 239   | 41      | 0                 | 0.109          | 17.2               |
| free1e-4  | caltech101_m    | siglip       | whole_image | 239   | 34      | 0                 | 0.00787        | 14.2               |
| free1e-5  | caltech101_m    | siglip       | whole_image | 239   | 12      | 0                 | 0              | 5.02               |
| free1e-6  | caltech101_m    | siglip       | whole_image | 239   | 8       | 0                 | 0              | 3.35               |
| param1e-6 | caltech101_m    | siglip       | whole_image | 239   | 0       | 0                 | 0              | 0                  |
| iter400   | caltech101_m    | siglip       | whole_image | 239   | 2       | 0                 | 0              | 0.837              |
| iter25    | caltech101_m    | siglip       | whole_image | 239   | 19      | 0                 | 0              | 7.95               |
| ll1e-3    | caltech101_m    | dinov3_patch | whole_image | 228   | 25      | 0                 | 0.00244        | 11                 |
| ll1e-4    | caltech101_m    | dinov3_patch | whole_image | 228   | 19      | 0                 | 0              | 8.33               |
| ll1e-5    | caltech101_m    | dinov3_patch | whole_image | 228   | 9       | 0                 | 0              | 3.95               |
| ll1e-6    | caltech101_m    | dinov3_patch | whole_image | 228   | 6       | 0                 | 0              | 2.63               |
| ll1e-7    | caltech101_m    | dinov3_patch | whole_image | 228   | 3       | 0                 | 0              | 1.32               |
| ll1e-8    | caltech101_m    | dinov3_patch | whole_image | 228   | 0       | 0                 | 0              | 0                  |
| ll1e-9    | caltech101_m    | dinov3_patch | whole_image | 228   | 0       | 0                 | 0              | 0                  |
| free1e-3  | caltech101_m    | dinov3_patch | whole_image | 228   | 44      | 0                 | 0.097          | 19.3               |
| free1e-4  | caltech101_m    | dinov3_patch | whole_image | 228   | 26      | 0                 | 0.00244        | 11.4               |
| free1e-5  | caltech101_m    | dinov3_patch | whole_image | 228   | 15      | 0                 | 0              | 6.58               |
| free1e-6  | caltech101_m    | dinov3_patch | whole_image | 228   | 4       | 0                 | 0              | 1.75               |
| param1e-6 | caltech101_m    | dinov3_patch | whole_image | 228   | 0       | 0                 | 0              | 0                  |
| iter400   | caltech101_m    | dinov3_patch | whole_image | 228   | 1       | 0                 | 0              | 0.439              |
| iter25    | caltech101_m    | dinov3_patch | whole_image | 228   | 10      | 0                 | 0              | 4.39               |
| ll1e-3    | coco_val        | siglip       | whole_image | 280   | 267     | 0.0703            | 0.183          | 95.4               |
| ll1e-4    | coco_val        | siglip       | whole_image | 280   | 267     | 0.0271            | 0.0641         | 95.4               |
| ll1e-5    | coco_val        | siglip       | whole_image | 280   | 260     | 0.00838           | 0.0206         | 92.9               |
| ll1e-6    | coco_val        | siglip       | whole_image | 280   | 254     | 0.0025            | 0.00622        | 90.7               |
| ll1e-7    | coco_val        | siglip       | whole_image | 280   | 210     | 0.000814          | 0.00206        | 75                 |
| ll1e-8    | coco_val        | siglip       | whole_image | 280   | 132     | 0                 | 0.000819       | 47.1               |
| ll1e-9    | coco_val        | siglip       | whole_image | 280   | 43      | 0                 | 0.000412       | 15.4               |
| free1e-3  | coco_val        | siglip       | whole_image | 280   | 267     | 0.0731            | 0.203          | 95.4               |
| free1e-4  | coco_val        | siglip       | whole_image | 280   | 266     | 0.0329            | 0.0849         | 95                 |
| free1e-5  | coco_val        | siglip       | whole_image | 280   | 264     | 0.0138            | 0.0356         | 94.3               |
| free1e-6  | coco_val        | siglip       | whole_image | 280   | 230     | 0.00577           | 0.0172         | 82.1               |
| param1e-6 | coco_val        | siglip       | whole_image | 280   | 5       | 0                 | 0              | 1.79               |
| iter400   | coco_val        | siglip       | whole_image | 280   | 35      | 0                 | 0.000408       | 12.5               |
| iter25    | coco_val        | siglip       | whole_image | 280   | 253     | 0.0134            | 0.0978         | 90.4               |
| ll1e-3    | coco_val        | dinov3_patch | max_patch   | 279   | 275     | 0.0191            | 0.1            | 98.6               |
| ll1e-4    | coco_val        | dinov3_patch | max_patch   | 279   | 274     | 0.0101            | 0.0327         | 98.2               |
| ll1e-5    | coco_val        | dinov3_patch | max_patch   | 279   | 269     | 0.00333           | 0.0108         | 96.4               |
| ll1e-6    | coco_val        | dinov3_patch | max_patch   | 279   | 241     | 0.00123           | 0.00336        | 86.4               |
| ll1e-7    | coco_val        | dinov3_patch | max_patch   | 279   | 181     | 0.000412          | 0.00124        | 64.9               |
| ll1e-8    | coco_val        | dinov3_patch | max_patch   | 279   | 96      | 0                 | 0.000417       | 34.4               |
| ll1e-9    | coco_val        | dinov3_patch | max_patch   | 279   | 27      | 0                 | 0              | 9.68               |
| free1e-3  | coco_val        | dinov3_patch | max_patch   | 279   | 276     | 0.0206            | 0.148          | 98.9               |
| free1e-4  | coco_val        | dinov3_patch | max_patch   | 279   | 275     | 0.0125            | 0.0525         | 98.6               |
| free1e-5  | coco_val        | dinov3_patch | max_patch   | 279   | 262     | 0.00537           | 0.0248         | 93.9               |
| free1e-6  | coco_val        | dinov3_patch | max_patch   | 279   | 182     | 0.000814          | 0.0121         | 65.2               |
| param1e-6 | coco_val        | dinov3_patch | max_patch   | 279   | 10      | 0                 | 0              | 3.58               |
| iter400   | coco_val        | dinov3_patch | max_patch   | 279   | 12      | 0                 | 0              | 4.3                |
| iter25    | coco_val        | dinov3_patch | max_patch   | 279   | 213     | 0.00207           | 0.0324         | 76.3               |
| ll1e-3    | coco_val        | dinov3_patch | whole_image | 279   | 275     | 0.102             | 0.338          | 98.6               |
| ll1e-4    | coco_val        | dinov3_patch | whole_image | 279   | 270     | 0.0373            | 0.0844         | 96.8               |
| ll1e-5    | coco_val        | dinov3_patch | whole_image | 279   | 262     | 0.0112            | 0.0267         | 93.9               |
| ll1e-6    | coco_val        | dinov3_patch | whole_image | 279   | 253     | 0.00329           | 0.00821        | 90.7               |
| ll1e-7    | coco_val        | dinov3_patch | whole_image | 279   | 223     | 0.000836          | 0.00248        | 79.9               |
| ll1e-8    | coco_val        | dinov3_patch | whole_image | 279   | 157     | 0.000407          | 0.000833       | 56.3               |
| ll1e-9    | coco_val        | dinov3_patch | whole_image | 279   | 68      | 0                 | 0.000415       | 24.4               |
| free1e-3  | coco_val        | dinov3_patch | whole_image | 279   | 275     | 0.117             | 0.381          | 98.6               |
| free1e-4  | coco_val        | dinov3_patch | whole_image | 279   | 274     | 0.0495            | 0.362          | 98.2               |
| free1e-5  | coco_val        | dinov3_patch | whole_image | 279   | 271     | 0.0214            | 0.217          | 97.1               |
| free1e-6  | coco_val        | dinov3_patch | whole_image | 279   | 253     | 0.008             | 0.0786         | 90.7               |
| param1e-6 | coco_val        | dinov3_patch | whole_image | 279   | 6       | 0                 | 0              | 2.15               |
| iter400   | coco_val        | dinov3_patch | whole_image | 279   | 30      | 0                 | 0.000408       | 10.8               |
| iter25    | coco_val        | dinov3_patch | whole_image | 279   | 263     | 0.0349            | 0.327          | 94.3               |

## incumbent

| dataset         | embedder     | style       | folds | n_median | anchors_median | iter_median | iter_p90 | hit_cap_pct | init_ms | refit_ms | refit_share_pct |
|-----------------|--------------|-------------|-------|----------|----------------|-------------|----------|-------------|---------|----------|-----------------|
| ALL             | ALL          | ALL         | 4493  | 2.06e+03 | 21             | 113         | 200      | 26.5        | 1.68    | 25.1     | 92.9            |
| visual_genome_m | siglip       | whole_image | 638   | 2.04e+03 | 16             | 200         | 200      | 51.1        | 1.82    | 42.5     | 94.5            |
| visual_genome_m | dinov3_patch | max_patch   | 633   | 2.04e+03 | 26             | 88          | 200      | 17.1        | 1.57    | 19.2     | 93              |
| visual_genome_m | dinov3_patch | whole_image | 627   | 2.04e+03 | 27             | 183         | 200      | 47          | 2.35    | 40.2     | 92.8            |
| caltech101_m    | siglip       | whole_image | 476   | 366      | 15.5           | 2           | 42.5     | 2.31        | 0.546   | 0.455    | 79.8            |
| caltech101_m    | dinov3_patch | whole_image | 451   | 366      | 27             | 5           | 30       | 0.665       | 0.576   | 0.916    | 72.8            |
| coco_val        | siglip       | whole_image | 558   | 2.42e+03 | 16             | 156         | 200      | 35.5        | 1.99    | 35.6     | 93.7            |
| coco_val        | dinov3_patch | max_patch   | 557   | 2.42e+03 | 26             | 93          | 200      | 13.6        | 1.5     | 20.9     | 93              |
| coco_val        | dinov3_patch | whole_image | 553   | 2.42e+03 | 26             | 156         | 200      | 31.1        | 2.62    | 35.4     | 92.4            |

## fits_by_arm

| arm       | folds | anchored_both | lost_anchored_fit | iter_median | iter_p90 | hit_cap_pct | refit_speedup | fit_speedup | obj_better | obj_worse | obj_d_median | free_better | free_worse | d_midpoint_median | d_midpoint_max |
|-----------|-------|---------------|-------------------|-------------|----------|-------------|---------------|-------------|------------|-----------|--------------|-------------|------------|-------------------|----------------|
| ll1e-3    | 4516  | 4491          | 2                 | 3           | 12       | 0           | 20.6          | 8.62        | 0          | 3786      | -0.00113     | 961         | 2865       | 0.00669           | 0.149          |
| ll1e-4    | 4516  | 4488          | 5                 | 9           | 38       | 0           | 6.67          | 4.76        | 0          | 3708      | -0.000193    | 1546        | 2205       | 0.00373           | 0.128          |
| ll1e-5    | 4516  | 4493          | 0                 | 19          | 63       | 0.266       | 3.69          | 3.1         | 0          | 3682      | -2.17e-05    | 2129        | 1600       | 0.0014            | 0.0687         |
| ll1e-6    | 4516  | 4493          | 0                 | 30          | 96       | 1.55        | 2.42          | 2.2         | 0          | 3604      | -2.13e-06    | 2391        | 1274       | 0.000447          | 0.0579         |
| ll1e-7    | 4516  | 4493          | 0                 | 40          | 130      | 4.69        | 1.84          | 1.74        | 0          | 3438      | -1.88e-07    | 2419        | 1105       | 0.000129          | 0.018          |
| ll1e-8    | 4516  | 4493          | 0                 | 50          | 168      | 7.66        | 1.54          | 1.48        | 0          | 3248      | -1.68e-08    | 2342        | 1035       | 3.71e-05          | 0.00102        |
| ll1e-9    | 4516  | 4493          | 0                 | 60          | 200      | 10.3        | 1.35          | 1.31        | 0          | 2651      | -1.5e-09     | 2245        | 1001       | 1.04e-05          | 0.000218       |
| free1e-3  | 4516  | 4491          | 2                 | 2           | 3        | 0           | 39.4          | 10.6        | 0          | 3870      | -0.00147     | 868         | 3039       | 0.00715           | 0.208          |
| free1e-4  | 4516  | 4491          | 2                 | 7           | 26       | 0           | 10.6          | 6.31        | 0          | 3753      | -0.000353    | 1215        | 2578       | 0.00457           | 0.199          |
| free1e-5  | 4516  | 4492          | 1                 | 16          | 49       | 0.0886      | 5.04          | 3.92        | 0          | 3672      | -3.8e-05     | 1947        | 1790       | 0.00152           | 0.144          |
| free1e-6  | 4516  | 4491          | 2                 | 30          | 83       | 1.46        | 2.9           | 2.56        | 0          | 3441      | -2.57e-07    | 2292        | 1363       | 0.000142          | 0.133          |
| param1e-6 | 4516  | 4493          | 0                 | 76          | 200      | 15.7        | 1.26          | 1.24        | 0          | 0         | 0            | 2066        | 917        | 1.26e-06          | 1.08e-05       |
| iter400   | 4516  | 4491          | 2                 | 113         | 400      | 10.9        | 0.775         | 0.788       | 597        | 0         | 0            | 300         | 783        | 0                 | 0.139          |
| iter25    | 4516  | 4485          | 8                 | 25          | 25       | 81.5        | 5.22          | 4.02        | 0          | 3385      | -9.07e-06    | 1659        | 1977       | 0.000733          | 0.146          |

## monotonicity

| objective | traces | iterations | decreasing_iterations | decreasing_pct | traces_with_a_decrease | worst_single_step |
|-----------|--------|------------|-----------------------|----------------|------------------------|-------------------|
| anchored  | 8      | 1989       | 0                     | 0              | 0                      | 0                 |
| free      | 8      | 2410       | 883                   | 36.6           | 6                      | -2.02e-06         |

## bench_by_n

| arm       | n_bucket | resampled | samples | n_median | init_ms | refit_ms | fit_ms | iter_median | refit_speedup_median | fit_speedup_median |
|-----------|----------|-----------|---------|----------|---------|----------|--------|-------------|----------------------|--------------------|
| baseline  | <1k      | 0         | 24      | 351      | 0.477   | 0.618    | 1.09   | 3           | 1                    | 1                  |
| free1e-3  | <1k      | 0         | 24      | 351      | 0.477   | 0.599    | 1.08   | 3           | 1.06                 | 1.04               |
| free1e-4  | <1k      | 0         | 24      | 351      | 0.477   | 0.607    | 1.09   | 3           | 1.05                 | 1.03               |
| free1e-5  | <1k      | 0         | 24      | 351      | 0.477   | 0.613    | 1.09   | 3           | 1.04                 | 1.02               |
| free1e-6  | <1k      | 0         | 24      | 351      | 0.477   | 0.612    | 1.09   | 3           | 1.01                 | 1.01               |
| iter25    | <1k      | 0         | 24      | 351      | 0.477   | 0.618    | 1.1    | 3           | 1                    | 1                  |
| iter400   | <1k      | 0         | 24      | 351      | 0.477   | 0.615    | 1.1    | 3           | 0.996                | 0.996              |
| ll1e-3    | <1k      | 0         | 24      | 351      | 0.477   | 0.704    | 1.17   | 3           | 0.91                 | 0.946              |
| ll1e-4    | <1k      | 0         | 24      | 351      | 0.477   | 0.703    | 1.18   | 3           | 0.873                | 0.917              |
| ll1e-5    | <1k      | 0         | 24      | 351      | 0.477   | 0.706    | 1.18   | 3           | 0.858                | 0.905              |
| ll1e-6    | <1k      | 0         | 24      | 351      | 0.477   | 0.7      | 1.18   | 3           | 0.866                | 0.911              |
| ll1e-7    | <1k      | 0         | 24      | 351      | 0.477   | 0.695    | 1.18   | 3           | 0.866                | 0.912              |
| ll1e-8    | <1k      | 0         | 24      | 351      | 0.477   | 0.695    | 1.18   | 3           | 0.87                 | 0.914              |
| ll1e-9    | <1k      | 0         | 24      | 351      | 0.477   | 0.697    | 1.17   | 3           | 0.864                | 0.906              |
| param1e-6 | <1k      | 0         | 24      | 351      | 0.477   | 0.49     | 0.966  | 2           | 0.998                | 1                  |
| baseline  | 1k-5k    | 0         | 60      | 2.08e+03 | 1.87    | 30.1     | 32.1   | 134         | 1                    | 1                  |
| free1e-3  | 1k-5k    | 0         | 60      | 2.08e+03 | 1.87    | 0.582    | 2.64   | 2           | 48.2                 | 12.4               |
| free1e-4  | 1k-5k    | 0         | 60      | 2.08e+03 | 1.87    | 1.87     | 3.72   | 8           | 15.8                 | 8.77               |
| free1e-5  | 1k-5k    | 0         | 60      | 2.08e+03 | 1.87    | 4.46     | 7.37   | 19.5        | 6.01                 | 4.64               |
| free1e-6  | 1k-5k    | 0         | 60      | 2.08e+03 | 1.87    | 8.22     | 11.3   | 37.5        | 3.64                 | 2.83               |
| iter25    | 1k-5k    | 0         | 60      | 2.08e+03 | 1.87    | 5.72     | 7.59   | 25          | 5.27                 | 4.23               |
| iter400   | 1k-5k    | 0         | 60      | 2.08e+03 | 1.87    | 30.2     | 32.5   | 134         | 0.998                | 0.998              |
| ll1e-3    | 1k-5k    | 0         | 60      | 2.08e+03 | 1.87    | 0.674    | 2.82   | 2           | 33.1                 | 10.8               |
| ll1e-4    | 1k-5k    | 0         | 60      | 2.08e+03 | 1.87    | 2.76     | 4.9    | 10.5        | 10.6                 | 6.14               |
| ll1e-5    | 1k-5k    | 0         | 60      | 2.08e+03 | 1.87    | 6.18     | 8.76   | 24.5        | 4.84                 | 3.76               |
| ll1e-6    | 1k-5k    | 0         | 60      | 2.08e+03 | 1.87    | 9.04     | 12     | 36          | 3.11                 | 2.67               |
| ll1e-7    | 1k-5k    | 0         | 60      | 2.08e+03 | 1.87    | 12.4     | 14.9   | 51          | 2.35                 | 2.16               |
| ll1e-8    | 1k-5k    | 0         | 60      | 2.08e+03 | 1.87    | 16.1     | 18.2   | 64.5        | 1.93                 | 1.78               |
| ll1e-9    | 1k-5k    | 0         | 60      | 2.08e+03 | 1.87    | 18.5     | 21     | 75          | 1.65                 | 1.56               |
| param1e-6 | 1k-5k    | 0         | 60      | 2.08e+03 | 1.87    | 20.3     | 23.2   | 90          | 1.46                 | 1.39               |
| baseline  | 20000    | 1         | 84      | 2e+04    | 5.45    | 98.5     | 105    | 142         | 1                    | 1                  |
| free1e-3  | 20000    | 1         | 84      | 2e+04    | 5.45    | 1.75     | 7.25   | 2           | 57.1                 | 11.1               |
| free1e-4  | 20000    | 1         | 84      | 2e+04    | 5.45    | 4.81     | 10.6   | 6           | 10.2                 | 5.79               |
| free1e-5  | 20000    | 1         | 84      | 2e+04    | 5.45    | 12.8     | 19.4   | 16.5        | 4.67                 | 3.54               |
| free1e-6  | 20000    | 1         | 84      | 2e+04    | 5.45    | 20.3     | 28.8   | 26.5        | 2.95                 | 2.38               |
| iter25    | 20000    | 1         | 84      | 2e+04    | 5.45    | 17.5     | 23     | 25          | 5.65                 | 4.25               |
| iter400   | 20000    | 1         | 84      | 2e+04    | 5.45    | 98       | 105    | 142         | 0.993                | 0.995              |
| ll1e-3    | 20000    | 1         | 84      | 2e+04    | 5.45    | 1.81     | 7.33   | 2           | 43                   | 10.6               |
| ll1e-4    | 20000    | 1         | 84      | 2e+04    | 5.45    | 5.76     | 11.3   | 7           | 10.2                 | 5.44               |
| ll1e-5    | 20000    | 1         | 84      | 2e+04    | 5.45    | 15.4     | 22.3   | 19          | 4.26                 | 3.29               |
| ll1e-6    | 20000    | 1         | 84      | 2e+04    | 5.45    | 25.5     | 34.8   | 32          | 2.61                 | 2.19               |
| ll1e-7    | 20000    | 1         | 84      | 2e+04    | 5.45    | 34.4     | 42.2   | 43.5        | 1.98                 | 1.72               |
| ll1e-8    | 20000    | 1         | 84      | 2e+04    | 5.45    | 44.7     | 50.6   | 56          | 1.63                 | 1.46               |
| ll1e-9    | 20000    | 1         | 84      | 2e+04    | 5.45    | 54.5     | 59.7   | 69          | 1.31                 | 1.24               |
| param1e-6 | 20000    | 1         | 84      | 2e+04    | 5.45    | 63.2     | 69.1   | 90.5        | 1.35                 | 1.26               |
| baseline  | 50000    | 1         | 84      | 5e+04    | 13.7    | 189      | 202    | 132         | 1                    | 1                  |
| free1e-3  | 50000    | 1         | 84      | 5e+04    | 13.7    | 4.47     | 18.2   | 2           | 41.7                 | 9.79               |
| free1e-4  | 50000    | 1         | 84      | 5e+04    | 13.7    | 12       | 26.8   | 6.5         | 9.12                 | 5.39               |
| free1e-5  | 50000    | 1         | 84      | 5e+04    | 13.7    | 26.4     | 46.3   | 15          | 4.25                 | 3.27               |
| free1e-6  | 50000    | 1         | 84      | 5e+04    | 13.7    | 50.3     | 66.2   | 29          | 2.57                 | 2.22               |
| iter25    | 50000    | 1         | 84      | 5e+04    | 13.7    | 36.7     | 50.5   | 25          | 5.14                 | 3.9                |
| iter400   | 50000    | 1         | 84      | 5e+04    | 13.7    | 189      | 202    | 132         | 0.998                | 0.998              |
| ll1e-3    | 50000    | 1         | 84      | 5e+04    | 13.7    | 4.52     | 18.2   | 2           | 39.9                 | 9.74               |
| ll1e-4    | 50000    | 1         | 84      | 5e+04    | 13.7    | 13       | 26.8   | 7           | 9                    | 5.27               |
| ll1e-5    | 50000    | 1         | 84      | 5e+04    | 13.7    | 28.5     | 49     | 16          | 4.17                 | 3.25               |
| ll1e-6    | 50000    | 1         | 84      | 5e+04    | 13.7    | 58.5     | 77.1   | 33.5        | 2.49                 | 2.14               |
| ll1e-7    | 50000    | 1         | 84      | 5e+04    | 13.7    | 76.8     | 95.6   | 44.5        | 1.8                  | 1.65               |
| ll1e-8    | 50000    | 1         | 84      | 5e+04    | 13.7    | 97.9     | 119    | 57          | 1.36                 | 1.32               |
| ll1e-9    | 50000    | 1         | 84      | 5e+04    | 13.7    | 115      | 138    | 67          | 1.13                 | 1.1                |
| param1e-6 | 50000    | 1         | 84      | 5e+04    | 13.7    | 120      | 142    | 83.5        | 1.18                 | 1.11               |
