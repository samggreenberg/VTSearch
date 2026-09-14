# Safe thresholds ON vs OFF — trajectory A/B (#2799)

Head: `linear_svm` · cells ON/OFF: 114/114 · ON `/expscratch/sgreenberg/anchem-3825/ab_ll1e-6/results` vs OFF `/expscratch/sgreenberg/anchem-3825/ab_baseline/results`

Both runs are full simulations; because the blended threshold drives Autopilot's
Hard pick, the two arms vote on different items, so cells (category × seed), not
steps, are the paired units.

The app shows a trained detector from **7 votes** onward; below that it
sorts by text/example cosine, so `scope=app_visible` is what users actually get and
`scope=all_steps` is the purely numerical reading.

## Per-window paired comparison (Δ = ON − OFF; negative = safe thresholds better)

```
      scope                                      arm            window            metric  n_cells  safe_on  safe_off  delta_on_minus_off  win_rate_on  p_wilcoxon              note
app_visible    caltech101_m/dinov3_patch/whole_image         ramp_6_20              cost       11 0.417395  0.384210        3.318514e-02     0.090909    0.312500
app_visible    caltech101_m/dinov3_patch/whole_image         ramp_6_20               fnr       11 0.001186  0.003162       -1.976273e-03     0.090909    1.000000
app_visible    caltech101_m/dinov3_patch/whole_image         ramp_6_20               fpr       11 0.416209  0.381048        3.516142e-02     0.090909    0.312500
app_visible    caltech101_m/dinov3_patch/whole_image         ramp_6_20            regret       11 0.411190  0.378005        3.318514e-02     0.090909    0.312500
app_visible    caltech101_m/dinov3_patch/whole_image         ramp_6_20 average_precision       11 0.998476  0.998476        0.000000e+00     0.000000         NaN all-zero or empty
app_visible    caltech101_m/dinov3_patch/whole_image         ramp_6_20             auroc       11 0.999173  0.999173        0.000000e+00     0.000000         NaN all-zero or empty
app_visible    caltech101_m/dinov3_patch/whole_image         ramp_6_20        degenerate       11 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image         ramp_6_20              cost       12 0.064321  0.061930        2.390876e-03     0.250000    0.578125
app_visible          caltech101_m/siglip/whole_image         ramp_6_20               fnr       12 0.001976  0.001976        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image         ramp_6_20               fpr       12 0.062345  0.059954        2.390876e-03     0.250000    0.578125
app_visible          caltech101_m/siglip/whole_image         ramp_6_20            regret       12 0.063701  0.061311        2.390876e-03     0.250000    0.578125
app_visible          caltech101_m/siglip/whole_image         ramp_6_20 average_precision       12 0.999937  0.999937        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image         ramp_6_20             auroc       12 0.999974  0.999974        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image         ramp_6_20        degenerate       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          coco_val/dinov3_patch/max_patch         ramp_6_20              cost       14 0.224300  0.230334       -6.034863e-03     0.428571    1.000000
app_visible          coco_val/dinov3_patch/max_patch         ramp_6_20               fnr       14 0.067259  0.080090       -1.283138e-02     0.500000    0.138641
app_visible          coco_val/dinov3_patch/max_patch         ramp_6_20               fpr       14 0.157041  0.150244        6.796518e-03     0.500000    1.000000
app_visible          coco_val/dinov3_patch/max_patch         ramp_6_20            regret       14 0.092125  0.084241        7.883315e-03     0.428571    0.241211
app_visible          coco_val/dinov3_patch/max_patch         ramp_6_20 average_precision       14 0.740045  0.733187        6.857723e-03     0.571429    1.000000
app_visible          coco_val/dinov3_patch/max_patch         ramp_6_20             auroc       14 0.968407  0.964392        4.014856e-03     0.428571    0.669800
app_visible          coco_val/dinov3_patch/max_patch         ramp_6_20        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible        coco_val/dinov3_patch/whole_image         ramp_6_20              cost       14 0.451526  0.450312        1.213495e-03     0.500000    0.951538
app_visible        coco_val/dinov3_patch/whole_image         ramp_6_20               fnr       14 0.128380  0.107476        2.090354e-02     0.285714    0.172955
app_visible        coco_val/dinov3_patch/whole_image         ramp_6_20               fpr       14 0.323146  0.342836       -1.969003e-02     0.714286    0.267578
app_visible        coco_val/dinov3_patch/whole_image         ramp_6_20            regret       14 0.200053  0.207048       -6.995149e-03     0.500000    0.951538
app_visible        coco_val/dinov3_patch/whole_image         ramp_6_20 average_precision       14 0.585606  0.591454       -5.848095e-03     0.500000    0.600179
app_visible        coco_val/dinov3_patch/whole_image         ramp_6_20             auroc       14 0.904999  0.912524       -7.524965e-03     0.357143    0.916512
app_visible        coco_val/dinov3_patch/whole_image         ramp_6_20        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible              coco_val/siglip/whole_image         ramp_6_20              cost       14 0.350375  0.370226       -1.985079e-02     0.785714    0.078491
app_visible              coco_val/siglip/whole_image         ramp_6_20               fnr       14 0.125910  0.122506        3.404592e-03     0.285714    0.480177
app_visible              coco_val/siglip/whole_image         ramp_6_20               fpr       14 0.224465  0.247720       -2.325529e-02     0.785714    0.016602
app_visible              coco_val/siglip/whole_image         ramp_6_20            regret       14 0.113111  0.122281       -9.170082e-03     0.714286    0.216553
app_visible              coco_val/siglip/whole_image         ramp_6_20 average_precision       14 0.590238  0.591723       -1.485036e-03     0.428571    0.669800
app_visible              coco_val/siglip/whole_image         ramp_6_20             auroc       14 0.917750  0.914152        3.597230e-03     0.285714    0.267578
app_visible              coco_val/siglip/whole_image         ramp_6_20        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible   visual_genome_m/dinov3_patch/max_patch         ramp_6_20              cost       16 0.291870  0.301984       -1.011435e-02     0.687500    0.274445
app_visible   visual_genome_m/dinov3_patch/max_patch         ramp_6_20               fnr       16 0.112288  0.110370        1.918624e-03     0.375000    0.806766
app_visible   visual_genome_m/dinov3_patch/max_patch         ramp_6_20               fpr       16 0.179581  0.191614       -1.203290e-02     0.687500    0.322510
app_visible   visual_genome_m/dinov3_patch/max_patch         ramp_6_20            regret       16 0.072210  0.076157       -3.947061e-03     0.687500    0.433197
app_visible   visual_genome_m/dinov3_patch/max_patch         ramp_6_20 average_precision       16 0.502620  0.476947        2.567289e-02     0.375000    0.211426
app_visible   visual_genome_m/dinov3_patch/max_patch         ramp_6_20             auroc       16 0.917561  0.915168        2.393346e-03     0.437500    0.860260
app_visible   visual_genome_m/dinov3_patch/max_patch         ramp_6_20        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible visual_genome_m/dinov3_patch/whole_image         ramp_6_20              cost       16 0.577702  0.579751       -2.048859e-03     0.437500    0.939880
app_visible visual_genome_m/dinov3_patch/whole_image         ramp_6_20               fnr       16 0.237977  0.243634       -5.656552e-03     0.500000    0.820892
app_visible visual_genome_m/dinov3_patch/whole_image         ramp_6_20               fpr       16 0.339724  0.336117        3.607685e-03     0.500000    0.860260
app_visible visual_genome_m/dinov3_patch/whole_image         ramp_6_20            regret       16 0.117955  0.121744       -3.788885e-03     0.437500    0.743561
app_visible visual_genome_m/dinov3_patch/whole_image         ramp_6_20 average_precision       16 0.315278  0.318277       -2.999099e-03     0.500000    0.632172
app_visible visual_genome_m/dinov3_patch/whole_image         ramp_6_20             auroc       16 0.791581  0.791562        1.918316e-05     0.250000    0.322510
app_visible visual_genome_m/dinov3_patch/whole_image         ramp_6_20        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible       visual_genome_m/siglip/whole_image         ramp_6_20              cost       16 0.485546  0.464855        2.069187e-02     0.437500    0.211426
app_visible       visual_genome_m/siglip/whole_image         ramp_6_20               fnr       16 0.166701  0.159322        7.378963e-03     0.437500    0.733271
app_visible       visual_genome_m/siglip/whole_image         ramp_6_20               fpr       16 0.318846  0.305533        1.331292e-02     0.312500    0.093445
app_visible       visual_genome_m/siglip/whole_image         ramp_6_20            regret       16 0.099355  0.098410        9.447659e-04     0.562500    0.899933
app_visible       visual_genome_m/siglip/whole_image         ramp_6_20 average_precision       16 0.325627  0.350310       -2.468304e-02     0.750000    0.024963
app_visible       visual_genome_m/siglip/whole_image         ramp_6_20             auroc       16 0.844835  0.852044       -7.209262e-03     0.437500    0.820892
app_visible       visual_genome_m/siglip/whole_image         ramp_6_20        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus              cost       12 0.063286  0.060939        2.347577e-03     0.000000    0.062500
app_visible    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus               fnr       12 0.007415  0.006392        1.023061e-03     0.000000    1.000000
app_visible    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus               fpr       12 0.055872  0.054547        1.324516e-03     0.083333    0.312500
app_visible    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus            regret       12 0.062481  0.060126        2.354364e-03     0.000000    0.062500
app_visible    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus average_precision       12 0.999884  0.999884        1.635417e-07     0.000000    1.000000
app_visible    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus             auroc       12 0.999938  0.999938        6.041667e-08     0.000000    1.000000
app_visible    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus        degenerate       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image post_ramp_21_plus              cost       12 0.007504  0.014302       -6.797766e-03     0.333333    0.125000
app_visible          caltech101_m/siglip/whole_image post_ramp_21_plus               fnr       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image post_ramp_21_plus               fpr       12 0.007504  0.014302       -6.797766e-03     0.333333    0.125000
app_visible          caltech101_m/siglip/whole_image post_ramp_21_plus            regret       12 0.007504  0.014302       -6.797766e-03     0.333333    0.125000
app_visible          caltech101_m/siglip/whole_image post_ramp_21_plus average_precision       12 1.000000  1.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image post_ramp_21_plus             auroc       12 1.000000  1.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image post_ramp_21_plus        degenerate       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          coco_val/dinov3_patch/max_patch post_ramp_21_plus              cost       14 0.186050  0.182593        3.456675e-03     0.357143    0.426270
app_visible          coco_val/dinov3_patch/max_patch post_ramp_21_plus               fnr       14 0.069580  0.065318        4.261574e-03     0.214286    0.332880
app_visible          coco_val/dinov3_patch/max_patch post_ramp_21_plus               fpr       14 0.116470  0.117275       -8.049223e-04     0.571429    0.855225
app_visible          coco_val/dinov3_patch/max_patch post_ramp_21_plus            regret       14 0.069281  0.071986       -2.705342e-03     0.571429    0.583008
app_visible          coco_val/dinov3_patch/max_patch post_ramp_21_plus average_precision       14 0.796775  0.791461        5.313473e-03     0.571429    0.855225
app_visible          coco_val/dinov3_patch/max_patch post_ramp_21_plus             auroc       14 0.974329  0.976726       -2.396751e-03     0.500000    0.325806
app_visible          coco_val/dinov3_patch/max_patch post_ramp_21_plus        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible        coco_val/dinov3_patch/whole_image post_ramp_21_plus              cost       14 0.260601  0.257909        2.691995e-03     0.571429    0.855225
app_visible        coco_val/dinov3_patch/whole_image post_ramp_21_plus               fnr       14 0.130779  0.127504        3.274643e-03     0.428571    0.583008
app_visible        coco_val/dinov3_patch/whole_image post_ramp_21_plus               fpr       14 0.129822  0.130405       -5.826152e-04     0.571429    0.855225
app_visible        coco_val/dinov3_patch/whole_image post_ramp_21_plus            regret       14 0.064700  0.066101       -1.401479e-03     0.642857    0.807739
app_visible        coco_val/dinov3_patch/whole_image post_ramp_21_plus average_precision       14 0.687915  0.694278       -6.363101e-03     0.500000    0.760864
app_visible        coco_val/dinov3_patch/whole_image post_ramp_21_plus             auroc       14 0.938011  0.935866        2.144571e-03     0.428571    0.541626
app_visible        coco_val/dinov3_patch/whole_image post_ramp_21_plus        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible              coco_val/siglip/whole_image post_ramp_21_plus              cost       14 0.245348  0.245748       -3.999848e-04     0.428571    0.855225
app_visible              coco_val/siglip/whole_image post_ramp_21_plus               fnr       14 0.111639  0.120311       -8.672262e-03     0.428571    0.346522
app_visible              coco_val/siglip/whole_image post_ramp_21_plus               fpr       14 0.133709  0.125437        8.272305e-03     0.357143    0.216553
app_visible              coco_val/siglip/whole_image post_ramp_21_plus            regret       14 0.056119  0.055458        6.614116e-04     0.500000    0.951538
app_visible              coco_val/siglip/whole_image post_ramp_21_plus average_precision       14 0.668080  0.678684       -1.060427e-02     0.500000    0.541626
app_visible              coco_val/siglip/whole_image post_ramp_21_plus             auroc       14 0.938817  0.939264       -4.471884e-04     0.428571    0.807739
app_visible              coco_val/siglip/whole_image post_ramp_21_plus        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus              cost       16 0.258347  0.247767        1.057968e-02     0.375000    0.274445
app_visible   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus               fnr       16 0.097930  0.089077        8.852977e-03     0.375000    0.432626
app_visible   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus               fpr       16 0.160417  0.158690        1.726721e-03     0.437500    0.860260
app_visible   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus            regret       16 0.063354  0.057993        5.361305e-03     0.437500    0.561890
app_visible   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus average_precision       16 0.532590  0.519367        1.322281e-02     0.562500    0.632172
app_visible   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus             auroc       16 0.925180  0.922402        2.778027e-03     0.375000    0.211426
app_visible   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus              cost       16 0.451361  0.446176        5.185217e-03     0.437500    0.632172
app_visible visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus               fnr       16 0.281752  0.292386       -1.063405e-02     0.500000    0.463745
app_visible visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus               fpr       16 0.169609  0.153790        1.581927e-02     0.312500    0.211426
app_visible visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus            regret       16 0.059722  0.057309        2.413228e-03     0.500000    0.979950
app_visible visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus average_precision       16 0.376011  0.412806       -3.679579e-02     0.687500    0.129730
app_visible visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus             auroc       16 0.832593  0.833232       -6.387336e-04     0.500000    0.860260
app_visible visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible       visual_genome_m/siglip/whole_image post_ramp_21_plus              cost       16 0.390294  0.371828        1.846564e-02     0.375000    0.211426
app_visible       visual_genome_m/siglip/whole_image post_ramp_21_plus               fnr       16 0.175374  0.168798        6.576253e-03     0.500000    0.743561
app_visible       visual_genome_m/siglip/whole_image post_ramp_21_plus               fpr       16 0.214920  0.203031        1.188940e-02     0.437500    0.348389
app_visible       visual_genome_m/siglip/whole_image post_ramp_21_plus            regret       16 0.074281  0.069882        4.398484e-03     0.375000    0.596588
app_visible       visual_genome_m/siglip/whole_image post_ramp_21_plus average_precision       16 0.399815  0.429099       -2.928414e-02     0.687500    0.116669
app_visible       visual_genome_m/siglip/whole_image post_ramp_21_plus             auroc       16 0.879063  0.884096       -5.032892e-03     0.625000    0.495422
app_visible       visual_genome_m/siglip/whole_image post_ramp_21_plus        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible    caltech101_m/dinov3_patch/whole_image         all_steps              cost       12 0.098899  0.092892        6.006562e-03     0.000000    0.031250
app_visible    caltech101_m/dinov3_patch/whole_image         all_steps               fnr       12 0.007050  0.006265        7.847963e-04     0.083333    1.000000
app_visible    caltech101_m/dinov3_patch/whole_image         all_steps               fpr       12 0.091849  0.086627        5.221766e-03     0.083333    0.156250
app_visible    caltech101_m/dinov3_patch/whole_image         all_steps            regret       12 0.097890  0.091876        6.013103e-03     0.000000    0.031250
app_visible    caltech101_m/dinov3_patch/whole_image         all_steps average_precision       12 0.999827  0.999827        1.576305e-07     0.000000    1.000000
app_visible    caltech101_m/dinov3_patch/whole_image         all_steps             auroc       12 0.999911  0.999911        5.823293e-08     0.000000    1.000000
app_visible    caltech101_m/dinov3_patch/whole_image         all_steps        degenerate       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image         all_steps              cost       12 0.015285  0.020882       -5.596705e-03     0.416667    0.218750
app_visible          caltech101_m/siglip/whole_image         all_steps               fnr       12 0.000294  0.000294        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image         all_steps               fpr       12 0.014990  0.020587       -5.596705e-03     0.416667    0.218750
app_visible          caltech101_m/siglip/whole_image         all_steps            regret       12 0.015193  0.020789       -5.596705e-03     0.416667    0.218750
app_visible          caltech101_m/siglip/whole_image         all_steps average_precision       12 0.999991  0.999991        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image         all_steps             auroc       12 0.999996  0.999996        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image         all_steps        degenerate       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          coco_val/dinov3_patch/max_patch         all_steps              cost       14 0.191638  0.189473        2.164691e-03     0.428571    0.463135
app_visible          coco_val/dinov3_patch/max_patch         all_steps               fnr       14 0.069100  0.067294        1.805450e-03     0.214286    0.444587
app_visible          coco_val/dinov3_patch/max_patch         all_steps               fpr       14 0.122538  0.122179        3.592204e-04     0.571429    0.807739
app_visible          coco_val/dinov3_patch/max_patch         all_steps            regret       14 0.072665  0.073833       -1.168235e-03     0.571429    0.583008
app_visible          coco_val/dinov3_patch/max_patch         all_steps average_precision       14 0.788550  0.783091        5.458840e-03     0.500000    1.000000
app_visible          coco_val/dinov3_patch/max_patch         all_steps             auroc       14 0.973478  0.974975       -1.496572e-03     0.571429    0.541626
app_visible          coco_val/dinov3_patch/max_patch         all_steps        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible        coco_val/dinov3_patch/whole_image         all_steps              cost       14 0.288559  0.286202        2.356979e-03     0.500000    0.760864
app_visible        coco_val/dinov3_patch/whole_image         all_steps               fnr       14 0.129956  0.124250        5.706246e-03     0.428571    0.501587
app_visible        coco_val/dinov3_patch/whole_image         all_steps               fpr       14 0.158603  0.161952       -3.349236e-03     0.500000    0.714844
app_visible        coco_val/dinov3_patch/whole_image         all_steps            regret       14 0.084673  0.086937       -2.264392e-03     0.571429    1.000000
app_visible        coco_val/dinov3_patch/whole_image         all_steps average_precision       14 0.673025  0.679297       -6.271955e-03     0.571429    0.714844
app_visible        coco_val/dinov3_patch/whole_image         all_steps             auroc       14 0.933261  0.932501        7.602739e-04     0.428571    0.583008
app_visible        coco_val/dinov3_patch/whole_image         all_steps        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible              coco_val/siglip/whole_image         all_steps              cost       14 0.260990  0.264287       -3.296913e-03     0.500000    0.760864
app_visible              coco_val/siglip/whole_image         all_steps               fnr       14 0.113764  0.120638       -6.873581e-03     0.428571    0.432768
app_visible              coco_val/siglip/whole_image         all_steps               fpr       14 0.147226  0.143649        3.576707e-03     0.428571    0.463135
app_visible              coco_val/siglip/whole_image         all_steps            regret       14 0.064607  0.065410       -8.028533e-04     0.428571    0.855225
app_visible              coco_val/siglip/whole_image         all_steps average_precision       14 0.656486  0.665732       -9.246084e-03     0.428571    0.541626
app_visible              coco_val/siglip/whole_image         all_steps             auroc       14 0.935679  0.935524        1.551717e-04     0.357143    0.501587
app_visible              coco_val/siglip/whole_image         all_steps        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible   visual_genome_m/dinov3_patch/max_patch         all_steps              cost       16 0.263299  0.255819        7.480065e-03     0.437500    0.528168
app_visible   visual_genome_m/dinov3_patch/max_patch         all_steps               fnr       16 0.100031  0.092287        7.744260e-03     0.375000    0.396726
app_visible   visual_genome_m/dinov3_patch/max_patch         all_steps               fpr       16 0.163268  0.163532       -2.641663e-04     0.625000    0.561890
app_visible   visual_genome_m/dinov3_patch/max_patch         all_steps            regret       16 0.064672  0.060716        3.956040e-03     0.437500    0.561890
app_visible   visual_genome_m/dinov3_patch/max_patch         all_steps average_precision       16 0.528137  0.513107        1.503078e-02     0.562500    0.495422
app_visible   visual_genome_m/dinov3_patch/max_patch         all_steps             auroc       16 0.924058  0.921339        2.719060e-03     0.375000    0.211426
app_visible   visual_genome_m/dinov3_patch/max_patch         all_steps        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible visual_genome_m/dinov3_patch/whole_image         all_steps              cost       16 0.469830  0.465868        3.962605e-03     0.375000    0.528168
app_visible visual_genome_m/dinov3_patch/whole_image         all_steps               fnr       16 0.275552  0.285308       -9.756217e-03     0.500000    0.348389
app_visible visual_genome_m/dinov3_patch/whole_image         all_steps               fpr       16 0.194278  0.180559        1.371882e-02     0.375000    0.231201
app_visible visual_genome_m/dinov3_patch/whole_image         all_steps            regret       16 0.068296  0.066841        1.454854e-03     0.437500    0.860260
app_visible visual_genome_m/dinov3_patch/whole_image         all_steps average_precision       16 0.367201  0.398868       -3.166739e-02     0.687500    0.129730
app_visible visual_genome_m/dinov3_patch/whole_image         all_steps             auroc       16 0.826635  0.827090       -4.548115e-04     0.500000    0.860260
app_visible visual_genome_m/dinov3_patch/whole_image         all_steps        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible       visual_genome_m/siglip/whole_image         all_steps              cost       16 0.403753  0.384966        1.878684e-02     0.375000    0.192810
app_visible       visual_genome_m/siglip/whole_image         all_steps               fnr       16 0.173806  0.167226        6.579929e-03     0.500000    0.899933
app_visible       visual_genome_m/siglip/whole_image         all_steps               fpr       16 0.229947  0.217740        1.220692e-02     0.437500    0.322510
app_visible       visual_genome_m/siglip/whole_image         all_steps            regret       16 0.078010  0.074022        3.987942e-03     0.375000    0.596588
app_visible       visual_genome_m/siglip/whole_image         all_steps average_precision       16 0.389156  0.417683       -2.852615e-02     0.687500    0.116669
app_visible       visual_genome_m/siglip/whole_image         all_steps             auroc       16 0.874417  0.879740       -5.322438e-03     0.625000    0.433197
app_visible       visual_genome_m/siglip/whole_image         all_steps        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image      pure_gmm_2_5              cost       10 0.541823  0.542442       -6.188500e-04     0.100000    1.000000
  all_steps    caltech101_m/dinov3_patch/whole_image      pure_gmm_2_5               fnr       10 0.004167  0.004167        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image      pure_gmm_2_5               fpr       10 0.537656  0.538275       -6.188500e-04     0.100000    1.000000
  all_steps    caltech101_m/dinov3_patch/whole_image      pure_gmm_2_5            regret       10 0.540215  0.540833       -6.188000e-04     0.100000    1.000000
  all_steps    caltech101_m/dinov3_patch/whole_image      pure_gmm_2_5 average_precision       10 0.996746  0.996746        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image      pure_gmm_2_5             auroc       10 0.999847  0.999847        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image      pure_gmm_2_5        degenerate       10 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image      pure_gmm_2_5              cost       12 0.494320  0.494730       -4.092500e-04     0.166667    0.500000
  all_steps          caltech101_m/siglip/whole_image      pure_gmm_2_5               fnr       12 0.016441  0.016441        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image      pure_gmm_2_5               fpr       12 0.477879  0.478288       -4.092500e-04     0.166667    0.500000
  all_steps          caltech101_m/siglip/whole_image      pure_gmm_2_5            regret       12 0.479599  0.480008       -4.092500e-04     0.166667    0.500000
  all_steps          caltech101_m/siglip/whole_image      pure_gmm_2_5 average_precision       12 0.985832  0.985832        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image      pure_gmm_2_5             auroc       12 0.998445  0.998445        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image      pure_gmm_2_5        degenerate       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          coco_val/dinov3_patch/max_patch      pure_gmm_2_5              cost       14 0.349001  0.348803        1.979643e-04     0.428571    0.929153
  all_steps          coco_val/dinov3_patch/max_patch      pure_gmm_2_5               fnr       14 0.089222  0.089222        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          coco_val/dinov3_patch/max_patch      pure_gmm_2_5               fpr       14 0.259779  0.259581        1.979286e-04     0.428571    0.929153
  all_steps          coco_val/dinov3_patch/max_patch      pure_gmm_2_5            regret       14 0.169581  0.169383        1.980000e-04     0.428571    0.858863
  all_steps          coco_val/dinov3_patch/max_patch      pure_gmm_2_5 average_precision       14 0.674755  0.674755        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          coco_val/dinov3_patch/max_patch      pure_gmm_2_5             auroc       14 0.936868  0.936868        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          coco_val/dinov3_patch/max_patch      pure_gmm_2_5        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps        coco_val/dinov3_patch/whole_image      pure_gmm_2_5              cost       14 0.633484  0.633312        1.720000e-04     0.285714    0.721277
  all_steps        coco_val/dinov3_patch/whole_image      pure_gmm_2_5               fnr       14 0.116248  0.116248        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps        coco_val/dinov3_patch/whole_image      pure_gmm_2_5               fpr       14 0.517236  0.517064        1.719643e-04     0.285714    0.721277
  all_steps        coco_val/dinov3_patch/whole_image      pure_gmm_2_5            regret       14 0.308168  0.307997        1.719643e-04     0.285714    0.721277
  all_steps        coco_val/dinov3_patch/whole_image      pure_gmm_2_5 average_precision       14 0.478819  0.478819        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps        coco_val/dinov3_patch/whole_image      pure_gmm_2_5             auroc       14 0.859874  0.859874        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps        coco_val/dinov3_patch/whole_image      pure_gmm_2_5        degenerate       14 0.035714  0.035714        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps              coco_val/siglip/whole_image      pure_gmm_2_5              cost       14 0.522288  0.521464        8.242024e-04     0.357143    0.789675
  all_steps              coco_val/siglip/whole_image      pure_gmm_2_5               fnr       14 0.077431  0.077431        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps              coco_val/siglip/whole_image      pure_gmm_2_5               fpr       14 0.444857  0.444033        8.242024e-04     0.357143    0.789675
  all_steps              coco_val/siglip/whole_image      pure_gmm_2_5            regret       14 0.241024  0.240200        8.241310e-04     0.357143    0.789675
  all_steps              coco_val/siglip/whole_image      pure_gmm_2_5 average_precision       14 0.500590  0.500590        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps              coco_val/siglip/whole_image      pure_gmm_2_5             auroc       14 0.901479  0.901479        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps              coco_val/siglip/whole_image      pure_gmm_2_5        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps   visual_genome_m/dinov3_patch/max_patch      pure_gmm_2_5              cost       16 0.362575  0.361742        8.326562e-04     0.250000    0.068217
  all_steps   visual_genome_m/dinov3_patch/max_patch      pure_gmm_2_5               fnr       16 0.117047  0.117047        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps   visual_genome_m/dinov3_patch/max_patch      pure_gmm_2_5               fpr       16 0.245528  0.244695        8.326562e-04     0.250000    0.075368
  all_steps   visual_genome_m/dinov3_patch/max_patch      pure_gmm_2_5            regret       16 0.120249  0.119416        8.326875e-04     0.250000    0.061884
  all_steps   visual_genome_m/dinov3_patch/max_patch      pure_gmm_2_5 average_precision       16 0.463686  0.463686        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps   visual_genome_m/dinov3_patch/max_patch      pure_gmm_2_5             auroc       16 0.907755  0.907755        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps   visual_genome_m/dinov3_patch/max_patch      pure_gmm_2_5        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps visual_genome_m/dinov3_patch/whole_image      pure_gmm_2_5              cost       16 0.739284  0.739388       -1.049063e-04     0.375000    0.964524
  all_steps visual_genome_m/dinov3_patch/whole_image      pure_gmm_2_5               fnr       16 0.134910  0.134910        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps visual_genome_m/dinov3_patch/whole_image      pure_gmm_2_5               fpr       16 0.604373  0.604478       -1.048750e-04     0.375000    1.000000
  all_steps visual_genome_m/dinov3_patch/whole_image      pure_gmm_2_5            regret       16 0.224263  0.224368       -1.048438e-04     0.375000    0.929153
  all_steps visual_genome_m/dinov3_patch/whole_image      pure_gmm_2_5 average_precision       16 0.236464  0.236464        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps visual_genome_m/dinov3_patch/whole_image      pure_gmm_2_5             auroc       16 0.762593  0.762593        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps visual_genome_m/dinov3_patch/whole_image      pure_gmm_2_5        degenerate       16 0.031250  0.031250        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps       visual_genome_m/siglip/whole_image      pure_gmm_2_5              cost       15 0.677047  0.677067       -1.960000e-05     0.400000    0.924978
  all_steps       visual_genome_m/siglip/whole_image      pure_gmm_2_5               fnr       15 0.145987  0.146021       -3.360000e-05     0.066667    0.654721
  all_steps       visual_genome_m/siglip/whole_image      pure_gmm_2_5               fpr       15 0.531060  0.531046        1.405556e-05     0.400000    0.875291
  all_steps       visual_genome_m/siglip/whole_image      pure_gmm_2_5            regret       15 0.202675  0.202694       -1.948889e-05     0.400000    0.924978
  all_steps       visual_genome_m/siglip/whole_image      pure_gmm_2_5 average_precision       15 0.249008  0.249008        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps       visual_genome_m/siglip/whole_image      pure_gmm_2_5             auroc       15 0.788571  0.788571        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps       visual_genome_m/siglip/whole_image      pure_gmm_2_5        degenerate       15 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image         ramp_6_20              cost       12 0.504056  0.480107        2.394865e-02     0.083333    0.187500
  all_steps    caltech101_m/dinov3_patch/whole_image         ramp_6_20               fnr       12 0.069859  0.071067       -1.207722e-03     0.083333    1.000000
  all_steps    caltech101_m/dinov3_patch/whole_image         ramp_6_20               fpr       12 0.434196  0.409040        2.515637e-02     0.083333    0.187500
  all_steps    caltech101_m/dinov3_patch/whole_image         ramp_6_20            regret       12 0.490639  0.466690        2.394865e-02     0.083333    0.187500
  all_steps    caltech101_m/dinov3_patch/whole_image         ramp_6_20 average_precision       12 0.996441  0.996441        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image         ramp_6_20             auroc       12 0.997770  0.997770        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image         ramp_6_20        degenerate       12 0.057234  0.057234        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image         ramp_6_20              cost       12 0.077890  0.076054        1.835722e-03     0.250000    0.687500
  all_steps          caltech101_m/siglip/whole_image         ramp_6_20               fnr       12 0.002980  0.002980        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image         ramp_6_20               fpr       12 0.074910  0.073075        1.835722e-03     0.250000    0.687500
  all_steps          caltech101_m/siglip/whole_image         ramp_6_20            regret       12 0.077108  0.075272        1.835722e-03     0.250000    0.687500
  all_steps          caltech101_m/siglip/whole_image         ramp_6_20 average_precision       12 0.999894  0.999894        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image         ramp_6_20             auroc       12 0.999948  0.999948        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image         ramp_6_20        degenerate       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          coco_val/dinov3_patch/max_patch         ramp_6_20              cost       14 0.231010  0.235657       -4.647129e-03     0.428571    1.000000
  all_steps          coco_val/dinov3_patch/max_patch         ramp_6_20               fnr       14 0.067229  0.078653       -1.142341e-02     0.500000    0.138641
  all_steps          coco_val/dinov3_patch/max_patch         ramp_6_20               fpr       14 0.163781  0.157005        6.776281e-03     0.500000    0.951538
  all_steps          coco_val/dinov3_patch/max_patch         ramp_6_20            regret       14 0.094644  0.087513        7.131386e-03     0.428571    0.267578
  all_steps          coco_val/dinov3_patch/max_patch         ramp_6_20 average_precision       14 0.734537  0.729004        5.533314e-03     0.571429    1.000000
  all_steps          coco_val/dinov3_patch/max_patch         ramp_6_20             auroc       14 0.965863  0.962553        3.310300e-03     0.428571    0.669800
  all_steps          coco_val/dinov3_patch/max_patch         ramp_6_20        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps        coco_val/dinov3_patch/whole_image         ramp_6_20              cost       14 0.460993  0.461013       -1.983333e-05     0.500000    0.951538
  all_steps        coco_val/dinov3_patch/whole_image         ramp_6_20               fnr       14 0.127787  0.110065        1.772175e-02     0.285714    0.172955
  all_steps        coco_val/dinov3_patch/whole_image         ramp_6_20               fpr       14 0.333206  0.350948       -1.774157e-02     0.714286    0.325806
  all_steps        coco_val/dinov3_patch/whole_image         ramp_6_20            regret       14 0.205068  0.211744       -6.676752e-03     0.500000    0.951538
  all_steps        coco_val/dinov3_patch/whole_image         ramp_6_20 average_precision       14 0.579812  0.584652       -4.839962e-03     0.500000    0.600179
  all_steps        coco_val/dinov3_patch/whole_image         ramp_6_20             auroc       14 0.901177  0.907670       -6.492590e-03     0.357143    0.916512
  all_steps        coco_val/dinov3_patch/whole_image         ramp_6_20        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps              coco_val/siglip/whole_image         ramp_6_20              cost       14 0.357552  0.376007       -1.845437e-02     0.785714    0.078491
  all_steps              coco_val/siglip/whole_image         ramp_6_20               fnr       14 0.124292  0.121172        3.120248e-03     0.285714    0.480177
  all_steps              coco_val/siglip/whole_image         ramp_6_20               fpr       14 0.233260  0.254835       -2.157454e-02     0.785714    0.016602
  all_steps              coco_val/siglip/whole_image         ramp_6_20            regret       14 0.118599  0.127084       -8.485714e-03     0.714286    0.241211
  all_steps              coco_val/siglip/whole_image         ramp_6_20 average_precision       14 0.586921  0.588307       -1.386033e-03     0.428571    0.669800
  all_steps              coco_val/siglip/whole_image         ramp_6_20             auroc       14 0.917315  0.913958        3.357414e-03     0.285714    0.267578
  all_steps              coco_val/siglip/whole_image         ramp_6_20        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps   visual_genome_m/dinov3_patch/max_patch         ramp_6_20              cost       16 0.291132  0.301631       -1.049927e-02     0.687500    0.274445
  all_steps   visual_genome_m/dinov3_patch/max_patch         ramp_6_20               fnr       16 0.111642  0.111520        1.218958e-04     0.375000    0.861304
  all_steps   visual_genome_m/dinov3_patch/max_patch         ramp_6_20               fpr       16 0.179490  0.190111       -1.062110e-02     0.687500    0.322510
  all_steps   visual_genome_m/dinov3_patch/max_patch         ramp_6_20            regret       16 0.070778  0.074699       -3.921729e-03     0.687500    0.375458
  all_steps   visual_genome_m/dinov3_patch/max_patch         ramp_6_20 average_precision       16 0.500915  0.476558        2.435676e-02     0.375000    0.211426
  all_steps   visual_genome_m/dinov3_patch/max_patch         ramp_6_20             auroc       16 0.917479  0.914922        2.557125e-03     0.437500    0.781952
  all_steps   visual_genome_m/dinov3_patch/max_patch         ramp_6_20        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps visual_genome_m/dinov3_patch/whole_image         ramp_6_20              cost       16 0.587663  0.589748       -2.084171e-03     0.437500    0.979950
  all_steps visual_genome_m/dinov3_patch/whole_image         ramp_6_20               fnr       16 0.232261  0.236472       -4.210971e-03     0.500000    0.860260
  all_steps visual_genome_m/dinov3_patch/whole_image         ramp_6_20               fpr       16 0.355403  0.353276        2.126804e-03     0.500000    0.899933
  all_steps visual_genome_m/dinov3_patch/whole_image         ramp_6_20            regret       16 0.125881  0.129488       -3.606417e-03     0.375000    0.743561
  all_steps visual_genome_m/dinov3_patch/whole_image         ramp_6_20 average_precision       16 0.311397  0.314061       -2.663833e-03     0.500000    0.596588
  all_steps visual_genome_m/dinov3_patch/whole_image         ramp_6_20             auroc       16 0.789869  0.789802        6.668750e-05     0.250000    0.322510
  all_steps visual_genome_m/dinov3_patch/whole_image         ramp_6_20        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps       visual_genome_m/siglip/whole_image         ramp_6_20              cost       16 0.497362  0.479799        1.756288e-02     0.375000    0.192810
  all_steps       visual_genome_m/siglip/whole_image         ramp_6_20               fnr       16 0.168324  0.163174        5.150158e-03     0.437500    0.733271
  all_steps       visual_genome_m/siglip/whole_image         ramp_6_20               fpr       16 0.329038  0.316625        1.241274e-02     0.312500    0.104584
  all_steps       visual_genome_m/siglip/whole_image         ramp_6_20            regret       16 0.106071  0.105138        9.331000e-04     0.562500    0.860260
  all_steps       visual_genome_m/siglip/whole_image         ramp_6_20 average_precision       16 0.322253  0.344372       -2.211974e-02     0.750000    0.028992
  all_steps       visual_genome_m/siglip/whole_image         ramp_6_20             auroc       16 0.839564  0.845169       -5.604883e-03     0.437500    0.820892
  all_steps       visual_genome_m/siglip/whole_image         ramp_6_20        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus              cost       12 0.069593  0.067245        2.347577e-03     0.000000    0.062500
  all_steps    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus               fnr       12 0.007264  0.006241        1.023061e-03     0.000000    1.000000
  all_steps    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus               fpr       12 0.062329  0.061004        1.324516e-03     0.083333    0.312500
  all_steps    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus            regret       12 0.068263  0.065909        2.354364e-03     0.000000    0.062500
  all_steps    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus average_precision       12 0.999737  0.999737        1.635417e-07     0.000000    1.000000
  all_steps    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus             auroc       12 0.999855  0.999855        6.041667e-08     0.000000    1.000000
  all_steps    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus        degenerate       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image post_ramp_21_plus              cost       12 0.007504  0.014302       -6.797766e-03     0.333333    0.125000
  all_steps          caltech101_m/siglip/whole_image post_ramp_21_plus               fnr       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image post_ramp_21_plus               fpr       12 0.007504  0.014302       -6.797766e-03     0.333333    0.125000
  all_steps          caltech101_m/siglip/whole_image post_ramp_21_plus            regret       12 0.007504  0.014302       -6.797766e-03     0.333333    0.125000
  all_steps          caltech101_m/siglip/whole_image post_ramp_21_plus average_precision       12 1.000000  1.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image post_ramp_21_plus             auroc       12 1.000000  1.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image post_ramp_21_plus        degenerate       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          coco_val/dinov3_patch/max_patch post_ramp_21_plus              cost       14 0.186050  0.182593        3.456675e-03     0.357143    0.426270
  all_steps          coco_val/dinov3_patch/max_patch post_ramp_21_plus               fnr       14 0.069580  0.065318        4.261574e-03     0.214286    0.332880
  all_steps          coco_val/dinov3_patch/max_patch post_ramp_21_plus               fpr       14 0.116470  0.117275       -8.049223e-04     0.571429    0.855225
  all_steps          coco_val/dinov3_patch/max_patch post_ramp_21_plus            regret       14 0.069281  0.071986       -2.705342e-03     0.571429    0.583008
  all_steps          coco_val/dinov3_patch/max_patch post_ramp_21_plus average_precision       14 0.796775  0.791461        5.313473e-03     0.571429    0.855225
  all_steps          coco_val/dinov3_patch/max_patch post_ramp_21_plus             auroc       14 0.974329  0.976726       -2.396751e-03     0.500000    0.325806
  all_steps          coco_val/dinov3_patch/max_patch post_ramp_21_plus        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps        coco_val/dinov3_patch/whole_image post_ramp_21_plus              cost       14 0.260601  0.257909        2.691995e-03     0.571429    0.855225
  all_steps        coco_val/dinov3_patch/whole_image post_ramp_21_plus               fnr       14 0.130779  0.127504        3.274643e-03     0.428571    0.583008
  all_steps        coco_val/dinov3_patch/whole_image post_ramp_21_plus               fpr       14 0.129822  0.130405       -5.826152e-04     0.571429    0.855225
  all_steps        coco_val/dinov3_patch/whole_image post_ramp_21_plus            regret       14 0.064700  0.066101       -1.401479e-03     0.642857    0.807739
  all_steps        coco_val/dinov3_patch/whole_image post_ramp_21_plus average_precision       14 0.687915  0.694278       -6.363101e-03     0.500000    0.760864
  all_steps        coco_val/dinov3_patch/whole_image post_ramp_21_plus             auroc       14 0.938011  0.935866        2.144571e-03     0.428571    0.541626
  all_steps        coco_val/dinov3_patch/whole_image post_ramp_21_plus        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps              coco_val/siglip/whole_image post_ramp_21_plus              cost       14 0.245348  0.245748       -3.999848e-04     0.428571    0.855225
  all_steps              coco_val/siglip/whole_image post_ramp_21_plus               fnr       14 0.111639  0.120311       -8.672262e-03     0.428571    0.346522
  all_steps              coco_val/siglip/whole_image post_ramp_21_plus               fpr       14 0.133709  0.125437        8.272305e-03     0.357143    0.216553
  all_steps              coco_val/siglip/whole_image post_ramp_21_plus            regret       14 0.056119  0.055458        6.614116e-04     0.500000    0.951538
  all_steps              coco_val/siglip/whole_image post_ramp_21_plus average_precision       14 0.668080  0.678684       -1.060427e-02     0.500000    0.541626
  all_steps              coco_val/siglip/whole_image post_ramp_21_plus             auroc       14 0.938817  0.939264       -4.471884e-04     0.428571    0.807739
  all_steps              coco_val/siglip/whole_image post_ramp_21_plus        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus              cost       16 0.258347  0.247767        1.057968e-02     0.375000    0.274445
  all_steps   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus               fnr       16 0.097930  0.089077        8.852977e-03     0.375000    0.432626
  all_steps   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus               fpr       16 0.160417  0.158690        1.726721e-03     0.437500    0.860260
  all_steps   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus            regret       16 0.063354  0.057993        5.361305e-03     0.437500    0.561890
  all_steps   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus average_precision       16 0.532590  0.519367        1.322281e-02     0.562500    0.632172
  all_steps   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus             auroc       16 0.925180  0.922402        2.778027e-03     0.375000    0.211426
  all_steps   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus              cost       16 0.451361  0.446176        5.185217e-03     0.437500    0.632172
  all_steps visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus               fnr       16 0.281752  0.292386       -1.063405e-02     0.500000    0.463745
  all_steps visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus               fpr       16 0.169609  0.153790        1.581927e-02     0.312500    0.211426
  all_steps visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus            regret       16 0.059722  0.057309        2.413228e-03     0.500000    0.979950
  all_steps visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus average_precision       16 0.376011  0.412806       -3.679579e-02     0.687500    0.129730
  all_steps visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus             auroc       16 0.832593  0.833232       -6.387336e-04     0.500000    0.860260
  all_steps visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps       visual_genome_m/siglip/whole_image post_ramp_21_plus              cost       16 0.390294  0.371828        1.846564e-02     0.375000    0.211426
  all_steps       visual_genome_m/siglip/whole_image post_ramp_21_plus               fnr       16 0.175374  0.168798        6.576253e-03     0.500000    0.743561
  all_steps       visual_genome_m/siglip/whole_image post_ramp_21_plus               fpr       16 0.214920  0.203031        1.188940e-02     0.437500    0.348389
  all_steps       visual_genome_m/siglip/whole_image post_ramp_21_plus            regret       16 0.074281  0.069882        4.398484e-03     0.375000    0.596588
  all_steps       visual_genome_m/siglip/whole_image post_ramp_21_plus average_precision       16 0.399815  0.429099       -2.928414e-02     0.687500    0.116669
  all_steps       visual_genome_m/siglip/whole_image post_ramp_21_plus             auroc       16 0.879063  0.884096       -5.032892e-03     0.625000    0.495422
  all_steps       visual_genome_m/siglip/whole_image post_ramp_21_plus        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image         all_steps              cost       12 0.142636  0.136993        5.643784e-03     0.000000    0.031250
  all_steps    caltech101_m/dinov3_patch/whole_image         all_steps               fnr       12 0.016079  0.015395        6.839294e-04     0.083333    1.000000
  all_steps    caltech101_m/dinov3_patch/whole_image         all_steps               fpr       12 0.126558  0.121598        4.959855e-03     0.083333    0.156250
  all_steps    caltech101_m/dinov3_patch/whole_image         all_steps            regret       12 0.139518  0.133868        5.649561e-03     0.000000    0.031250
  all_steps    caltech101_m/dinov3_patch/whole_image         all_steps average_precision       12 0.999200  0.999199        1.391844e-07     0.000000    1.000000
  all_steps    caltech101_m/dinov3_patch/whole_image         all_steps             auroc       12 0.999549  0.999549        5.141844e-08     0.000000    1.000000
  all_steps    caltech101_m/dinov3_patch/whole_image         all_steps        degenerate       12 0.008055  0.008055        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image         all_steps              cost       12 0.027708  0.033055       -5.346443e-03     0.500000    0.195312
  all_steps          caltech101_m/siglip/whole_image         all_steps               fnr       12 0.000800  0.000800        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image         all_steps               fpr       12 0.026908  0.032255       -5.346443e-03     0.500000    0.195312
  all_steps          caltech101_m/siglip/whole_image         all_steps            regret       12 0.027304  0.032651       -5.346443e-03     0.500000    0.195312
  all_steps          caltech101_m/siglip/whole_image         all_steps average_precision       12 0.999741  0.999741        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image         all_steps             auroc       12 0.999962  0.999962        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image         all_steps        degenerate       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          coco_val/dinov3_patch/max_patch         all_steps              cost       14 0.196362  0.194226        2.136320e-03     0.428571    0.463135
  all_steps          coco_val/dinov3_patch/max_patch         all_steps               fnr       14 0.069621  0.067873        1.748194e-03     0.214286    0.444587
  all_steps          coco_val/dinov3_patch/max_patch         all_steps               fpr       14 0.126741  0.126353        3.881060e-04     0.571429    0.807739
  all_steps          coco_val/dinov3_patch/max_patch         all_steps            regret       14 0.075271  0.076395       -1.124336e-03     0.571429    0.583008
  all_steps          coco_val/dinov3_patch/max_patch         all_steps average_precision       14 0.784634  0.779397        5.237913e-03     0.500000    1.000000
  all_steps          coco_val/dinov3_patch/max_patch         all_steps             auroc       14 0.972248  0.973713       -1.464800e-03     0.571429    0.541626
  all_steps          coco_val/dinov3_patch/max_patch         all_steps        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps        coco_val/dinov3_patch/whole_image         all_steps              cost       14 0.299278  0.297057        2.220681e-03     0.500000    0.760864
  all_steps        coco_val/dinov3_patch/whole_image         all_steps               fnr       14 0.130017  0.124575        5.441213e-03     0.428571    0.501587
  all_steps        coco_val/dinov3_patch/whole_image         all_steps               fpr       14 0.169261  0.172482       -3.220504e-03     0.500000    0.714844
  all_steps        coco_val/dinov3_patch/whole_image         all_steps            regret       14 0.091426  0.093611       -2.184801e-03     0.571429    1.000000
  all_steps        coco_val/dinov3_patch/whole_image         all_steps average_precision       14 0.666887  0.672883       -5.996366e-03     0.571429    0.714844
  all_steps        coco_val/dinov3_patch/whole_image         all_steps             auroc       14 0.930704  0.929939        7.647099e-04     0.428571    0.583008
  all_steps        coco_val/dinov3_patch/whole_image         all_steps        degenerate       14 0.000736  0.000736        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps              coco_val/siglip/whole_image         all_steps              cost       14 0.268593  0.271720       -3.127113e-03     0.500000    0.855225
  all_steps              coco_val/siglip/whole_image         all_steps               fnr       14 0.112896  0.119551       -6.655030e-03     0.428571    0.432768
  all_steps              coco_val/siglip/whole_image         all_steps               fpr       14 0.155698  0.152170        3.527953e-03     0.428571    0.463135
  all_steps              coco_val/siglip/whole_image         all_steps            regret       14 0.069684  0.070417       -7.326769e-04     0.500000    0.807739
  all_steps              coco_val/siglip/whole_image         all_steps average_precision       14 0.652038  0.660957       -8.919242e-03     0.428571    0.541626
  all_steps              coco_val/siglip/whole_image         all_steps             auroc       14 0.934673  0.934542        1.310631e-04     0.357143    0.501587
  all_steps              coco_val/siglip/whole_image         all_steps        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps   visual_genome_m/dinov3_patch/max_patch         all_steps              cost       16 0.265565  0.258446        7.119077e-03     0.437500    0.528168
  all_steps   visual_genome_m/dinov3_patch/max_patch         all_steps               fnr       16 0.100444  0.093124        7.320274e-03     0.375000    0.396726
  all_steps   visual_genome_m/dinov3_patch/max_patch         all_steps               fpr       16 0.165121  0.165322       -2.011695e-04     0.625000    0.561890
  all_steps   visual_genome_m/dinov3_patch/max_patch         all_steps            regret       16 0.065675  0.061843        3.832410e-03     0.437500    0.561890
  all_steps   visual_genome_m/dinov3_patch/max_patch         all_steps average_precision       16 0.526271  0.511599        1.467192e-02     0.562500    0.495422
  all_steps   visual_genome_m/dinov3_patch/max_patch         all_steps             auroc       16 0.923630  0.920943        2.686588e-03     0.375000    0.211426
  all_steps   visual_genome_m/dinov3_patch/max_patch         all_steps        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps visual_genome_m/dinov3_patch/whole_image         all_steps              cost       16 0.478376  0.474424        3.952010e-03     0.375000    0.528168
  all_steps visual_genome_m/dinov3_patch/whole_image         all_steps               fnr       16 0.271071  0.280493       -9.421535e-03     0.500000    0.348389
  all_steps visual_genome_m/dinov3_patch/whole_image         all_steps               fpr       16 0.207304  0.193931        1.337354e-02     0.375000    0.231201
  all_steps visual_genome_m/dinov3_patch/whole_image         all_steps            regret       16 0.073346  0.071915        1.430436e-03     0.437500    0.820892
  all_steps visual_genome_m/dinov3_patch/whole_image         all_steps average_precision       16 0.363142  0.393901       -3.075898e-02     0.687500    0.129730
  all_steps visual_genome_m/dinov3_patch/whole_image         all_steps             auroc       16 0.824543  0.825060       -5.164781e-04     0.500000    0.860260
  all_steps visual_genome_m/dinov3_patch/whole_image         all_steps        degenerate       16 0.000644  0.000644        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps       visual_genome_m/siglip/whole_image         all_steps              cost       16 0.413645  0.395604        1.804037e-02     0.375000    0.192810
  all_steps       visual_genome_m/siglip/whole_image         all_steps               fnr       16 0.174324  0.168032        6.291761e-03     0.500000    0.899933
  all_steps       visual_genome_m/siglip/whole_image         all_steps               fpr       16 0.239321  0.227572        1.174862e-02     0.437500    0.322510
  all_steps       visual_genome_m/siglip/whole_image         all_steps            regret       16 0.081615  0.077912        3.702142e-03     0.375000    0.596588
  all_steps       visual_genome_m/siglip/whole_image         all_steps average_precision       16 0.384246  0.411940       -2.769435e-02     0.687500    0.116669
  all_steps       visual_genome_m/siglip/whole_image         all_steps             auroc       16 0.870418  0.875533       -5.114612e-03     0.625000    0.433197
  all_steps       visual_genome_m/siglip/whole_image         all_steps        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
```

## Verdict (read on the production max_patch arm, app-visible steps)

```json
{
  "app_visible": {
    "scope": "app_visible",
    "pure_gmm_2_5": {
      "n_cells": 0,
      "reading": "no steps in this window at this scope"
    },
    "ramp_6_20": {
      "n_cells": 16,
      "delta_cost_on_minus_off": -0.008074607699592066,
      "p": 0.6372222900390625,
      "reading": "safe ON better (n.s.)"
    },
    "post_ramp_21_plus": {
      "n_cells": 16,
      "delta_cost_on_minus_off": 0.0070181765624999955,
      "p": 0.3503570556640625,
      "reading": "safe ON worse (n.s.)"
    },
    "all_steps": {
      "n_cells": 16,
      "delta_cost_on_minus_off": 0.004822377742284386,
      "p": 0.4956512451171875,
      "reading": "safe ON worse (n.s.)"
    },
    "force_on_for_all_users": true
  },
  "all_steps": {
    "scope": "all_steps",
    "pure_gmm_2_5": {
      "n_cells": 16,
      "delta_cost_on_minus_off": 0.0005153102678571404,
      "p": 0.498684668936212,
      "reading": "safe ON worse (n.s.)"
    },
    "ramp_6_20": {
      "n_cells": 16,
      "delta_cost_on_minus_off": -0.007573199702380951,
      "p": 0.6372222900390625,
      "reading": "safe ON better (n.s.)"
    },
    "post_ramp_21_plus": {
      "n_cells": 16,
      "delta_cost_on_minus_off": 0.0070181765624999955,
      "p": 0.3503570556640625,
      "reading": "safe ON worse (n.s.)"
    },
    "all_steps": {
      "n_cells": 16,
      "delta_cost_on_minus_off": 0.004627698453608245,
      "p": 0.4956512451171875,
      "reading": "safe ON worse (n.s.)"
    },
    "force_on_for_all_users": false
  }
}
```

Curves: `agg/ab_curves_vs_votes.csv` (1556 rows) · paired cells: `agg/ab_paired_cells.csv` · figures: ab_cost_vs_votes.png, ab_fnr_vs_votes.png, ab_degenerate_vs_votes.png
