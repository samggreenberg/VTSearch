# Safe thresholds ON vs OFF — trajectory A/B (#2799)

Head: `linear_svm` · cells ON/OFF: 114/114 · ON `/expscratch/sgreenberg/anchem-3825/ab_ll1e-3/results` vs OFF `/expscratch/sgreenberg/anchem-3825/ab_baseline/results`

Both runs are full simulations; because the blended threshold drives Autopilot's
Hard pick, the two arms vote on different items, so cells (category × seed), not
steps, are the paired units.

The app shows a trained detector from **7 votes** onward; below that it
sorts by text/example cosine, so `scope=app_visible` is what users actually get and
`scope=all_steps` is the purely numerical reading.

## Per-window paired comparison (Δ = ON − OFF; negative = safe thresholds better)

```
      scope                                      arm            window            metric  n_cells  safe_on  safe_off  delta_on_minus_off  win_rate_on  p_wilcoxon              note
app_visible    caltech101_m/dinov3_patch/whole_image         ramp_6_20              cost       11 0.384845  0.384210        6.355221e-04     0.363636    0.812500
app_visible    caltech101_m/dinov3_patch/whole_image         ramp_6_20               fnr       11 0.003162  0.003162        0.000000e+00     0.000000         NaN all-zero or empty
app_visible    caltech101_m/dinov3_patch/whole_image         ramp_6_20               fpr       11 0.381683  0.381048        6.355221e-04     0.363636    0.812500
app_visible    caltech101_m/dinov3_patch/whole_image         ramp_6_20            regret       11 0.378640  0.378005        6.355221e-04     0.363636    0.812500
app_visible    caltech101_m/dinov3_patch/whole_image         ramp_6_20 average_precision       11 0.998476  0.998476        0.000000e+00     0.000000         NaN all-zero or empty
app_visible    caltech101_m/dinov3_patch/whole_image         ramp_6_20             auroc       11 0.999173  0.999173        0.000000e+00     0.000000         NaN all-zero or empty
app_visible    caltech101_m/dinov3_patch/whole_image         ramp_6_20        degenerate       11 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image         ramp_6_20              cost       12 0.053501  0.061930       -8.429038e-03     0.333333    0.910156
app_visible          caltech101_m/siglip/whole_image         ramp_6_20               fnr       12 0.002388  0.001976        4.115417e-04     0.000000    0.500000
app_visible          caltech101_m/siglip/whole_image         ramp_6_20               fpr       12 0.051113  0.059954       -8.840586e-03     0.333333    0.820312
app_visible          caltech101_m/siglip/whole_image         ramp_6_20            regret       12 0.052678  0.061311       -8.632604e-03     0.333333    0.910156
app_visible          caltech101_m/siglip/whole_image         ramp_6_20 average_precision       12 0.999930  0.999937       -7.511905e-06     0.166667    0.500000
app_visible          caltech101_m/siglip/whole_image         ramp_6_20             auroc       12 0.999970  0.999974       -3.291667e-06     0.166667    0.500000
app_visible          caltech101_m/siglip/whole_image         ramp_6_20        degenerate       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          coco_val/dinov3_patch/max_patch         ramp_6_20              cost       14 0.232562  0.230334        2.227951e-03     0.571429    0.669800
app_visible          coco_val/dinov3_patch/max_patch         ramp_6_20               fnr       14 0.052391  0.080090       -2.769903e-02     0.500000    0.059336
app_visible          coco_val/dinov3_patch/max_patch         ramp_6_20               fpr       14 0.180171  0.150244        2.992701e-02     0.428571    0.172607
app_visible          coco_val/dinov3_patch/max_patch         ramp_6_20            regret       14 0.110145  0.084241        2.590351e-02     0.357143    0.016602
app_visible          coco_val/dinov3_patch/max_patch         ramp_6_20 average_precision       14 0.747441  0.733187        1.425334e-02     0.285714    0.193726
app_visible          coco_val/dinov3_patch/max_patch         ramp_6_20             auroc       14 0.970367  0.964392        5.974626e-03     0.214286    0.041870
app_visible          coco_val/dinov3_patch/max_patch         ramp_6_20        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible        coco_val/dinov3_patch/whole_image         ramp_6_20              cost       14 0.494436  0.450312        4.412407e-02     0.285714    0.041870
app_visible        coco_val/dinov3_patch/whole_image         ramp_6_20               fnr       14 0.114323  0.107476        6.846752e-03     0.500000    0.916512
app_visible        coco_val/dinov3_patch/whole_image         ramp_6_20               fpr       14 0.380113  0.342836        3.727743e-02     0.428571    0.135254
app_visible        coco_val/dinov3_patch/whole_image         ramp_6_20            regret       14 0.236839  0.207048        2.979093e-02     0.285714    0.172607
app_visible        coco_val/dinov3_patch/whole_image         ramp_6_20 average_precision       14 0.571295  0.591454       -2.015920e-02     0.571429    0.345448
app_visible        coco_val/dinov3_patch/whole_image         ramp_6_20             auroc       14 0.906704  0.912524       -5.820071e-03     0.571429    0.310897
app_visible        coco_val/dinov3_patch/whole_image         ramp_6_20        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible              coco_val/siglip/whole_image         ramp_6_20              cost       14 0.370418  0.370226        1.915561e-04     0.428571    0.714844
app_visible              coco_val/siglip/whole_image         ramp_6_20               fnr       14 0.079266  0.122506       -4.323979e-02     0.642857    0.034170
app_visible              coco_val/siglip/whole_image         ramp_6_20               fpr       14 0.291152  0.247720        4.343138e-02     0.285714    0.016602
app_visible              coco_val/siglip/whole_image         ramp_6_20            regret       14 0.149599  0.122281        2.731784e-02     0.142857    0.004028
app_visible              coco_val/siglip/whole_image         ramp_6_20 average_precision       14 0.614984  0.591723        2.326085e-02     0.214286    0.029541
app_visible              coco_val/siglip/whole_image         ramp_6_20             auroc       14 0.931859  0.914152        1.770655e-02     0.214286    0.020264
app_visible              coco_val/siglip/whole_image         ramp_6_20        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible   visual_genome_m/dinov3_patch/max_patch         ramp_6_20              cost       16 0.317748  0.301984        1.576356e-02     0.500000    0.939880
app_visible   visual_genome_m/dinov3_patch/max_patch         ramp_6_20               fnr       16 0.105585  0.110370       -4.784865e-03     0.437500    0.700703
app_visible   visual_genome_m/dinov3_patch/max_patch         ramp_6_20               fpr       16 0.212163  0.191614        2.054847e-02     0.562500    0.899933
app_visible   visual_genome_m/dinov3_patch/max_patch         ramp_6_20            regret       16 0.097631  0.076157        2.147431e-02     0.437500    0.433197
app_visible   visual_genome_m/dinov3_patch/max_patch         ramp_6_20 average_precision       16 0.486617  0.476947        9.670213e-03     0.500000    0.433197
app_visible   visual_genome_m/dinov3_patch/max_patch         ramp_6_20             auroc       16 0.914638  0.915168       -5.303279e-04     0.562500    0.705719
app_visible   visual_genome_m/dinov3_patch/max_patch         ramp_6_20        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible visual_genome_m/dinov3_patch/whole_image         ramp_6_20              cost       16 0.648349  0.579751        6.859785e-02     0.062500    0.001312
app_visible visual_genome_m/dinov3_patch/whole_image         ramp_6_20               fnr       16 0.226101  0.243634       -1.753270e-02     0.812500    0.083252
app_visible visual_genome_m/dinov3_patch/whole_image         ramp_6_20               fpr       16 0.422247  0.336117        8.613057e-02     0.062500    0.000305
app_visible visual_genome_m/dinov3_patch/whole_image         ramp_6_20            regret       16 0.170392  0.121744        4.864838e-02     0.187500    0.007629
app_visible visual_genome_m/dinov3_patch/whole_image         ramp_6_20 average_precision       16 0.302337  0.318277       -1.593973e-02     0.812500    0.038635
app_visible visual_genome_m/dinov3_patch/whole_image         ramp_6_20             auroc       16 0.780659  0.791562       -1.090336e-02     0.625000    0.116669
app_visible visual_genome_m/dinov3_patch/whole_image         ramp_6_20        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible       visual_genome_m/siglip/whole_image         ramp_6_20              cost       16 0.491382  0.464855        2.652739e-02     0.375000    0.129730
app_visible       visual_genome_m/siglip/whole_image         ramp_6_20               fnr       16 0.141636  0.159322       -1.768577e-02     0.625000    0.255989
app_visible       visual_genome_m/siglip/whole_image         ramp_6_20               fpr       16 0.349746  0.305533        4.421311e-02     0.187500    0.002686
app_visible       visual_genome_m/siglip/whole_image         ramp_6_20            regret       16 0.127115  0.098410        2.870497e-02     0.375000    0.083252
app_visible       visual_genome_m/siglip/whole_image         ramp_6_20 average_precision       16 0.352313  0.350310        2.003320e-03     0.562500    0.899933
app_visible       visual_genome_m/siglip/whole_image         ramp_6_20             auroc       16 0.856143  0.852044        4.098875e-03     0.437500    0.596588
app_visible       visual_genome_m/siglip/whole_image         ramp_6_20        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus              cost       12 0.071867  0.060939        1.092802e-02     0.333333    0.640625
app_visible    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus               fnr       12 0.006088  0.006392       -3.036142e-04     0.083333    1.000000
app_visible    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus               fpr       12 0.065779  0.054547        1.123163e-02     0.333333    0.546875
app_visible    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus            regret       12 0.071149  0.060126        1.102223e-02     0.333333    0.640625
app_visible    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus average_precision       12 0.999890  0.999884        6.707523e-06     0.000000    0.500000
app_visible    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus             auroc       12 0.999941  0.999938        2.845139e-06     0.000000    0.500000
app_visible    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus        degenerate       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image post_ramp_21_plus              cost       12 0.000920  0.014302       -1.338259e-02     0.250000    0.437500
app_visible          caltech101_m/siglip/whole_image post_ramp_21_plus               fnr       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image post_ramp_21_plus               fpr       12 0.000920  0.014302       -1.338259e-02     0.250000    0.437500
app_visible          caltech101_m/siglip/whole_image post_ramp_21_plus            regret       12 0.000920  0.014302       -1.338259e-02     0.250000    0.437500
app_visible          caltech101_m/siglip/whole_image post_ramp_21_plus average_precision       12 1.000000  1.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image post_ramp_21_plus             auroc       12 1.000000  1.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image post_ramp_21_plus        degenerate       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          coco_val/dinov3_patch/max_patch post_ramp_21_plus              cost       14 0.193847  0.182593        1.125370e-02     0.428571    0.390991
app_visible          coco_val/dinov3_patch/max_patch post_ramp_21_plus               fnr       14 0.055065  0.065318       -1.025304e-02     0.500000    0.202622
app_visible          coco_val/dinov3_patch/max_patch post_ramp_21_plus               fpr       14 0.138782  0.117275        2.150675e-02     0.428571    0.325806
app_visible          coco_val/dinov3_patch/max_patch post_ramp_21_plus            regret       14 0.084642  0.071986        1.265538e-02     0.357143    0.135254
app_visible          coco_val/dinov3_patch/max_patch post_ramp_21_plus average_precision       14 0.783907  0.791461       -7.554346e-03     0.500000    0.807739
app_visible          coco_val/dinov3_patch/max_patch post_ramp_21_plus             auroc       14 0.975886  0.976726       -8.404643e-04     0.500000    0.463135
app_visible          coco_val/dinov3_patch/max_patch post_ramp_21_plus        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible        coco_val/dinov3_patch/whole_image post_ramp_21_plus              cost       14 0.300961  0.257909        4.305204e-02     0.214286    0.078491
app_visible        coco_val/dinov3_patch/whole_image post_ramp_21_plus               fnr       14 0.113832  0.127504       -1.367216e-02     0.500000    0.583008
app_visible        coco_val/dinov3_patch/whole_image post_ramp_21_plus               fpr       14 0.187129  0.130405        5.672424e-02     0.214286    0.035278
app_visible        coco_val/dinov3_patch/whole_image post_ramp_21_plus            regret       14 0.097619  0.066101        3.151768e-02     0.214286    0.057983
app_visible        coco_val/dinov3_patch/whole_image post_ramp_21_plus average_precision       14 0.656364  0.694278       -3.791474e-02     0.642857    0.135254
app_visible        coco_val/dinov3_patch/whole_image post_ramp_21_plus             auroc       14 0.936380  0.935866        5.135723e-04     0.428571    1.000000
app_visible        coco_val/dinov3_patch/whole_image post_ramp_21_plus        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible              coco_val/siglip/whole_image post_ramp_21_plus              cost       14 0.287236  0.245748        4.148819e-02     0.142857    0.003052
app_visible              coco_val/siglip/whole_image post_ramp_21_plus               fnr       14 0.095017  0.120311       -2.529382e-02     0.785714    0.007649
app_visible              coco_val/siglip/whole_image post_ramp_21_plus               fpr       14 0.192219  0.125437        6.678202e-02     0.071429    0.000244
app_visible              coco_val/siglip/whole_image post_ramp_21_plus            regret       14 0.088859  0.055458        3.340093e-02     0.142857    0.000610
app_visible              coco_val/siglip/whole_image post_ramp_21_plus average_precision       14 0.658562  0.678684       -2.012207e-02     0.714286    0.067627
app_visible              coco_val/siglip/whole_image post_ramp_21_plus             auroc       14 0.938100  0.939264       -1.163781e-03     0.714286    0.501587
app_visible              coco_val/siglip/whole_image post_ramp_21_plus        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus              cost       16 0.296198  0.247767        4.843110e-02     0.187500    0.002686
app_visible   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus               fnr       16 0.087249  0.089077       -1.827880e-03     0.375000    0.924978
app_visible   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus               fpr       16 0.208949  0.158690        5.025903e-02     0.187500    0.004181
app_visible   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus            regret       16 0.089509  0.057993        3.151640e-02     0.375000    0.073914
app_visible   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus average_precision       16 0.506658  0.519367       -1.270895e-02     0.625000    0.252228
app_visible   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus             auroc       16 0.921326  0.922402       -1.075312e-03     0.500000    0.860260
app_visible   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus              cost       16 0.481229  0.446176        3.505313e-02     0.500000    0.433197
app_visible visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus               fnr       16 0.227350  0.292386       -6.503660e-02     0.937500    0.004181
app_visible visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus               fpr       16 0.253880  0.153790        1.000897e-01     0.000000    0.000031
app_visible visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus            regret       16 0.096268  0.057309        3.895910e-02     0.312500    0.038635
app_visible visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus average_precision       16 0.361884  0.412806       -5.092201e-02     0.562500    0.159058
app_visible visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus             auroc       16 0.836908  0.833232        3.676454e-03     0.375000    0.129730
app_visible visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible       visual_genome_m/siglip/whole_image post_ramp_21_plus              cost       16 0.397748  0.371828        2.591998e-02     0.250000    0.028992
app_visible       visual_genome_m/siglip/whole_image post_ramp_21_plus               fnr       16 0.143060  0.168798       -2.573729e-02     0.750000    0.044312
app_visible       visual_genome_m/siglip/whole_image post_ramp_21_plus               fpr       16 0.254688  0.203031        5.165728e-02     0.125000    0.000153
app_visible       visual_genome_m/siglip/whole_image post_ramp_21_plus            regret       16 0.092708  0.069882        2.282544e-02     0.375000    0.231201
app_visible       visual_genome_m/siglip/whole_image post_ramp_21_plus average_precision       16 0.415686  0.429099       -1.341326e-02     0.625000    0.528168
app_visible       visual_genome_m/siglip/whole_image post_ramp_21_plus             auroc       16 0.886258  0.884096        2.161555e-03     0.500000    1.000000
app_visible       visual_genome_m/siglip/whole_image post_ramp_21_plus        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible    caltech101_m/dinov3_patch/whole_image         all_steps              cost       12 0.102841  0.092892        9.948328e-03     0.333333    0.652344
app_visible    caltech101_m/dinov3_patch/whole_image         all_steps               fnr       12 0.005947  0.006265       -3.187418e-04     0.083333    1.000000
app_visible    caltech101_m/dinov3_patch/whole_image         all_steps               fpr       12 0.096894  0.086627        1.026707e-02     0.333333    0.652344
app_visible    caltech101_m/dinov3_patch/whole_image         all_steps            regret       12 0.101919  0.091876        1.004229e-02     0.333333    0.570312
app_visible    caltech101_m/dinov3_patch/whole_image         all_steps average_precision       12 0.999834  0.999827        6.701612e-06     0.000000    0.500000
app_visible    caltech101_m/dinov3_patch/whole_image         all_steps             auroc       12 0.999913  0.999911        2.842955e-06     0.000000    0.500000
app_visible    caltech101_m/dinov3_patch/whole_image         all_steps        degenerate       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image         all_steps              cost       12 0.008478  0.020882       -1.240381e-02     0.416667    0.250000
app_visible          caltech101_m/siglip/whole_image         all_steps               fnr       12 0.000356  0.000294        6.129344e-05     0.000000    0.500000
app_visible          caltech101_m/siglip/whole_image         all_steps               fpr       12 0.008122  0.020587       -1.246510e-02     0.500000    0.203125
app_visible          caltech101_m/siglip/whole_image         all_steps            regret       12 0.008355  0.020789       -1.243412e-02     0.416667    0.250000
app_visible          caltech101_m/siglip/whole_image         all_steps average_precision       12 0.999990  0.999991       -1.118794e-06     0.166667    0.500000
app_visible          caltech101_m/siglip/whole_image         all_steps             auroc       12 0.999996  0.999996       -4.902482e-07     0.166667    0.500000
app_visible          caltech101_m/siglip/whole_image         all_steps        degenerate       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          coco_val/dinov3_patch/max_patch         all_steps              cost       14 0.199537  0.189473        1.006405e-02     0.500000    0.501587
app_visible          coco_val/dinov3_patch/max_patch         all_steps               fnr       14 0.054603  0.067294       -1.269121e-02     0.571429    0.114128
app_visible          coco_val/dinov3_patch/max_patch         all_steps               fpr       14 0.144934  0.122179        2.275528e-02     0.428571    0.295776
app_visible          coco_val/dinov3_patch/max_patch         all_steps            regret       14 0.088422  0.073833        1.458880e-02     0.214286    0.024536
app_visible          coco_val/dinov3_patch/max_patch         all_steps average_precision       14 0.778667  0.783091       -4.424011e-03     0.500000    0.951538
app_visible          coco_val/dinov3_patch/max_patch         all_steps             auroc       14 0.975096  0.974975        1.214849e-04     0.428571    0.669800
app_visible          coco_val/dinov3_patch/max_patch         all_steps        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible        coco_val/dinov3_patch/whole_image         all_steps              cost       14 0.329454  0.286202        4.325225e-02     0.214286    0.057983
app_visible        coco_val/dinov3_patch/whole_image         all_steps               fnr       14 0.113675  0.124250       -1.057524e-02     0.571429    0.541626
app_visible        coco_val/dinov3_patch/whole_image         all_steps               fpr       14 0.215780  0.161952        5.382753e-02     0.214286    0.029541
app_visible        coco_val/dinov3_patch/whole_image         all_steps            regret       14 0.118165  0.086937        3.122732e-02     0.214286    0.041870
app_visible        coco_val/dinov3_patch/whole_image         all_steps average_precision       14 0.643908  0.679297       -3.538858e-02     0.571429    0.172607
app_visible        coco_val/dinov3_patch/whole_image         all_steps             auroc       14 0.932062  0.932501       -4.390822e-04     0.428571    0.855225
app_visible        coco_val/dinov3_patch/whole_image         all_steps        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible              coco_val/siglip/whole_image         all_steps              cost       14 0.299625  0.264287        3.533763e-02     0.142857    0.006714
app_visible              coco_val/siglip/whole_image         all_steps               fnr       14 0.092671  0.120638       -2.796662e-02     0.785714    0.007649
app_visible              coco_val/siglip/whole_image         all_steps               fpr       14 0.206954  0.143649        6.330426e-02     0.000000    0.000122
app_visible              coco_val/siglip/whole_image         all_steps            regret       14 0.097905  0.065410        3.249494e-02     0.000000    0.000122
app_visible              coco_val/siglip/whole_image         all_steps average_precision       14 0.652072  0.665732       -1.366079e-02     0.714286    0.118896
app_visible              coco_val/siglip/whole_image         all_steps             auroc       14 0.937171  0.935524        1.646693e-03     0.642857    1.000000
app_visible              coco_val/siglip/whole_image         all_steps        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible   visual_genome_m/dinov3_patch/max_patch         all_steps              cost       16 0.299296  0.255819        4.347745e-02     0.187500    0.003357
app_visible   visual_genome_m/dinov3_patch/max_patch         all_steps               fnr       16 0.089795  0.092287       -2.492156e-03     0.437500    0.777565
app_visible   visual_genome_m/dinov3_patch/max_patch         all_steps               fpr       16 0.209502  0.163532        4.596965e-02     0.187500    0.009186
app_visible   visual_genome_m/dinov3_patch/max_patch         all_steps            regret       16 0.090706  0.060716        2.999034e-02     0.312500    0.050659
app_visible   visual_genome_m/dinov3_patch/max_patch         all_steps average_precision       16 0.503698  0.513107       -9.408593e-03     0.562500    0.403748
app_visible   visual_genome_m/dinov3_patch/max_patch         all_steps             auroc       16 0.920398  0.921339       -9.409897e-04     0.500000    0.705719
app_visible   visual_genome_m/dinov3_patch/max_patch         all_steps        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible visual_genome_m/dinov3_patch/whole_image         all_steps              cost       16 0.505772  0.465868        3.990457e-02     0.500000    0.297852
app_visible visual_genome_m/dinov3_patch/whole_image         all_steps               fnr       16 0.227229  0.285308       -5.807882e-02     0.875000    0.007629
app_visible visual_genome_m/dinov3_patch/whole_image         all_steps               fpr       16 0.278543  0.180559        9.798337e-02     0.000000    0.000031
app_visible visual_genome_m/dinov3_patch/whole_image         all_steps            regret       16 0.107145  0.066841        4.030442e-02     0.250000    0.006287
app_visible visual_genome_m/dinov3_patch/whole_image         all_steps average_precision       16 0.353148  0.398868       -4.572040e-02     0.562500    0.116669
app_visible visual_genome_m/dinov3_patch/whole_image         all_steps             auroc       16 0.828644  0.827090        1.553993e-03     0.375000    0.231201
app_visible visual_genome_m/dinov3_patch/whole_image         all_steps        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible       visual_genome_m/siglip/whole_image         all_steps              cost       16 0.410921  0.384966        2.595527e-02     0.250000    0.024963
app_visible       visual_genome_m/siglip/whole_image         all_steps               fnr       16 0.142493  0.167226       -2.473293e-02     0.750000    0.028992
app_visible       visual_genome_m/siglip/whole_image         all_steps               fpr       16 0.268428  0.217740        5.068821e-02     0.125000    0.000153
app_visible       visual_genome_m/siglip/whole_image         all_steps            regret       16 0.097915  0.074022        2.389306e-02     0.250000    0.038635
app_visible       visual_genome_m/siglip/whole_image         all_steps average_precision       16 0.406772  0.417683       -1.091049e-02     0.625000    0.495422
app_visible       visual_genome_m/siglip/whole_image         all_steps             auroc       16 0.882304  0.879740        2.564415e-03     0.562500    0.939880
app_visible       visual_genome_m/siglip/whole_image         all_steps        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image      pure_gmm_2_5              cost       10 0.537000  0.542442       -5.441300e-03     0.300000    0.250000
  all_steps    caltech101_m/dinov3_patch/whole_image      pure_gmm_2_5               fnr       10 0.004167  0.004167        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image      pure_gmm_2_5               fpr       10 0.532834  0.538275       -5.441300e-03     0.300000    0.250000
  all_steps    caltech101_m/dinov3_patch/whole_image      pure_gmm_2_5            regret       10 0.535392  0.540833       -5.441300e-03     0.300000    0.250000
  all_steps    caltech101_m/dinov3_patch/whole_image      pure_gmm_2_5 average_precision       10 0.996746  0.996746        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image      pure_gmm_2_5             auroc       10 0.999847  0.999847        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image      pure_gmm_2_5        degenerate       10 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image      pure_gmm_2_5              cost       12 0.489095  0.494730       -5.635000e-03     0.583333    0.015625
  all_steps          caltech101_m/siglip/whole_image      pure_gmm_2_5               fnr       12 0.016441  0.016441        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image      pure_gmm_2_5               fpr       12 0.472654  0.478288       -5.635000e-03     0.583333    0.015625
  all_steps          caltech101_m/siglip/whole_image      pure_gmm_2_5            regret       12 0.474373  0.480008       -5.635000e-03     0.583333    0.015625
  all_steps          caltech101_m/siglip/whole_image      pure_gmm_2_5 average_precision       12 0.985832  0.985832        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image      pure_gmm_2_5             auroc       12 0.998445  0.998445        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image      pure_gmm_2_5        degenerate       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          coco_val/dinov3_patch/max_patch      pure_gmm_2_5              cost       14 0.351113  0.348803        2.309571e-03     0.428571    0.861304
  all_steps          coco_val/dinov3_patch/max_patch      pure_gmm_2_5               fnr       14 0.089039  0.089222       -1.825357e-04     0.142857    1.000000
  all_steps          coco_val/dinov3_patch/max_patch      pure_gmm_2_5               fpr       14 0.262073  0.259581        2.492000e-03     0.500000    0.916512
  all_steps          coco_val/dinov3_patch/max_patch      pure_gmm_2_5            regret       14 0.171692  0.169383        2.309607e-03     0.428571    0.861304
  all_steps          coco_val/dinov3_patch/max_patch      pure_gmm_2_5 average_precision       14 0.674755  0.674755        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          coco_val/dinov3_patch/max_patch      pure_gmm_2_5             auroc       14 0.936868  0.936868        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          coco_val/dinov3_patch/max_patch      pure_gmm_2_5        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps        coco_val/dinov3_patch/whole_image      pure_gmm_2_5              cost       14 0.637126  0.633312        3.813964e-03     0.357143    0.530285
  all_steps        coco_val/dinov3_patch/whole_image      pure_gmm_2_5               fnr       14 0.116215  0.116248       -3.250000e-05     0.142857    0.892738
  all_steps        coco_val/dinov3_patch/whole_image      pure_gmm_2_5               fpr       14 0.520911  0.517064        3.846500e-03     0.428571    0.694887
  all_steps        coco_val/dinov3_patch/whole_image      pure_gmm_2_5            regret       14 0.311810  0.307997        3.813929e-03     0.357143    0.530285
  all_steps        coco_val/dinov3_patch/whole_image      pure_gmm_2_5 average_precision       14 0.478819  0.478819        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps        coco_val/dinov3_patch/whole_image      pure_gmm_2_5             auroc       14 0.859874  0.859874        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps        coco_val/dinov3_patch/whole_image      pure_gmm_2_5        degenerate       14 0.035714  0.035714        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps              coco_val/siglip/whole_image      pure_gmm_2_5              cost       14 0.517700  0.521464       -3.764179e-03     0.500000    0.760864
  all_steps              coco_val/siglip/whole_image      pure_gmm_2_5               fnr       14 0.073337  0.077431       -4.094060e-03     0.142857    0.179712
  all_steps              coco_val/siglip/whole_image      pure_gmm_2_5               fpr       14 0.444363  0.444033        3.298810e-04     0.500000    1.000000
  all_steps              coco_val/siglip/whole_image      pure_gmm_2_5            regret       14 0.236436  0.240200       -3.764321e-03     0.500000    0.760864
  all_steps              coco_val/siglip/whole_image      pure_gmm_2_5 average_precision       14 0.500590  0.500590        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps              coco_val/siglip/whole_image      pure_gmm_2_5             auroc       14 0.901479  0.901479        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps              coco_val/siglip/whole_image      pure_gmm_2_5        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps   visual_genome_m/dinov3_patch/max_patch      pure_gmm_2_5              cost       16 0.363341  0.361742        1.598844e-03     0.375000    0.972125
  all_steps   visual_genome_m/dinov3_patch/max_patch      pure_gmm_2_5               fnr       16 0.108705  0.117047       -8.342188e-03     0.125000    0.592980
  all_steps   visual_genome_m/dinov3_patch/max_patch      pure_gmm_2_5               fpr       16 0.254636  0.244695        9.941000e-03     0.375000    0.753152
  all_steps   visual_genome_m/dinov3_patch/max_patch      pure_gmm_2_5            regret       16 0.121015  0.119416        1.598781e-03     0.375000    0.972125
  all_steps   visual_genome_m/dinov3_patch/max_patch      pure_gmm_2_5 average_precision       16 0.463686  0.463686        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps   visual_genome_m/dinov3_patch/max_patch      pure_gmm_2_5             auroc       16 0.907755  0.907755        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps   visual_genome_m/dinov3_patch/max_patch      pure_gmm_2_5        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps visual_genome_m/dinov3_patch/whole_image      pure_gmm_2_5              cost       16 0.737596  0.739388       -1.792562e-03     0.437500    0.806766
  all_steps visual_genome_m/dinov3_patch/whole_image      pure_gmm_2_5               fnr       16 0.138223  0.134910        3.313062e-03     0.125000    0.400381
  all_steps visual_genome_m/dinov3_patch/whole_image      pure_gmm_2_5               fpr       16 0.599372  0.604478       -5.105625e-03     0.437500    0.506746
  all_steps visual_genome_m/dinov3_patch/whole_image      pure_gmm_2_5            regret       16 0.222575  0.224368       -1.792563e-03     0.437500    0.806766
  all_steps visual_genome_m/dinov3_patch/whole_image      pure_gmm_2_5 average_precision       16 0.236464  0.236464        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps visual_genome_m/dinov3_patch/whole_image      pure_gmm_2_5             auroc       16 0.762593  0.762593        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps visual_genome_m/dinov3_patch/whole_image      pure_gmm_2_5        degenerate       16 0.031250  0.031250        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps       visual_genome_m/siglip/whole_image      pure_gmm_2_5              cost       15 0.680962  0.677067        3.895456e-03     0.266667    0.300290
  all_steps       visual_genome_m/siglip/whole_image      pure_gmm_2_5               fnr       15 0.149491  0.146021        3.470456e-03     0.066667    0.224916
  all_steps       visual_genome_m/siglip/whole_image      pure_gmm_2_5               fpr       15 0.531471  0.531046        4.250111e-04     0.400000    0.974960
  all_steps       visual_genome_m/siglip/whole_image      pure_gmm_2_5            regret       15 0.206590  0.202694        3.895578e-03     0.266667    0.300290
  all_steps       visual_genome_m/siglip/whole_image      pure_gmm_2_5 average_precision       15 0.249008  0.249008        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps       visual_genome_m/siglip/whole_image      pure_gmm_2_5             auroc       15 0.788571  0.788571        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps       visual_genome_m/siglip/whole_image      pure_gmm_2_5        degenerate       15 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image         ramp_6_20              cost       12 0.481286  0.480107        1.179294e-03     0.500000    0.359375
  all_steps    caltech101_m/dinov3_patch/whole_image         ramp_6_20               fnr       12 0.071067  0.071067        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image         ramp_6_20               fpr       12 0.410219  0.409040        1.179294e-03     0.500000    0.359375
  all_steps    caltech101_m/dinov3_patch/whole_image         ramp_6_20            regret       12 0.467869  0.466690        1.179294e-03     0.500000    0.359375
  all_steps    caltech101_m/dinov3_patch/whole_image         ramp_6_20 average_precision       12 0.996441  0.996441        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image         ramp_6_20             auroc       12 0.997770  0.997770        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image         ramp_6_20        degenerate       12 0.057234  0.057234        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image         ramp_6_20              cost       12 0.076523  0.076054        4.682056e-04     0.333333    0.910156
  all_steps          caltech101_m/siglip/whole_image         ramp_6_20               fnr       12 0.003364  0.002980        3.841056e-04     0.000000    0.500000
  all_steps          caltech101_m/siglip/whole_image         ramp_6_20               fpr       12 0.073159  0.073075        8.409444e-05     0.333333    0.820312
  all_steps          caltech101_m/siglip/whole_image         ramp_6_20            regret       12 0.075551  0.075272        2.782111e-04     0.333333    0.910156
  all_steps          caltech101_m/siglip/whole_image         ramp_6_20 average_precision       12 0.999887  0.999894       -7.011111e-06     0.166667    0.500000
  all_steps          caltech101_m/siglip/whole_image         ramp_6_20             auroc       12 0.999945  0.999948       -3.072222e-06     0.166667    0.500000
  all_steps          caltech101_m/siglip/whole_image         ramp_6_20        degenerate       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          coco_val/dinov3_patch/max_patch         ramp_6_20              cost       14 0.239458  0.235657        3.800919e-03     0.500000    0.669800
  all_steps          coco_val/dinov3_patch/max_patch         ramp_6_20               fnr       14 0.053987  0.078653       -2.466553e-02     0.500000    0.074462
  all_steps          coco_val/dinov3_patch/max_patch         ramp_6_20               fpr       14 0.185471  0.157005        2.846648e-02     0.357143    0.153076
  all_steps          coco_val/dinov3_patch/max_patch         ramp_6_20            regret       14 0.112163  0.087513        2.465005e-02     0.285714    0.013428
  all_steps          coco_val/dinov3_patch/max_patch         ramp_6_20 average_precision       14 0.742812  0.729004        1.380844e-02     0.285714    0.193726
  all_steps          coco_val/dinov3_patch/max_patch         ramp_6_20             auroc       14 0.967783  0.962553        5.229714e-03     0.214286    0.041870
  all_steps          coco_val/dinov3_patch/max_patch         ramp_6_20        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps        coco_val/dinov3_patch/whole_image         ramp_6_20              cost       14 0.502927  0.461013        4.191441e-02     0.285714    0.057983
  all_steps        coco_val/dinov3_patch/whole_image         ramp_6_20               fnr       14 0.115758  0.110065        5.693538e-03     0.500000    0.972125
  all_steps        coco_val/dinov3_patch/whole_image         ramp_6_20               fpr       14 0.387169  0.350948        3.622098e-02     0.428571    0.104004
  all_steps        coco_val/dinov3_patch/whole_image         ramp_6_20            regret       14 0.240267  0.211744        2.852274e-02     0.285714    0.193726
  all_steps        coco_val/dinov3_patch/whole_image         ramp_6_20 average_precision       14 0.565957  0.584652       -1.869446e-02     0.571429    0.345448
  all_steps        coco_val/dinov3_patch/whole_image         ramp_6_20             auroc       14 0.902302  0.907670       -5.367943e-03     0.571429    0.345448
  all_steps        coco_val/dinov3_patch/whole_image         ramp_6_20        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps              coco_val/siglip/whole_image         ramp_6_20              cost       14 0.380012  0.376007        4.005019e-03     0.428571    0.583008
  all_steps              coco_val/siglip/whole_image         ramp_6_20               fnr       14 0.078614  0.121172       -4.255807e-02     0.642857    0.034170
  all_steps              coco_val/siglip/whole_image         ramp_6_20               fpr       14 0.301398  0.254835        4.656311e-02     0.285714    0.010742
  all_steps              coco_val/siglip/whole_image         ramp_6_20            regret       14 0.156407  0.127084        2.932288e-02     0.142857    0.002319
  all_steps              coco_val/siglip/whole_image         ramp_6_20 average_precision       14 0.610017  0.588307        2.171012e-02     0.214286    0.029541
  all_steps              coco_val/siglip/whole_image         ramp_6_20             auroc       14 0.930484  0.913958        1.652611e-02     0.214286    0.020264
  all_steps              coco_val/siglip/whole_image         ramp_6_20        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps   visual_genome_m/dinov3_patch/max_patch         ramp_6_20              cost       16 0.317985  0.301631        1.635369e-02     0.500000    0.820892
  all_steps   visual_genome_m/dinov3_patch/max_patch         ramp_6_20               fnr       16 0.105271  0.111520       -6.249408e-03     0.437500    0.506746
  all_steps   visual_genome_m/dinov3_patch/max_patch         ramp_6_20               fpr       16 0.212714  0.190111        2.260315e-02     0.500000    0.668549
  all_steps   visual_genome_m/dinov3_patch/max_patch         ramp_6_20            regret       16 0.097041  0.074699        2.234152e-02     0.437500    0.375458
  all_steps   visual_genome_m/dinov3_patch/max_patch         ramp_6_20 average_precision       16 0.486326  0.476558        9.768208e-03     0.500000    0.433197
  all_steps   visual_genome_m/dinov3_patch/max_patch         ramp_6_20             auroc       16 0.914901  0.914922       -2.134583e-05     0.562500    0.743561
  all_steps   visual_genome_m/dinov3_patch/max_patch         ramp_6_20        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps visual_genome_m/dinov3_patch/whole_image         ramp_6_20              cost       16 0.654247  0.589748        6.449953e-02     0.062500    0.001312
  all_steps visual_genome_m/dinov3_patch/whole_image         ramp_6_20               fnr       16 0.220659  0.236472       -1.581229e-02     0.750000    0.083252
  all_steps visual_genome_m/dinov3_patch/whole_image         ramp_6_20               fpr       16 0.433588  0.353276        8.031185e-02     0.062500    0.000580
  all_steps visual_genome_m/dinov3_patch/whole_image         ramp_6_20            regret       16 0.175482  0.129488        4.599465e-02     0.125000    0.004181
  all_steps visual_genome_m/dinov3_patch/whole_image         ramp_6_20 average_precision       16 0.299320  0.314061       -1.474157e-02     0.812500    0.033539
  all_steps visual_genome_m/dinov3_patch/whole_image         ramp_6_20             auroc       16 0.779740  0.789802       -1.006253e-02     0.625000    0.129730
  all_steps visual_genome_m/dinov3_patch/whole_image         ramp_6_20        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps       visual_genome_m/siglip/whole_image         ramp_6_20              cost       16 0.502872  0.479799        2.307277e-02     0.375000    0.129730
  all_steps       visual_genome_m/siglip/whole_image         ramp_6_20               fnr       16 0.144723  0.163174       -1.845144e-02     0.625000    0.280531
  all_steps       visual_genome_m/siglip/whole_image         ramp_6_20               fpr       16 0.358149  0.316625        4.152417e-02     0.187500    0.002136
  all_steps       visual_genome_m/siglip/whole_image         ramp_6_20            regret       16 0.132335  0.105138        2.719702e-02     0.375000    0.116669
  all_steps       visual_genome_m/siglip/whole_image         ramp_6_20 average_precision       16 0.347248  0.344372        2.875421e-03     0.562500    0.899933
  all_steps       visual_genome_m/siglip/whole_image         ramp_6_20             auroc       16 0.850310  0.845169        5.141538e-03     0.437500    0.596588
  all_steps       visual_genome_m/siglip/whole_image         ramp_6_20        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus              cost       12 0.078271  0.067245        1.102514e-02     0.333333    0.640625
  all_steps    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus               fnr       12 0.006010  0.006241       -2.314000e-04     0.083333    1.000000
  all_steps    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus               fpr       12 0.072261  0.061004        1.125654e-02     0.333333    0.546875
  all_steps    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus            regret       12 0.077019  0.065909        1.111061e-02     0.333333    0.546875
  all_steps    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus average_precision       12 0.999743  0.999737        6.053125e-06     0.000000    0.500000
  all_steps    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus             auroc       12 0.999857  0.999855        2.566667e-06     0.000000    0.500000
  all_steps    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus        degenerate       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image post_ramp_21_plus              cost       12 0.000920  0.014302       -1.338259e-02     0.250000    0.437500
  all_steps          caltech101_m/siglip/whole_image post_ramp_21_plus               fnr       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image post_ramp_21_plus               fpr       12 0.000920  0.014302       -1.338259e-02     0.250000    0.437500
  all_steps          caltech101_m/siglip/whole_image post_ramp_21_plus            regret       12 0.000920  0.014302       -1.338259e-02     0.250000    0.437500
  all_steps          caltech101_m/siglip/whole_image post_ramp_21_plus average_precision       12 1.000000  1.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image post_ramp_21_plus             auroc       12 1.000000  1.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image post_ramp_21_plus        degenerate       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          coco_val/dinov3_patch/max_patch post_ramp_21_plus              cost       14 0.193847  0.182593        1.125370e-02     0.428571    0.390991
  all_steps          coco_val/dinov3_patch/max_patch post_ramp_21_plus               fnr       14 0.055065  0.065318       -1.025304e-02     0.500000    0.202622
  all_steps          coco_val/dinov3_patch/max_patch post_ramp_21_plus               fpr       14 0.138782  0.117275        2.150675e-02     0.428571    0.325806
  all_steps          coco_val/dinov3_patch/max_patch post_ramp_21_plus            regret       14 0.084642  0.071986        1.265538e-02     0.357143    0.135254
  all_steps          coco_val/dinov3_patch/max_patch post_ramp_21_plus average_precision       14 0.783907  0.791461       -7.554346e-03     0.500000    0.807739
  all_steps          coco_val/dinov3_patch/max_patch post_ramp_21_plus             auroc       14 0.975886  0.976726       -8.404643e-04     0.500000    0.463135
  all_steps          coco_val/dinov3_patch/max_patch post_ramp_21_plus        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps        coco_val/dinov3_patch/whole_image post_ramp_21_plus              cost       14 0.300961  0.257909        4.305204e-02     0.214286    0.078491
  all_steps        coco_val/dinov3_patch/whole_image post_ramp_21_plus               fnr       14 0.113832  0.127504       -1.367216e-02     0.500000    0.583008
  all_steps        coco_val/dinov3_patch/whole_image post_ramp_21_plus               fpr       14 0.187129  0.130405        5.672424e-02     0.214286    0.035278
  all_steps        coco_val/dinov3_patch/whole_image post_ramp_21_plus            regret       14 0.097619  0.066101        3.151768e-02     0.214286    0.057983
  all_steps        coco_val/dinov3_patch/whole_image post_ramp_21_plus average_precision       14 0.656364  0.694278       -3.791474e-02     0.642857    0.135254
  all_steps        coco_val/dinov3_patch/whole_image post_ramp_21_plus             auroc       14 0.936380  0.935866        5.135723e-04     0.428571    1.000000
  all_steps        coco_val/dinov3_patch/whole_image post_ramp_21_plus        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps              coco_val/siglip/whole_image post_ramp_21_plus              cost       14 0.287236  0.245748        4.148819e-02     0.142857    0.003052
  all_steps              coco_val/siglip/whole_image post_ramp_21_plus               fnr       14 0.095017  0.120311       -2.529382e-02     0.785714    0.007649
  all_steps              coco_val/siglip/whole_image post_ramp_21_plus               fpr       14 0.192219  0.125437        6.678202e-02     0.071429    0.000244
  all_steps              coco_val/siglip/whole_image post_ramp_21_plus            regret       14 0.088859  0.055458        3.340093e-02     0.142857    0.000610
  all_steps              coco_val/siglip/whole_image post_ramp_21_plus average_precision       14 0.658562  0.678684       -2.012207e-02     0.714286    0.067627
  all_steps              coco_val/siglip/whole_image post_ramp_21_plus             auroc       14 0.938100  0.939264       -1.163781e-03     0.714286    0.501587
  all_steps              coco_val/siglip/whole_image post_ramp_21_plus        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus              cost       16 0.296198  0.247767        4.843110e-02     0.187500    0.002686
  all_steps   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus               fnr       16 0.087249  0.089077       -1.827880e-03     0.375000    0.924978
  all_steps   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus               fpr       16 0.208949  0.158690        5.025903e-02     0.187500    0.004181
  all_steps   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus            regret       16 0.089509  0.057993        3.151640e-02     0.375000    0.073914
  all_steps   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus average_precision       16 0.506658  0.519367       -1.270895e-02     0.625000    0.252228
  all_steps   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus             auroc       16 0.921326  0.922402       -1.075312e-03     0.500000    0.860260
  all_steps   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus              cost       16 0.481229  0.446176        3.505313e-02     0.500000    0.433197
  all_steps visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus               fnr       16 0.227350  0.292386       -6.503660e-02     0.937500    0.004181
  all_steps visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus               fpr       16 0.253880  0.153790        1.000897e-01     0.000000    0.000031
  all_steps visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus            regret       16 0.096268  0.057309        3.895910e-02     0.312500    0.038635
  all_steps visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus average_precision       16 0.361884  0.412806       -5.092201e-02     0.562500    0.159058
  all_steps visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus             auroc       16 0.836908  0.833232        3.676454e-03     0.375000    0.129730
  all_steps visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps       visual_genome_m/siglip/whole_image post_ramp_21_plus              cost       16 0.397748  0.371828        2.591998e-02     0.250000    0.028992
  all_steps       visual_genome_m/siglip/whole_image post_ramp_21_plus               fnr       16 0.143060  0.168798       -2.573729e-02     0.750000    0.044312
  all_steps       visual_genome_m/siglip/whole_image post_ramp_21_plus               fpr       16 0.254688  0.203031        5.165728e-02     0.125000    0.000153
  all_steps       visual_genome_m/siglip/whole_image post_ramp_21_plus            regret       16 0.092708  0.069882        2.282544e-02     0.375000    0.231201
  all_steps       visual_genome_m/siglip/whole_image post_ramp_21_plus average_precision       16 0.415686  0.429099       -1.341326e-02     0.625000    0.528168
  all_steps       visual_genome_m/siglip/whole_image post_ramp_21_plus             auroc       16 0.886258  0.884096        2.161555e-03     0.500000    1.000000
  all_steps       visual_genome_m/siglip/whole_image post_ramp_21_plus        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image         all_steps              cost       12 0.146223  0.136993        9.230455e-03     0.500000    0.898438
  all_steps    caltech101_m/dinov3_patch/whole_image         all_steps               fnr       12 0.015192  0.015395       -2.028838e-04     0.083333    1.000000
  all_steps    caltech101_m/dinov3_patch/whole_image         all_steps               fpr       12 0.131031  0.121598        9.433339e-03     0.500000    0.898438
  all_steps    caltech101_m/dinov3_patch/whole_image         all_steps            regret       12 0.143172  0.133868        9.303914e-03     0.500000    0.898438
  all_steps    caltech101_m/dinov3_patch/whole_image         all_steps average_precision       12 0.999205  0.999199        5.205493e-06     0.000000    0.500000
  all_steps    caltech101_m/dinov3_patch/whole_image         all_steps             auroc       12 0.999551  0.999549        2.207332e-06     0.000000    0.500000
  all_steps    caltech101_m/dinov3_patch/whole_image         all_steps        degenerate       12 0.008055  0.008055        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image         all_steps              cost       12 0.021884  0.033055       -1.117031e-02     0.583333    0.130859
  all_steps          caltech101_m/siglip/whole_image         all_steps               fnr       12 0.000859  0.000800        5.939777e-05     0.000000    0.500000
  all_steps          caltech101_m/siglip/whole_image         all_steps               fpr       12 0.021025  0.032255       -1.122971e-02     0.666667    0.064453
  all_steps          caltech101_m/siglip/whole_image         all_steps            regret       12 0.021451  0.032651       -1.119969e-02     0.583333    0.083984
  all_steps          caltech101_m/siglip/whole_image         all_steps average_precision       12 0.999740  0.999741       -1.084192e-06     0.166667    0.500000
  all_steps          caltech101_m/siglip/whole_image         all_steps             auroc       12 0.999962  0.999962       -4.750859e-07     0.166667    0.500000
  all_steps          caltech101_m/siglip/whole_image         all_steps        degenerate       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          coco_val/dinov3_patch/max_patch         all_steps              cost       14 0.204143  0.194226        9.916789e-03     0.500000    0.463135
  all_steps          coco_val/dinov3_patch/max_patch         all_steps               fnr       14 0.055599  0.067873       -1.227414e-02     0.571429    0.114128
  all_steps          coco_val/dinov3_patch/max_patch         all_steps               fpr       14 0.148544  0.126353        2.219094e-02     0.428571    0.325806
  all_steps          coco_val/dinov3_patch/max_patch         all_steps            regret       14 0.090692  0.076395        1.429691e-02     0.357143    0.041870
  all_steps          coco_val/dinov3_patch/max_patch         all_steps average_precision       14 0.775301  0.779397       -4.095063e-03     0.500000    0.951538
  all_steps          coco_val/dinov3_patch/max_patch         all_steps             auroc       14 0.973828  0.973713        1.155523e-04     0.428571    0.669800
  all_steps          coco_val/dinov3_patch/max_patch         all_steps        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps        coco_val/dinov3_patch/whole_image         all_steps              cost       14 0.339124  0.297057        4.206709e-02     0.214286    0.057983
  all_steps        coco_val/dinov3_patch/whole_image         all_steps               fnr       14 0.114179  0.124575       -1.039624e-02     0.571429    0.541626
  all_steps        coco_val/dinov3_patch/whole_image         all_steps               fpr       14 0.224945  0.172482        5.246337e-02     0.214286    0.029541
  all_steps        coco_val/dinov3_patch/whole_image         all_steps            regret       14 0.124094  0.093611        3.048333e-02     0.214286    0.035278
  all_steps        coco_val/dinov3_patch/whole_image         all_steps average_precision       14 0.638723  0.672883       -3.416078e-02     0.571429    0.172607
  all_steps        coco_val/dinov3_patch/whole_image         all_steps             auroc       14 0.929532  0.929939       -4.065295e-04     0.428571    0.903198
  all_steps        coco_val/dinov3_patch/whole_image         all_steps        degenerate       14 0.000736  0.000736        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps              coco_val/siglip/whole_image         all_steps              cost       14 0.306497  0.271720        3.477602e-02     0.142857    0.006714
  all_steps              coco_val/siglip/whole_image         all_steps               fnr       14 0.092070  0.119551       -2.748104e-02     0.785714    0.007649
  all_steps              coco_val/siglip/whole_image         all_steps               fpr       14 0.214427  0.152170        6.225707e-02     0.000000    0.000122
  all_steps              coco_val/siglip/whole_image         all_steps            regret       14 0.102407  0.070417        3.198954e-02     0.000000    0.000122
  all_steps              coco_val/siglip/whole_image         all_steps average_precision       14 0.647738  0.660957       -1.321906e-02     0.714286    0.118896
  all_steps              coco_val/siglip/whole_image         all_steps             auroc       14 0.936122  0.934542        1.579745e-03     0.642857    1.000000
  all_steps              coco_val/siglip/whole_image         all_steps        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps   visual_genome_m/dinov3_patch/max_patch         all_steps              cost       16 0.300951  0.258446        4.250506e-02     0.187500    0.003357
  all_steps   visual_genome_m/dinov3_patch/max_patch         all_steps               fnr       16 0.090478  0.093124       -2.645938e-03     0.437500    0.777565
  all_steps   visual_genome_m/dinov3_patch/max_patch         all_steps               fpr       16 0.210473  0.165322        4.515105e-02     0.125000    0.007629
  all_steps   visual_genome_m/dinov3_patch/max_patch         all_steps            regret       16 0.091324  0.061843        2.948074e-02     0.312500    0.050659
  all_steps   visual_genome_m/dinov3_patch/max_patch         all_steps average_precision       16 0.502628  0.511599       -8.971059e-03     0.562500    0.403748
  all_steps   visual_genome_m/dinov3_patch/max_patch         all_steps             auroc       16 0.920053  0.920943       -8.901566e-04     0.500000    0.705719
  all_steps   visual_genome_m/dinov3_patch/max_patch         all_steps        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps visual_genome_m/dinov3_patch/whole_image         all_steps              cost       16 0.513271  0.474424        3.884699e-02     0.500000    0.297852
  all_steps visual_genome_m/dinov3_patch/whole_image         all_steps               fnr       16 0.224477  0.280493       -5.601533e-02     0.875000    0.007629
  all_steps visual_genome_m/dinov3_patch/whole_image         all_steps               fpr       16 0.288793  0.193931        9.486231e-02     0.000000    0.000031
  all_steps visual_genome_m/dinov3_patch/whole_image         all_steps            regret       16 0.111122  0.071915        3.920683e-02     0.250000    0.006287
  all_steps visual_genome_m/dinov3_patch/whole_image         all_steps average_precision       16 0.349623  0.393901       -4.427716e-02     0.562500    0.116669
  all_steps visual_genome_m/dinov3_patch/whole_image         all_steps             auroc       16 0.826536  0.825060        1.476066e-03     0.375000    0.231201
  all_steps visual_genome_m/dinov3_patch/whole_image         all_steps        degenerate       16 0.000644  0.000644        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps       visual_genome_m/siglip/whole_image         all_steps              cost       16 0.420626  0.395604        2.502139e-02     0.250000    0.021393
  all_steps       visual_genome_m/siglip/whole_image         all_steps               fnr       16 0.144103  0.168032       -2.392961e-02     0.750000    0.028992
  all_steps       visual_genome_m/siglip/whole_image         all_steps               fpr       16 0.276523  0.227572        4.895101e-02     0.125000    0.000153
  all_steps       visual_genome_m/siglip/whole_image         all_steps            regret       16 0.100890  0.077912        2.297744e-02     0.250000    0.038635
  all_steps       visual_genome_m/siglip/whole_image         all_steps average_precision       16 0.401238  0.411940       -1.070205e-02     0.625000    0.495422
  all_steps       visual_genome_m/siglip/whole_image         all_steps             auroc       16 0.878026  0.875533        2.492899e-03     0.562500    0.939880
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
      "delta_cost_on_minus_off": 0.00899575529455664,
      "p": 0.804840087890625,
      "reading": "safe ON worse (n.s.)"
    },
    "post_ramp_21_plus": {
      "n_cells": 16,
      "delta_cost_on_minus_off": 0.029842397377232145,
      "p": 0.19683837890625,
      "reading": "safe ON worse (n.s.)"
    },
    "all_steps": {
      "n_cells": 16,
      "delta_cost_on_minus_off": 0.026770749113113102,
      "p": 0.252471923828125,
      "reading": "safe ON worse (n.s.)"
    },
    "force_on_for_all_users": false
  },
  "all_steps": {
    "scope": "all_steps",
    "pure_gmm_2_5": {
      "n_cells": 16,
      "delta_cost_on_minus_off": 0.0019542075892857165,
      "p": 0.9167147933599851,
      "reading": "safe ON worse (n.s.)"
    },
    "ramp_6_20": {
      "n_cells": 16,
      "delta_cost_on_minus_off": 0.010077303273809523,
      "p": 0.7453460693359375,
      "reading": "safe ON worse (n.s.)"
    },
    "post_ramp_21_plus": {
      "n_cells": 16,
      "delta_cost_on_minus_off": 0.029842397377232145,
      "p": 0.19683837890625,
      "reading": "safe ON worse (n.s.)"
    },
    "all_steps": {
      "n_cells": 16,
      "delta_cost_on_minus_off": 0.026210925303755517,
      "p": 0.233245849609375,
      "reading": "safe ON worse (n.s.)"
    },
    "force_on_for_all_users": false
  }
}
```

Curves: `agg/ab_curves_vs_votes.csv` (1556 rows) · paired cells: `agg/ab_paired_cells.csv` · figures: ab_cost_vs_votes.png, ab_fnr_vs_votes.png, ab_degenerate_vs_votes.png
