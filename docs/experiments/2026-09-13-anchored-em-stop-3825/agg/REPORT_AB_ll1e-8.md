# Safe thresholds ON vs OFF — trajectory A/B (#2799)

Head: `linear_svm` · cells ON/OFF: 114/114 · ON `/expscratch/sgreenberg/anchem-3825/ab_ll1e-8/results` vs OFF `/expscratch/sgreenberg/anchem-3825/ab_baseline/results`

Both runs are full simulations; because the blended threshold drives Autopilot's
Hard pick, the two arms vote on different items, so cells (category × seed), not
steps, are the paired units.

The app shows a trained detector from **7 votes** onward; below that it
sorts by text/example cosine, so `scope=app_visible` is what users actually get and
`scope=all_steps` is the purely numerical reading.

## Per-window paired comparison (Δ = ON − OFF; negative = safe thresholds better)

```
      scope                                      arm            window            metric  n_cells  safe_on  safe_off  delta_on_minus_off  win_rate_on  p_wilcoxon              note
app_visible    caltech101_m/dinov3_patch/whole_image         ramp_6_20              cost       11 0.384200  0.384210           -0.000009     0.090909    1.000000
app_visible    caltech101_m/dinov3_patch/whole_image         ramp_6_20               fnr       11 0.003162  0.003162            0.000000     0.000000         NaN all-zero or empty
app_visible    caltech101_m/dinov3_patch/whole_image         ramp_6_20               fpr       11 0.381038  0.381048           -0.000009     0.090909    1.000000
app_visible    caltech101_m/dinov3_patch/whole_image         ramp_6_20            regret       11 0.377995  0.378005           -0.000009     0.090909    1.000000
app_visible    caltech101_m/dinov3_patch/whole_image         ramp_6_20 average_precision       11 0.998476  0.998476            0.000000     0.000000         NaN all-zero or empty
app_visible    caltech101_m/dinov3_patch/whole_image         ramp_6_20             auroc       11 0.999173  0.999173            0.000000     0.000000         NaN all-zero or empty
app_visible    caltech101_m/dinov3_patch/whole_image         ramp_6_20        degenerate       11 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image         ramp_6_20              cost       12 0.059538  0.061930           -0.002392     0.166667    0.625000
app_visible          caltech101_m/siglip/whole_image         ramp_6_20               fnr       12 0.001976  0.001976            0.000000     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image         ramp_6_20               fpr       12 0.057562  0.059954           -0.002392     0.166667    0.625000
app_visible          caltech101_m/siglip/whole_image         ramp_6_20            regret       12 0.058919  0.061311           -0.002392     0.166667    0.625000
app_visible          caltech101_m/siglip/whole_image         ramp_6_20 average_precision       12 0.999937  0.999937            0.000000     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image         ramp_6_20             auroc       12 0.999974  0.999974            0.000000     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image         ramp_6_20        degenerate       12 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
app_visible          coco_val/dinov3_patch/max_patch         ramp_6_20              cost       14 0.230154  0.230334           -0.000180     0.642857    0.357544
app_visible          coco_val/dinov3_patch/max_patch         ramp_6_20               fnr       14 0.066639  0.080090           -0.013452     0.571429    0.036658
app_visible          coco_val/dinov3_patch/max_patch         ramp_6_20               fpr       14 0.163516  0.150244            0.013271     0.428571    0.501587
app_visible          coco_val/dinov3_patch/max_patch         ramp_6_20            regret       14 0.087675  0.084241            0.003434     0.357143    0.541626
app_visible          coco_val/dinov3_patch/max_patch         ramp_6_20 average_precision       14 0.732206  0.733187           -0.000981     0.500000    1.000000
app_visible          coco_val/dinov3_patch/max_patch         ramp_6_20             auroc       14 0.963446  0.964392           -0.000947     0.214286    0.426270
app_visible          coco_val/dinov3_patch/max_patch         ramp_6_20        degenerate       14 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
app_visible        coco_val/dinov3_patch/whole_image         ramp_6_20              cost       14 0.440473  0.450312           -0.009839     0.571429    0.625732
app_visible        coco_val/dinov3_patch/whole_image         ramp_6_20               fnr       14 0.122957  0.107476            0.015481     0.285714    0.099481
app_visible        coco_val/dinov3_patch/whole_image         ramp_6_20               fpr       14 0.317516  0.342836           -0.025320     0.714286    0.241211
app_visible        coco_val/dinov3_patch/whole_image         ramp_6_20            regret       14 0.187833  0.207048           -0.019215     0.571429    0.426270
app_visible        coco_val/dinov3_patch/whole_image         ramp_6_20 average_precision       14 0.577559  0.591454           -0.013895     0.714286    0.278707
app_visible        coco_val/dinov3_patch/whole_image         ramp_6_20             auroc       14 0.911768  0.912524           -0.000757     0.500000    0.421579
app_visible        coco_val/dinov3_patch/whole_image         ramp_6_20        degenerate       14 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
app_visible              coco_val/siglip/whole_image         ramp_6_20              cost       14 0.372489  0.370226            0.002263     0.428571    0.951538
app_visible              coco_val/siglip/whole_image         ramp_6_20               fnr       14 0.137929  0.122506            0.015423     0.142857    0.028056
app_visible              coco_val/siglip/whole_image         ramp_6_20               fpr       14 0.234560  0.247720           -0.013161     0.642857    0.193726
app_visible              coco_val/siglip/whole_image         ramp_6_20            regret       14 0.108954  0.122281           -0.013328     0.642857    0.193726
app_visible              coco_val/siglip/whole_image         ramp_6_20 average_precision       14 0.588518  0.591723           -0.003205     0.428571    0.807739
app_visible              coco_val/siglip/whole_image         ramp_6_20             auroc       14 0.905571  0.914152           -0.008582     0.428571    0.855225
app_visible              coco_val/siglip/whole_image         ramp_6_20        degenerate       14 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
app_visible   visual_genome_m/dinov3_patch/max_patch         ramp_6_20              cost       16 0.294616  0.301984           -0.007368     0.562500    0.532130
app_visible   visual_genome_m/dinov3_patch/max_patch         ramp_6_20               fnr       16 0.114838  0.110370            0.004468     0.375000    0.813945
app_visible   visual_genome_m/dinov3_patch/max_patch         ramp_6_20               fpr       16 0.179778  0.191614           -0.011836     0.562500    0.394246
app_visible   visual_genome_m/dinov3_patch/max_patch         ramp_6_20            regret       16 0.067554  0.076157           -0.008602     0.625000    0.306624
app_visible   visual_genome_m/dinov3_patch/max_patch         ramp_6_20 average_precision       16 0.491933  0.476947            0.014986     0.312500    0.191446
app_visible   visual_genome_m/dinov3_patch/max_patch         ramp_6_20             auroc       16 0.912514  0.915168           -0.002654     0.375000    0.649563
app_visible   visual_genome_m/dinov3_patch/max_patch         ramp_6_20        degenerate       16 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
app_visible visual_genome_m/dinov3_patch/whole_image         ramp_6_20              cost       16 0.598975  0.579751            0.019225     0.312500    0.129730
app_visible visual_genome_m/dinov3_patch/whole_image         ramp_6_20               fnr       16 0.257557  0.243634            0.013924     0.250000    0.157811
app_visible visual_genome_m/dinov3_patch/whole_image         ramp_6_20               fpr       16 0.341418  0.336117            0.005301     0.562500    0.939880
app_visible visual_genome_m/dinov3_patch/whole_image         ramp_6_20            regret       16 0.132103  0.121744            0.010359     0.375000    0.211426
app_visible visual_genome_m/dinov3_patch/whole_image         ramp_6_20 average_precision       16 0.314248  0.318277           -0.004029     0.500000    0.532130
app_visible visual_genome_m/dinov3_patch/whole_image         ramp_6_20             auroc       16 0.787357  0.791562           -0.004206     0.500000    0.570061
app_visible visual_genome_m/dinov3_patch/whole_image         ramp_6_20        degenerate       16 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
app_visible       visual_genome_m/siglip/whole_image         ramp_6_20              cost       16 0.457879  0.464855           -0.006975     0.625000    0.104584
app_visible       visual_genome_m/siglip/whole_image         ramp_6_20               fnr       16 0.166254  0.159322            0.006932     0.312500    0.310897
app_visible       visual_genome_m/siglip/whole_image         ramp_6_20               fpr       16 0.291626  0.305533           -0.013907     0.812500    0.002136
app_visible       visual_genome_m/siglip/whole_image         ramp_6_20            regret       16 0.095261  0.098410           -0.003149     0.562500    0.231201
app_visible       visual_genome_m/siglip/whole_image         ramp_6_20 average_precision       16 0.336662  0.350310           -0.013648     0.812500    0.143860
app_visible       visual_genome_m/siglip/whole_image         ramp_6_20             auroc       16 0.853800  0.852044            0.001755     0.625000    0.899933
app_visible       visual_genome_m/siglip/whole_image         ramp_6_20        degenerate       16 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
app_visible    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus              cost       12 0.060939  0.060939            0.000000     0.000000         NaN all-zero or empty
app_visible    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus               fnr       12 0.006392  0.006392            0.000000     0.000000         NaN all-zero or empty
app_visible    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus               fpr       12 0.054547  0.054547            0.000000     0.000000         NaN all-zero or empty
app_visible    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus            regret       12 0.060126  0.060126            0.000000     0.000000         NaN all-zero or empty
app_visible    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus average_precision       12 0.999884  0.999884            0.000000     0.000000         NaN all-zero or empty
app_visible    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus             auroc       12 0.999938  0.999938            0.000000     0.000000         NaN all-zero or empty
app_visible    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus        degenerate       12 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image post_ramp_21_plus              cost       12 0.010050  0.014302           -0.004252     0.250000    0.625000
app_visible          caltech101_m/siglip/whole_image post_ramp_21_plus               fnr       12 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image post_ramp_21_plus               fpr       12 0.010050  0.014302           -0.004252     0.250000    0.625000
app_visible          caltech101_m/siglip/whole_image post_ramp_21_plus            regret       12 0.010050  0.014302           -0.004252     0.250000    0.625000
app_visible          caltech101_m/siglip/whole_image post_ramp_21_plus average_precision       12 1.000000  1.000000            0.000000     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image post_ramp_21_plus             auroc       12 1.000000  1.000000            0.000000     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image post_ramp_21_plus        degenerate       12 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
app_visible          coco_val/dinov3_patch/max_patch post_ramp_21_plus              cost       14 0.181207  0.182593           -0.001386     0.642857    0.583008
app_visible          coco_val/dinov3_patch/max_patch post_ramp_21_plus               fnr       14 0.064733  0.065318           -0.000586     0.428571    0.959354
app_visible          coco_val/dinov3_patch/max_patch post_ramp_21_plus               fpr       14 0.116475  0.117275           -0.000800     0.571429    0.426270
app_visible          coco_val/dinov3_patch/max_patch post_ramp_21_plus            regret       14 0.067089  0.071986           -0.004897     0.571429    0.390991
app_visible          coco_val/dinov3_patch/max_patch post_ramp_21_plus average_precision       14 0.785124  0.791461           -0.006337     0.642857    0.714844
app_visible          coco_val/dinov3_patch/max_patch post_ramp_21_plus             auroc       14 0.974364  0.976726           -0.002362     0.500000    0.669800
app_visible          coco_val/dinov3_patch/max_patch post_ramp_21_plus        degenerate       14 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
app_visible        coco_val/dinov3_patch/whole_image post_ramp_21_plus              cost       14 0.258385  0.257909            0.000475     0.500000    0.951538
app_visible        coco_val/dinov3_patch/whole_image post_ramp_21_plus               fnr       14 0.134864  0.127504            0.007360     0.357143    0.357544
app_visible        coco_val/dinov3_patch/whole_image post_ramp_21_plus               fpr       14 0.123521  0.130405           -0.006884     0.428571    0.855225
app_visible        coco_val/dinov3_patch/whole_image post_ramp_21_plus            regret       14 0.059418  0.066101           -0.006683     0.571429    0.541626
app_visible        coco_val/dinov3_patch/whole_image post_ramp_21_plus average_precision       14 0.696785  0.694278            0.002506     0.357143    0.463135
app_visible        coco_val/dinov3_patch/whole_image post_ramp_21_plus             auroc       14 0.936582  0.935866            0.000716     0.500000    1.000000
app_visible        coco_val/dinov3_patch/whole_image post_ramp_21_plus        degenerate       14 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
app_visible              coco_val/siglip/whole_image post_ramp_21_plus              cost       14 0.286769  0.245748            0.041021     0.214286    0.006714
app_visible              coco_val/siglip/whole_image post_ramp_21_plus               fnr       14 0.150536  0.120311            0.030225     0.071429    0.004742
app_visible              coco_val/siglip/whole_image post_ramp_21_plus               fpr       14 0.136233  0.125437            0.010796     0.500000    0.625732
app_visible              coco_val/siglip/whole_image post_ramp_21_plus            regret       14 0.054162  0.055458           -0.001296     0.500000    0.807739
app_visible              coco_val/siglip/whole_image post_ramp_21_plus average_precision       14 0.645622  0.678684           -0.033062     0.714286    0.049438
app_visible              coco_val/siglip/whole_image post_ramp_21_plus             auroc       14 0.918105  0.939264           -0.021159     0.857143    0.002319
app_visible              coco_val/siglip/whole_image post_ramp_21_plus        degenerate       14 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
app_visible   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus              cost       16 0.250082  0.247767            0.002315     0.375000    0.596588
app_visible   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus               fnr       16 0.098275  0.089077            0.009198     0.125000    0.015906
app_visible   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus               fpr       16 0.151807  0.158690           -0.006883     0.625000    0.274445
app_visible   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus            regret       16 0.058555  0.057993            0.000562     0.500000    0.979950
app_visible   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus average_precision       16 0.544652  0.519367            0.025285     0.375000    0.159058
app_visible   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus             auroc       16 0.925924  0.922402            0.003522     0.312500    0.116669
app_visible   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus        degenerate       16 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
app_visible visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus              cost       16 0.454520  0.446176            0.008344     0.562500    0.899933
app_visible visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus               fnr       16 0.296425  0.292386            0.004039     0.500000    0.979950
app_visible visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus               fpr       16 0.158095  0.153790            0.004305     0.500000    1.000000
app_visible visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus            regret       16 0.065073  0.057309            0.007764     0.500000    0.463745
app_visible visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus average_precision       16 0.390408  0.412806           -0.022398     0.437500    0.495422
app_visible visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus             auroc       16 0.831539  0.833232           -0.001693     0.437500    0.596588
app_visible visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus        degenerate       16 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
app_visible       visual_genome_m/siglip/whole_image post_ramp_21_plus              cost       16 0.371546  0.371828           -0.000282     0.500000    0.939880
app_visible       visual_genome_m/siglip/whole_image post_ramp_21_plus               fnr       16 0.184325  0.168798            0.015528     0.375000    0.363488
app_visible       visual_genome_m/siglip/whole_image post_ramp_21_plus               fpr       16 0.187220  0.203031           -0.015810     0.875000    0.001007
app_visible       visual_genome_m/siglip/whole_image post_ramp_21_plus            regret       16 0.062509  0.069882           -0.007373     0.687500    0.211426
app_visible       visual_genome_m/siglip/whole_image post_ramp_21_plus average_precision       16 0.421770  0.429099           -0.007330     0.500000    0.899933
app_visible       visual_genome_m/siglip/whole_image post_ramp_21_plus             auroc       16 0.881487  0.884096           -0.002609     0.562500    0.632172
app_visible       visual_genome_m/siglip/whole_image post_ramp_21_plus        degenerate       16 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
app_visible    caltech101_m/dinov3_patch/whole_image         all_steps              cost       12 0.092890  0.092892           -0.000002     0.083333    1.000000
app_visible    caltech101_m/dinov3_patch/whole_image         all_steps               fnr       12 0.006265  0.006265            0.000000     0.000000         NaN all-zero or empty
app_visible    caltech101_m/dinov3_patch/whole_image         all_steps               fpr       12 0.086625  0.086627           -0.000002     0.083333    1.000000
app_visible    caltech101_m/dinov3_patch/whole_image         all_steps            regret       12 0.091874  0.091876           -0.000002     0.083333    1.000000
app_visible    caltech101_m/dinov3_patch/whole_image         all_steps average_precision       12 0.999827  0.999827            0.000000     0.000000         NaN all-zero or empty
app_visible    caltech101_m/dinov3_patch/whole_image         all_steps             auroc       12 0.999911  0.999911            0.000000     0.000000         NaN all-zero or empty
app_visible    caltech101_m/dinov3_patch/whole_image         all_steps        degenerate       12 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image         all_steps              cost       12 0.016932  0.020882           -0.003950     0.166667    0.875000
app_visible          caltech101_m/siglip/whole_image         all_steps               fnr       12 0.000294  0.000294            0.000000     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image         all_steps               fpr       12 0.016637  0.020587           -0.003950     0.166667    0.875000
app_visible          caltech101_m/siglip/whole_image         all_steps            regret       12 0.016839  0.020789           -0.003950     0.166667    0.875000
app_visible          caltech101_m/siglip/whole_image         all_steps average_precision       12 0.999991  0.999991            0.000000     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image         all_steps             auroc       12 0.999996  0.999996            0.000000     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image         all_steps        degenerate       12 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
app_visible          coco_val/dinov3_patch/max_patch         all_steps              cost       14 0.188309  0.189473           -0.001164     0.642857    0.669800
app_visible          coco_val/dinov3_patch/max_patch         all_steps               fnr       14 0.064858  0.067294           -0.002437     0.428571    0.721277
app_visible          coco_val/dinov3_patch/max_patch         all_steps               fpr       14 0.123452  0.122179            0.001273     0.571429    0.583008
app_visible          coco_val/dinov3_patch/max_patch         all_steps            regret       14 0.070140  0.073833           -0.003693     0.642857    0.390991
app_visible          coco_val/dinov3_patch/max_patch         all_steps average_precision       14 0.777507  0.783091           -0.005584     0.642857    0.714844
app_visible          coco_val/dinov3_patch/max_patch         all_steps             auroc       14 0.972812  0.974975           -0.002163     0.428571    0.760864
app_visible          coco_val/dinov3_patch/max_patch         all_steps        degenerate       14 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
app_visible        coco_val/dinov3_patch/whole_image         all_steps              cost       14 0.285114  0.286202           -0.001088     0.571429    1.000000
app_visible        coco_val/dinov3_patch/whole_image         all_steps               fnr       14 0.132759  0.124250            0.008509     0.357143    0.267578
app_visible        coco_val/dinov3_patch/whole_image         all_steps               fpr       14 0.152355  0.161952           -0.009597     0.500000    0.541626
app_visible        coco_val/dinov3_patch/whole_image         all_steps            regret       14 0.078321  0.086937           -0.008616     0.500000    0.501587
app_visible        coco_val/dinov3_patch/whole_image         all_steps average_precision       14 0.679398  0.679297            0.000102     0.357143    0.541626
app_visible        coco_val/dinov3_patch/whole_image         all_steps             auroc       14 0.932987  0.932501            0.000487     0.500000    1.000000
app_visible        coco_val/dinov3_patch/whole_image         all_steps        degenerate       14 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
app_visible              coco_val/siglip/whole_image         all_steps              cost       14 0.299536  0.264287            0.035249     0.214286    0.020264
app_visible              coco_val/siglip/whole_image         all_steps               fnr       14 0.148658  0.120638            0.028020     0.071429    0.003702
app_visible              coco_val/siglip/whole_image         all_steps               fpr       14 0.150878  0.143649            0.007228     0.500000    0.807739
app_visible              coco_val/siglip/whole_image         all_steps            regret       14 0.062322  0.065410           -0.003088     0.571429    0.625732
app_visible              coco_val/siglip/whole_image         all_steps average_precision       14 0.637117  0.665732           -0.028615     0.642857    0.090576
app_visible              coco_val/siglip/whole_image         all_steps             auroc       14 0.916238  0.935524           -0.019286     0.714286    0.006714
app_visible              coco_val/siglip/whole_image         all_steps        degenerate       14 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
app_visible   visual_genome_m/dinov3_patch/max_patch         all_steps              cost       16 0.256490  0.255819            0.000671     0.500000    0.939880
app_visible   visual_genome_m/dinov3_patch/max_patch         all_steps               fnr       16 0.100658  0.092287            0.008371     0.187500    0.047990
app_visible   visual_genome_m/dinov3_patch/max_patch         all_steps               fpr       16 0.155832  0.163532           -0.007700     0.625000    0.274445
app_visible   visual_genome_m/dinov3_patch/max_patch         all_steps            regret       16 0.059848  0.060716           -0.000868     0.562500    0.743561
app_visible   visual_genome_m/dinov3_patch/max_patch         all_steps average_precision       16 0.536938  0.513107            0.023831     0.437500    0.175354
app_visible   visual_genome_m/dinov3_patch/max_patch         all_steps             auroc       16 0.924020  0.921339            0.002681     0.312500    0.175354
app_visible   visual_genome_m/dinov3_patch/max_patch         all_steps        degenerate       16 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
app_visible visual_genome_m/dinov3_patch/whole_image         all_steps              cost       16 0.475757  0.465868            0.009890     0.437500    0.705719
app_visible visual_genome_m/dinov3_patch/whole_image         all_steps               fnr       16 0.290838  0.285308            0.005530     0.500000    0.899933
app_visible visual_genome_m/dinov3_patch/whole_image         all_steps               fpr       16 0.184919  0.180559            0.004360     0.500000    1.000000
app_visible visual_genome_m/dinov3_patch/whole_image         all_steps            regret       16 0.074985  0.066841            0.008144     0.437500    0.375458
app_visible visual_genome_m/dinov3_patch/whole_image         all_steps average_precision       16 0.379231  0.398868           -0.019638     0.437500    0.495422
app_visible visual_genome_m/dinov3_patch/whole_image         all_steps             auroc       16 0.825076  0.827090           -0.002014     0.437500    0.596588
app_visible visual_genome_m/dinov3_patch/whole_image         all_steps        degenerate       16 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
app_visible       visual_genome_m/siglip/whole_image         all_steps              cost       16 0.383623  0.384966           -0.001343     0.500000    0.820892
app_visible       visual_genome_m/siglip/whole_image         all_steps               fnr       16 0.181371  0.167226            0.014145     0.375000    0.280531
app_visible       visual_genome_m/siglip/whole_image         all_steps               fpr       16 0.202252  0.217740           -0.015488     0.875000    0.000427
app_visible       visual_genome_m/siglip/whole_image         all_steps            regret       16 0.067275  0.074022           -0.006747     0.687500    0.175354
app_visible       visual_genome_m/siglip/whole_image         all_steps average_precision       16 0.409494  0.417683           -0.008189     0.562500    0.939880
app_visible       visual_genome_m/siglip/whole_image         all_steps             auroc       16 0.877816  0.879740           -0.001924     0.562500    0.668549
app_visible       visual_genome_m/siglip/whole_image         all_steps        degenerate       16 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image      pure_gmm_2_5              cost       10 0.542442  0.542442            0.000000     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image      pure_gmm_2_5               fnr       10 0.004167  0.004167            0.000000     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image      pure_gmm_2_5               fpr       10 0.538275  0.538275            0.000000     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image      pure_gmm_2_5            regret       10 0.540833  0.540833            0.000000     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image      pure_gmm_2_5 average_precision       10 0.996746  0.996746            0.000000     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image      pure_gmm_2_5             auroc       10 0.999847  0.999847            0.000000     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image      pure_gmm_2_5        degenerate       10 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image      pure_gmm_2_5              cost       12 0.494730  0.494730            0.000000     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image      pure_gmm_2_5               fnr       12 0.016441  0.016441            0.000000     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image      pure_gmm_2_5               fpr       12 0.478288  0.478288            0.000000     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image      pure_gmm_2_5            regret       12 0.480008  0.480008            0.000000     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image      pure_gmm_2_5 average_precision       12 0.985832  0.985832            0.000000     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image      pure_gmm_2_5             auroc       12 0.998445  0.998445            0.000000     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image      pure_gmm_2_5        degenerate       12 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
  all_steps          coco_val/dinov3_patch/max_patch      pure_gmm_2_5              cost       14 0.348877  0.348803            0.000074     0.071429    0.224916
  all_steps          coco_val/dinov3_patch/max_patch      pure_gmm_2_5               fnr       14 0.089222  0.089222            0.000000     0.000000         NaN all-zero or empty
  all_steps          coco_val/dinov3_patch/max_patch      pure_gmm_2_5               fpr       14 0.259655  0.259581            0.000074     0.071429    0.224916
  all_steps          coco_val/dinov3_patch/max_patch      pure_gmm_2_5            regret       14 0.169457  0.169383            0.000074     0.071429    0.224916
  all_steps          coco_val/dinov3_patch/max_patch      pure_gmm_2_5 average_precision       14 0.674755  0.674755            0.000000     0.000000         NaN all-zero or empty
  all_steps          coco_val/dinov3_patch/max_patch      pure_gmm_2_5             auroc       14 0.936868  0.936868            0.000000     0.000000         NaN all-zero or empty
  all_steps          coco_val/dinov3_patch/max_patch      pure_gmm_2_5        degenerate       14 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
  all_steps        coco_val/dinov3_patch/whole_image      pure_gmm_2_5              cost       14 0.633326  0.633312            0.000014     0.142857    0.753152
  all_steps        coco_val/dinov3_patch/whole_image      pure_gmm_2_5               fnr       14 0.116248  0.116248            0.000000     0.000000         NaN all-zero or empty
  all_steps        coco_val/dinov3_patch/whole_image      pure_gmm_2_5               fpr       14 0.517078  0.517064            0.000014     0.142857    0.753152
  all_steps        coco_val/dinov3_patch/whole_image      pure_gmm_2_5            regret       14 0.308010  0.307997            0.000014     0.142857    0.753152
  all_steps        coco_val/dinov3_patch/whole_image      pure_gmm_2_5 average_precision       14 0.478819  0.478819            0.000000     0.000000         NaN all-zero or empty
  all_steps        coco_val/dinov3_patch/whole_image      pure_gmm_2_5             auroc       14 0.859874  0.859874            0.000000     0.000000         NaN all-zero or empty
  all_steps        coco_val/dinov3_patch/whole_image      pure_gmm_2_5        degenerate       14 0.035714  0.035714            0.000000     0.000000         NaN all-zero or empty
  all_steps              coco_val/siglip/whole_image      pure_gmm_2_5              cost       14 0.521449  0.521464           -0.000015     0.071429    0.317311
  all_steps              coco_val/siglip/whole_image      pure_gmm_2_5               fnr       14 0.077431  0.077431            0.000000     0.000000         NaN all-zero or empty
  all_steps              coco_val/siglip/whole_image      pure_gmm_2_5               fpr       14 0.444018  0.444033           -0.000015     0.071429    0.317311
  all_steps              coco_val/siglip/whole_image      pure_gmm_2_5            regret       14 0.240185  0.240200           -0.000015     0.071429    0.317311
  all_steps              coco_val/siglip/whole_image      pure_gmm_2_5 average_precision       14 0.500590  0.500590            0.000000     0.000000         NaN all-zero or empty
  all_steps              coco_val/siglip/whole_image      pure_gmm_2_5             auroc       14 0.901479  0.901479            0.000000     0.000000         NaN all-zero or empty
  all_steps              coco_val/siglip/whole_image      pure_gmm_2_5        degenerate       14 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
  all_steps   visual_genome_m/dinov3_patch/max_patch      pure_gmm_2_5              cost       16 0.361787  0.361742            0.000045     0.125000    0.500184
  all_steps   visual_genome_m/dinov3_patch/max_patch      pure_gmm_2_5               fnr       16 0.117047  0.117047            0.000000     0.000000         NaN all-zero or empty
  all_steps   visual_genome_m/dinov3_patch/max_patch      pure_gmm_2_5               fpr       16 0.244740  0.244695            0.000045     0.125000    0.500184
  all_steps   visual_genome_m/dinov3_patch/max_patch      pure_gmm_2_5            regret       16 0.119461  0.119416            0.000045     0.125000    0.500184
  all_steps   visual_genome_m/dinov3_patch/max_patch      pure_gmm_2_5 average_precision       16 0.463686  0.463686            0.000000     0.000000         NaN all-zero or empty
  all_steps   visual_genome_m/dinov3_patch/max_patch      pure_gmm_2_5             auroc       16 0.907755  0.907755            0.000000     0.000000         NaN all-zero or empty
  all_steps   visual_genome_m/dinov3_patch/max_patch      pure_gmm_2_5        degenerate       16 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
  all_steps visual_genome_m/dinov3_patch/whole_image      pure_gmm_2_5              cost       16 0.739358  0.739388           -0.000030     0.125000    0.285049
  all_steps visual_genome_m/dinov3_patch/whole_image      pure_gmm_2_5               fnr       16 0.134910  0.134910            0.000000     0.000000         NaN all-zero or empty
  all_steps visual_genome_m/dinov3_patch/whole_image      pure_gmm_2_5               fpr       16 0.604448  0.604478           -0.000030     0.125000    0.285049
  all_steps visual_genome_m/dinov3_patch/whole_image      pure_gmm_2_5            regret       16 0.224338  0.224368           -0.000030     0.125000    0.285049
  all_steps visual_genome_m/dinov3_patch/whole_image      pure_gmm_2_5 average_precision       16 0.236464  0.236464            0.000000     0.000000         NaN all-zero or empty
  all_steps visual_genome_m/dinov3_patch/whole_image      pure_gmm_2_5             auroc       16 0.762593  0.762593            0.000000     0.000000         NaN all-zero or empty
  all_steps visual_genome_m/dinov3_patch/whole_image      pure_gmm_2_5        degenerate       16 0.031250  0.031250            0.000000     0.000000         NaN all-zero or empty
  all_steps       visual_genome_m/siglip/whole_image      pure_gmm_2_5              cost       15 0.676924  0.677067           -0.000143     0.266667    0.067889
  all_steps       visual_genome_m/siglip/whole_image      pure_gmm_2_5               fnr       15 0.146021  0.146021            0.000000     0.000000         NaN all-zero or empty
  all_steps       visual_genome_m/siglip/whole_image      pure_gmm_2_5               fpr       15 0.530903  0.531046           -0.000143     0.266667    0.067889
  all_steps       visual_genome_m/siglip/whole_image      pure_gmm_2_5            regret       15 0.202551  0.202694           -0.000143     0.266667    0.067889
  all_steps       visual_genome_m/siglip/whole_image      pure_gmm_2_5 average_precision       15 0.249008  0.249008            0.000000     0.000000         NaN all-zero or empty
  all_steps       visual_genome_m/siglip/whole_image      pure_gmm_2_5             auroc       15 0.788571  0.788571            0.000000     0.000000         NaN all-zero or empty
  all_steps       visual_genome_m/siglip/whole_image      pure_gmm_2_5        degenerate       15 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image         ramp_6_20              cost       12 0.480094  0.480107           -0.000013     0.083333    1.000000
  all_steps    caltech101_m/dinov3_patch/whole_image         ramp_6_20               fnr       12 0.071067  0.071067            0.000000     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image         ramp_6_20               fpr       12 0.409026  0.409040           -0.000013     0.083333    1.000000
  all_steps    caltech101_m/dinov3_patch/whole_image         ramp_6_20            regret       12 0.466677  0.466690           -0.000013     0.083333    1.000000
  all_steps    caltech101_m/dinov3_patch/whole_image         ramp_6_20 average_precision       12 0.996441  0.996441            0.000000     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image         ramp_6_20             auroc       12 0.997770  0.997770            0.000000     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image         ramp_6_20        degenerate       12 0.057234  0.057234            0.000000     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image         ramp_6_20              cost       12 0.073807  0.076054           -0.002247     0.166667    0.625000
  all_steps          caltech101_m/siglip/whole_image         ramp_6_20               fnr       12 0.002980  0.002980            0.000000     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image         ramp_6_20               fpr       12 0.070827  0.073075           -0.002247     0.166667    0.625000
  all_steps          caltech101_m/siglip/whole_image         ramp_6_20            regret       12 0.073025  0.075272           -0.002247     0.166667    0.625000
  all_steps          caltech101_m/siglip/whole_image         ramp_6_20 average_precision       12 0.999894  0.999894            0.000000     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image         ramp_6_20             auroc       12 0.999948  0.999948            0.000000     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image         ramp_6_20        degenerate       12 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
  all_steps          coco_val/dinov3_patch/max_patch         ramp_6_20              cost       14 0.235799  0.235657            0.000142     0.642857    0.357544
  all_steps          coco_val/dinov3_patch/max_patch         ramp_6_20               fnr       14 0.066218  0.078653           -0.012434     0.571429    0.046853
  all_steps          coco_val/dinov3_patch/max_patch         ramp_6_20               fpr       14 0.169581  0.157005            0.012576     0.428571    0.501587
  all_steps          coco_val/dinov3_patch/max_patch         ramp_6_20            regret       14 0.090679  0.087513            0.003167     0.357143    0.541626
  all_steps          coco_val/dinov3_patch/max_patch         ramp_6_20 average_precision       14 0.728553  0.729004           -0.000451     0.500000    0.951538
  all_steps          coco_val/dinov3_patch/max_patch         ramp_6_20             auroc       14 0.961664  0.962553           -0.000889     0.214286    0.426270
  all_steps          coco_val/dinov3_patch/max_patch         ramp_6_20        degenerate       14 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
  all_steps        coco_val/dinov3_patch/whole_image         ramp_6_20              cost       14 0.451265  0.461013           -0.009747     0.571429    0.625732
  all_steps        coco_val/dinov3_patch/whole_image         ramp_6_20               fnr       14 0.123770  0.110065            0.013705     0.285714    0.099481
  all_steps        coco_val/dinov3_patch/whole_image         ramp_6_20               fpr       14 0.327495  0.350948           -0.023452     0.714286    0.241211
  all_steps        coco_val/dinov3_patch/whole_image         ramp_6_20            regret       14 0.193270  0.211744           -0.018474     0.571429    0.390991
  all_steps        coco_val/dinov3_patch/whole_image         ramp_6_20 average_precision       14 0.571791  0.584652           -0.012860     0.714286    0.278707
  all_steps        coco_val/dinov3_patch/whole_image         ramp_6_20             auroc       14 0.906954  0.907670           -0.000716     0.500000    0.421579
  all_steps        coco_val/dinov3_patch/whole_image         ramp_6_20        degenerate       14 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
  all_steps              coco_val/siglip/whole_image         ramp_6_20              cost       14 0.378118  0.376007            0.002112     0.428571    0.951538
  all_steps              coco_val/siglip/whole_image         ramp_6_20               fnr       14 0.135567  0.121172            0.014395     0.142857    0.028056
  all_steps              coco_val/siglip/whole_image         ramp_6_20               fpr       14 0.242552  0.254835           -0.012283     0.642857    0.193726
  all_steps              coco_val/siglip/whole_image         ramp_6_20            regret       14 0.114645  0.127084           -0.012439     0.642857    0.193726
  all_steps              coco_val/siglip/whole_image         ramp_6_20 average_precision       14 0.585315  0.588307           -0.002992     0.428571    0.807739
  all_steps              coco_val/siglip/whole_image         ramp_6_20             auroc       14 0.905948  0.913958           -0.008010     0.428571    0.855225
  all_steps              coco_val/siglip/whole_image         ramp_6_20        degenerate       14 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
  all_steps   visual_genome_m/dinov3_patch/max_patch         ramp_6_20              cost       16 0.292883  0.301631           -0.008748     0.562500    0.495521
  all_steps   visual_genome_m/dinov3_patch/max_patch         ramp_6_20               fnr       16 0.114294  0.111520            0.002774     0.375000    0.813945
  all_steps   visual_genome_m/dinov3_patch/max_patch         ramp_6_20               fpr       16 0.178589  0.190111           -0.011522     0.562500    0.363488
  all_steps   visual_genome_m/dinov3_patch/max_patch         ramp_6_20            regret       16 0.066219  0.074699           -0.008480     0.625000    0.280531
  all_steps   visual_genome_m/dinov3_patch/max_patch         ramp_6_20 average_precision       16 0.491417  0.476558            0.014858     0.312500    0.155635
  all_steps   visual_genome_m/dinov3_patch/max_patch         ramp_6_20             auroc       16 0.913184  0.914922           -0.001738     0.375000    0.649563
  all_steps   visual_genome_m/dinov3_patch/max_patch         ramp_6_20        degenerate       16 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
  all_steps visual_genome_m/dinov3_patch/whole_image         ramp_6_20              cost       16 0.607627  0.589748            0.017879     0.312500    0.143860
  all_steps visual_genome_m/dinov3_patch/whole_image         ramp_6_20               fnr       16 0.249174  0.236472            0.012703     0.250000    0.157811
  all_steps visual_genome_m/dinov3_patch/whole_image         ramp_6_20               fpr       16 0.358453  0.353276            0.005177     0.562500    0.939880
  all_steps visual_genome_m/dinov3_patch/whole_image         ramp_6_20            regret       16 0.139323  0.129488            0.009835     0.375000    0.211426
  all_steps visual_genome_m/dinov3_patch/whole_image         ramp_6_20 average_precision       16 0.310342  0.314061           -0.003719     0.500000    0.532130
  all_steps visual_genome_m/dinov3_patch/whole_image         ramp_6_20             auroc       16 0.785973  0.789802           -0.003829     0.500000    0.570061
  all_steps visual_genome_m/dinov3_patch/whole_image         ramp_6_20        degenerate       16 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
  all_steps       visual_genome_m/siglip/whole_image         ramp_6_20              cost       16 0.472394  0.479799           -0.007406     0.625000    0.093445
  all_steps       visual_genome_m/siglip/whole_image         ramp_6_20               fnr       16 0.168197  0.163174            0.005022     0.312500    0.310897
  all_steps       visual_genome_m/siglip/whole_image         ramp_6_20               fpr       16 0.304197  0.316625           -0.012428     0.812500    0.002136
  all_steps       visual_genome_m/siglip/whole_image         ramp_6_20            regret       16 0.102155  0.105138           -0.002983     0.562500    0.192810
  all_steps       visual_genome_m/siglip/whole_image         ramp_6_20 average_precision       16 0.331940  0.344372           -0.012432     0.812500    0.143860
  all_steps       visual_genome_m/siglip/whole_image         ramp_6_20             auroc       16 0.847273  0.845169            0.002104     0.625000    0.899933
  all_steps       visual_genome_m/siglip/whole_image         ramp_6_20        degenerate       16 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus              cost       12 0.067245  0.067245            0.000000     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus               fnr       12 0.006241  0.006241            0.000000     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus               fpr       12 0.061004  0.061004            0.000000     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus            regret       12 0.065909  0.065909            0.000000     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus average_precision       12 0.999737  0.999737            0.000000     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus             auroc       12 0.999855  0.999855            0.000000     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus        degenerate       12 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image post_ramp_21_plus              cost       12 0.010050  0.014302           -0.004252     0.250000    0.625000
  all_steps          caltech101_m/siglip/whole_image post_ramp_21_plus               fnr       12 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image post_ramp_21_plus               fpr       12 0.010050  0.014302           -0.004252     0.250000    0.625000
  all_steps          caltech101_m/siglip/whole_image post_ramp_21_plus            regret       12 0.010050  0.014302           -0.004252     0.250000    0.625000
  all_steps          caltech101_m/siglip/whole_image post_ramp_21_plus average_precision       12 1.000000  1.000000            0.000000     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image post_ramp_21_plus             auroc       12 1.000000  1.000000            0.000000     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image post_ramp_21_plus        degenerate       12 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
  all_steps          coco_val/dinov3_patch/max_patch post_ramp_21_plus              cost       14 0.181207  0.182593           -0.001386     0.642857    0.583008
  all_steps          coco_val/dinov3_patch/max_patch post_ramp_21_plus               fnr       14 0.064733  0.065318           -0.000586     0.428571    0.959354
  all_steps          coco_val/dinov3_patch/max_patch post_ramp_21_plus               fpr       14 0.116475  0.117275           -0.000800     0.571429    0.426270
  all_steps          coco_val/dinov3_patch/max_patch post_ramp_21_plus            regret       14 0.067089  0.071986           -0.004897     0.571429    0.390991
  all_steps          coco_val/dinov3_patch/max_patch post_ramp_21_plus average_precision       14 0.785124  0.791461           -0.006337     0.642857    0.714844
  all_steps          coco_val/dinov3_patch/max_patch post_ramp_21_plus             auroc       14 0.974364  0.976726           -0.002362     0.500000    0.669800
  all_steps          coco_val/dinov3_patch/max_patch post_ramp_21_plus        degenerate       14 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
  all_steps        coco_val/dinov3_patch/whole_image post_ramp_21_plus              cost       14 0.258385  0.257909            0.000475     0.500000    0.951538
  all_steps        coco_val/dinov3_patch/whole_image post_ramp_21_plus               fnr       14 0.134864  0.127504            0.007360     0.357143    0.357544
  all_steps        coco_val/dinov3_patch/whole_image post_ramp_21_plus               fpr       14 0.123521  0.130405           -0.006884     0.428571    0.855225
  all_steps        coco_val/dinov3_patch/whole_image post_ramp_21_plus            regret       14 0.059418  0.066101           -0.006683     0.571429    0.541626
  all_steps        coco_val/dinov3_patch/whole_image post_ramp_21_plus average_precision       14 0.696785  0.694278            0.002506     0.357143    0.463135
  all_steps        coco_val/dinov3_patch/whole_image post_ramp_21_plus             auroc       14 0.936582  0.935866            0.000716     0.500000    1.000000
  all_steps        coco_val/dinov3_patch/whole_image post_ramp_21_plus        degenerate       14 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
  all_steps              coco_val/siglip/whole_image post_ramp_21_plus              cost       14 0.286769  0.245748            0.041021     0.214286    0.006714
  all_steps              coco_val/siglip/whole_image post_ramp_21_plus               fnr       14 0.150536  0.120311            0.030225     0.071429    0.004742
  all_steps              coco_val/siglip/whole_image post_ramp_21_plus               fpr       14 0.136233  0.125437            0.010796     0.500000    0.625732
  all_steps              coco_val/siglip/whole_image post_ramp_21_plus            regret       14 0.054162  0.055458           -0.001296     0.500000    0.807739
  all_steps              coco_val/siglip/whole_image post_ramp_21_plus average_precision       14 0.645622  0.678684           -0.033062     0.714286    0.049438
  all_steps              coco_val/siglip/whole_image post_ramp_21_plus             auroc       14 0.918105  0.939264           -0.021159     0.857143    0.002319
  all_steps              coco_val/siglip/whole_image post_ramp_21_plus        degenerate       14 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
  all_steps   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus              cost       16 0.250082  0.247767            0.002315     0.375000    0.596588
  all_steps   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus               fnr       16 0.098275  0.089077            0.009198     0.125000    0.015906
  all_steps   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus               fpr       16 0.151807  0.158690           -0.006883     0.625000    0.274445
  all_steps   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus            regret       16 0.058555  0.057993            0.000562     0.500000    0.979950
  all_steps   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus average_precision       16 0.544652  0.519367            0.025285     0.375000    0.159058
  all_steps   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus             auroc       16 0.925924  0.922402            0.003522     0.312500    0.116669
  all_steps   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus        degenerate       16 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
  all_steps visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus              cost       16 0.454520  0.446176            0.008344     0.562500    0.899933
  all_steps visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus               fnr       16 0.296425  0.292386            0.004039     0.500000    0.979950
  all_steps visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus               fpr       16 0.158095  0.153790            0.004305     0.500000    1.000000
  all_steps visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus            regret       16 0.065073  0.057309            0.007764     0.500000    0.463745
  all_steps visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus average_precision       16 0.390408  0.412806           -0.022398     0.437500    0.495422
  all_steps visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus             auroc       16 0.831539  0.833232           -0.001693     0.437500    0.596588
  all_steps visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus        degenerate       16 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
  all_steps       visual_genome_m/siglip/whole_image post_ramp_21_plus              cost       16 0.371546  0.371828           -0.000282     0.500000    0.939880
  all_steps       visual_genome_m/siglip/whole_image post_ramp_21_plus               fnr       16 0.184325  0.168798            0.015528     0.375000    0.363488
  all_steps       visual_genome_m/siglip/whole_image post_ramp_21_plus               fpr       16 0.187220  0.203031           -0.015810     0.875000    0.001007
  all_steps       visual_genome_m/siglip/whole_image post_ramp_21_plus            regret       16 0.062509  0.069882           -0.007373     0.687500    0.211426
  all_steps       visual_genome_m/siglip/whole_image post_ramp_21_plus average_precision       16 0.421770  0.429099           -0.007330     0.500000    0.899933
  all_steps       visual_genome_m/siglip/whole_image post_ramp_21_plus             auroc       16 0.881487  0.884096           -0.002609     0.562500    0.632172
  all_steps       visual_genome_m/siglip/whole_image post_ramp_21_plus        degenerate       16 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image         all_steps              cost       12 0.136990  0.136993           -0.000002     0.083333    1.000000
  all_steps    caltech101_m/dinov3_patch/whole_image         all_steps               fnr       12 0.015395  0.015395            0.000000     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image         all_steps               fpr       12 0.121596  0.121598           -0.000002     0.083333    1.000000
  all_steps    caltech101_m/dinov3_patch/whole_image         all_steps            regret       12 0.133866  0.133868           -0.000002     0.083333    1.000000
  all_steps    caltech101_m/dinov3_patch/whole_image         all_steps average_precision       12 0.999199  0.999199            0.000000     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image         all_steps             auroc       12 0.999549  0.999549            0.000000     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image         all_steps        degenerate       12 0.008055  0.008055            0.000000     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image         all_steps              cost       12 0.029205  0.033055           -0.003850     0.166667    0.875000
  all_steps          caltech101_m/siglip/whole_image         all_steps               fnr       12 0.000800  0.000800            0.000000     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image         all_steps               fpr       12 0.028405  0.032255           -0.003850     0.166667    0.875000
  all_steps          caltech101_m/siglip/whole_image         all_steps            regret       12 0.028801  0.032651           -0.003850     0.166667    0.875000
  all_steps          caltech101_m/siglip/whole_image         all_steps average_precision       12 0.999741  0.999741            0.000000     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image         all_steps             auroc       12 0.999962  0.999962            0.000000     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image         all_steps        degenerate       12 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
  all_steps          coco_val/dinov3_patch/max_patch         all_steps              cost       14 0.193107  0.194226           -0.001120     0.642857    0.669800
  all_steps          coco_val/dinov3_patch/max_patch         all_steps               fnr       14 0.065467  0.067873           -0.002406     0.428571    0.721277
  all_steps          coco_val/dinov3_patch/max_patch         all_steps               fpr       14 0.127639  0.126353            0.001286     0.571429    0.583008
  all_steps          coco_val/dinov3_patch/max_patch         all_steps            regret       14 0.072848  0.076395           -0.003548     0.642857    0.390991
  all_steps          coco_val/dinov3_patch/max_patch         all_steps average_precision       14 0.774101  0.779397           -0.005296     0.642857    0.714844
  all_steps          coco_val/dinov3_patch/max_patch         all_steps             auroc       14 0.971627  0.973713           -0.002085     0.428571    0.760864
  all_steps          coco_val/dinov3_patch/max_patch         all_steps        degenerate       14 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
  all_steps        coco_val/dinov3_patch/whole_image         all_steps              cost       14 0.295942  0.297057           -0.001115     0.571429    1.000000
  all_steps        coco_val/dinov3_patch/whole_image         all_steps               fnr       14 0.132765  0.124575            0.008189     0.357143    0.267578
  all_steps        coco_val/dinov3_patch/whole_image         all_steps               fpr       14 0.163178  0.172482           -0.009304     0.500000    0.541626
  all_steps        coco_val/dinov3_patch/whole_image         all_steps            regret       14 0.085242  0.093611           -0.008369     0.500000    0.501587
  all_steps        coco_val/dinov3_patch/whole_image         all_steps average_precision       14 0.672962  0.672883            0.000078     0.357143    0.541626
  all_steps        coco_val/dinov3_patch/whole_image         all_steps             auroc       14 0.930418  0.929939            0.000479     0.500000    1.000000
  all_steps        coco_val/dinov3_patch/whole_image         all_steps        degenerate       14 0.000736  0.000736            0.000000     0.000000         NaN all-zero or empty
  all_steps              coco_val/siglip/whole_image         all_steps              cost       14 0.305862  0.271720            0.034142     0.214286    0.020264
  all_steps              coco_val/siglip/whole_image         all_steps               fnr       14 0.146704  0.119551            0.027153     0.071429    0.003702
  all_steps              coco_val/siglip/whole_image         all_steps               fpr       14 0.159159  0.152170            0.006989     0.500000    0.807739
  all_steps              coco_val/siglip/whole_image         all_steps            regret       14 0.067437  0.070417           -0.002980     0.571429    0.625732
  all_steps              coco_val/siglip/whole_image         all_steps average_precision       14 0.633252  0.660957           -0.027706     0.642857    0.090576
  all_steps              coco_val/siglip/whole_image         all_steps             auroc       14 0.915852  0.934542           -0.018690     0.714286    0.006714
  all_steps              coco_val/siglip/whole_image         all_steps        degenerate       14 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
  all_steps   visual_genome_m/dinov3_patch/max_patch         all_steps              cost       16 0.259004  0.258446            0.000557     0.500000    0.939880
  all_steps   visual_genome_m/dinov3_patch/max_patch         all_steps               fnr       16 0.101139  0.093124            0.008015     0.187500    0.047990
  all_steps   visual_genome_m/dinov3_patch/max_patch         all_steps               fpr       16 0.157865  0.165322           -0.007458     0.625000    0.252228
  all_steps   visual_genome_m/dinov3_patch/max_patch         all_steps            regret       16 0.060996  0.061843           -0.000847     0.562500    0.743561
  all_steps   visual_genome_m/dinov3_patch/max_patch         all_steps average_precision       16 0.534750  0.511599            0.023151     0.437500    0.175354
  all_steps   visual_genome_m/dinov3_patch/max_patch         all_steps             auroc       16 0.923579  0.920943            0.002636     0.312500    0.175354
  all_steps   visual_genome_m/dinov3_patch/max_patch         all_steps        degenerate       16 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
  all_steps visual_genome_m/dinov3_patch/whole_image         all_steps              cost       16 0.484069  0.474424            0.009646     0.437500    0.705719
  all_steps visual_genome_m/dinov3_patch/whole_image         all_steps               fnr       16 0.285788  0.280493            0.005295     0.500000    0.899933
  all_steps visual_genome_m/dinov3_patch/whole_image         all_steps               fpr       16 0.198281  0.193931            0.004351     0.500000    1.000000
  all_steps visual_genome_m/dinov3_patch/whole_image         all_steps            regret       16 0.079839  0.071915            0.007923     0.437500    0.322510
  all_steps visual_genome_m/dinov3_patch/whole_image         all_steps average_precision       16 0.374853  0.393901           -0.019048     0.437500    0.495422
  all_steps visual_genome_m/dinov3_patch/whole_image         all_steps             auroc       16 0.823071  0.825060           -0.001989     0.437500    0.596588
  all_steps visual_genome_m/dinov3_patch/whole_image         all_steps        degenerate       16 0.000644  0.000644            0.000000     0.000000         NaN all-zero or empty
  all_steps       visual_genome_m/siglip/whole_image         all_steps              cost       16 0.394253  0.395604           -0.001352     0.500000    0.820892
  all_steps       visual_genome_m/siglip/whole_image         all_steps               fnr       16 0.181666  0.168032            0.013634     0.375000    0.280531
  all_steps       visual_genome_m/siglip/whole_image         all_steps               fpr       16 0.212587  0.227572           -0.014985     0.875000    0.000427
  all_steps       visual_genome_m/siglip/whole_image         all_steps            regret       16 0.071370  0.077912           -0.006542     0.687500    0.175354
  all_steps       visual_genome_m/siglip/whole_image         all_steps average_precision       16 0.403950  0.411940           -0.007990     0.562500    0.939880
  all_steps       visual_genome_m/siglip/whole_image         all_steps             auroc       16 0.873674  0.875533           -0.001859     0.562500    0.668549
  all_steps       visual_genome_m/siglip/whole_image         all_steps        degenerate       16 0.000000  0.000000            0.000000     0.000000         NaN all-zero or empty
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
      "delta_cost_on_minus_off": -0.0037739800162634968,
      "p": 0.4448369168645606,
      "reading": "safe ON better (n.s.)"
    },
    "post_ramp_21_plus": {
      "n_cells": 16,
      "delta_cost_on_minus_off": 0.0004645467075892831,
      "p": 0.5897979736328125,
      "reading": "safe ON worse (n.s.)"
    },
    "all_steps": {
      "n_cells": 16,
      "delta_cost_on_minus_off": -0.0002464700575960985,
      "p": 0.804840087890625,
      "reading": "safe ON better (n.s.)"
    },
    "force_on_for_all_users": true
  },
  "all_steps": {
    "scope": "all_steps",
    "pure_gmm_2_5": {
      "n_cells": 16,
      "delta_cost_on_minus_off": 5.9729910714286404e-05,
      "p": 0.3625500705433781,
      "reading": "safe ON worse (n.s.)"
    },
    "ramp_6_20": {
      "n_cells": 16,
      "delta_cost_on_minus_off": -0.004303211309523808,
      "p": 0.42653241418455756,
      "reading": "safe ON better (n.s.)"
    },
    "post_ramp_21_plus": {
      "n_cells": 16,
      "delta_cost_on_minus_off": 0.0004645467075892831,
      "p": 0.5897979736328125,
      "reading": "safe ON worse (n.s.)"
    },
    "all_steps": {
      "n_cells": 16,
      "delta_cost_on_minus_off": -0.0002810821980854238,
      "p": 0.804840087890625,
      "reading": "safe ON better (n.s.)"
    },
    "force_on_for_all_users": false
  }
}
```

Curves: `agg/ab_curves_vs_votes.csv` (1556 rows) · paired cells: `agg/ab_paired_cells.csv` · figures: ab_cost_vs_votes.png, ab_fnr_vs_votes.png, ab_degenerate_vs_votes.png
