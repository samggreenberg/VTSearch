# Safe thresholds ON vs OFF — trajectory A/B (#2799)

Head: `linear_svm` · cells ON/OFF: 114/114 · ON `/expscratch/sgreenberg/gmm-3585/ab_native/results` vs OFF `/expscratch/sgreenberg/gmm-3585/ab_baseline/results`

Both runs are full simulations; because the blended threshold drives Autopilot's
Hard pick, the two arms vote on different items, so cells (category × seed), not
steps, are the paired units.

The app shows a trained detector from **7 votes** onward; below that it
sorts by text/example cosine, so `scope=app_visible` is what users actually get and
`scope=all_steps` is the purely numerical reading.

## Per-window paired comparison (Δ = ON − OFF; negative = safe thresholds better)

```
      scope                                      arm            window            metric  n_cells  safe_on  safe_off  delta_on_minus_off  win_rate_on  p_wilcoxon              note
app_visible    caltech101_m/dinov3_patch/whole_image         ramp_6_20              cost       11 0.384210  0.534617       -1.504078e-01     0.545455    0.431641                  
app_visible    caltech101_m/dinov3_patch/whole_image         ramp_6_20               fnr       11 0.003162  0.000135        3.026748e-03     0.090909    1.000000                  
app_visible    caltech101_m/dinov3_patch/whole_image         ramp_6_20               fpr       11 0.381048  0.534482       -1.534345e-01     0.545455    0.431641                  
app_visible    caltech101_m/dinov3_patch/whole_image         ramp_6_20            regret       11 0.378005  0.530849       -1.528446e-01     0.545455    0.431641                  
app_visible    caltech101_m/dinov3_patch/whole_image         ramp_6_20 average_precision       11 0.998476  0.999178       -7.016818e-04     0.090909    1.000000                  
app_visible    caltech101_m/dinov3_patch/whole_image         ramp_6_20             auroc       11 0.999173  0.999607       -4.340455e-04     0.090909    1.000000                  
app_visible    caltech101_m/dinov3_patch/whole_image         ramp_6_20        degenerate       11 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image         ramp_6_20              cost       12 0.061930  0.056203        5.727563e-03     0.250000    1.000000                  
app_visible          caltech101_m/siglip/whole_image         ramp_6_20               fnr       12 0.001976  0.001976        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image         ramp_6_20               fpr       12 0.059954  0.054226        5.727563e-03     0.250000    1.000000                  
app_visible          caltech101_m/siglip/whole_image         ramp_6_20            regret       12 0.061311  0.055583        5.727563e-03     0.250000    1.000000                  
app_visible          caltech101_m/siglip/whole_image         ramp_6_20 average_precision       12 0.999937  0.999937        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image         ramp_6_20             auroc       12 0.999974  0.999974        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image         ramp_6_20        degenerate       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          coco_val/dinov3_patch/max_patch         ramp_6_20              cost       14 0.230334  0.231710       -1.375556e-03     0.357143    1.000000                  
app_visible          coco_val/dinov3_patch/max_patch         ramp_6_20               fnr       14 0.080090  0.076554        3.536638e-03     0.357143    0.952765                  
app_visible          coco_val/dinov3_patch/max_patch         ramp_6_20               fpr       14 0.150244  0.155156       -4.912163e-03     0.428571    0.937473                  
app_visible          coco_val/dinov3_patch/max_patch         ramp_6_20            regret       14 0.084241  0.089316       -5.074327e-03     0.500000    0.582920                  
app_visible          coco_val/dinov3_patch/max_patch         ramp_6_20 average_precision       14 0.733187  0.734915       -1.727724e-03     0.571429    0.346522                  
app_visible          coco_val/dinov3_patch/max_patch         ramp_6_20             auroc       14 0.964392  0.964355        3.711735e-05     0.357143    1.000000                  
app_visible          coco_val/dinov3_patch/max_patch         ramp_6_20        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible        coco_val/dinov3_patch/whole_image         ramp_6_20              cost       14 0.450312  0.447384        2.928016e-03     0.357143    0.813945                  
app_visible        coco_val/dinov3_patch/whole_image         ramp_6_20               fnr       14 0.107476  0.113502       -6.025624e-03     0.428571    0.813945                  
app_visible        coco_val/dinov3_patch/whole_image         ramp_6_20               fpr       14 0.342836  0.333882        8.953522e-03     0.357143    0.875329                  
app_visible        coco_val/dinov3_patch/whole_image         ramp_6_20            regret       14 0.207048  0.201850        5.197841e-03     0.428571    0.937473                  
app_visible        coco_val/dinov3_patch/whole_image         ramp_6_20 average_precision       14 0.591454  0.569062        2.239177e-02     0.285714    0.136097                  
app_visible        coco_val/dinov3_patch/whole_image         ramp_6_20             auroc       14 0.912524  0.911588        9.364702e-04     0.357143    0.937473                  
app_visible        coco_val/dinov3_patch/whole_image         ramp_6_20        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible              coco_val/siglip/whole_image         ramp_6_20              cost       14 0.370226  0.382562       -1.233569e-02     0.500000    0.951538                  
app_visible              coco_val/siglip/whole_image         ramp_6_20               fnr       14 0.122506  0.142991       -2.048543e-02     0.571429    0.272095                  
app_visible              coco_val/siglip/whole_image         ramp_6_20               fpr       14 0.247720  0.239571        8.149694e-03     0.500000    0.583008                  
app_visible              coco_val/siglip/whole_image         ramp_6_20            regret       14 0.122281  0.113996        8.284959e-03     0.500000    0.583008                  
app_visible              coco_val/siglip/whole_image         ramp_6_20 average_precision       14 0.591723  0.581297        1.042567e-02     0.500000    0.501587                  
app_visible              coco_val/siglip/whole_image         ramp_6_20             auroc       14 0.914152  0.901976        1.217629e-02     0.357143    0.501587                  
app_visible              coco_val/siglip/whole_image         ramp_6_20        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible   visual_genome_m/dinov3_patch/max_patch         ramp_6_20              cost       16 0.301937  0.295666        6.270355e-03     0.375000    0.382352                  
app_visible   visual_genome_m/dinov3_patch/max_patch         ramp_6_20               fnr       16 0.110068  0.125072       -1.500342e-02     0.562500    0.075368                  
app_visible   visual_genome_m/dinov3_patch/max_patch         ramp_6_20               fpr       16 0.191868  0.170595        2.127374e-02     0.250000    0.046399                  
app_visible   visual_genome_m/dinov3_patch/max_patch         ramp_6_20            regret       16 0.076030  0.070391        5.638987e-03     0.437500    0.700703                  
app_visible   visual_genome_m/dinov3_patch/max_patch         ramp_6_20 average_precision       16 0.476773  0.478471       -1.698539e-03     0.312500    1.000000                  
app_visible   visual_genome_m/dinov3_patch/max_patch         ramp_6_20             auroc       16 0.915032  0.912325        2.706643e-03     0.125000    0.109511                  
app_visible   visual_genome_m/dinov3_patch/max_patch         ramp_6_20        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible visual_genome_m/dinov3_patch/whole_image         ramp_6_20              cost       16 0.579751  0.573700        6.050975e-03     0.500000    0.974960                  
app_visible visual_genome_m/dinov3_patch/whole_image         ramp_6_20               fnr       16 0.243634  0.228733        1.490073e-02     0.250000    0.300290                  
app_visible visual_genome_m/dinov3_patch/whole_image         ramp_6_20               fpr       16 0.336117  0.344967       -8.849762e-03     0.437500    0.777565                  
app_visible visual_genome_m/dinov3_patch/whole_image         ramp_6_20            regret       16 0.121744  0.127745       -6.000824e-03     0.437500    0.729891                  
app_visible visual_genome_m/dinov3_patch/whole_image         ramp_6_20 average_precision       16 0.318277  0.317022        1.254646e-03     0.375000    0.470338                  
app_visible visual_genome_m/dinov3_patch/whole_image         ramp_6_20             auroc       16 0.791562  0.804061       -1.249892e-02     0.437500    0.509797                  
app_visible visual_genome_m/dinov3_patch/whole_image         ramp_6_20        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible       visual_genome_m/siglip/whole_image         ramp_6_20              cost       16 0.464855  0.484636       -1.978089e-02     0.562500    0.375458                  
app_visible       visual_genome_m/siglip/whole_image         ramp_6_20               fnr       16 0.159322  0.172885       -1.356387e-02     0.437500    0.245494                  
app_visible       visual_genome_m/siglip/whole_image         ramp_6_20               fpr       16 0.305533  0.311750       -6.216951e-03     0.625000    0.322510                  
app_visible       visual_genome_m/siglip/whole_image         ramp_6_20            regret       16 0.098410  0.111436       -1.302603e-02     0.562500    0.495422                  
app_visible       visual_genome_m/siglip/whole_image         ramp_6_20 average_precision       16 0.350310  0.346155        4.155418e-03     0.625000    1.000000                  
app_visible       visual_genome_m/siglip/whole_image         ramp_6_20             auroc       16 0.852044  0.848083        3.960978e-03     0.562500    0.979950                  
app_visible       visual_genome_m/siglip/whole_image         ramp_6_20        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus              cost       12 0.065456  0.066588       -1.132876e-03     0.500000    0.965820                  
app_visible    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus               fnr       12 0.006392  0.007068       -6.766566e-04     0.250000    0.875000                  
app_visible    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus               fpr       12 0.059064  0.059520       -4.562197e-04     0.500000    0.965820                  
app_visible    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus            regret       12 0.064643  0.066275       -1.631867e-03     0.500000    0.965820                  
app_visible    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus average_precision       12 0.999884  0.999972       -8.835718e-05     0.166667    0.500000                  
app_visible    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus             auroc       12 0.999938  0.999985       -4.729178e-05     0.166667    0.500000                  
app_visible    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus        degenerate       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image post_ramp_21_plus              cost       12 0.014582  0.014283        2.989708e-04     0.166667    1.000000                  
app_visible          caltech101_m/siglip/whole_image post_ramp_21_plus               fnr       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image post_ramp_21_plus               fpr       12 0.014582  0.014283        2.989708e-04     0.166667    1.000000                  
app_visible          caltech101_m/siglip/whole_image post_ramp_21_plus            regret       12 0.014582  0.014283        2.989708e-04     0.166667    1.000000                  
app_visible          caltech101_m/siglip/whole_image post_ramp_21_plus average_precision       12 1.000000  1.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image post_ramp_21_plus             auroc       12 1.000000  1.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image post_ramp_21_plus        degenerate       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          coco_val/dinov3_patch/max_patch post_ramp_21_plus              cost       14 0.185568  0.185748       -1.800000e-04     0.500000    0.875329                  
app_visible          coco_val/dinov3_patch/max_patch post_ramp_21_plus               fnr       14 0.068365  0.067687        6.774473e-04     0.357143    0.952765                  
app_visible          coco_val/dinov3_patch/max_patch post_ramp_21_plus               fpr       14 0.117203  0.118060       -8.574241e-04     0.357143    0.753684                  
app_visible          coco_val/dinov3_patch/max_patch post_ramp_21_plus            regret       14 0.069556  0.074870       -5.313787e-03     0.714286    0.157939                  
app_visible          coco_val/dinov3_patch/max_patch post_ramp_21_plus average_precision       14 0.790257  0.801354       -1.109677e-02     0.571429    0.388186                  
app_visible          coco_val/dinov3_patch/max_patch post_ramp_21_plus             auroc       14 0.974093  0.976774       -2.680875e-03     0.642857    0.059739                  
app_visible          coco_val/dinov3_patch/max_patch post_ramp_21_plus        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible        coco_val/dinov3_patch/whole_image post_ramp_21_plus              cost       14 0.266383  0.264920        1.462983e-03     0.285714    0.426270                  
app_visible        coco_val/dinov3_patch/whole_image post_ramp_21_plus               fnr       14 0.128068  0.123520        4.547979e-03     0.285714    0.295776                  
app_visible        coco_val/dinov3_patch/whole_image post_ramp_21_plus               fpr       14 0.138316  0.141401       -3.085050e-03     0.571429    0.903198                  
app_visible        coco_val/dinov3_patch/whole_image post_ramp_21_plus            regret       14 0.068954  0.075570       -6.615982e-03     0.500000    1.000000                  
app_visible        coco_val/dinov3_patch/whole_image post_ramp_21_plus average_precision       14 0.688527  0.696676       -8.149005e-03     0.571429    0.153076                  
app_visible        coco_val/dinov3_patch/whole_image post_ramp_21_plus             auroc       14 0.934493  0.935680       -1.186479e-03     0.500000    0.903198                  
app_visible        coco_val/dinov3_patch/whole_image post_ramp_21_plus        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible              coco_val/siglip/whole_image post_ramp_21_plus              cost       14 0.241227  0.275053       -3.382553e-02     0.857143    0.006714                  
app_visible              coco_val/siglip/whole_image post_ramp_21_plus               fnr       14 0.115131  0.145151       -3.002086e-02     0.642857    0.034170                  
app_visible              coco_val/siglip/whole_image post_ramp_21_plus               fpr       14 0.126096  0.129901       -3.804623e-03     0.428571    0.855225                  
app_visible              coco_val/siglip/whole_image post_ramp_21_plus            regret       14 0.055976  0.054102        1.873452e-03     0.428571    0.807739                  
app_visible              coco_val/siglip/whole_image post_ramp_21_plus average_precision       14 0.682203  0.647506        3.469696e-02     0.214286    0.016602                  
app_visible              coco_val/siglip/whole_image post_ramp_21_plus             auroc       14 0.942561  0.921634        2.092709e-02     0.214286    0.006714                  
app_visible              coco_val/siglip/whole_image post_ramp_21_plus        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus              cost       16 0.247483  0.254136       -6.652888e-03     0.375000    0.694887                  
app_visible   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus               fnr       16 0.092265  0.110372       -1.810690e-02     0.562500    0.061884                  
app_visible   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus               fpr       16 0.155218  0.143764        1.145402e-02     0.312500    0.182338                  
app_visible   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus            regret       16 0.054843  0.053190        1.653536e-03     0.312500    0.875329                  
app_visible   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus average_precision       16 0.520422  0.525527       -5.105069e-03     0.375000    0.753684                  
app_visible   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus             auroc       16 0.923371  0.918574        4.796791e-03     0.375000    0.272095                  
app_visible   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus              cost       16 0.450877  0.445566        5.310580e-03     0.437500    0.570061                  
app_visible visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus               fnr       16 0.297565  0.287381        1.018374e-02     0.312500    0.334277                  
app_visible visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus               fpr       16 0.153312  0.158185       -4.873150e-03     0.562500    0.690945                  
app_visible visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus            regret       16 0.056441  0.060987       -4.545989e-03     0.437500    0.864705                  
app_visible visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus average_precision       16 0.411778  0.387733        2.404522e-02     0.437500    0.733271                  
app_visible visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus             auroc       16 0.829309  0.831738       -2.428615e-03     0.625000    0.191446                  
app_visible visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible       visual_genome_m/siglip/whole_image post_ramp_21_plus              cost       16 0.377820  0.383382       -5.562185e-03     0.625000    0.561890                  
app_visible       visual_genome_m/siglip/whole_image post_ramp_21_plus               fnr       16 0.186046  0.195880       -9.833688e-03     0.437500    0.460302                  
app_visible       visual_genome_m/siglip/whole_image post_ramp_21_plus               fpr       16 0.191774  0.187502        4.271472e-03     0.437500    0.528168                  
app_visible       visual_genome_m/siglip/whole_image post_ramp_21_plus            regret       16 0.062477  0.065388       -2.910927e-03     0.437500    0.820892                  
app_visible       visual_genome_m/siglip/whole_image post_ramp_21_plus average_precision       16 0.421505  0.426363       -4.858449e-03     0.562500    0.433197                  
app_visible       visual_genome_m/siglip/whole_image post_ramp_21_plus             auroc       16 0.878194  0.876489        1.704994e-03     0.437500    0.561890                  
app_visible       visual_genome_m/siglip/whole_image post_ramp_21_plus        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible    caltech101_m/dinov3_patch/whole_image         all_steps              cost       12 0.096883  0.112796       -1.591232e-02     0.583333    0.622070                  
app_visible    caltech101_m/dinov3_patch/whole_image         all_steps               fnr       12 0.006265  0.006465       -1.999732e-04     0.250000    0.875000                  
app_visible    caltech101_m/dinov3_patch/whole_image         all_steps               fpr       12 0.090618  0.106330       -1.571235e-02     0.583333    0.518555                  
app_visible    caltech101_m/dinov3_patch/whole_image         all_steps            regret       12 0.095868  0.112239       -1.637141e-02     0.583333    0.622070                  
app_visible    caltech101_m/dinov3_patch/whole_image         all_steps average_precision       12 0.999827  0.999914       -8.742219e-05     0.166667    0.500000                  
app_visible    caltech101_m/dinov3_patch/whole_image         all_steps             auroc       12 0.999911  0.999960       -4.907339e-05     0.166667    0.500000                  
app_visible    caltech101_m/dinov3_patch/whole_image         all_steps        degenerate       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image         all_steps              cost       12 0.021133  0.020018        1.115244e-03     0.333333    0.937500                  
app_visible          caltech101_m/siglip/whole_image         all_steps               fnr       12 0.000294  0.000294        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image         all_steps               fpr       12 0.020838  0.019723        1.115244e-03     0.333333    0.937500                  
app_visible          caltech101_m/siglip/whole_image         all_steps            regret       12 0.021040  0.019925        1.115244e-03     0.333333    0.937500                  
app_visible          caltech101_m/siglip/whole_image         all_steps average_precision       12 0.999991  0.999991        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image         all_steps             auroc       12 0.999996  0.999996        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          caltech101_m/siglip/whole_image         all_steps        degenerate       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible          coco_val/dinov3_patch/max_patch         all_steps              cost       14 0.191998  0.192356       -3.580616e-04     0.500000    0.875329                  
app_visible          coco_val/dinov3_patch/max_patch         all_steps               fnr       14 0.069888  0.068785        1.103284e-03     0.428571    0.767097                  
app_visible          coco_val/dinov3_patch/max_patch         all_steps               fpr       14 0.122109  0.123571       -1.461321e-03     0.357143    0.937473                  
app_visible          coco_val/dinov3_patch/max_patch         all_steps            regret       14 0.071754  0.077032       -5.278123e-03     0.642857    0.099481                  
app_visible          coco_val/dinov3_patch/max_patch         all_steps average_precision       14 0.782095  0.791796       -9.701379e-03     0.571429    0.432768                  
app_visible          coco_val/dinov3_patch/max_patch         all_steps             auroc       14 0.972735  0.975011       -2.276068e-03     0.642857    0.071189                  
app_visible          coco_val/dinov3_patch/max_patch         all_steps        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible        coco_val/dinov3_patch/whole_image         all_steps              cost       14 0.293462  0.291724        1.738145e-03     0.428571    0.541626                  
app_visible        coco_val/dinov3_patch/whole_image         all_steps               fnr       14 0.124768  0.121705        3.062335e-03     0.285714    0.357544                  
app_visible        coco_val/dinov3_patch/whole_image         all_steps               fpr       14 0.168695  0.170019       -1.324253e-03     0.500000    0.951538                  
app_visible        coco_val/dinov3_patch/whole_image         all_steps            regret       14 0.089367  0.094199       -4.832569e-03     0.500000    0.807739                  
app_visible        coco_val/dinov3_patch/whole_image         all_steps average_precision       14 0.674388  0.678087       -3.699269e-03     0.571429    0.193726                  
app_visible        coco_val/dinov3_patch/whole_image         all_steps             auroc       14 0.931319  0.932189       -8.694797e-04     0.571429    0.855225                  
app_visible        coco_val/dinov3_patch/whole_image         all_steps        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible              coco_val/siglip/whole_image         all_steps              cost       14 0.260440  0.291065       -3.062492e-02     0.857143    0.006714                  
app_visible              coco_val/siglip/whole_image         all_steps               fnr       14 0.116229  0.144830       -2.860069e-02     0.642857    0.028056                  
app_visible              coco_val/siglip/whole_image         all_steps               fpr       14 0.144211  0.146235       -2.024193e-03     0.428571    0.903198                  
app_visible              coco_val/siglip/whole_image         all_steps            regret       14 0.065851  0.063023        2.828357e-03     0.500000    0.426270                  
app_visible              coco_val/siglip/whole_image         all_steps average_precision       14 0.668727  0.637645        3.108209e-02     0.214286    0.016602                  
app_visible              coco_val/siglip/whole_image         all_steps             auroc       14 0.938330  0.918706        1.962378e-02     0.214286    0.005249                  
app_visible              coco_val/siglip/whole_image         all_steps        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible   visual_genome_m/dinov3_patch/max_patch         all_steps              cost       16 0.255541  0.260315       -4.774021e-03     0.437500    0.916512                  
app_visible   visual_genome_m/dinov3_patch/max_patch         all_steps               fnr       16 0.094910  0.112600       -1.769015e-02     0.625000    0.039243                  
app_visible   visual_genome_m/dinov3_patch/max_patch         all_steps               fpr       16 0.160631  0.147714        1.291613e-02     0.312500    0.151956                  
app_visible   visual_genome_m/dinov3_patch/max_patch         all_steps            regret       16 0.058010  0.055754        2.255717e-03     0.375000    0.753152                  
app_visible   visual_genome_m/dinov3_patch/max_patch         all_steps average_precision       16 0.513948  0.518563       -4.615579e-03     0.375000    0.937473                  
app_visible   visual_genome_m/dinov3_patch/max_patch         all_steps             auroc       16 0.922154  0.917633        4.520780e-03     0.312500    0.272095                  
app_visible   visual_genome_m/dinov3_patch/max_patch         all_steps        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible visual_genome_m/dinov3_patch/whole_image         all_steps              cost       16 0.469834  0.464273        5.560397e-03     0.437500    0.532130                  
app_visible visual_genome_m/dinov3_patch/whole_image         all_steps               fnr       16 0.289705  0.278663        1.104225e-02     0.375000    0.155635                  
app_visible visual_genome_m/dinov3_patch/whole_image         all_steps               fpr       16 0.180129  0.185611       -5.481849e-03     0.625000    0.495521                  
app_visible visual_genome_m/dinov3_patch/whole_image         all_steps            regret       16 0.066084  0.070732       -4.648561e-03     0.437500    0.495521                  
app_visible visual_genome_m/dinov3_patch/whole_image         all_steps average_precision       16 0.398006  0.377537        2.046985e-02     0.437500    0.570061                  
app_visible visual_genome_m/dinov3_patch/whole_image         all_steps             auroc       16 0.823767  0.827686       -3.918504e-03     0.687500    0.172848                  
app_visible visual_genome_m/dinov3_patch/whole_image         all_steps        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
app_visible       visual_genome_m/siglip/whole_image         all_steps              cost       16 0.390086  0.397861       -7.775158e-03     0.562500    0.433197                  
app_visible       visual_genome_m/siglip/whole_image         all_steps               fnr       16 0.181928  0.192344       -1.041632e-02     0.437500    0.561890                  
app_visible       visual_genome_m/siglip/whole_image         all_steps               fpr       16 0.208158  0.205517        2.641143e-03     0.375000    0.528168                  
app_visible       visual_genome_m/siglip/whole_image         all_steps            regret       16 0.067714  0.072126       -4.412563e-03     0.562500    0.528168                  
app_visible       visual_genome_m/siglip/whole_image         all_steps average_precision       16 0.411192  0.414679       -3.487325e-03     0.562500    0.297852                  
app_visible       visual_genome_m/siglip/whole_image         all_steps             auroc       16 0.874702  0.872575        2.127181e-03     0.437500    0.596588                  
app_visible       visual_genome_m/siglip/whole_image         all_steps        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image      pure_gmm_2_5              cost       10 0.542442  0.566170       -2.372805e-02     0.500000    0.687500                  
  all_steps    caltech101_m/dinov3_patch/whole_image      pure_gmm_2_5               fnr       10 0.004167  0.000000        4.166650e-03     0.000000    1.000000                  
  all_steps    caltech101_m/dinov3_patch/whole_image      pure_gmm_2_5               fpr       10 0.538275  0.566170       -2.789470e-02     0.500000    0.687500                  
  all_steps    caltech101_m/dinov3_patch/whole_image      pure_gmm_2_5            regret       10 0.540833  0.564561       -2.372805e-02     0.500000    0.687500                  
  all_steps    caltech101_m/dinov3_patch/whole_image      pure_gmm_2_5 average_precision       10 0.996746  0.996746        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image      pure_gmm_2_5             auroc       10 0.999847  0.999847        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image      pure_gmm_2_5        degenerate       10 0.000000  0.100000       -1.000000e-01     0.100000    1.000000                  
  all_steps          caltech101_m/siglip/whole_image      pure_gmm_2_5              cost       12 0.494730  0.436583        5.814658e-02     0.000000    0.031250                  
  all_steps          caltech101_m/siglip/whole_image      pure_gmm_2_5               fnr       12 0.016441  0.017325       -8.838333e-04     0.083333    1.000000                  
  all_steps          caltech101_m/siglip/whole_image      pure_gmm_2_5               fpr       12 0.478288  0.419258        5.903042e-02     0.000000    0.031250                  
  all_steps          caltech101_m/siglip/whole_image      pure_gmm_2_5            regret       12 0.480008  0.421964        5.804421e-02     0.000000    0.031250                  
  all_steps          caltech101_m/siglip/whole_image      pure_gmm_2_5 average_precision       12 0.985832  0.986030       -1.984167e-04     0.083333    1.000000                  
  all_steps          caltech101_m/siglip/whole_image      pure_gmm_2_5             auroc       12 0.998445  0.998454       -8.500000e-06     0.083333    1.000000                  
  all_steps          caltech101_m/siglip/whole_image      pure_gmm_2_5        degenerate       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          coco_val/dinov3_patch/max_patch      pure_gmm_2_5              cost       14 0.348803  0.334822        1.398082e-02     0.285714    0.248864                  
  all_steps          coco_val/dinov3_patch/max_patch      pure_gmm_2_5               fnr       14 0.089222  0.092750       -3.527964e-03     0.285714    0.600179                  
  all_steps          coco_val/dinov3_patch/max_patch      pure_gmm_2_5               fpr       14 0.259581  0.242073        1.750879e-02     0.428571    0.753152                  
  all_steps          coco_val/dinov3_patch/max_patch      pure_gmm_2_5            regret       14 0.169383  0.139604        2.977879e-02     0.214286    0.064030                  
  all_steps          coco_val/dinov3_patch/max_patch      pure_gmm_2_5 average_precision       14 0.674755  0.670145        4.609429e-03     0.285714    0.575062                  
  all_steps          coco_val/dinov3_patch/max_patch      pure_gmm_2_5             auroc       14 0.936868  0.933957        2.910786e-03     0.142857    0.139414                  
  all_steps          coco_val/dinov3_patch/max_patch      pure_gmm_2_5        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps        coco_val/dinov3_patch/whole_image      pure_gmm_2_5              cost       14 0.633312  0.641723       -8.411393e-03     0.571429    0.669800                  
  all_steps        coco_val/dinov3_patch/whole_image      pure_gmm_2_5               fnr       14 0.116248  0.139480       -2.323225e-02     0.500000    0.066316                  
  all_steps        coco_val/dinov3_patch/whole_image      pure_gmm_2_5               fpr       14 0.517064  0.502243        1.482082e-02     0.500000    0.903198                  
  all_steps        coco_val/dinov3_patch/whole_image      pure_gmm_2_5            regret       14 0.307997  0.282944        2.505200e-02     0.428571    0.501587                  
  all_steps        coco_val/dinov3_patch/whole_image      pure_gmm_2_5 average_precision       14 0.478819  0.451136        2.768329e-02     0.285714    0.168807                  
  all_steps        coco_val/dinov3_patch/whole_image      pure_gmm_2_5             auroc       14 0.859874  0.840654        1.922018e-02     0.285714    0.168807                  
  all_steps        coco_val/dinov3_patch/whole_image      pure_gmm_2_5        degenerate       14 0.035714  0.000000        3.571429e-02     0.000000    0.317311                  
  all_steps              coco_val/siglip/whole_image      pure_gmm_2_5              cost       14 0.521464  0.597451       -7.598738e-02     0.785714    0.002319                  
  all_steps              coco_val/siglip/whole_image      pure_gmm_2_5               fnr       14 0.077431  0.118600       -4.116854e-02     0.642857    0.028056                  
  all_steps              coco_val/siglip/whole_image      pure_gmm_2_5               fpr       14 0.444033  0.478852       -3.481893e-02     0.642857    0.104004                  
  all_steps              coco_val/siglip/whole_image      pure_gmm_2_5            regret       14 0.240200  0.237822        2.378464e-03     0.571429    0.807739                  
  all_steps              coco_val/siglip/whole_image      pure_gmm_2_5 average_precision       14 0.500590  0.476451        2.413939e-02     0.357143    0.325806                  
  all_steps              coco_val/siglip/whole_image      pure_gmm_2_5             auroc       14 0.901479  0.856568        4.491119e-02     0.071429    0.001221                  
  all_steps              coco_val/siglip/whole_image      pure_gmm_2_5        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps   visual_genome_m/dinov3_patch/max_patch      pure_gmm_2_5              cost       16 0.361742  0.380575       -1.883303e-02     0.375000    0.864705                  
  all_steps   visual_genome_m/dinov3_patch/max_patch      pure_gmm_2_5               fnr       16 0.117047  0.120607       -3.560438e-03     0.125000    0.753152                  
  all_steps   visual_genome_m/dinov3_patch/max_patch      pure_gmm_2_5               fpr       16 0.244695  0.259968       -1.527262e-02     0.437500    0.495521                  
  all_steps   visual_genome_m/dinov3_patch/max_patch      pure_gmm_2_5            regret       16 0.119416  0.125436       -6.019594e-03     0.500000    0.334277                  
  all_steps   visual_genome_m/dinov3_patch/max_patch      pure_gmm_2_5 average_precision       16 0.463686  0.460000        3.685250e-03     0.312500    0.483840                  
  all_steps   visual_genome_m/dinov3_patch/max_patch      pure_gmm_2_5             auroc       16 0.907755  0.902759        4.996063e-03     0.250000    0.779435                  
  all_steps   visual_genome_m/dinov3_patch/max_patch      pure_gmm_2_5        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps visual_genome_m/dinov3_patch/whole_image      pure_gmm_2_5              cost       16 0.739388  0.706077        3.331131e-02     0.312500    0.078292                  
  all_steps visual_genome_m/dinov3_patch/whole_image      pure_gmm_2_5               fnr       16 0.134910  0.142183       -7.272375e-03     0.125000    0.865772                  
  all_steps visual_genome_m/dinov3_patch/whole_image      pure_gmm_2_5               fpr       16 0.604478  0.563894        4.058369e-02     0.375000    0.172848                  
  all_steps visual_genome_m/dinov3_patch/whole_image      pure_gmm_2_5            regret       16 0.224368  0.201903        2.246547e-02     0.312500    0.111769                  
  all_steps visual_genome_m/dinov3_patch/whole_image      pure_gmm_2_5 average_precision       16 0.236464  0.246171       -9.706750e-03     0.250000    1.000000                  
  all_steps visual_genome_m/dinov3_patch/whole_image      pure_gmm_2_5             auroc       16 0.762593  0.774598       -1.200431e-02     0.375000    0.326989                  
  all_steps visual_genome_m/dinov3_patch/whole_image      pure_gmm_2_5        degenerate       16 0.031250  0.000000        3.125000e-02     0.000000    0.317311                  
  all_steps       visual_genome_m/siglip/whole_image      pure_gmm_2_5              cost       15 0.677067  0.649584        2.748284e-02     0.533333    0.846924                  
  all_steps       visual_genome_m/siglip/whole_image      pure_gmm_2_5               fnr       15 0.146021  0.148608       -2.587300e-03     0.333333    0.858863                  
  all_steps       visual_genome_m/siglip/whole_image      pure_gmm_2_5               fpr       15 0.531046  0.500976        3.007009e-02     0.400000    0.187622                  
  all_steps       visual_genome_m/siglip/whole_image      pure_gmm_2_5            regret       15 0.202694  0.209193       -6.498133e-03     0.466667    0.803955                  
  all_steps       visual_genome_m/siglip/whole_image      pure_gmm_2_5 average_precision       15 0.249008  0.245513        3.494689e-03     0.400000    0.972125                  
  all_steps       visual_genome_m/siglip/whole_image      pure_gmm_2_5             auroc       15 0.788571  0.794181       -5.609444e-03     0.466667    0.753152                  
  all_steps       visual_genome_m/siglip/whole_image      pure_gmm_2_5        degenerate       15 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image         ramp_6_20              cost       12 0.480107  0.567526       -8.741896e-02     0.416667    0.464844                  
  all_steps    caltech101_m/dinov3_patch/whole_image         ramp_6_20               fnr       12 0.071067  0.072792       -1.724746e-03     0.166667    0.812500                  
  all_steps    caltech101_m/dinov3_patch/whole_image         ramp_6_20               fpr       12 0.409040  0.494734       -8.569421e-02     0.416667    0.519531                  
  all_steps    caltech101_m/dinov3_patch/whole_image         ramp_6_20            regret       12 0.466690  0.557241       -9.055065e-02     0.416667    0.413086                  
  all_steps    caltech101_m/dinov3_patch/whole_image         ramp_6_20 average_precision       12 0.996441  0.997410       -9.696857e-04     0.083333    1.000000                  
  all_steps    caltech101_m/dinov3_patch/whole_image         ramp_6_20             auroc       12 0.997770  0.998460       -6.906167e-04     0.083333    1.000000                  
  all_steps    caltech101_m/dinov3_patch/whole_image         ramp_6_20        degenerate       12 0.057234  0.071581       -1.434676e-02     0.250000    0.250000                  
  all_steps          caltech101_m/siglip/whole_image         ramp_6_20              cost       12 0.076054  0.073398        2.656500e-03     0.333333    0.843750                  
  all_steps          caltech101_m/siglip/whole_image         ramp_6_20               fnr       12 0.002980  0.002980        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image         ramp_6_20               fpr       12 0.073075  0.070418        2.656500e-03     0.333333    0.843750                  
  all_steps          caltech101_m/siglip/whole_image         ramp_6_20            regret       12 0.075272  0.072616        2.656500e-03     0.333333    0.843750                  
  all_steps          caltech101_m/siglip/whole_image         ramp_6_20 average_precision       12 0.999894  0.999894        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image         ramp_6_20             auroc       12 0.999948  0.999948        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image         ramp_6_20        degenerate       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          coco_val/dinov3_patch/max_patch         ramp_6_20              cost       14 0.235657  0.239667       -4.009433e-03     0.357143    0.972125                  
  all_steps          coco_val/dinov3_patch/max_patch         ramp_6_20               fnr       14 0.078653  0.075189        3.463562e-03     0.428571    0.959354                  
  all_steps          coco_val/dinov3_patch/max_patch         ramp_6_20               fpr       14 0.157005  0.164478       -7.472967e-03     0.428571    0.972125                  
  all_steps          coco_val/dinov3_patch/max_patch         ramp_6_20            regret       14 0.087513  0.094337       -6.823643e-03     0.571429    0.278707                  
  all_steps          coco_val/dinov3_patch/max_patch         ramp_6_20 average_precision       14 0.729004  0.729711       -7.069810e-04     0.500000    0.388186                  
  all_steps          coco_val/dinov3_patch/max_patch         ramp_6_20             auroc       14 0.962553  0.962345        2.074619e-04     0.357143    0.875329                  
  all_steps          coco_val/dinov3_patch/max_patch         ramp_6_20        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps        coco_val/dinov3_patch/whole_image         ramp_6_20              cost       14 0.461013  0.462865       -1.852076e-03     0.357143    0.937473                  
  all_steps        coco_val/dinov3_patch/whole_image         ramp_6_20               fnr       14 0.110065  0.115394       -5.328671e-03     0.357143    0.875329                  
  all_steps        coco_val/dinov3_patch/whole_image         ramp_6_20               fpr       14 0.350948  0.347471        3.476486e-03     0.357143    1.000000                  
  all_steps        coco_val/dinov3_patch/whole_image         ramp_6_20            regret       14 0.211744  0.209484        2.260910e-03     0.428571    0.937473                  
  all_steps        coco_val/dinov3_patch/whole_image         ramp_6_20 average_precision       14 0.584652  0.561461        2.319110e-02     0.285714    0.116664                  
  all_steps        coco_val/dinov3_patch/whole_image         ramp_6_20             auroc       14 0.907670  0.906049        1.621086e-03     0.357143    1.000000                  
  all_steps        coco_val/dinov3_patch/whole_image         ramp_6_20        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps              coco_val/siglip/whole_image         ramp_6_20              cost       14 0.376007  0.390860       -1.485343e-02     0.500000    0.903198                  
  all_steps              coco_val/siglip/whole_image         ramp_6_20               fnr       14 0.121172  0.140281       -1.910935e-02     0.571429    0.239317                  
  all_steps              coco_val/siglip/whole_image         ramp_6_20               fpr       14 0.254835  0.250579        4.255895e-03     0.571429    1.000000                  
  all_steps              coco_val/siglip/whole_image         ramp_6_20            regret       14 0.127084  0.120622        6.462657e-03     0.500000    0.714844                  
  all_steps              coco_val/siglip/whole_image         ramp_6_20 average_precision       14 0.588307  0.577026        1.128075e-02     0.500000    0.583008                  
  all_steps              coco_val/siglip/whole_image         ramp_6_20             auroc       14 0.913958  0.901243        1.271451e-02     0.357143    0.463135                  
  all_steps              coco_val/siglip/whole_image         ramp_6_20        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps   visual_genome_m/dinov3_patch/max_patch         ramp_6_20              cost       16 0.301590  0.297126        4.464633e-03     0.437500    0.506746                  
  all_steps   visual_genome_m/dinov3_patch/max_patch         ramp_6_20               fnr       16 0.111259  0.122697       -1.143792e-02     0.500000    0.182314                  
  all_steps   visual_genome_m/dinov3_patch/max_patch         ramp_6_20               fpr       16 0.190331  0.174429        1.590252e-02     0.312500    0.196051                  
  all_steps   visual_genome_m/dinov3_patch/max_patch         ramp_6_20            regret       16 0.074590  0.071427        3.163417e-03     0.437500    0.916512                  
  all_steps   visual_genome_m/dinov3_patch/max_patch         ramp_6_20 average_precision       16 0.476407  0.478019       -1.611396e-03     0.312500    0.929153                  
  all_steps   visual_genome_m/dinov3_patch/max_patch         ramp_6_20             auroc       16 0.914804  0.912141        2.662587e-03     0.187500    0.213223                  
  all_steps   visual_genome_m/dinov3_patch/max_patch         ramp_6_20        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps visual_genome_m/dinov3_patch/whole_image         ramp_6_20              cost       16 0.589748  0.581683        8.064883e-03     0.437500    0.924978                  
  all_steps visual_genome_m/dinov3_patch/whole_image         ramp_6_20               fnr       16 0.236472  0.223605        1.286695e-02     0.250000    0.396726                  
  all_steps visual_genome_m/dinov3_patch/whole_image         ramp_6_20               fpr       16 0.353276  0.358078       -4.802063e-03     0.500000    0.777565                  
  all_steps visual_genome_m/dinov3_patch/whole_image         ramp_6_20            regret       16 0.129488  0.132891       -3.403517e-03     0.437500    0.777565                  
  all_steps visual_genome_m/dinov3_patch/whole_image         ramp_6_20 average_precision       16 0.314061  0.312532        1.529417e-03     0.375000    0.593618                  
  all_steps visual_genome_m/dinov3_patch/whole_image         ramp_6_20             auroc       16 0.789802  0.802692       -1.288975e-02     0.500000    0.362686                  
  all_steps visual_genome_m/dinov3_patch/whole_image         ramp_6_20        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps       visual_genome_m/siglip/whole_image         ramp_6_20              cost       16 0.479799  0.496914       -1.711455e-02     0.562500    0.463745                  
  all_steps       visual_genome_m/siglip/whole_image         ramp_6_20               fnr       16 0.163174  0.174598       -1.142365e-02     0.437500    0.330536                  
  all_steps       visual_genome_m/siglip/whole_image         ramp_6_20               fpr       16 0.316625  0.322316       -5.690863e-03     0.625000    0.375458                  
  all_steps       visual_genome_m/siglip/whole_image         ramp_6_20            regret       16 0.105138  0.117795       -1.265649e-02     0.562500    0.433197                  
  all_steps       visual_genome_m/siglip/whole_image         ramp_6_20 average_precision       16 0.344372  0.341931        2.441296e-03     0.625000    0.899933                  
  all_steps       visual_genome_m/siglip/whole_image         ramp_6_20             auroc       16 0.845169  0.842966        2.203021e-03     0.562500    1.000000                  
  all_steps       visual_genome_m/siglip/whole_image         ramp_6_20        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus              cost       12 0.071762  0.068912        2.850299e-03     0.500000    1.000000                  
  all_steps    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus               fnr       12 0.006241  0.007171       -9.299010e-04     0.250000    0.875000                  
  all_steps    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus               fpr       12 0.065521  0.061741        3.780200e-03     0.500000    1.000000                  
  all_steps    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus            regret       12 0.070426  0.068042        2.383158e-03     0.500000    1.000000                  
  all_steps    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus average_precision       12 0.999737  0.999820       -8.254271e-05     0.166667    0.500000                  
  all_steps    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus             auroc       12 0.999855  0.999899       -4.418333e-05     0.166667    0.500000                  
  all_steps    caltech101_m/dinov3_patch/whole_image post_ramp_21_plus        degenerate       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image post_ramp_21_plus              cost       12 0.014582  0.014283        2.989708e-04     0.166667    1.000000                  
  all_steps          caltech101_m/siglip/whole_image post_ramp_21_plus               fnr       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image post_ramp_21_plus               fpr       12 0.014582  0.014283        2.989708e-04     0.166667    1.000000                  
  all_steps          caltech101_m/siglip/whole_image post_ramp_21_plus            regret       12 0.014582  0.014283        2.989708e-04     0.166667    1.000000                  
  all_steps          caltech101_m/siglip/whole_image post_ramp_21_plus average_precision       12 1.000000  1.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image post_ramp_21_plus             auroc       12 1.000000  1.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          caltech101_m/siglip/whole_image post_ramp_21_plus        degenerate       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          coco_val/dinov3_patch/max_patch post_ramp_21_plus              cost       14 0.185568  0.185748       -1.800000e-04     0.500000    0.875329                  
  all_steps          coco_val/dinov3_patch/max_patch post_ramp_21_plus               fnr       14 0.068365  0.067687        6.774473e-04     0.357143    0.952765                  
  all_steps          coco_val/dinov3_patch/max_patch post_ramp_21_plus               fpr       14 0.117203  0.118060       -8.574241e-04     0.357143    0.753684                  
  all_steps          coco_val/dinov3_patch/max_patch post_ramp_21_plus            regret       14 0.069556  0.074870       -5.313787e-03     0.714286    0.157939                  
  all_steps          coco_val/dinov3_patch/max_patch post_ramp_21_plus average_precision       14 0.790257  0.801354       -1.109677e-02     0.571429    0.388186                  
  all_steps          coco_val/dinov3_patch/max_patch post_ramp_21_plus             auroc       14 0.974093  0.976774       -2.680875e-03     0.642857    0.059739                  
  all_steps          coco_val/dinov3_patch/max_patch post_ramp_21_plus        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps        coco_val/dinov3_patch/whole_image post_ramp_21_plus              cost       14 0.266383  0.264920        1.462983e-03     0.285714    0.426270                  
  all_steps        coco_val/dinov3_patch/whole_image post_ramp_21_plus               fnr       14 0.128068  0.123520        4.547979e-03     0.285714    0.295776                  
  all_steps        coco_val/dinov3_patch/whole_image post_ramp_21_plus               fpr       14 0.138316  0.141401       -3.085050e-03     0.571429    0.903198                  
  all_steps        coco_val/dinov3_patch/whole_image post_ramp_21_plus            regret       14 0.068954  0.075570       -6.615982e-03     0.500000    1.000000                  
  all_steps        coco_val/dinov3_patch/whole_image post_ramp_21_plus average_precision       14 0.688527  0.696676       -8.149005e-03     0.571429    0.153076                  
  all_steps        coco_val/dinov3_patch/whole_image post_ramp_21_plus             auroc       14 0.934493  0.935680       -1.186479e-03     0.500000    0.903198                  
  all_steps        coco_val/dinov3_patch/whole_image post_ramp_21_plus        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps              coco_val/siglip/whole_image post_ramp_21_plus              cost       14 0.241227  0.275053       -3.382553e-02     0.857143    0.006714                  
  all_steps              coco_val/siglip/whole_image post_ramp_21_plus               fnr       14 0.115131  0.145151       -3.002086e-02     0.642857    0.034170                  
  all_steps              coco_val/siglip/whole_image post_ramp_21_plus               fpr       14 0.126096  0.129901       -3.804623e-03     0.428571    0.855225                  
  all_steps              coco_val/siglip/whole_image post_ramp_21_plus            regret       14 0.055976  0.054102        1.873452e-03     0.428571    0.807739                  
  all_steps              coco_val/siglip/whole_image post_ramp_21_plus average_precision       14 0.682203  0.647506        3.469696e-02     0.214286    0.016602                  
  all_steps              coco_val/siglip/whole_image post_ramp_21_plus             auroc       14 0.942561  0.921634        2.092709e-02     0.214286    0.006714                  
  all_steps              coco_val/siglip/whole_image post_ramp_21_plus        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus              cost       16 0.247483  0.254136       -6.652888e-03     0.375000    0.694887                  
  all_steps   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus               fnr       16 0.092265  0.110372       -1.810690e-02     0.562500    0.061884                  
  all_steps   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus               fpr       16 0.155218  0.143764        1.145402e-02     0.312500    0.182338                  
  all_steps   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus            regret       16 0.054843  0.053190        1.653536e-03     0.312500    0.875329                  
  all_steps   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus average_precision       16 0.520422  0.525527       -5.105069e-03     0.375000    0.753684                  
  all_steps   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus             auroc       16 0.923371  0.918574        4.796791e-03     0.375000    0.272095                  
  all_steps   visual_genome_m/dinov3_patch/max_patch post_ramp_21_plus        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus              cost       16 0.450877  0.445566        5.310580e-03     0.437500    0.570061                  
  all_steps visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus               fnr       16 0.297565  0.287381        1.018374e-02     0.312500    0.334277                  
  all_steps visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus               fpr       16 0.153312  0.158185       -4.873150e-03     0.562500    0.690945                  
  all_steps visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus            regret       16 0.056441  0.060987       -4.545989e-03     0.437500    0.864705                  
  all_steps visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus average_precision       16 0.411778  0.387733        2.404522e-02     0.437500    0.733271                  
  all_steps visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus             auroc       16 0.829309  0.831738       -2.428615e-03     0.625000    0.191446                  
  all_steps visual_genome_m/dinov3_patch/whole_image post_ramp_21_plus        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps       visual_genome_m/siglip/whole_image post_ramp_21_plus              cost       16 0.377820  0.383382       -5.562185e-03     0.625000    0.561890                  
  all_steps       visual_genome_m/siglip/whole_image post_ramp_21_plus               fnr       16 0.186046  0.195880       -9.833688e-03     0.437500    0.460302                  
  all_steps       visual_genome_m/siglip/whole_image post_ramp_21_plus               fpr       16 0.191774  0.187502        4.271472e-03     0.437500    0.528168                  
  all_steps       visual_genome_m/siglip/whole_image post_ramp_21_plus            regret       16 0.062477  0.065388       -2.910927e-03     0.437500    0.820892                  
  all_steps       visual_genome_m/siglip/whole_image post_ramp_21_plus average_precision       16 0.421505  0.426363       -4.858449e-03     0.562500    0.433197                  
  all_steps       visual_genome_m/siglip/whole_image post_ramp_21_plus             auroc       16 0.878194  0.876489        1.704994e-03     0.437500    0.561890                  
  all_steps       visual_genome_m/siglip/whole_image post_ramp_21_plus        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps    caltech101_m/dinov3_patch/whole_image         all_steps              cost       12 0.140751  0.149984       -9.233205e-03     0.666667    0.518555                  
  all_steps    caltech101_m/dinov3_patch/whole_image         all_steps               fnr       12 0.015395  0.016086       -6.915559e-04     0.333333    0.687500                  
  all_steps    caltech101_m/dinov3_patch/whole_image         all_steps               fpr       12 0.125356  0.133898       -8.541650e-03     0.666667    0.622070                  
  all_steps    caltech101_m/dinov3_patch/whole_image         all_steps            regret       12 0.137627  0.147907       -1.028019e-02     0.666667    0.518555                  
  all_steps    caltech101_m/dinov3_patch/whole_image         all_steps average_precision       12 0.999199  0.999462       -2.622663e-04     0.166667    0.500000                  
  all_steps    caltech101_m/dinov3_patch/whole_image         all_steps             auroc       12 0.999549  0.999714       -1.654524e-04     0.166667    0.500000                  
  all_steps    caltech101_m/dinov3_patch/whole_image         all_steps        degenerate       12 0.008055  0.010727       -2.671570e-03     0.250000    0.250000                  
  all_steps          caltech101_m/siglip/whole_image         all_steps              cost       12 0.033288  0.031432        1.855617e-03     0.333333    0.425781                  
  all_steps          caltech101_m/siglip/whole_image         all_steps               fnr       12 0.000800  0.000818       -1.822337e-05     0.083333    1.000000                  
  all_steps          caltech101_m/siglip/whole_image         all_steps               fpr       12 0.032488  0.030614        1.873840e-03     0.333333    0.425781                  
  all_steps          caltech101_m/siglip/whole_image         all_steps            regret       12 0.032884  0.031030        1.853506e-03     0.333333    0.425781                  
  all_steps          caltech101_m/siglip/whole_image         all_steps average_precision       12 0.999741  0.999745       -4.091065e-06     0.083333    1.000000                  
  all_steps          caltech101_m/siglip/whole_image         all_steps             auroc       12 0.999962  0.999962       -1.752577e-07     0.083333    1.000000                  
  all_steps          caltech101_m/siglip/whole_image         all_steps        degenerate       12 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps          coco_val/dinov3_patch/max_patch         all_steps              cost       14 0.196679  0.197159       -4.802047e-04     0.428571    1.000000                  
  all_steps          coco_val/dinov3_patch/max_patch         all_steps               fnr       14 0.070386  0.069364        1.021580e-03     0.428571    0.646462                  
  all_steps          coco_val/dinov3_patch/max_patch         all_steps               fpr       14 0.126293  0.127795       -1.501761e-03     0.428571    1.000000                  
  all_steps          coco_val/dinov3_patch/max_patch         all_steps            regret       14 0.074391  0.079215       -4.823712e-03     0.642857    0.135254                  
  all_steps          coco_val/dinov3_patch/max_patch         all_steps average_precision       14 0.778404  0.787570       -9.166261e-03     0.571429    0.480177                  
  all_steps          coco_val/dinov3_patch/max_patch         all_steps             auroc       14 0.971541  0.973660       -2.118933e-03     0.642857    0.084379                  
  all_steps          coco_val/dinov3_patch/max_patch         all_steps        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps        coco_val/dinov3_patch/whole_image         all_steps              cost       14 0.304046  0.303299        7.467496e-04     0.428571    0.583008                  
  all_steps        coco_val/dinov3_patch/whole_image         all_steps               fnr       14 0.125040  0.122592        2.447874e-03     0.285714    0.357544                  
  all_steps        coco_val/dinov3_patch/whole_image         all_steps               fpr       14 0.179006  0.180707       -1.701186e-03     0.500000    0.951538                  
  all_steps        coco_val/dinov3_patch/whole_image         all_steps            regret       14 0.095964  0.100554       -4.590319e-03     0.428571    0.855225                  
  all_steps        coco_val/dinov3_patch/whole_image         all_steps average_precision       14 0.668140  0.670703       -2.563787e-03     0.571429    0.193726                  
  all_steps        coco_val/dinov3_patch/whole_image         all_steps             auroc       14 0.928807  0.929138       -3.315641e-04     0.571429    0.855225                  
  all_steps        coco_val/dinov3_patch/whole_image         all_steps        degenerate       14 0.000736  0.000000        7.363770e-04     0.000000    0.317311                  
  all_steps              coco_val/siglip/whole_image         all_steps              cost       14 0.267993  0.299709       -3.171672e-02     0.857143    0.005249                  
  all_steps              coco_val/siglip/whole_image         all_steps               fnr       14 0.115270  0.143811       -2.854105e-02     0.642857    0.028056                  
  all_steps              coco_val/siglip/whole_image         all_steps               fpr       14 0.152722  0.155898       -3.175645e-03     0.500000    0.714844                  
  all_steps              coco_val/siglip/whole_image         all_steps            regret       14 0.070856  0.068254        2.601967e-03     0.500000    0.390991                  
  all_steps              coco_val/siglip/whole_image         all_steps average_precision       14 0.663856  0.633060        3.079676e-02     0.214286    0.016602                  
  all_steps              coco_val/siglip/whole_image         all_steps             auroc       14 0.937265  0.917122        2.014219e-02     0.214286    0.002319                  
  all_steps              coco_val/siglip/whole_image         all_steps        degenerate       14 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps   visual_genome_m/dinov3_patch/max_patch         all_steps              cost       16 0.258206  0.263391       -5.184820e-03     0.500000    0.820892                  
  all_steps   visual_genome_m/dinov3_patch/max_patch         all_steps               fnr       16 0.095713  0.112489       -1.677569e-02     0.562500    0.041389                  
  all_steps   visual_genome_m/dinov3_patch/max_patch         all_steps               fpr       16 0.162493  0.150902        1.159087e-02     0.375000    0.322510                  
  all_steps   visual_genome_m/dinov3_patch/max_patch         all_steps            regret       16 0.059228  0.057500        1.728814e-03     0.437500    0.860260                  
  all_steps   visual_genome_m/dinov3_patch/max_patch         all_steps average_precision       16 0.512446  0.516830       -4.383566e-03     0.375000    0.875329                  
  all_steps   visual_genome_m/dinov3_patch/max_patch         all_steps             auroc       16 0.921724  0.917253        4.470869e-03     0.312500    0.239317                  
  all_steps   visual_genome_m/dinov3_patch/max_patch         all_steps        degenerate       16 0.000000  0.000000        0.000000e+00     0.000000         NaN all-zero or empty
  all_steps visual_genome_m/dinov3_patch/whole_image         all_steps              cost       16 0.478300  0.471987        6.313838e-03     0.312500    0.348389                  
  all_steps visual_genome_m/dinov3_patch/whole_image         all_steps               fnr       16 0.284764  0.274525        1.023875e-02     0.437500    0.232979                  
  all_steps visual_genome_m/dinov3_patch/whole_image         all_steps               fpr       16 0.193536  0.197461       -3.924903e-03     0.562500    0.705719                  
  all_steps visual_genome_m/dinov3_patch/whole_image         all_steps            regret       16 0.071200  0.075012       -3.812381e-03     0.437500    0.781952                  
  all_steps visual_genome_m/dinov3_patch/whole_image         all_steps average_precision       16 0.393052  0.373185        1.986748e-02     0.437500    0.609235                  
  all_steps visual_genome_m/dinov3_patch/whole_image         all_steps             auroc       16 0.821824  0.826068       -4.243753e-03     0.687500    0.139756                  
  all_steps visual_genome_m/dinov3_patch/whole_image         all_steps        degenerate       16 0.000644  0.000000        6.443299e-04     0.000000    0.317311                  
  all_steps       visual_genome_m/siglip/whole_image         all_steps              cost       16 0.400558  0.407267       -6.709231e-03     0.562500    0.561890                  
  all_steps       visual_genome_m/siglip/whole_image         all_steps               fnr       16 0.182267  0.192093       -9.826013e-03     0.437500    0.668549                  
  all_steps       visual_genome_m/siglip/whole_image         all_steps               fpr       16 0.218291  0.215174        3.116759e-03     0.375000    0.495422                  
  all_steps       visual_genome_m/siglip/whole_image         all_steps            regret       16 0.071815  0.076260       -4.444895e-03     0.625000    0.463745                  
  all_steps       visual_genome_m/siglip/whole_image         all_steps average_precision       16 0.405681  0.409160       -3.479756e-03     0.562500    0.348389                  
  all_steps       visual_genome_m/siglip/whole_image         all_steps             auroc       16 0.870659  0.868988        1.670427e-03     0.500000    0.781952                  
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
      "delta_cost_on_minus_off": 0.0024473995736406405,
      "p": 0.6911760449287577,
      "reading": "safe ON worse (n.s.)"
    },
    "post_ramp_21_plus": {
      "n_cells": 16,
      "delta_cost_on_minus_off": -0.003416443750000005,
      "p": 0.7851079257601976,
      "reading": "safe ON better (n.s.)"
    },
    "all_steps": {
      "n_cells": 16,
      "delta_cost_on_minus_off": -0.0025660413302523396,
      "p": 0.895920578505908,
      "reading": "safe ON better (n.s.)"
    },
    "force_on_for_all_users": false
  },
  "all_steps": {
    "scope": "all_steps",
    "pure_gmm_2_5": {
      "n_cells": 16,
      "delta_cost_on_minus_off": -0.002426104910714286,
      "p": 0.5567842254877393,
      "reading": "safe ON better (n.s.)"
    },
    "ramp_6_20": {
      "n_cells": 16,
      "delta_cost_on_minus_off": 0.00022760000000000488,
      "p": 0.7394354289421676,
      "reading": "safe ON worse (n.s.)"
    },
    "post_ramp_21_plus": {
      "n_cells": 16,
      "delta_cost_on_minus_off": -0.003416443750000005,
      "p": 0.7851079257601976,
      "reading": "safe ON better (n.s.)"
    },
    "all_steps": {
      "n_cells": 16,
      "delta_cost_on_minus_off": -0.002832512472385863,
      "p": 0.9104461669921875,
      "reading": "safe ON better (n.s.)"
    },
    "force_on_for_all_users": false
  }
}
```

Curves: `agg/ab_curves_vs_votes.csv` (1556 rows) · paired cells: `agg/ab_paired_cells.csv` · figures: ab_cost_vs_votes.png, ab_fnr_vs_votes.png, ab_degenerate_vs_votes.png
