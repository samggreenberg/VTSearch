# #4220 precision frames - machine summary

Frames: 5533 (cell x step); arms ['h0.05', 'natural'].

```
    arm scenario estimator      fit reading    X  frames  violation_rate  empty_rate  empty_though_reachable  recall_share_of_oracle  median_achieved
  h0.05     same  foldrank logistic     lcb 0.25    2871           0.048       0.361                   0.356                   0.482            0.521
  h0.05     same  foldrank isotonic     lcb 0.25    2871           0.059       0.299                   0.295                   0.553            0.509
  h0.05     same  foldrank isotonic  lcb+em 0.25    2871           0.081       0.292                   0.288                   0.594            0.488
  h0.05     same  insample logistic  lcb+em 0.25    2871           0.084       0.031                   0.027                   0.861            0.317
  h0.05     same  foldrank logistic  lcb+em 0.25    2871           0.088       0.349                   0.345                   0.548            0.456
  h0.05     same  insample logistic     lcb 0.25    2871           0.092       0.009                   0.007                   0.885            0.315
  h0.05     same  insample isotonic  lcb+em 0.25    2871           0.139       0.039                   0.037                   0.892            0.314
  h0.05     same  insample isotonic     lcb 0.25    2871           0.156       0.002                   0.001                   0.944            0.307
  h0.05     same   foldraw logistic  lcb+em 0.25    2871           0.194       0.251                   0.247                   0.720            0.278
  h0.05     same   foldraw logistic     lcb 0.25    2871           0.194       0.215                   0.211                   0.754            0.278
  h0.05     same   foldraw isotonic     lcb 0.25    2871           0.196       0.227                   0.223                   0.777            0.275
  h0.05     same   foldraw isotonic  lcb+em 0.25    2871           0.237       0.165                   0.161                   0.864            0.273
  h0.05     same  foldrank logistic     lcb 0.50    2871           0.031       0.592                   0.575                   0.316            0.891
  h0.05     same  foldrank logistic  lcb+em 0.50    2871           0.038       0.557                   0.541                   0.338            0.895
  h0.05     same  foldrank isotonic     lcb 0.50    2871           0.046       0.511                   0.496                   0.382            0.873
  h0.05     same  foldrank isotonic  lcb+em 0.50    2871           0.051       0.503                   0.487                   0.391            0.889
  h0.05     same   foldraw logistic  lcb+em 0.50    2871           0.158       0.326                   0.312                   0.712            0.547
  h0.05     same   foldraw logistic     lcb 0.50    2871           0.161       0.308                   0.294                   0.720            0.547
  h0.05     same   foldraw isotonic     lcb 0.50    2871           0.173       0.393                   0.378                   0.667            0.531
  h0.05     same   foldraw isotonic  lcb+em 0.50    2871           0.185       0.377                   0.362                   0.712            0.532
  h0.05     same  insample logistic  lcb+em 0.50    2871           0.196       0.035                   0.026                   1.009            0.565
  h0.05     same  insample isotonic  lcb+em 0.50    2871           0.200       0.061                   0.053                   0.944            0.577
  h0.05     same  insample logistic     lcb 0.50    2871           0.219       0.018                   0.013                   1.051            0.559
  h0.05     same  insample isotonic     lcb 0.50    2871           0.232       0.020                   0.014                   1.033            0.568
  h0.05     same  foldrank logistic     lcb 0.75    2871           0.031       0.700                   0.635                   0.255            1.000
  h0.05     same  foldrank logistic  lcb+em 0.75    2871           0.033       0.690                   0.625                   0.258            1.000
  h0.05     same  foldrank isotonic  lcb+em 0.75    2871           0.051       0.584                   0.521                   0.323            1.000
  h0.05     same  foldrank isotonic     lcb 0.75    2871           0.051       0.606                   0.543                   0.330            1.000
  h0.05     same   foldraw logistic  lcb+em 0.75    2871           0.168       0.405                   0.349                   0.687            0.800
  h0.05     same   foldraw logistic     lcb 0.75    2871           0.169       0.393                   0.337                   0.695            0.799
  h0.05     same   foldraw isotonic  lcb+em 0.75    2871           0.177       0.457                   0.395                   0.680            0.783
  h0.05     same   foldraw isotonic     lcb 0.75    2871           0.179       0.457                   0.396                   0.676            0.782
  h0.05     same  insample isotonic  lcb+em 0.75    2871           0.268       0.124                   0.090                   1.021            0.810
  h0.05     same  insample isotonic     lcb 0.75    2871           0.301       0.088                   0.060                   1.118            0.800
  h0.05     same  insample logistic  lcb+em 0.75    2871           0.366       0.043                   0.026                   1.377            0.782
  h0.05     same  insample logistic     lcb 0.75    2871           0.393       0.028                   0.015                   1.450            0.778
  h0.05     same  foldrank logistic  lcb+em 0.90    2871           0.032       0.756                   0.665                   0.236            1.000
  h0.05     same  foldrank logistic     lcb 0.90    2871           0.033       0.755                   0.664                   0.235            1.000
  h0.05     same  foldrank isotonic  lcb+em 0.90    2871           0.052       0.627                   0.539                   0.317            1.000
  h0.05     same  foldrank isotonic     lcb 0.90    2871           0.054       0.645                   0.557                   0.318            1.000
  h0.05     same   foldraw logistic  lcb+em 0.90    2871           0.159       0.487                   0.403                   0.637            0.940
  h0.05     same   foldraw logistic     lcb 0.90    2871           0.162       0.482                   0.398                   0.643            0.938
  h0.05     same   foldraw isotonic  lcb+em 0.90    2871           0.205       0.476                   0.393                   0.827            0.917
  h0.05     same   foldraw isotonic     lcb 0.90    2871           0.209       0.476                   0.393                   0.833            0.917
  h0.05     same  insample isotonic  lcb+em 0.90    2871           0.311       0.196                   0.144                   1.223            0.933
  h0.05     same  insample isotonic     lcb 0.90    2871           0.326       0.184                   0.132                   1.304            0.927
  h0.05     same  insample logistic  lcb+em 0.90    2871           0.478       0.061                   0.037                   1.901            0.896
  h0.05     same  insample logistic     lcb 0.90    2871           0.509       0.044                   0.024                   2.001            0.887
  h0.05  shifted  foldrank logistic  lcb+em 0.25    2871           0.059       0.763                   0.667                   0.432            0.459
  h0.05  shifted  foldrank isotonic  lcb+em 0.25    2871           0.099       0.599                   0.512                   0.631            0.467
  h0.05  shifted   foldraw logistic  lcb+em 0.25    2871           0.226       0.525                   0.439                   0.834            0.255
  h0.05  shifted  foldrank logistic     lcb 0.25    2871           0.242       0.350                   0.272                   1.124            0.353
  h0.05  shifted  foldrank isotonic     lcb 0.25    2871           0.309       0.289                   0.227                   1.530            0.315
  h0.05  shifted   foldraw isotonic  lcb+em 0.25    2871           0.373       0.415                   0.333                   1.155            0.219
  h0.05  shifted  insample isotonic  lcb+em 0.25    2871           0.442       0.171                   0.138                   1.602            0.236
  h0.05  shifted  insample logistic  lcb+em 0.25    2871           0.622       0.044                   0.029                   2.168            0.196
  h0.05  shifted   foldraw isotonic     lcb 0.25    2871           0.720       0.226                   0.164                   2.160            0.108
  h0.05  shifted   foldraw logistic     lcb 0.25    2871           0.720       0.202                   0.150                   1.966            0.116
  h0.05  shifted  insample logistic     lcb 0.25    2871           0.889       0.001                   0.000                   2.692            0.129
  h0.05  shifted  insample isotonic     lcb 0.25    2871           0.924       0.000                   0.000                   3.032            0.101
  h0.05  shifted  foldrank logistic  lcb+em 0.50    2871           0.047       0.790                   0.614                   0.503            0.853
  h0.05  shifted  foldrank isotonic  lcb+em 0.50    2871           0.086       0.632                   0.468                   0.659            0.836
  h0.05  shifted  foldrank logistic     lcb 0.50    2871           0.117       0.593                   0.426                   0.870            0.743
  h0.05  shifted  foldrank isotonic     lcb 0.50    2871           0.175       0.502                   0.350                   1.100            0.683
  h0.05  shifted   foldraw logistic  lcb+em 0.50    2871           0.217       0.556                   0.400                   0.979            0.500
  h0.05  shifted   foldraw isotonic  lcb+em 0.50    2871           0.340       0.463                   0.301                   1.268            0.446
  h0.05  shifted  insample isotonic  lcb+em 0.50    2871           0.464       0.184                   0.123                   1.822            0.442
  h0.05  shifted   foldraw isotonic     lcb 0.50    2871           0.567       0.385                   0.236                   1.858            0.287
  h0.05  shifted   foldraw logistic     lcb 0.50    2871           0.621       0.293                   0.174                   2.028            0.284
  h0.05  shifted  insample logistic  lcb+em 0.50    2871           0.662       0.052                   0.026                   2.661            0.340
  h0.05  shifted  insample isotonic     lcb 0.50    2871           0.908       0.006                   0.001                   3.316            0.223
  h0.05  shifted  insample logistic     lcb 0.50    2871           0.908       0.005                   0.000                   3.340            0.231
  h0.05  shifted  foldrank logistic  lcb+em 0.75    2871           0.048       0.809                   0.558                   0.468            0.976
  h0.05  shifted  foldrank logistic     lcb 0.75    2871           0.079       0.704                   0.458                   0.691            0.952
  h0.05  shifted  foldrank isotonic  lcb+em 0.75    2871           0.096       0.645                   0.414                   0.608            0.971
  h0.05  shifted  foldrank isotonic     lcb 0.75    2871           0.137       0.599                   0.374                   0.854            0.923
  h0.05  shifted   foldraw logistic  lcb+em 0.75    2871           0.201       0.603                   0.378                   0.932            0.743
  h0.05  shifted   foldraw isotonic  lcb+em 0.75    2871           0.357       0.473                   0.250                   1.464            0.660
  h0.05  shifted  insample isotonic  lcb+em 0.75    2871           0.481       0.204                   0.104                   1.885            0.638
  h0.05  shifted   foldraw isotonic     lcb 0.75    2871           0.496       0.454                   0.234                   1.766            0.508
  h0.05  shifted   foldraw logistic     lcb 0.75    2871           0.534       0.380                   0.193                   1.814            0.522
  h0.05  shifted  insample logistic  lcb+em 0.75    2871           0.697       0.067                   0.028                   2.805            0.467
  h0.05  shifted  insample isotonic     lcb 0.75    2871           0.804       0.061                   0.018                   2.885            0.421
  h0.05  shifted  insample logistic     lcb 0.75    2871           0.918       0.010                   0.002                   3.638            0.333
  h0.05  shifted  foldrank logistic  lcb+em 0.90    2871           0.043       0.827                   0.555                   0.485            1.000
  h0.05  shifted  foldrank logistic     lcb 0.90    2871           0.069       0.752                   0.487                   0.667            1.000
  h0.05  shifted  foldrank isotonic  lcb+em 0.90    2871           0.108       0.653                   0.406                   0.687            1.000
  h0.05  shifted  foldrank isotonic     lcb 0.90    2871           0.137       0.635                   0.392                   0.873            0.974
  h0.05  shifted   foldraw logistic  lcb+em 0.90    2871           0.185       0.657                   0.408                   0.890            0.872
  h0.05  shifted   foldraw isotonic  lcb+em 0.90    2871           0.394       0.476                   0.244                   1.921            0.772
  h0.05  shifted   foldraw logistic     lcb 0.90    2871           0.439       0.470                   0.244                   1.706            0.724
  h0.05  shifted   foldraw isotonic     lcb 0.90    2871           0.476       0.472                   0.240                   2.149            0.676
  h0.05  shifted  insample isotonic  lcb+em 0.90    2871           0.512       0.211                   0.105                   2.276            0.742
  h0.05  shifted  insample isotonic     lcb 0.90    2871           0.694       0.146                   0.063                   2.966            0.600
  h0.05  shifted  insample logistic  lcb+em 0.90    2871           0.720       0.094                   0.044                   3.200            0.577
  h0.05  shifted  insample logistic     lcb 0.90    2871           0.918       0.020                   0.004                   4.254            0.426
natural     same  insample logistic  lcb+em 0.25    2662           0.234       0.195                   0.122                   0.902            0.333
natural     same  insample logistic     lcb 0.25    2662           0.311       0.069                   0.028                   1.017            0.320
natural     same   foldraw logistic  lcb+em 0.25    2662           0.467       0.222                   0.131                   1.658            0.211
natural     same   foldraw logistic     lcb 0.25    2662           0.476       0.207                   0.118                   1.690            0.209
natural     same   foldraw isotonic  lcb+em 0.25    2662           0.512       0.277                   0.171                   1.605            0.168
natural     same  insample isotonic  lcb+em 0.25    2662           0.512       0.161                   0.112                   1.246            0.196
natural     same   foldraw isotonic     lcb 0.25    2662           0.522       0.272                   0.167                   1.621            0.168
natural     same  foldrank logistic     lcb 0.25    2662           0.576       0.285                   0.181                   1.575            0.099
natural     same  foldrank logistic  lcb+em 0.25    2662           0.579       0.285                   0.183                   1.583            0.097
natural     same  foldrank isotonic  lcb+em 0.25    2662           0.579       0.291                   0.184                   1.576            0.094
natural     same  foldrank isotonic     lcb 0.25    2662           0.583       0.281                   0.178                   1.598            0.095
natural     same  insample isotonic     lcb 0.25    2662           0.637       0.017                   0.006                   1.497            0.186
natural     same  insample logistic  lcb+em 0.50    2662           0.259       0.220                   0.112                   0.924            0.595
natural     same  insample isotonic  lcb+em 0.50    2662           0.268       0.269                   0.156                   0.850            0.587
natural     same  insample logistic     lcb 0.50    2662           0.319       0.139                   0.057                   1.001            0.571
natural     same  insample isotonic     lcb 0.50    2662           0.323       0.155                   0.066                   0.947            0.571
natural     same   foldraw isotonic  lcb+em 0.50    2662           0.434       0.350                   0.194                   1.426            0.379
natural     same   foldraw isotonic     lcb 0.50    2662           0.443       0.343                   0.188                   1.472            0.375
natural     same   foldraw logistic  lcb+em 0.50    2662           0.489       0.245                   0.123                   1.762            0.385
natural     same   foldraw logistic     lcb 0.50    2662           0.500       0.234                   0.116                   1.783            0.380
natural     same  foldrank logistic     lcb 0.50    2662           0.522       0.365                   0.212                   1.622            0.177
natural     same  foldrank logistic  lcb+em 0.50    2662           0.523       0.373                   0.220                   1.614            0.172
natural     same  foldrank isotonic  lcb+em 0.50    2662           0.534       0.360                   0.204                   1.598            0.171
natural     same  foldrank isotonic     lcb 0.50    2662           0.538       0.350                   0.196                   1.628            0.172
natural     same  insample isotonic     lcb 0.75    2662           0.061       0.484                   0.264                   0.350            1.000
natural     same  insample isotonic  lcb+em 0.75    2662           0.062       0.524                   0.302                   0.368            1.000
natural     same  insample logistic  lcb+em 0.75    2662           0.260       0.264                   0.100                   0.937            0.846
natural     same  insample logistic     lcb 0.75    2662           0.310       0.206                   0.069                   0.983            0.818
natural     same   foldraw isotonic  lcb+em 0.75    2662           0.366       0.376                   0.173                   1.427            0.667
natural     same   foldraw isotonic     lcb 0.75    2662           0.368       0.373                   0.172                   1.427            0.667
natural     same  foldrank logistic     lcb 0.75    2662           0.470       0.437                   0.233                   1.778            0.253
natural     same  foldrank logistic  lcb+em 0.75    2662           0.471       0.439                   0.236                   1.778            0.253
natural     same   foldraw logistic  lcb+em 0.75    2662           0.507       0.273                   0.113                   1.971            0.549
natural     same  foldrank isotonic  lcb+em 0.75    2662           0.510       0.386                   0.186                   1.839            0.287
natural     same  foldrank isotonic     lcb 0.75    2662           0.511       0.381                   0.181                   1.842            0.287
natural     same   foldraw logistic     lcb 0.75    2662           0.511       0.266                   0.109                   1.985            0.547
natural     same  insample isotonic  lcb+em 0.90    2662           0.026       0.761                   0.510                   0.143            1.000
natural     same  insample isotonic     lcb 0.90    2662           0.027       0.757                   0.506                   0.147            1.000
natural     same  insample logistic  lcb+em 0.90    2662           0.222       0.337                   0.139                   0.828            0.966
natural     same  insample logistic     lcb 0.90    2662           0.244       0.287                   0.105                   0.867            0.971
natural     same   foldraw isotonic     lcb 0.90    2662           0.322       0.390                   0.180                   1.432            0.874
natural     same   foldraw isotonic  lcb+em 0.90    2662           0.325       0.391                   0.181                   1.431            0.875
natural     same  foldrank logistic     lcb 0.90    2662           0.394       0.527                   0.307                   1.865            0.328
natural     same  foldrank logistic  lcb+em 0.90    2662           0.394       0.536                   0.316                   1.891            0.322
natural     same  foldrank isotonic  lcb+em 0.90    2662           0.457       0.405                   0.196                   1.875            0.518
natural     same  foldrank isotonic     lcb 0.90    2662           0.458       0.403                   0.193                   1.873            0.520
natural     same   foldraw logistic  lcb+em 0.90    2662           0.516       0.298                   0.126                   2.164            0.688
natural     same   foldraw logistic     lcb 0.90    2662           0.520       0.292                   0.122                   2.174            0.683
```

## Reference cuts at t=150
```
    arm scenario                  cut   t  median_returned  median_precision  median_recall
  h0.05     same              shipped 150            140.0             0.336          0.958
  h0.05     same            tau_cross 150             67.0             0.638          0.896
  h0.05     same tau_gumbel_priorfree 150             45.0             0.850          0.800
  h0.05     same              tau_mid 150             66.0             0.672          0.898
  h0.05     same        tau_priorfree 150             85.0             0.532          0.926
  h0.05     same             tau_rate 150             85.0             0.532          0.926
  h0.05     same        tau_tail_a040 150             73.0             0.574          0.897
  h0.05     same        tau_tail_a110 150            135.0             0.343          0.960
  h0.05     same        tau_tail_a220 150            243.0             0.197          0.981
  h0.05     same        tau_tail_a400 150            439.0             0.112          1.000
  h0.05  shifted              shipped 150           1147.0             0.042          0.958
  h0.05  shifted            tau_cross 150            333.0             0.128          0.896
  h0.05  shifted tau_gumbel_priorfree 150            105.5             0.300          0.800
  h0.05  shifted              tau_mid 150            290.0             0.151          0.898
  h0.05  shifted        tau_priorfree 150            518.0             0.085          0.926
  h0.05  shifted             tau_rate 150            518.0             0.085          0.926
  h0.05  shifted        tau_tail_a040 150            408.0             0.101          0.897
  h0.05  shifted        tau_tail_a110 150           1122.0             0.041          0.960
  h0.05  shifted        tau_tail_a220 150           2395.5             0.020          0.981
  h0.05  shifted        tau_tail_a400 150           4712.0             0.010          1.000
natural     same              shipped 150           2284.5             0.020          0.961
natural     same            tau_cross 150           1451.0             0.029          0.932
natural     same tau_gumbel_priorfree 150           2810.0             0.016          0.968
natural     same              tau_mid 150           2611.0             0.018          0.960
natural     same        tau_priorfree 150           2311.0             0.020          0.962
natural     same             tau_rate 150           2311.0             0.020          0.962
natural     same        tau_tail_a040 150            475.0             0.059          0.830
natural     same        tau_tail_a110 150           1253.5             0.031          0.921
natural     same        tau_tail_a220 150           2541.0             0.017          0.963
natural     same        tau_tail_a400 150           4867.0             0.010          0.981
```
