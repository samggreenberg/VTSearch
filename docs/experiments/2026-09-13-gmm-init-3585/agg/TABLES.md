# #3585 gate tables

## gate_by_arm

| arm               | kind | cases | changed | changed_pct | rate_reportable | prov_changed | d_thr_median | d_thr_p90 | d_thr_max | d_adm_median_of_changed | d_adm_max | d_adm_frac_max | speedup_median |
|-------------------|------|-------|---------|-------------|-----------------|--------------|--------------|-----------|-----------|-------------------------|-----------|----------------|----------------|
| native            | fold | 2258  | 158     | 7           | yes             | 3            | 0            | 0         | 0.197     | 2                       | 365       | 0.799          | 1.45           |
| native_ll1e-4     | fold | 2258  | 234     | 10.4        | yes             | 10           | 0            | 1.43e-05  | 0.197     | 2                       | 471       | 0.799          | 1.39           |
| native_ll1e-5     | fold | 2258  | 285     | 12.6        | yes             | 15           | 0            | 8.19e-05  | 0.197     | 4                       | 486       | 0.799          | 1.29           |
| native_param1e-8  | fold | 2258  | 325     | 14.4        | yes             | 17           | 0            | 0.000197  | 0.197     | 9                       | 486       | 0.799          | 0.754          |
| native_iter50     | fold | 2258  | 301     | 13.3        | yes             | 13           | 0            | 9.77e-05  | 0.197     | 4                       | 486       | 0.799          | 1.15           |
| sklearn_kmeanspp  | fold | 2258  | 364     | 16.1        | yes             | 41           | 0            | 0.000396  | 0.167     | 11.5                    | 1372      | 0.684          | 0.947          |
| sklearn_spherical | fold | 2258  | 0       | 0           | yes             | 0            | 0            | 0         | 0         |                         | 0         | 0              | 1.04           |
| native_10k        | fold | 2258  | 158     | 7           | yes             | 3            | 0            | 0         | 0.197     | 2                       | 365       | 0.799          | 1.45           |
| native            | sort | 195   | 119     | 61          | yes             | 0            | 8.17e-05     | 0.00156   | 0.141     | 10                      | 872       | 0.416          | 6.4            |
| native_ll1e-4     | sort | 195   | 160     | 82.1        | yes             | 0            | 0.00418      | 0.0179    | 0.127     | 40                      | 1201      | 0.412          | 3.94           |
| native_ll1e-5     | sort | 195   | 164     | 84.1        | yes             | 0            | 0.00775      | 0.0224    | 0.115     | 117                     | 1551      | 0.404          | 2.53           |
| native_param1e-8  | sort | 195   | 164     | 84.1        | yes             | 0            | 0.00986      | 0.0279    | 0.109     | 215                     | 4797      | 0.427          | 0.602          |
| native_iter50     | sort | 195   | 164     | 84.1        | yes             | 0            | 0.00771      | 0.0203    | 0.109     | 97                      | 1011      | 0.4            | 1.48           |
| sklearn_kmeanspp  | sort | 195   | 156     | 80          | yes             | 0            | 0.00176      | 0.021     | 0.052     | 78.5                    | 3694      | 0.375          | 0.913          |
| sklearn_spherical | sort | 195   | 0       | 0           | yes             | 0            | 0            | 5.97e-16  | 7.66e-15  |                         | 0         | 0              | 1.12           |
| native_10k        | sort | 195   | 119     | 61          | yes             | 0            | 9.03e-05     | 0.00156   | 0.141     | 9                       | 872       | 0.416          | 6.4            |

## gate_by_env

| arm               | kind | dataset         | embedder     | style       | cases | changed | d_thr_p90 | changed_pct |
|-------------------|------|-----------------|--------------|-------------|-------|---------|-----------|-------------|
| native            | fold | visual_genome_m | siglip       | whole_image | 319   | 45      | 6.03e-05  | 14.1        |
| native_ll1e-4     | fold | visual_genome_m | siglip       | whole_image | 319   | 68      | 0.000165  | 21.3        |
| native_ll1e-5     | fold | visual_genome_m | siglip       | whole_image | 319   | 84      | 0.000341  | 26.3        |
| native_param1e-8  | fold | visual_genome_m | siglip       | whole_image | 319   | 97      | 0.00127   | 30.4        |
| native_iter50     | fold | visual_genome_m | siglip       | whole_image | 319   | 92      | 0.00037   | 28.8        |
| sklearn_kmeanspp  | fold | visual_genome_m | siglip       | whole_image | 319   | 97      | 0.00182   | 30.4        |
| sklearn_spherical | fold | visual_genome_m | siglip       | whole_image | 319   | 0       | 0         | 0           |
| native_10k        | fold | visual_genome_m | siglip       | whole_image | 319   | 45      | 6.03e-05  | 14.1        |
| native            | sort | visual_genome_m | siglip       | whole_image | 16    | 15      | 0.000697  | 93.8        |
| native_ll1e-4     | sort | visual_genome_m | siglip       | whole_image | 16    | 15      | 0.0118    | 93.8        |
| native_ll1e-5     | sort | visual_genome_m | siglip       | whole_image | 16    | 16      | 0.022     | 100         |
| native_param1e-8  | sort | visual_genome_m | siglip       | whole_image | 16    | 16      | 0.0347    | 100         |
| native_iter50     | sort | visual_genome_m | siglip       | whole_image | 16    | 16      | 0.0104    | 100         |
| sklearn_kmeanspp  | sort | visual_genome_m | siglip       | whole_image | 16    | 16      | 0.0397    | 100         |
| sklearn_spherical | sort | visual_genome_m | siglip       | whole_image | 16    | 0       | 4.11e-15  | 0           |
| native_10k        | sort | visual_genome_m | siglip       | whole_image | 16    | 15      | 0.000697  | 93.8        |
| native            | fold | visual_genome_m | dinov3_patch | max_patch   | 317   | 3       | 0         | 0.946       |
| native_ll1e-4     | fold | visual_genome_m | dinov3_patch | max_patch   | 317   | 6       | 0         | 1.89        |
| native_ll1e-5     | fold | visual_genome_m | dinov3_patch | max_patch   | 317   | 8       | 0         | 2.52        |
| native_param1e-8  | fold | visual_genome_m | dinov3_patch | max_patch   | 317   | 12      | 0         | 3.79        |
| native_iter50     | fold | visual_genome_m | dinov3_patch | max_patch   | 317   | 9       | 0         | 2.84        |
| sklearn_kmeanspp  | fold | visual_genome_m | dinov3_patch | max_patch   | 317   | 19      | 0         | 5.99        |
| sklearn_spherical | fold | visual_genome_m | dinov3_patch | max_patch   | 317   | 0       | 0         | 0           |
| native_10k        | fold | visual_genome_m | dinov3_patch | max_patch   | 317   | 3       | 0         | 0.946       |
| native            | fold | visual_genome_m | dinov3_patch | whole_image | 317   | 48      | 4.69e-05  | 15.1        |
| native_ll1e-4     | fold | visual_genome_m | dinov3_patch | whole_image | 317   | 77      | 0.000163  | 24.3        |
| native_ll1e-5     | fold | visual_genome_m | dinov3_patch | whole_image | 317   | 91      | 0.000358  | 28.7        |
| native_param1e-8  | fold | visual_genome_m | dinov3_patch | whole_image | 317   | 101     | 0.000892  | 31.9        |
| native_iter50     | fold | visual_genome_m | dinov3_patch | whole_image | 317   | 97      | 0.00031   | 30.6        |
| sklearn_kmeanspp  | fold | visual_genome_m | dinov3_patch | whole_image | 317   | 129     | 0.00255   | 40.7        |
| sklearn_spherical | fold | visual_genome_m | dinov3_patch | whole_image | 317   | 0       | 0         | 0           |
| native_10k        | fold | visual_genome_m | dinov3_patch | whole_image | 317   | 48      | 4.69e-05  | 15.1        |
| native            | sort | visual_genome_m | dinov3_patch | max_patch   | 17    | 9       | 0.0161    | 52.9        |
| native_ll1e-4     | sort | visual_genome_m | dinov3_patch | max_patch   | 17    | 17      | 0.0241    | 100         |
| native_ll1e-5     | sort | visual_genome_m | dinov3_patch | max_patch   | 17    | 17      | 0.0269    | 100         |
| native_param1e-8  | sort | visual_genome_m | dinov3_patch | max_patch   | 17    | 17      | 0.0302    | 100         |
| native_iter50     | sort | visual_genome_m | dinov3_patch | max_patch   | 17    | 17      | 0.0284    | 100         |
| sklearn_kmeanspp  | sort | visual_genome_m | dinov3_patch | max_patch   | 17    | 16      | 0.0273    | 94.1        |
| sklearn_spherical | sort | visual_genome_m | dinov3_patch | max_patch   | 17    | 0       | 2.05e-16  | 0           |
| native_10k        | sort | visual_genome_m | dinov3_patch | max_patch   | 17    | 9       | 0.0161    | 52.9        |
| native            | sort | visual_genome_m | dinov3_patch | whole_image | 17    | 9       | 0.0161    | 52.9        |
| native_ll1e-4     | sort | visual_genome_m | dinov3_patch | whole_image | 17    | 17      | 0.0241    | 100         |
| native_ll1e-5     | sort | visual_genome_m | dinov3_patch | whole_image | 17    | 17      | 0.0269    | 100         |
| native_param1e-8  | sort | visual_genome_m | dinov3_patch | whole_image | 17    | 17      | 0.0302    | 100         |
| native_iter50     | sort | visual_genome_m | dinov3_patch | whole_image | 17    | 17      | 0.0284    | 100         |
| sklearn_kmeanspp  | sort | visual_genome_m | dinov3_patch | whole_image | 17    | 16      | 0.0273    | 94.1        |
| sklearn_spherical | sort | visual_genome_m | dinov3_patch | whole_image | 17    | 0       | 2.05e-16  | 0           |
| native_10k        | sort | visual_genome_m | dinov3_patch | whole_image | 17    | 9       | 0.0161    | 52.9        |
| native            | fold | caltech101_m    | siglip       | whole_image | 239   | 4       | 0         | 1.67        |
| native_ll1e-4     | fold | caltech101_m    | siglip       | whole_image | 239   | 5       | 0         | 2.09        |
| native_ll1e-5     | fold | caltech101_m    | siglip       | whole_image | 239   | 7       | 0         | 2.93        |
| native_param1e-8  | fold | caltech101_m    | siglip       | whole_image | 239   | 7       | 0         | 2.93        |
| native_iter50     | fold | caltech101_m    | siglip       | whole_image | 239   | 7       | 0         | 2.93        |
| sklearn_kmeanspp  | fold | caltech101_m    | siglip       | whole_image | 239   | 10      | 0         | 4.18        |
| sklearn_spherical | fold | caltech101_m    | siglip       | whole_image | 239   | 0       | 0         | 0           |
| native_10k        | fold | caltech101_m    | siglip       | whole_image | 239   | 4       | 0         | 1.67        |
| native            | sort | caltech101_m    | siglip       | whole_image | 13    | 4       | 0.0297    | 30.8        |
| native_ll1e-4     | sort | caltech101_m    | siglip       | whole_image | 13    | 8       | 0.0352    | 61.5        |
| native_ll1e-5     | sort | caltech101_m    | siglip       | whole_image | 13    | 8       | 0.0352    | 61.5        |
| native_param1e-8  | sort | caltech101_m    | siglip       | whole_image | 13    | 8       | 0.0352    | 61.5        |
| native_iter50     | sort | caltech101_m    | siglip       | whole_image | 13    | 8       | 0.0352    | 61.5        |
| sklearn_kmeanspp  | sort | caltech101_m    | siglip       | whole_image | 13    | 11      | 0.0421    | 84.6        |
| sklearn_spherical | sort | caltech101_m    | siglip       | whole_image | 13    | 0       | 4.97e-16  | 0           |
| native_10k        | sort | caltech101_m    | siglip       | whole_image | 13    | 4       | 0.0297    | 30.8        |
| native            | fold | caltech101_m    | dinov3_patch | whole_image | 228   | 27      | 0.0458    | 11.8        |
| native_ll1e-4     | fold | caltech101_m    | dinov3_patch | whole_image | 228   | 29      | 0.0458    | 12.7        |
| native_ll1e-5     | fold | caltech101_m    | dinov3_patch | whole_image | 228   | 31      | 0.0458    | 13.6        |
| native_param1e-8  | fold | caltech101_m    | dinov3_patch | whole_image | 228   | 31      | 0.0458    | 13.6        |
| native_iter50     | fold | caltech101_m    | dinov3_patch | whole_image | 228   | 31      | 0.0458    | 13.6        |
| sklearn_kmeanspp  | fold | caltech101_m    | dinov3_patch | whole_image | 228   | 19      | 0         | 8.33        |
| sklearn_spherical | fold | caltech101_m    | dinov3_patch | whole_image | 228   | 0       | 0         | 0           |
| native_10k        | fold | caltech101_m    | dinov3_patch | whole_image | 228   | 27      | 0.0458    | 11.8        |
| native            | sort | caltech101_m    | dinov3_patch | whole_image | 26    | 0       | 5.14e-05  | 0           |
| native_ll1e-4     | sort | caltech101_m    | dinov3_patch | whole_image | 26    | 3       | 0.00555   | 11.5        |
| native_ll1e-5     | sort | caltech101_m    | dinov3_patch | whole_image | 26    | 5       | 0.0101    | 19.2        |
| native_param1e-8  | sort | caltech101_m    | dinov3_patch | whole_image | 26    | 5       | 0.0117    | 19.2        |
| native_iter50     | sort | caltech101_m    | dinov3_patch | whole_image | 26    | 5       | 0.0117    | 19.2        |
| sklearn_kmeanspp  | sort | caltech101_m    | dinov3_patch | whole_image | 26    | 0       | 0.000581  | 0           |
| sklearn_spherical | sort | caltech101_m    | dinov3_patch | whole_image | 26    | 0       | 2.64e-16  | 0           |
| native_10k        | sort | caltech101_m    | dinov3_patch | whole_image | 26    | 0       | 5.14e-05  | 0           |
| native            | fold | coco_val        | siglip       | whole_image | 280   | 18      | 0         | 6.43        |
| native_ll1e-4     | fold | coco_val        | siglip       | whole_image | 280   | 23      | 0         | 8.21        |
| native_ll1e-5     | fold | coco_val        | siglip       | whole_image | 280   | 28      | 1.59e-06  | 10          |
| native_param1e-8  | fold | coco_val        | siglip       | whole_image | 280   | 36      | 7.3e-05   | 12.9        |
| native_iter50     | fold | coco_val        | siglip       | whole_image | 280   | 30      | 2.09e-05  | 10.7        |
| sklearn_kmeanspp  | fold | coco_val        | siglip       | whole_image | 280   | 34      | 0.000116  | 12.1        |
| sklearn_spherical | fold | coco_val        | siglip       | whole_image | 280   | 0       | 0         | 0           |
| native_10k        | fold | coco_val        | siglip       | whole_image | 280   | 18      | 0         | 6.43        |
| native            | sort | coco_val        | siglip       | whole_image | 14    | 14      | 0.00623   | 100         |
| native_ll1e-4     | sort | coco_val        | siglip       | whole_image | 14    | 14      | 0.0156    | 100         |
| native_ll1e-5     | sort | coco_val        | siglip       | whole_image | 14    | 14      | 0.0175    | 100         |
| native_param1e-8  | sort | coco_val        | siglip       | whole_image | 14    | 14      | 0.0203    | 100         |
| native_iter50     | sort | coco_val        | siglip       | whole_image | 14    | 14      | 0.017     | 100         |
| sklearn_kmeanspp  | sort | coco_val        | siglip       | whole_image | 14    | 14      | 0.0152    | 100         |
| sklearn_spherical | sort | coco_val        | siglip       | whole_image | 14    | 0       | 1.69e-16  | 0           |
| native_10k        | sort | coco_val        | siglip       | whole_image | 14    | 14      | 0.00623   | 100         |
| native            | fold | coco_val        | dinov3_patch | max_patch   | 279   | 5       | 0         | 1.79        |
| native_ll1e-4     | fold | coco_val        | dinov3_patch | max_patch   | 279   | 8       | 0         | 2.87        |
| native_ll1e-5     | fold | coco_val        | dinov3_patch | max_patch   | 279   | 12      | 0         | 4.3         |
| native_param1e-8  | fold | coco_val        | dinov3_patch | max_patch   | 279   | 13      | 0         | 4.66        |
| native_iter50     | fold | coco_val        | dinov3_patch | max_patch   | 279   | 12      | 0         | 4.3         |
| sklearn_kmeanspp  | fold | coco_val        | dinov3_patch | max_patch   | 279   | 14      | 0         | 5.02        |
| sklearn_spherical | fold | coco_val        | dinov3_patch | max_patch   | 279   | 0       | 0         | 0           |
| native_10k        | fold | coco_val        | dinov3_patch | max_patch   | 279   | 5       | 0         | 1.79        |
| native            | fold | coco_val        | dinov3_patch | whole_image | 279   | 8       | 0         | 2.87        |
| native_ll1e-4     | fold | coco_val        | dinov3_patch | whole_image | 279   | 18      | 0         | 6.45        |
| native_ll1e-5     | fold | coco_val        | dinov3_patch | whole_image | 279   | 24      | 0         | 8.6         |
| native_param1e-8  | fold | coco_val        | dinov3_patch | whole_image | 279   | 28      | 3.82e-06  | 10          |
| native_iter50     | fold | coco_val        | dinov3_patch | whole_image | 279   | 23      | 0         | 8.24        |
| sklearn_kmeanspp  | fold | coco_val        | dinov3_patch | whole_image | 279   | 42      | 0.000132  | 15.1        |
| sklearn_spherical | fold | coco_val        | dinov3_patch | whole_image | 279   | 0       | 0         | 0           |
| native_10k        | fold | coco_val        | dinov3_patch | whole_image | 279   | 8       | 0         | 2.87        |
| native            | sort | coco_val        | dinov3_patch | max_patch   | 15    | 10      | 0.0041    | 66.7        |
| native_ll1e-4     | sort | coco_val        | dinov3_patch | max_patch   | 15    | 15      | 0.0171    | 100         |
| native_ll1e-5     | sort | coco_val        | dinov3_patch | max_patch   | 15    | 15      | 0.022     | 100         |
| native_param1e-8  | sort | coco_val        | dinov3_patch | max_patch   | 15    | 15      | 0.0244    | 100         |
| native_iter50     | sort | coco_val        | dinov3_patch | max_patch   | 15    | 15      | 0.024     | 100         |
| sklearn_kmeanspp  | sort | coco_val        | dinov3_patch | max_patch   | 15    | 13      | 0.0206    | 86.7        |
| sklearn_spherical | sort | coco_val        | dinov3_patch | max_patch   | 15    | 0       | 2.03e-16  | 0           |
| native_10k        | sort | coco_val        | dinov3_patch | max_patch   | 15    | 10      | 0.0041    | 66.7        |
| native            | sort | coco_val        | dinov3_patch | whole_image | 15    | 10      | 0.0041    | 66.7        |
| native_ll1e-4     | sort | coco_val        | dinov3_patch | whole_image | 15    | 15      | 0.0171    | 100         |
| native_ll1e-5     | sort | coco_val        | dinov3_patch | whole_image | 15    | 15      | 0.022     | 100         |
| native_param1e-8  | sort | coco_val        | dinov3_patch | whole_image | 15    | 15      | 0.0244    | 100         |
| native_iter50     | sort | coco_val        | dinov3_patch | whole_image | 15    | 15      | 0.024     | 100         |
| sklearn_kmeanspp  | sort | coco_val        | dinov3_patch | whole_image | 15    | 13      | 0.0206    | 86.7        |
| sklearn_spherical | sort | coco_val        | dinov3_patch | whole_image | 15    | 0       | 2.03e-16  | 0           |
| native_10k        | sort | coco_val        | dinov3_patch | whole_image | 15    | 10      | 0.0041    | 66.7        |
| native            | sort | caltech101_m    | siglip       | text_sort   | 6     | 2       | 0.0187    | 33.3        |
| native_ll1e-4     | sort | caltech101_m    | siglip       | text_sort   | 6     | 4       | 0.0323    | 66.7        |
| native_ll1e-5     | sort | caltech101_m    | siglip       | text_sort   | 6     | 4       | 0.0323    | 66.7        |
| native_param1e-8  | sort | caltech101_m    | siglip       | text_sort   | 6     | 4       | 0.0323    | 66.7        |
| native_iter50     | sort | caltech101_m    | siglip       | text_sort   | 6     | 4       | 0.0323    | 66.7        |
| sklearn_kmeanspp  | sort | caltech101_m    | siglip       | text_sort   | 6     | 5       | 0.0362    | 83.3        |
| sklearn_spherical | sort | caltech101_m    | siglip       | text_sort   | 6     | 0       | 3.47e-16  | 0           |
| native_10k        | sort | caltech101_m    | siglip       | text_sort   | 6     | 2       | 0.0187    | 33.3        |
| native            | sort | caltech101_m    | siglip2_l    | text_sort   | 6     | 0       | 3.79e-05  | 0           |
| native_ll1e-4     | sort | caltech101_m    | siglip2_l    | text_sort   | 6     | 3       | 0.0221    | 50          |
| native_ll1e-5     | sort | caltech101_m    | siglip2_l    | text_sort   | 6     | 3       | 0.0226    | 50          |
| native_param1e-8  | sort | caltech101_m    | siglip2_l    | text_sort   | 6     | 3       | 0.0229    | 50          |
| native_iter50     | sort | caltech101_m    | siglip2_l    | text_sort   | 6     | 3       | 0.0229    | 50          |
| sklearn_kmeanspp  | sort | caltech101_m    | siglip2_l    | text_sort   | 6     | 3       | 0.00999   | 50          |
| sklearn_spherical | sort | caltech101_m    | siglip2_l    | text_sort   | 6     | 0       | 1.53e-16  | 0           |
| native_10k        | sort | caltech101_m    | siglip2_l    | text_sort   | 6     | 0       | 3.79e-05  | 0           |
| native            | sort | visual_genome_m | siglip       | text_sort   | 8     | 8       | 0.000113  | 100         |
| native_ll1e-4     | sort | visual_genome_m | siglip       | text_sort   | 8     | 8       | 0.0112    | 100         |
| native_ll1e-5     | sort | visual_genome_m | siglip       | text_sort   | 8     | 8       | 0.0232    | 100         |
| native_param1e-8  | sort | visual_genome_m | siglip       | text_sort   | 8     | 8       | 0.0315    | 100         |
| native_iter50     | sort | visual_genome_m | siglip       | text_sort   | 8     | 8       | 0.0115    | 100         |
| sklearn_kmeanspp  | sort | visual_genome_m | siglip       | text_sort   | 8     | 8       | 0.00592   | 100         |
| sklearn_spherical | sort | visual_genome_m | siglip       | text_sort   | 8     | 0       | 9.82e-17  | 0           |
| native_10k        | sort | visual_genome_m | siglip       | text_sort   | 8     | 8       | 0.000113  | 100         |
| native            | sort | visual_genome_m | siglip2_l    | text_sort   | 8     | 7       | 0.000304  | 87.5        |
| native_ll1e-4     | sort | visual_genome_m | siglip2_l    | text_sort   | 8     | 8       | 0.00733   | 100         |
| native_ll1e-5     | sort | visual_genome_m | siglip2_l    | text_sort   | 8     | 8       | 0.0105    | 100         |
| native_param1e-8  | sort | visual_genome_m | siglip2_l    | text_sort   | 8     | 8       | 0.0132    | 100         |
| native_iter50     | sort | visual_genome_m | siglip2_l    | text_sort   | 8     | 8       | 0.0106    | 100         |
| sklearn_kmeanspp  | sort | visual_genome_m | siglip2_l    | text_sort   | 8     | 8       | 0.0185    | 100         |
| sklearn_spherical | sort | visual_genome_m | siglip2_l    | text_sort   | 8     | 0       | 0         | 0           |
| native_10k        | sort | visual_genome_m | siglip2_l    | text_sort   | 8     | 7       | 0.000304  | 87.5        |
| native            | sort | coco_val        | siglip       | text_sort   | 7     | 7       | 0.00368   | 100         |
| native_ll1e-4     | sort | coco_val        | siglip       | text_sort   | 7     | 7       | 0.0133    | 100         |
| native_ll1e-5     | sort | coco_val        | siglip       | text_sort   | 7     | 7       | 0.0167    | 100         |
| native_param1e-8  | sort | coco_val        | siglip       | text_sort   | 7     | 7       | 0.0195    | 100         |
| native_iter50     | sort | coco_val        | siglip       | text_sort   | 7     | 7       | 0.0156    | 100         |
| sklearn_kmeanspp  | sort | coco_val        | siglip       | text_sort   | 7     | 7       | 0.0111    | 100         |
| sklearn_spherical | sort | coco_val        | siglip       | text_sort   | 7     | 0       | 1.39e-16  | 0           |
| native_10k        | sort | coco_val        | siglip       | text_sort   | 7     | 7       | 0.00368   | 100         |
| native            | sort | coco_val        | siglip2_l    | text_sort   | 7     | 5       | 0.000239  | 71.4        |
| native_ll1e-4     | sort | coco_val        | siglip2_l    | text_sort   | 7     | 6       | 0.00949   | 85.7        |
| native_ll1e-5     | sort | coco_val        | siglip2_l    | text_sort   | 7     | 7       | 0.0196    | 100         |
| native_param1e-8  | sort | coco_val        | siglip2_l    | text_sort   | 7     | 7       | 0.0284    | 100         |
| native_iter50     | sort | coco_val        | siglip2_l    | text_sort   | 7     | 7       | 0.0154    | 100         |
| sklearn_kmeanspp  | sort | coco_val        | siglip2_l    | text_sort   | 7     | 6       | 0.0065    | 85.7        |
| sklearn_spherical | sort | coco_val        | siglip2_l    | text_sort   | 7     | 0       | 3.89e-17  | 0           |
| native_10k        | sort | coco_val        | siglip2_l    | text_sort   | 7     | 5       | 0.000239  | 71.4        |
| native            | sort | vg_scale        | clip         | text_sort   | 10    | 9       | 0.00013   | 90          |
| native_ll1e-4     | sort | vg_scale        | clip         | text_sort   | 10    | 10      | 0.000386  | 100         |
| native_ll1e-5     | sort | vg_scale        | clip         | text_sort   | 10    | 10      | 0.00278   | 100         |
| native_param1e-8  | sort | vg_scale        | clip         | text_sort   | 10    | 10      | 0.00833   | 100         |
| native_iter50     | sort | vg_scale        | clip         | text_sort   | 10    | 10      | 0.000744  | 100         |
| sklearn_kmeanspp  | sort | vg_scale        | clip         | text_sort   | 10    | 10      | 0.006     | 100         |
| sklearn_spherical | sort | vg_scale        | clip         | text_sort   | 10    | 0       | 3.78e-15  | 0           |
| native_10k        | sort | vg_scale        | clip         | text_sort   | 10    | 9       | 0.000361  | 90          |
| native            | sort | vg_scale        | siglip       | text_sort   | 10    | 10      | 0.000208  | 100         |
| native_ll1e-4     | sort | vg_scale        | siglip       | text_sort   | 10    | 10      | 0.00136   | 100         |
| native_ll1e-5     | sort | vg_scale        | siglip       | text_sort   | 10    | 10      | 0.00434   | 100         |
| native_param1e-8  | sort | vg_scale        | siglip       | text_sort   | 10    | 10      | 0.00817   | 100         |
| native_iter50     | sort | vg_scale        | siglip       | text_sort   | 10    | 10      | 0.00182   | 100         |
| sklearn_kmeanspp  | sort | vg_scale        | siglip       | text_sort   | 10    | 10      | 0.0177    | 100         |
| sklearn_spherical | sort | vg_scale        | siglip       | text_sort   | 10    | 0       | 1.21e-16  | 0           |
| native_10k        | sort | vg_scale        | siglip       | text_sort   | 10    | 10      | 0.000418  | 100         |

## fits_by_arm

| arm               | kind | samples | better | tied | worse | failed_here_only | d_loglik_median | d_loglik_min | speedup_median | seconds_median | base_seconds_median |
|-------------------|------|---------|--------|------|-------|------------------|-----------------|--------------|----------------|----------------|---------------------|
| native            | fold | 4516    | 2369   | 1    | 2146  | 0                | 5.36e-07        | -0.214       | 7.18           | 0.00171        | 0.0131              |
| native_ll1e-4     | fold | 4516    | 4442   | 0    | 74    | 0                | 0.00139         | -0.214       | 4.06           | 0.00318        | 0.0131              |
| native_ll1e-5     | fold | 4516    | 4488   | 0    | 28    | 0                | 0.00181         | -0.214       | 2.34           | 0.00619        | 0.0131              |
| native_param1e-8  | fold | 4516    | 4501   | 0    | 15    | 0                | 0.00194         | -0.214       | 0.429          | 0.0341         | 0.0131              |
| native_iter50     | fold | 4516    | 4500   | 0    | 16    | 0                | 0.00183         | -0.214       | 1.31           | 0.0115         | 0.0131              |
| sklearn_kmeanspp  | fold | 4516    | 1651   | 631  | 2234  | 0                | 0               | -0.269       | 0.937          | 0.0135         | 0.0131              |
| sklearn_spherical | fold | 4516    | 0      | 4516 | 0     | 0                | 0               | 0            | 1.13           | 0.0116         | 0.0131              |
| native_10k        | fold | 4515    | 2368   | 1    | 2146  | 0                | 5.34e-07        | -0.214       | 7.17           | 0.00171        | 0.0131              |
| native            | sort | 195     | 103    | 0    | 92    | 0                | 5.75e-08        | -0.000582    | 7.55           | 0.00191        | 0.0148              |
| native_ll1e-4     | sort | 195     | 195    | 0    | 0     | 0                | 0.00142         | 4.8e-08      | 4.29           | 0.00362        | 0.0148              |
| native_ll1e-5     | sort | 195     | 195    | 0    | 0     | 0                | 0.00196         | 4.8e-08      | 2.68           | 0.00671        | 0.0148              |
| native_param1e-8  | sort | 195     | 195    | 0    | 0     | 0                | 0.00244         | 4.8e-08      | 0.597          | 0.0235         | 0.0148              |
| native_iter50     | sort | 195     | 195    | 0    | 0     | 0                | 0.00198         | 4.8e-08      | 1.49           | 0.0115         | 0.0148              |
| sklearn_kmeanspp  | sort | 195     | 74     | 20   | 101   | 0                | -4.12e-06       | -0.0993      | 0.918          | 0.0179         | 0.0148              |
| sklearn_spherical | sort | 195     | 0      | 195  | 0     | 0                | 0               | 0            | 1.12           | 0.0132         | 0.0148              |
| native_10k        | sort | 195     | 110    | 0    | 85    | 0                | 1.41e-07        | -0.00164     | 7.53           | 0.00191        | 0.0148              |

## sort_latency

| arm               | sorts | same_process_timing | n_median | rest_ms | fit_ms | fit_share_pct | sort_ms | sort_speedup |
|-------------------|-------|---------------------|----------|---------|--------|---------------|---------|--------------|
| baseline          | 62    | 62                  | 4952     | 25.1    | 26.2   | 47.8          | 49.7    | 1            |
| native            | 62    | 62                  | 4952     | 25.1    | 3.69   | 10.4          | 28      | 1.71         |
| native_ll1e-4     | 62    | 0                   | 4952     | 25.1    | 9.76   | 22.1          | 34.9    | 1.44         |
| native_ll1e-5     | 62    | 0                   | 4952     | 25.1    | 20.2   | 40.6          | 46.6    | 1.2          |
| native_param1e-8  | 62    | 0                   | 4952     | 25.1    | 57.4   | 67.5          | 80.3    | 0.68         |
| native_iter50     | 62    | 0                   | 4952     | 25.1    | 15.7   | 37            | 40.6    | 1.24         |
| sklearn_kmeanspp  | 62    | 0                   | 4952     | 25.1    | 34.2   | 52.2          | 62.6    | 0.981        |
| sklearn_spherical | 62    | 0                   | 4952     | 25.1    | 25.1   | 47.4          | 48.7    | 1.01         |
| native_10k        | 62    | 0                   | 4952     | 25.1    | 4.22   | 12.1          | 28.7    | 1.69         |

## bench_by_n

| arm               | n_bucket | resampled | samples | n_median | seconds_median | seconds_p90 | speedup_median | speedup_min |
|-------------------|----------|-----------|---------|----------|----------------|-------------|----------------|-------------|
| baseline          | <1k      | 0         | 48      | 374      | 0.00538        | 0.00914     | 1              | 1           |
| native            | <1k      | 0         | 48      | 374      | 0.000537       | 0.00112     | 9.5            | 4.43        |
| native_10k        | <1k      | 0         | 48      | 374      | 0.000539       | 0.00117     | 9.55           | 4.33        |
| native_iter50     | <1k      | 0         | 48      | 374      | 0.000447       | 0.00891     | 13.3           | 0.802       |
| native_ll1e-4     | <1k      | 0         | 48      | 374      | 0.000508       | 0.00188     | 9.77           | 0.537       |
| native_ll1e-5     | <1k      | 0         | 48      | 374      | 0.000512       | 0.0128      | 9.68           | 0.475       |
| native_param1e-8  | <1k      | 0         | 48      | 374      | 0.000456       | 0.0353      | 13.3           | 0.207       |
| sklearn_kmeanspp  | <1k      | 0         | 48      | 374      | 0.00557        | 0.0109      | 0.934          | 0.497       |
| sklearn_spherical | <1k      | 0         | 48      | 374      | 0.00421        | 0.00706     | 1.28           | 1.15        |
| baseline          | 1k-5k    | 0         | 120     | 2.09e+03 | 0.0142         | 0.0283      | 1              | 1           |
| native            | 1k-5k    | 0         | 120     | 2.09e+03 | 0.00181        | 0.0039      | 7.52           | 5.36        |
| native_10k        | 1k-5k    | 0         | 120     | 2.09e+03 | 0.00181        | 0.00402     | 7.57           | 5.36        |
| native_iter50     | 1k-5k    | 0         | 120     | 2.09e+03 | 0.0118         | 0.0126      | 1.22           | 0.588       |
| native_ll1e-4     | 1k-5k    | 0         | 120     | 2.09e+03 | 0.00347        | 0.00804     | 4.13           | 1.05        |
| native_ll1e-5     | 1k-5k    | 0         | 120     | 2.09e+03 | 0.00623        | 0.0157      | 2.52           | 0.559       |
| native_param1e-8  | 1k-5k    | 0         | 120     | 2.09e+03 | 0.0293         | 0.0468      | 0.431          | 0.215       |
| sklearn_kmeanspp  | 1k-5k    | 0         | 120     | 2.09e+03 | 0.0144         | 0.0382      | 0.94           | 0.245       |
| sklearn_spherical | 1k-5k    | 0         | 120     | 2.09e+03 | 0.0125         | 0.0248      | 1.12           | 0.945       |
| baseline          | 20000    | 1         | 168     | 2e+04    | 0.055          | 0.109       | 1              | 1           |
| native            | 20000    | 1         | 168     | 2e+04    | 0.00487        | 0.00965     | 10.1           | 6.56        |
| native_10k        | 20000    | 1         | 168     | 2e+04    | 0.00343        | 0.00682     | 14.4           | 8.25        |
| native_iter50     | 20000    | 1         | 168     | 2e+04    | 0.0361         | 0.0382      | 1.87           | 0.687       |
| native_ll1e-4     | 20000    | 1         | 168     | 2e+04    | 0.00896        | 0.0247      | 5.99           | 0.684       |
| native_ll1e-5     | 20000    | 1         | 168     | 2e+04    | 0.0164         | 0.0551      | 3.47           | 0.569       |
| native_param1e-8  | 20000    | 1         | 168     | 2e+04    | 0.0844         | 0.145       | 0.765          | 0.291       |
| sklearn_kmeanspp  | 20000    | 1         | 168     | 2e+04    | 0.0543         | 0.151       | 0.891          | 0.184       |
| sklearn_spherical | 20000    | 1         | 168     | 2e+04    | 0.0579         | 0.115       | 0.933          | 0.761       |
| baseline          | 50000    | 1         | 168     | 5e+04    | 0.143          | 0.274       | 1              | 1           |
| native            | 50000    | 1         | 168     | 5e+04    | 0.0105         | 0.0206      | 12.3           | 6.14        |
| native_10k        | 50000    | 1         | 168     | 5e+04    | 0.00354        | 0.00675     | 37.5           | 24.6        |
| native_iter50     | 50000    | 1         | 168     | 5e+04    | 0.0735         | 0.076       | 2.43           | 0.896       |
| native_ll1e-4     | 50000    | 1         | 168     | 5e+04    | 0.0204         | 0.0556      | 7.11           | 0.843       |
| native_ll1e-5     | 50000    | 1         | 168     | 5e+04    | 0.0363         | 0.12        | 4.42           | 0.707       |
| native_param1e-8  | 50000    | 1         | 168     | 5e+04    | 0.168          | 0.293       | 1.01           | 0.379       |
| sklearn_kmeanspp  | 50000    | 1         | 168     | 5e+04    | 0.161          | 0.44        | 0.867          | 0.21        |
| sklearn_spherical | 50000    | 1         | 168     | 5e+04    | 0.142          | 0.28        | 1              | 0.828       |
