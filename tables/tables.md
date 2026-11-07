## table1_scenes

| scene | height | width | bands | classes | labelled | smallest_class | block | classes_retained | supports_protocol |
|---|---|---|---|---|---|---|---|---|---|
| Pavia University | 610 | 340 | 103 | 9 | 42776 | 947 | 33.0 | 9 | yes |
| Pavia Centre | 1096 | 715 | 102 | 9 | 148152 | 2685 | 18.0 | 9 | yes |
| Salinas | 512 | 217 | 204 | 16 | 54129 | 916 | 18.0 | 13 | yes |
| Houston 2013 | 349 | 1905 | 144 | 15 | 14999 | 323 | 33.0 | 13 | yes |
| IndianPines | 145 | 145 | 200 | 16 | 10249 | 20 |  | 6 | no |
| KSC | 512 | 614 | 176 | 13 | 5211 | 105 |  | 7 | no |
| Botswana | 1476 | 256 | 145 | 14 | 3248 | 95 |  | 7 | no |

## table2_leakage_audit

| scene | protocol | patch | frac_test_patches_touching_train | mean_train_pixels_in_window |
|---|---|---|---|---|
| Pavia University | random | 5 | 0.8457 | 1.94 |
| Pavia University | random | 7 | 0.9438 | 3.61 |
| Pavia University | random | 9 | 0.9719 | 5.67 |
| Pavia University | random | 11 | 0.9843 | 8.07 |
| Pavia University | random | 15 | 0.9944 | 13.82 |
| Pavia University | block-disjoint | 15 | 0.0 | 0.0 |
| Pavia Centre | random | 5 | 0.8649 | 2.04 |
| Pavia Centre | random | 7 | 0.9603 | 3.87 |
| Pavia Centre | random | 9 | 0.9827 | 6.12 |
| Pavia Centre | random | 11 | 0.9904 | 8.78 |
| Pavia Centre | random | 15 | 0.9958 | 15.22 |
| Pavia Centre | block-disjoint | 15 | 0.0 | 0.0 |
| Salinas | random | 5 | 0.9018 | 2.26 |
| Salinas | random | 7 | 0.986 | 4.41 |
| Salinas | random | 9 | 0.9981 | 7.2 |
| Salinas | random | 11 | 0.9996 | 10.61 |
| Salinas | random | 15 | 1.0 | 19.19 |
| Salinas | block-disjoint | 15 | 0.0 | 0.0 |
| Houston 2013 | random | 5 | 0.7467 | 1.46 |
| Houston 2013 | random | 7 | 0.8578 | 2.36 |
| Houston 2013 | random | 9 | 0.8987 | 3.23 |
| Houston 2013 | random | 11 | 0.9164 | 4.08 |
| Houston 2013 | random | 15 | 0.9344 | 5.85 |
| Houston 2013 | block-disjoint | 15 | 0.0 | 0.0 |

## table3_protocol_cross

| model | random_OA | random_sd | disjoint_OA | disjoint_sd | drop | AA | kappa | params | MFLOPs |
|---|---|---|---|---|---|---|---|---|---|
| 2-D CNN | 0.9894 | 0.0091 | 0.8417 | 0.111 | 0.1477 | 0.8195 | 0.8108 | 28075 | 3.01 |
| SSFTT | 0.996 | 0.0029 | 0.833 | 0.1253 | 0.163 | 0.8183 | 0.8022 | 184411 | 19.88 |
| LSSA | 0.9932 | 0.0051 | 0.8215 | 0.1497 | 0.1717 | 0.7897 | 0.7892 | 32411 | 3.73 |
| 3-D CNN | 0.9909 | 0.0068 | 0.821 | 0.1287 | 0.1699 | 0.8111 | 0.7876 | 3709739 | 33.07 |
| HybridSN | 0.9874 | 0.0095 | 0.8034 | 0.1383 | 0.184 | 0.7902 | 0.7664 | 534011 | 32.03 |
| ASCN | 0.9505 | 0.0345 | 0.8024 | 0.13 | 0.1481 | 0.786 | 0.7639 | 45579 | 0.66 |
| spectral SVM | 0.9304 | 0.0385 | 0.7794 | 0.136 | 0.151 | 0.7663 | 0.7377 | 13410 |  |

## table4_random_pairs

| model_a | model_b | diff | p | p_BH | significant |
|---|---|---|---|---|---|
| LSSA | spectral SVM | 0.0628 | <0.0001 | <0.0001 | yes |
| HybridSN | ASCN | 0.0368 | <0.0001 | <0.0001 | yes |
| 2-D CNN | spectral SVM | 0.0591 | <0.0001 | <0.0001 | yes |
| SSFTT | HybridSN | 0.0086 | <0.0001 | <0.0001 | yes |
| SSFTT | ASCN | 0.0454 | <0.0001 | <0.0001 | yes |
| SSFTT | spectral SVM | 0.0656 | <0.0001 | <0.0001 | yes |
| 2-D CNN | ASCN | 0.0389 | <0.0001 | <0.0001 | yes |
| 3-D CNN | spectral SVM | 0.0605 | <0.0001 | <0.0001 | yes |
| LSSA | HybridSN | 0.0058 | <0.0001 | <0.0001 | yes |
| LSSA | ASCN | 0.0426 | <0.0001 | <0.0001 | yes |
| HybridSN | spectral SVM | 0.057 | <0.0001 | <0.0001 | yes |
| 3-D CNN | ASCN | 0.0403 | <0.0001 | <0.0001 | yes |
| ASCN | spectral SVM | 0.0202 | <0.0001 | <0.0001 | yes |
| SSFTT | 3-D CNN | 0.0051 | <0.0001 | <0.0001 | yes |
| 3-D CNN | HybridSN | 0.0035 | <0.0001 | <0.0001 | yes |
| SSFTT | LSSA | 0.0028 | <0.0001 | <0.0001 | yes |
| SSFTT | 2-D CNN | 0.0065 | <0.0001 | 0.0001 | yes |
| LSSA | 3-D CNN | 0.0023 | 0.0006 | 0.0007 | yes |
| 2-D CNN | HybridSN | 0.002 | 0.0017 | 0.0019 | yes |
| LSSA | 2-D CNN | 0.0037 | 0.0215 | 0.0226 | yes |
| 3-D CNN | 2-D CNN | 0.0014 | 0.2162 | 0.2162 | no |

## table5_model_pairs

| model_a | model_b | diff | n_pairs | p | p_BH | significant |
|---|---|---|---|---|---|---|
| 2-D CNN | spectral SVM | 0.0623 | 20 | <0.0001 | 0.0001 | yes |
| 3-D CNN | spectral SVM | 0.0416 | 20 | <0.0001 | 0.0003 | yes |
| SSFTT | spectral SVM | 0.0536 | 20 | <0.0001 | 0.0006 | yes |
| 2-D CNN | ASCN | 0.0393 | 20 | 0.0005 | 0.0025 | yes |
| 2-D CNN | HybridSN | 0.0383 | 20 | 0.0017 | 0.0069 | yes |
| SSFTT | ASCN | 0.0306 | 20 | 0.0020 | 0.0069 | yes |
| 2-D CNN | 3-D CNN | 0.0207 | 20 | 0.0042 | 0.0127 | yes |
| SSFTT | HybridSN | 0.0296 | 20 | 0.0049 | 0.0128 | yes |
| LSSA | spectral SVM | 0.0421 | 20 | 0.0083 | 0.0194 | yes |
| ASCN | spectral SVM | 0.023 | 20 | 0.0153 | 0.0322 | yes |
| HybridSN | spectral SVM | 0.024 | 20 | 0.0172 | 0.0328 | yes |
| 3-D CNN | ASCN | 0.0186 | 20 | 0.0215 | 0.0376 | yes |
| 3-D CNN | HybridSN | 0.0176 | 20 | 0.0328 | 0.0529 | no |
| SSFTT | 3-D CNN | 0.0119 | 20 | 0.0362 | 0.0544 | no |
| 2-D CNN | LSSA | 0.0203 | 20 | 0.0583 | 0.0816 | no |
| SSFTT | LSSA | 0.0115 | 20 | 0.1650 | 0.2165 | no |
| LSSA | ASCN | 0.0191 | 20 | 0.1769 | 0.2185 | no |
| 2-D CNN | SSFTT | 0.0087 | 20 | 0.2162 | 0.2522 | no |
| LSSA | HybridSN | 0.0181 | 20 | 0.4091 | 0.4522 | no |
| LSSA | 3-D CNN | 0.0004 | 20 | 0.5459 | 0.5732 | no |
| HybridSN | ASCN | 0.001 | 20 | 0.8695 | 0.8695 | no |

## table6_scene_cluster

| contrast | mean | ci_lo | ci_hi | scenes_positive | Houston 2013 | Pavia Centre | Pavia University | Salinas |
|---|---|---|---|---|---|---|---|---|
| 2-D CNN - HybridSN | 0.0383 | 0.0089 | 0.0677 | 3/4 | 0.0671 | -0.0021 | 0.0684 | 0.0199 |
| 2-D CNN - 3-D CNN | 0.0207 | 0.0053 | 0.0373 | 4/4 | 0.0464 | 0.0007 | 0.0257 | 0.0099 |
| 2-D CNN - SSFTT | 0.0087 | -0.011 | 0.0285 | 2/4 | 0.029 | -0.0022 | 0.028 | -0.0198 |
| 2-D CNN - ASCN | 0.0393 | 0.0165 | 0.0578 | 4/4 | 0.0438 | 0.0074 | 0.0435 | 0.0626 |
| SSFTT - ASCN | 0.0306 | 0.0111 | 0.0655 | 4/4 | 0.0148 | 0.0096 | 0.0155 | 0.0824 |
| 2-D CNN - spectral SVM | 0.0623 | 0.0277 | 0.0828 | 4/4 | 0.0857 | 0.0107 | 0.0741 | 0.0788 |
| SSFTT - spectral SVM | 0.0536 | 0.0238 | 0.0855 | 4/4 | 0.0566 | 0.0129 | 0.0462 | 0.0986 |
| LSSA - spectral SVM | 0.0421 | 0.0047 | 0.0836 | 4/4 | 0.002 | 0.0074 | 0.109 | 0.0498 |
| 3-D CNN - spectral SVM | 0.0416 | 0.0196 | 0.0615 | 4/4 | 0.0392 | 0.0099 | 0.0485 | 0.0689 |
| HybridSN - spectral SVM | 0.024 | 0.009 | 0.0473 | 4/4 | 0.0186 | 0.0128 | 0.0058 | 0.0588 |
| ASCN - spectral SVM | 0.023 | 0.0098 | 0.0362 | 4/4 | 0.0418 | 0.0033 | 0.0306 | 0.0162 |
| dropping the buffer, pooled over models | 0.0622 | 0.0308 | 0.0972 | 4/4 | 0.1118 | 0.0197 | 0.0642 | 0.0531 |
| blocked to interleaved geography, pooled over models | 0.1 | 0.0307 | 0.181 | 4/4 | 0.2148 | 0.0085 | 0.0971 | 0.0795 |
| 2-D CNN window 5 to 15, buffered | 0.0122 | -0.0064 | 0.0307 | 2/4 | -0.0082 | -0.0046 | 0.0296 | 0.0319 |
| SSFTT window 3 to 15, buffered | 0.0274 | 0.0022 | 0.0612 | 3/4 | 0.0267 | -0.0049 | 0.0093 | 0.0785 |
| 2-D CNN window 5 to 15, random | 0.02 | 0.0048 | 0.0398 | 4/4 | 0.0199 | 0.002 | 0.0076 | 0.0506 |
| SSFTT window 3 to 15, random | 0.0439 | 0.0169 | 0.0708 | 4/4 | 0.0595 | 0.0069 | 0.027 | 0.0821 |
| 2-D CNN width 16 to 32 | 0.0359 | 0.0128 | 0.0531 | 4/4 | 0.0384 | 0.0021 | 0.045 | 0.0581 |
| 2-D CNN width 32 to 64 | 0.0091 | -0.0004 | 0.0195 | 3/4 | 0.0241 | 0.0057 | 0.0107 | -0.0041 |
| 2-D CNN width 64 to 128 | -0.0006 | -0.0087 | 0.0056 | 2/4 | 0.0064 | -0.0003 | -0.0132 | 0.0048 |
| SSFTT - 2-D CNN at 10 epochs | 0.0804 | 0.0216 | 0.1688 | 4/4 | 0.065 | 0.0071 | 0.0366 | 0.2129 |
| SSFTT - 2-D CNN at 60 epochs | -0.0097 | -0.0273 | 0.0089 | 1/4 | -0.0242 | -0.0041 | -0.0303 | 0.02 |
| SSFTT - 2-D CNN at 5 labels per class | 0.0335 | -0.0074 | 0.0744 | 3/4 | 0.0827 | 0.0046 | -0.0193 | 0.0661 |
| SSFTT - 2-D CNN at 10 labels per class | 0.0661 | -0.0053 | 0.1375 | 3/4 | 0.1138 | -0.0181 | 0.0075 | 0.1612 |
| SSFTT - 2-D CNN at 25 labels per class | 0.0532 | 0.0153 | 0.0876 | 3/4 | 0.0464 | -0.0086 | 0.0881 | 0.0871 |

## table7_decomposition

| model | buffered | block_no_buffer | random | buffered_to_unbuffered | unbuffered_to_random | total | first_contrast_share |
|---|---|---|---|---|---|---|---|
| 2-D CNN | 0.8417 | 0.8904 | 0.9894 | 0.0487 | 0.099 | 0.1477 | 0.33 |
| SSFTT | 0.833 | 0.9038 | 0.996 | 0.0708 | 0.0922 | 0.163 | 0.434 |
| LSSA | 0.8215 | 0.8869 | 0.9932 | 0.0654 | 0.1063 | 0.1717 | 0.381 |
| 3-D CNN | 0.821 | 0.8843 | 0.9909 | 0.0633 | 0.1066 | 0.1699 | 0.373 |
| HybridSN | 0.8034 | 0.8719 | 0.9874 | 0.0685 | 0.1155 | 0.184 | 0.372 |
| ASCN | 0.8024 | 0.8596 | 0.9505 | 0.0572 | 0.0909 | 0.1481 | 0.386 |
| spectral SVM | 0.7794 | 0.841 | 0.9304 | 0.0616 | 0.0894 | 0.151 | 0.408 |

## table8_window_sweep

| model | protocol | from_patch | to_patch | gain | n_pairs | p |
|---|---|---|---|---|---|---|
| 2-D CNN | random | 5 | 15 | 0.02 | 20 | <0.0001 |
| 2-D CNN | disjoint | 5 | 15 | 0.0122 | 20 | 0.4091 |
| SSFTT | random | 3 | 15 | 0.0439 | 20 | <0.0001 |
| SSFTT | disjoint | 3 | 15 | 0.0274 | 20 | 0.0266 |

## table9_capacity

| width | OA | sd | params | gain_vs_previous | p |
|---|---|---|---|---|---|
| 8 | 0.7134 | 0.1895 | 3571 |  |  |
| 16 | 0.8042 | 0.1395 | 9435 | 0.0907 | 0.0001 |
| 32 | 0.84 | 0.1141 | 28075 | 0.0359 | 0.0215 |
| 64 | 0.8491 | 0.1058 | 93003 | 0.0091 | 0.1327 |
| 128 | 0.8486 | 0.1027 | 333451 | -0.0006 | 0.6477 |

## table10_training_budget

| epochs | diff_SSFTT_minus_CNN2D | p | n_pairs | 2-D CNN | HybridSN | SSFTT |
|---|---|---|---|---|---|---|
| 10 | 0.0804 | 0.0005 | 20 | 0.7243 | 0.6958 | 0.8047 |
| 20 | 0.0316 | 0.0973 | 20 | 0.7954 | 0.7752 | 0.827 |
| 40 | 0.0051 | 0.8408 | 20 | 0.8283 | 0.7958 | 0.8334 |
| 60 | -0.0097 | 0.1231 | 20 | 0.8426 | 0.809 | 0.833 |
| 100 | -0.0086 | 0.3118 | 20 | 0.8434 | 0.8083 | 0.8348 |

## table11_low_label

| labels_per_class | diff_SSFTT_minus_CNN2D | p | n_pairs | 2-D CNN | LSSA | SSFTT | spectral SVM |
|---|---|---|---|---|---|---|---|
| 5 | 0.0335 | 0.2611 | 20 | 0.7195 | 0.4859 | 0.7531 | 0.7039 |
| 10 | 0.0661 | 0.0153 | 20 | 0.7122 | 0.4801 | 0.7783 | 0.7296 |
| 25 | 0.0532 | 0.0032 | 20 | 0.7431 | 0.6374 | 0.7963 | 0.742 |
