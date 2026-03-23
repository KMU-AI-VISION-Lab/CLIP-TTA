# Keyword Hypothesis Report

This report focuses on within-group and cross-group relational changes for selected keywords.

## ImageNet_v1_vs_ImageNet_A_nc180_sc10_knn5_10_seed1

- keywords: dog, bird, vehicle
- match_mode: both
- filtered_pairs: 231
- overall within-group: pairs=102, mean_delta_similarity=0.0681, mean_abs_diff=0.1256
- overall cross-group: pairs=129, mean_delta_similarity=-0.0242, mean_abs_diff=0.0790

Top increase:
- hummingbird (94) / goose (99) | source=0.2663, target=0.5381, delta=0.2719
Top decrease:
- sulphur-crested cockatoo (89) / toucan (96) | source=0.4839, target=0.1610, delta=-0.3229

Within-group summary:
- bird: pairs=66, mean_delta_similarity=0.0899, mean_abs_diff=0.1376
- vehicle: pairs=36, mean_delta_similarity=0.0280, mean_abs_diff=0.1035

Cross-group summary:
- bird__dog: pairs=12, mean_delta_similarity=-0.1042, mean_abs_diff=0.1042
- bird__vehicle: pairs=108, mean_delta_similarity=-0.0209, mean_abs_diff=0.0750
- dog__vehicle: pairs=9, mean_delta_similarity=0.0429, mean_abs_diff=0.0933

## ImageNet_v1_vs_ImageNet_R_nc200_sc10_knn5_10_seed1

- keywords: dog, bird, vehicle
- match_mode: both
- filtered_pairs: 595
- overall within-group: pairs=189, mean_delta_similarity=-0.0375, mean_abs_diff=0.1020
- overall cross-group: pairs=406, mean_delta_similarity=0.0149, mean_abs_diff=0.0739

Top increase:
- vulture (23) / Chihuahua (151) | source=-0.1489, target=0.1522, delta=0.3011
Top decrease:
- junco (13) / vulture (23) | source=0.4371, target=0.0546, delta=-0.3825

Within-group summary:
- bird: pairs=66, mean_delta_similarity=-0.0495, mean_abs_diff=0.1265
- dog: pairs=45, mean_delta_similarity=-0.0398, mean_abs_diff=0.1054
- vehicle: pairs=78, mean_delta_similarity=-0.0259, mean_abs_diff=0.0794

Cross-group summary:
- bird__dog: pairs=120, mean_delta_similarity=0.0490, mean_abs_diff=0.0874
- bird__vehicle: pairs=156, mean_delta_similarity=0.0153, mean_abs_diff=0.0712
- dog__vehicle: pairs=130, mean_delta_similarity=-0.0170, mean_abs_diff=0.0648

## ImageNet_v1_vs_ImageNet_Sketch_nc200_sc10_knn5_10_seed1

- keywords: dog, bird, vehicle
- match_mode: both
- filtered_pairs: 190
- overall within-group: pairs=65, mean_delta_similarity=0.0114, mean_abs_diff=0.1025
- overall cross-group: pairs=125, mean_delta_similarity=0.0074, mean_abs_diff=0.0697

Top increase:
- freight car (565) / semi-trailer truck (867) | source=0.3804, target=0.8313, delta=0.4509
Top decrease:
- tandem bicycle (444) / limousine (627) | source=0.3791, target=0.1568, delta=-0.2223

Within-group summary:
- bird: pairs=10, mean_delta_similarity=0.0105, mean_abs_diff=0.0835
- dog: pairs=10, mean_delta_similarity=0.0072, mean_abs_diff=0.0463
- vehicle: pairs=45, mean_delta_similarity=0.0125, mean_abs_diff=0.1191

Cross-group summary:
- bird__dog: pairs=25, mean_delta_similarity=0.0988, mean_abs_diff=0.1088
- bird__vehicle: pairs=50, mean_delta_similarity=-0.0262, mean_abs_diff=0.0621
- dog__vehicle: pairs=50, mean_delta_similarity=-0.0047, mean_abs_diff=0.0577

## ImageNet_v1_vs_ImageNet_v2_nc200_sc10_knn5_10_seed1

- keywords: dog, bird, vehicle
- match_mode: both
- filtered_pairs: 190
- overall within-group: pairs=65, mean_delta_similarity=-0.0128, mean_abs_diff=0.0598
- overall cross-group: pairs=125, mean_delta_similarity=-0.0033, mean_abs_diff=0.0442

Top increase:
- station wagon (436) / semi-trailer truck (867) | source=0.2626, target=0.4375, delta=0.1749
Top decrease:
- aircraft carrier (403) / military aircraft (895) | source=0.7389, target=0.5149, delta=-0.2240

Within-group summary:
- bird: pairs=10, mean_delta_similarity=-0.0041, mean_abs_diff=0.0481
- dog: pairs=10, mean_delta_similarity=-0.0048, mean_abs_diff=0.0646
- vehicle: pairs=45, mean_delta_similarity=-0.0165, mean_abs_diff=0.0613

Cross-group summary:
- bird__dog: pairs=25, mean_delta_similarity=-0.0013, mean_abs_diff=0.0463
- bird__vehicle: pairs=50, mean_delta_similarity=-0.0016, mean_abs_diff=0.0377
- dog__vehicle: pairs=50, mean_delta_similarity=-0.0061, mean_abs_diff=0.0497
