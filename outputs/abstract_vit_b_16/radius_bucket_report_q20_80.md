# Radius Bucket Report

This report tests whether classes near the global center move further inward and classes far from the center move further outward.

Bucket rule:
- inner: source radius <= 20th percentile
- mid: between the two thresholds
- outer: source radius >= 80th percentile

## ImageNet_v1_vs_ImageNet_A_nc180_sc10_knn5_10_seed1

- bucket_definition: source-radius-quantiles
- thresholds: inner <= 0.4482, outer >= 0.5491
- num_classes_total: 180
- mean_delta_radius_overall: -0.1018
- mean_abs_delta_radius_overall: 0.1033

- inner: classes=36, mean_source_radius=0.4183, mean_target_radius=0.3560, mean_delta_radius=-0.0623, mean_abs_delta_radius=0.0640
- mid: classes=108, mean_source_radius=0.4973, mean_target_radius=0.3975, mean_delta_radius=-0.0997, mean_abs_delta_radius=0.1017
- outer: classes=36, mean_source_radius=0.5827, mean_target_radius=0.4353, mean_delta_radius=-0.1474, mean_abs_delta_radius=0.1474

Inner representative classes:
- chain (488) | source_radius=0.3383, target_radius=0.3126, delta_radius=-0.0257
- doormat (539) | source_radius=0.3574, target_radius=0.2703, delta_radius=-0.0871
- wheelbarrow (428) | source_radius=0.3633, target_radius=0.2959, delta_radius=-0.0674

Mid representative classes:
- sundial (835) | source_radius=0.4977, target_radius=0.4274, delta_radius=-0.0702
- bell pepper (945) | source_radius=0.4970, target_radius=0.4499, delta_radius=-0.0471
- red fox (277) | source_radius=0.4979, target_radius=0.4850, delta_radius=-0.0129

Outer representative classes:
- volleyball (890) | source_radius=0.6528, target_radius=0.4458, delta_radius=-0.2070
- revolver (763) | source_radius=0.6472, target_radius=0.3963, delta_radius=-0.2509
- basketball (430) | source_radius=0.6395, target_radius=0.5142, delta_radius=-0.1253

Inner top radius increase:
- pomegranate (957) | source_radius=0.4429, target_radius=0.4731, delta_radius=0.0303
Inner top radius decrease:
- flagpole (557) | source_radius=0.4458, target_radius=0.3155, delta_radius=-0.1303

Mid top radius increase:
- African bush elephant (386) | source_radius=0.5270, target_radius=0.5982, delta_radius=0.0712
Mid top radius decrease:
- acorn (988) | source_radius=0.4654, target_radius=0.2564, delta_radius=-0.2090

Outer top radius increase:
- steam locomotive (820) | source_radius=0.5505, target_radius=0.5114, delta_radius=-0.0392
Outer top radius decrease:
- guacamole (924) | source_radius=0.5984, target_radius=0.3461, delta_radius=-0.2523

## ImageNet_v1_vs_ImageNet_R_nc200_sc10_knn5_10_seed1

- bucket_definition: source-radius-quantiles
- thresholds: inner <= 0.4498, outer >= 0.5302
- num_classes_total: 200
- mean_delta_radius_overall: -0.1081
- mean_abs_delta_radius_overall: 0.1086

- inner: classes=40, mean_source_radius=0.4255, mean_target_radius=0.3627, mean_delta_radius=-0.0628, mean_abs_delta_radius=0.0654
- mid: classes=120, mean_source_radius=0.4866, mean_target_radius=0.3778, mean_delta_radius=-0.1087, mean_abs_delta_radius=0.1087
- outer: classes=40, mean_source_radius=0.5722, mean_target_radius=0.4208, mean_delta_radius=-0.1514, mean_abs_delta_radius=0.1514

Inner representative classes:
- pig (341) | source_radius=0.3876, target_radius=0.3726, delta_radius=-0.0151
- binoculars (447) | source_radius=0.3888, target_radius=0.3290, delta_radius=-0.0598
- bucket (463) | source_radius=0.3962, target_radius=0.3320, delta_radius=-0.0642

Mid representative classes:
- Italian Greyhound (171) | source_radius=0.4836, target_radius=0.4075, delta_radius=-0.0761
- cottontail rabbit (330) | source_radius=0.4831, target_radius=0.3801, delta_radius=-0.1030
- hummingbird (94) | source_radius=0.4836, target_radius=0.3665, delta_radius=-0.1170

Outer representative classes:
- canoe (472) | source_radius=0.6827, target_radius=0.3438, delta_radius=-0.3388
- basketball (430) | source_radius=0.6724, target_radius=0.4320, delta_radius=-0.2405
- ambulance (407) | source_radius=0.6541, target_radius=0.5113, delta_radius=-0.1429

Inner top radius increase:
- cockroach (314) | source_radius=0.4097, target_radius=0.4328, delta_radius=0.0231
Inner top radius decrease:
- snail (113) | source_radius=0.4492, target_radius=0.2813, delta_radius=-0.1679

Mid top radius increase:
- wine bottle (907) | source_radius=0.4502, target_radius=0.4380, delta_radius=-0.0122
Mid top radius decrease:
- hammerhead shark (4) | source_radius=0.5138, target_radius=0.3073, delta_radius=-0.2065

Outer top radius increase:
- military aircraft (895) | source_radius=0.5572, target_radius=0.5065, delta_radius=-0.0507
Outer top radius decrease:
- canoe (472) | source_radius=0.6827, target_radius=0.3438, delta_radius=-0.3388

## ImageNet_v1_vs_ImageNet_Sketch_nc200_sc10_knn5_10_seed1

- bucket_definition: source-radius-quantiles
- thresholds: inner <= 0.4394, outer >= 0.5440
- num_classes_total: 200
- mean_delta_radius_overall: -0.0583
- mean_abs_delta_radius_overall: 0.0713

- inner: classes=40, mean_source_radius=0.3942, mean_target_radius=0.4016, mean_delta_radius=0.0074, mean_abs_delta_radius=0.0365
- mid: classes=120, mean_source_radius=0.4871, mean_target_radius=0.4315, mean_delta_radius=-0.0557, mean_abs_delta_radius=0.0626
- outer: classes=40, mean_source_radius=0.5768, mean_target_radius=0.4449, mean_delta_radius=-0.1319, mean_abs_delta_radius=0.1319

Inner representative classes:
- plunger (731) | source_radius=0.3223, target_radius=0.3788, delta_radius=0.0565
- window screen (904) | source_radius=0.3334, target_radius=0.5451, delta_radius=0.2117
- bucket (463) | source_radius=0.3357, target_radius=0.3732, delta_radius=0.0375

Mid representative classes:
- Komodo dragon (48) | source_radius=0.4814, target_radius=0.4248, delta_radius=-0.0565
- Kuvasz (222) | source_radius=0.4814, target_radius=0.4550, delta_radius=-0.0264
- table lamp (846) | source_radius=0.4813, target_radius=0.3764, delta_radius=-0.1049

Outer representative classes:
- football helmet (560) | source_radius=0.6968, target_radius=0.5916, delta_radius=-0.1052
- dishcloth (533) | source_radius=0.6150, target_radius=0.3146, delta_radius=-0.3004
- photocopier (713) | source_radius=0.6122, target_radius=0.4482, delta_radius=-0.1640

Inner top radius increase:
- window screen (904) | source_radius=0.3334, target_radius=0.5451, delta_radius=0.2117
Inner top radius decrease:
- bath towel (434) | source_radius=0.3643, target_radius=0.2982, delta_radius=-0.0660

Mid top radius increase:
- Lhasa Apso (204) | source_radius=0.4623, target_radius=0.5360, delta_radius=0.0737
Mid top radius decrease:
- chocolate syrup (960) | source_radius=0.5058, target_radius=0.2993, delta_radius=-0.2066

Outer top radius increase:
- tank (847) | source_radius=0.5604, target_radius=0.5448, delta_radius=-0.0156
Outer top radius decrease:
- dishcloth (533) | source_radius=0.6150, target_radius=0.3146, delta_radius=-0.3004

## ImageNet_v1_vs_ImageNet_v2_nc200_sc10_knn5_10_seed1

- bucket_definition: source-radius-quantiles
- thresholds: inner <= 0.4290, outer >= 0.5457
- num_classes_total: 200
- mean_delta_radius_overall: -0.0148
- mean_abs_delta_radius_overall: 0.0341

- inner: classes=40, mean_source_radius=0.3963, mean_target_radius=0.3984, mean_delta_radius=0.0021, mean_abs_delta_radius=0.0256
- mid: classes=120, mean_source_radius=0.4894, mean_target_radius=0.4769, mean_delta_radius=-0.0125, mean_abs_delta_radius=0.0342
- outer: classes=40, mean_source_radius=0.5809, mean_target_radius=0.5425, mean_delta_radius=-0.0385, mean_abs_delta_radius=0.0422

Inner representative classes:
- plunger (731) | source_radius=0.3411, target_radius=0.3743, delta_radius=0.0332
- plastic bag (728) | source_radius=0.3466, target_radius=0.3450, delta_radius=-0.0016
- mousetrap (674) | source_radius=0.3541, target_radius=0.3790, delta_radius=0.0250

Mid representative classes:
- clothes iron (606) | source_radius=0.4948, target_radius=0.4501, delta_radius=-0.0448
- shopping cart (791) | source_radius=0.4942, target_radius=0.4670, delta_radius=-0.0272
- stinkhorn mushroom (994) | source_radius=0.4932, target_radius=0.4856, delta_radius=-0.0076

Outer representative classes:
- football helmet (560) | source_radius=0.7161, target_radius=0.6146, delta_radius=-0.1015
- high-speed train (466) | source_radius=0.6283, target_radius=0.5794, delta_radius=-0.0490
- lotion (631) | source_radius=0.6085, target_radius=0.5451, delta_radius=-0.0634

Inner top radius increase:
- turnstile (877) | source_radius=0.4204, target_radius=0.4819, delta_radius=0.0614
Inner top radius decrease:
- soap dispenser (804) | source_radius=0.4142, target_radius=0.3532, delta_radius=-0.0610

Mid top radius increase:
- drink pitcher (725) | source_radius=0.4391, target_radius=0.5508, delta_radius=0.1117
Mid top radius decrease:
- couch (831) | source_radius=0.5142, target_radius=0.4078, delta_radius=-0.1064

Outer top radius increase:
- sulphur butterfly (325) | source_radius=0.5468, target_radius=0.5881, delta_radius=0.0413
Outer top radius decrease:
- crash helmet (518) | source_radius=0.5904, target_radius=0.3844, delta_radius=-0.2061
