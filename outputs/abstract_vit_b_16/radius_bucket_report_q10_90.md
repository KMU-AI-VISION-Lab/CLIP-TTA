# Radius Bucket Report

This report tests whether classes near the global center move further inward and classes far from the center move further outward.

Bucket rule:
- inner: source radius <= 10th percentile
- mid: between the two thresholds
- outer: source radius >= 90th percentile

## ImageNet_v1_vs_ImageNet_A_nc180_sc10_knn5_10_seed1

- bucket_definition: source-radius-quantiles
- thresholds: inner <= 0.4246, outer >= 0.5704
- num_classes_total: 180
- mean_delta_radius_overall: -0.1018
- mean_abs_delta_radius_overall: 0.1033

- inner: classes=18, mean_source_radius=0.3988, mean_target_radius=0.3319, mean_delta_radius=-0.0669, mean_abs_delta_radius=0.0669
- mid: classes=144, mean_source_radius=0.4978, mean_target_radius=0.4005, mean_delta_radius=-0.0972, mean_abs_delta_radius=0.0992
- outer: classes=18, mean_source_radius=0.6046, mean_target_radius=0.4318, mean_delta_radius=-0.1728, mean_abs_delta_radius=0.1728

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
- mask (643) | source_radius=0.3892, target_radius=0.3701, delta_radius=-0.0191
Inner top radius decrease:
- fishing casting reel (758) | source_radius=0.4138, target_radius=0.2860, delta_radius=-0.1279

Mid top radius increase:
- African bush elephant (386) | source_radius=0.5270, target_radius=0.5982, delta_radius=0.0712
Mid top radius decrease:
- acorn (988) | source_radius=0.4654, target_radius=0.2564, delta_radius=-0.2090

Outer top radius increase:
- carbonara (959) | source_radius=0.5949, target_radius=0.4849, delta_radius=-0.1100
Outer top radius decrease:
- guacamole (924) | source_radius=0.5984, target_radius=0.3461, delta_radius=-0.2523

## ImageNet_v1_vs_ImageNet_R_nc200_sc10_knn5_10_seed1

- bucket_definition: source-radius-quantiles
- thresholds: inner <= 0.4313, outer >= 0.5668
- num_classes_total: 200
- mean_delta_radius_overall: -0.1081
- mean_abs_delta_radius_overall: 0.1086

- inner: classes=20, mean_source_radius=0.4107, mean_target_radius=0.3578, mean_delta_radius=-0.0529, mean_abs_delta_radius=0.0562
- mid: classes=160, mean_source_radius=0.4882, mean_target_radius=0.3796, mean_delta_radius=-0.1086, mean_abs_delta_radius=0.1089
- outer: classes=20, mean_source_radius=0.5986, mean_target_radius=0.4396, mean_delta_radius=-0.1590, mean_abs_delta_radius=0.1590

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
- bow tie (457) | source_radius=0.4190, target_radius=0.3091, delta_radius=-0.1099

Mid top radius increase:
- Labrador Retriever (208) | source_radius=0.4364, target_radius=0.4559, delta_radius=0.0195
Mid top radius decrease:
- mobile phone (487) | source_radius=0.5668, target_radius=0.3350, delta_radius=-0.2318

Outer top radius increase:
- steam locomotive (820) | source_radius=0.5676, target_radius=0.5081, delta_radius=-0.0595
Outer top radius decrease:
- canoe (472) | source_radius=0.6827, target_radius=0.3438, delta_radius=-0.3388

## ImageNet_v1_vs_ImageNet_Sketch_nc200_sc10_knn5_10_seed1

- bucket_definition: source-radius-quantiles
- thresholds: inner <= 0.4006, outer >= 0.5751
- num_classes_total: 200
- mean_delta_radius_overall: -0.0583
- mean_abs_delta_radius_overall: 0.0713

- inner: classes=20, mean_source_radius=0.3726, mean_target_radius=0.3863, mean_delta_radius=0.0137, mean_abs_delta_radius=0.0427
- mid: classes=160, mean_source_radius=0.4870, mean_target_radius=0.4307, mean_delta_radius=-0.0563, mean_abs_delta_radius=0.0654
- outer: classes=20, mean_source_radius=0.5961, mean_target_radius=0.4496, mean_delta_radius=-0.1465, mean_abs_delta_radius=0.1465

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
- sliding door (799) | source_radius=0.4010, target_radius=0.4830, delta_radius=0.0821
Mid top radius decrease:
- chocolate syrup (960) | source_radius=0.5058, target_radius=0.2993, delta_radius=-0.2066

Outer top radius increase:
- mosque (668) | source_radius=0.5960, target_radius=0.5565, delta_radius=-0.0395
Outer top radius decrease:
- dishcloth (533) | source_radius=0.6150, target_radius=0.3146, delta_radius=-0.3004

## ImageNet_v1_vs_ImageNet_v2_nc200_sc10_knn5_10_seed1

- bucket_definition: source-radius-quantiles
- thresholds: inner <= 0.4030, outer >= 0.5805
- num_classes_total: 200
- mean_delta_radius_overall: -0.0148
- mean_abs_delta_radius_overall: 0.0341

- inner: classes=20, mean_source_radius=0.3766, mean_target_radius=0.3777, mean_delta_radius=0.0011, mean_abs_delta_radius=0.0224
- mid: classes=160, mean_source_radius=0.4890, mean_target_radius=0.4758, mean_delta_radius=-0.0133, mean_abs_delta_radius=0.0342
- outer: classes=20, mean_source_radius=0.6017, mean_target_radius=0.5592, mean_delta_radius=-0.0425, mean_abs_delta_radius=0.0446

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
- table lamp (846) | source_radius=0.3971, target_radius=0.4446, delta_radius=0.0475
Inner top radius decrease:
- shopping basket (790) | source_radius=0.3923, target_radius=0.3451, delta_radius=-0.0472

Mid top radius increase:
- drink pitcher (725) | source_radius=0.4391, target_radius=0.5508, delta_radius=0.1117
Mid top radius decrease:
- couch (831) | source_radius=0.5142, target_radius=0.4078, delta_radius=-0.1064

Outer top radius increase:
- spotted salamander (28) | source_radius=0.5875, target_radius=0.5954, delta_radius=0.0079
Outer top radius decrease:
- crash helmet (518) | source_radius=0.5904, target_radius=0.3844, delta_radius=-0.2061
