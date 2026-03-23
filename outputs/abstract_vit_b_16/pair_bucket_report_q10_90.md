# Pair Bucket Hypothesis Report

This report tests whether originally similar class pairs become closer and originally dissimilar class pairs become farther apart.

Bucket rule:
- far: source similarity <= 10th percentile
- mid: between the two thresholds
- near: source similarity >= 90th percentile

## ImageNet_v1_vs_ImageNet_A_nc180_sc10_knn5_10_seed1

- bucket_definition: source-similarity-quantiles
- thresholds: far <= -0.2204, near >= 0.2554
- num_pairs_total: 16110

- far: pairs=1611, mean_source_similarity=-0.2585, mean_target_similarity=-0.2838, mean_delta_similarity=-0.0252, mean_abs_diff=0.0756
- mid: pairs=12888, mean_source_similarity=-0.0219, mean_target_similarity=-0.0217, mean_delta_similarity=0.0002, mean_abs_diff=0.1131
- near: pairs=1611, mean_source_similarity=0.3802, mean_target_similarity=0.4080, mean_delta_similarity=0.0278, mean_abs_diff=0.1176

Far representative pairs:
- agama (42) / rocking chair (765) | source=-0.3723, target=-0.2785, delta=0.0938
- mongoose (298) / sewing machine (786) | source=-0.3698, target=-0.4243, delta=-0.0546
- chameleon (47) / shovel (792) | source=-0.3618, target=-0.2588, delta=0.1030

Mid representative pairs:
- vulture (23) / parachute (701) | source=-0.0369, target=-0.1153, delta=-0.0784
- hermit crab (125) / broccoli (937) | source=-0.0369, target=-0.2106, delta=-0.1738
- sewing machine (786) / spider web (815) | source=-0.0368, target=0.2431, delta=0.2799

Near representative pairs:
- grasshopper (311) / leafhopper (317) | source=0.8315, target=0.5268, delta=-0.3047
- Rottweiler (234) / German Shepherd Dog (235) | source=0.8218, target=0.4358, delta=-0.3860
- crayfish (124) / hermit crab (125) | source=0.8199, target=0.6830, delta=-0.1369

Far top increase:
- teddy bear (850) / acorn (988) | source=-0.2280, target=0.1754, delta=0.4033
Far top decrease:
- dragonfly (319) / bow tie (457) | source=-0.2740, target=-0.5529, delta=-0.2790

Mid top increase:
- crayfish (124) / small white butterfly (324) | source=0.0216, target=0.5649, delta=0.5433
Mid top decrease:
- bee (309) / rapeseed (984) | source=0.2378, target=-0.2319, delta=-0.4697

Near top increase:
- lion (291) / white-headed capuchin (378) | source=0.3081, target=0.7273, delta=0.4192
Near top decrease:
- harvestman (70) / spider web (815) | source=0.6417, target=0.1112, delta=-0.5305

## ImageNet_v1_vs_ImageNet_R_nc200_sc10_knn5_10_seed1

- bucket_definition: source-similarity-quantiles
- thresholds: far <= -0.2050, near >= 0.2474
- num_pairs_total: 19900

- far: pairs=1990, mean_source_similarity=-0.2404, mean_target_similarity=-0.1587, mean_delta_similarity=0.0817, mean_abs_diff=0.0981
- mid: pairs=15920, mean_source_similarity=-0.0295, mean_target_similarity=-0.0262, mean_delta_similarity=0.0034, mean_abs_diff=0.0837
- near: pairs=1990, mean_source_similarity=0.4294, mean_target_similarity=0.3211, mean_delta_similarity=-0.1083, mean_abs_diff=0.1349

Far representative pairs:
- guinea pig (338) / binoculars (447) | source=-0.3619, target=-0.1392, delta=0.2227
- snail (113) / Siberian Husky (250) | source=-0.3590, target=-0.3134, delta=0.0456
- Scottish Terrier (199) / ant (310) | source=-0.3499, target=-0.1225, delta=0.2275

Mid representative pairs:
- Indian cobra (63) / king penguin (145) | source=-0.0477, target=-0.0838, delta=-0.0360
- porcupine (334) / cabbage (936) | source=-0.0477, target=0.0092, delta=0.0569
- starfish (327) / missile (657) | source=-0.0477, target=-0.1512, delta=-0.1035

Near representative pairs:
- Boston Terrier (195) / French Bulldog (245) | source=0.8963, target=0.7701, delta=-0.1262
- Basset Hound (161) / Beagle (162) | source=0.8869, target=0.8155, delta=-0.0714
- grasshopper (311) / praying mantis (315) | source=0.8551, target=0.7017, delta=-0.1534

Far top increase:
- Indian cobra (63) / flute (558) | source=-0.2356, target=0.1426, delta=0.3781
Far top decrease:
- Standard Poodle (267) / hammer (587) | source=-0.2136, target=-0.4269, delta=-0.2133

Mid top increase:
- vulture (23) / jellyfish (107) | source=-0.1130, target=0.4413, delta=0.5543
Mid top decrease:
- Indian cobra (63) / cottontail rabbit (330) | source=0.2214, target=-0.2373, delta=-0.4587

Near top increase:
- Siberian Husky (250) / grey wolf (269) | source=0.4063, target=0.7278, delta=0.3215
Near top decrease:
- mitten (658) / scarf (824) | source=0.7459, target=0.1834, delta=-0.5626

## ImageNet_v1_vs_ImageNet_Sketch_nc200_sc10_knn5_10_seed1

- bucket_definition: source-similarity-quantiles
- thresholds: far <= -0.2176, near >= 0.2547
- num_pairs_total: 19900

- far: pairs=1990, mean_source_similarity=-0.2572, mean_target_similarity=-0.2003, mean_delta_similarity=0.0569, mean_abs_diff=0.0815
- mid: pairs=15920, mean_source_similarity=-0.0243, mean_target_similarity=-0.0219, mean_delta_similarity=0.0024, mean_abs_diff=0.0953
- near: pairs=1990, mean_source_similarity=0.4080, mean_target_similarity=0.3293, mean_delta_similarity=-0.0787, mean_abs_diff=0.1323

Far representative pairs:
- vulture (23) / clothes iron (606) | source=-0.3769, target=-0.2604, delta=0.1165
- beaver (337) / couch (831) | source=-0.3768, target=-0.2352, delta=0.1417
- vulture (23) / toaster (859) | source=-0.3657, target=-0.3146, delta=0.0510

Mid representative pairs:
- high-speed train (466) / lampshade (619) | source=-0.0407, target=0.0050, delta=0.0457
- ocean liner (628) / scabbard (777) | source=-0.0407, target=0.1111, delta=0.1518
- Pomeranian (259) / envelope (549) | source=-0.0407, target=-0.2119, delta=-0.1711

Near representative pairs:
- Rottweiler (234) / Dobermann (236) | source=0.9015, target=0.7747, delta=-0.1268
- common redshank (141) / dowitcher (142) | source=0.8821, target=0.8442, delta=-0.0378
- macaque (373) / langur (374) | source=0.8751, target=0.6682, delta=-0.2069

Far top increase:
- chiton (116) / neck brace (678) | source=-0.2387, target=0.2196, delta=0.4583
Far top decrease:
- Sussex Spaniel (220) / turnstile (877) | source=-0.2195, target=-0.4390, delta=-0.2195

Mid top increase:
- mousetrap (674) / turnstile (877) | source=-0.0403, target=0.6339, delta=0.6743
Mid top decrease:
- Sussex Spaniel (220) / bath towel (434) | source=0.1673, target=-0.3622, delta=-0.5295

Near top increase:
- freight car (565) / semi-trailer truck (867) | source=0.3804, target=0.8313, delta=0.4509
Near top decrease:
- Miniature Pinscher (237) / bath towel (434) | source=0.3470, target=-0.3401, delta=-0.6872

## ImageNet_v1_vs_ImageNet_v2_nc200_sc10_knn5_10_seed1

- bucket_definition: source-similarity-quantiles
- thresholds: far <= -0.2154, near >= 0.2471
- num_pairs_total: 19900

- far: pairs=1990, mean_source_similarity=-0.2550, mean_target_similarity=-0.2353, mean_delta_similarity=0.0197, mean_abs_diff=0.0443
- mid: pairs=15920, mean_source_similarity=-0.0240, mean_target_similarity=-0.0234, mean_delta_similarity=0.0005, mean_abs_diff=0.0559
- near: pairs=1990, mean_source_similarity=0.4028, mean_target_similarity=0.3811, mean_delta_similarity=-0.0217, mean_abs_diff=0.0724

Far representative pairs:
- vulture (23) / toaster (859) | source=-0.4115, target=-0.2969, delta=0.1147
- great egret (132) / toaster (859) | source=-0.3889, target=-0.3210, delta=0.0679
- bald eagle (22) / toaster (859) | source=-0.3887, target=-0.2589, delta=0.1298

Mid representative pairs:
- bald eagle (22) / holster (597) | source=-0.0386, target=-0.1220, delta=-0.0834
- Otterhound (175) / fur coat (568) | source=-0.0386, target=0.0344, delta=0.0730
- electrical switch (844) / comic book (917) | source=-0.0386, target=0.0027, delta=0.0412

Near representative pairs:
- night snake (60) / African rock python (62) | source=0.9090, target=0.8674, delta=-0.0416
- eastern hog-nosed snake (54) / night snake (60) | source=0.9050, target=0.9206, delta=0.0156
- eastern hog-nosed snake (54) / African rock python (62) | source=0.8923, target=0.8400, delta=-0.0523

Far top increase:
- gar fish (395) / toaster (859) | source=-0.3090, target=-0.0656, delta=0.2434
Far top decrease:
- gorilla (366) / necklace (679) | source=-0.2182, target=-0.3549, delta=-0.1367

Mid top increase:
- crash helmet (518) / tricycle (870) | source=0.1134, target=0.5265, delta=0.4131
Mid top decrease:
- beaver (337) / mousetrap (674) | source=0.1174, target=-0.3333, delta=-0.4506

Near top increase:
- spatula (813) / chocolate syrup (960) | source=0.2895, target=0.6031, delta=0.3136
Near top decrease:
- bottle cap (455) / necklace (679) | source=0.6052, target=0.0584, delta=-0.5468
