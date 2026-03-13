아래 step 2까지 하고 결과가 나왔어 이를 분석해줘.
그리고 UMAP이 차원을 축소해서 보기 때문에, 원래 512차원에서의 toplogy를 못볼까봐 걱정돼. 괜한걱정인지, 아니면 보다 나은방법이 있는지 추천해줘.

# 결과
Cross-dataset geometry evaluation
pairwise distance correlation is not reliable for cross-dataset comparison without sample correspondence
Source dataset: imagenet
Target dataset: imagenet_v2
Backbone: vit_b16
Selected classes: 200
centroid_cosine: same=0.943081 different=0.676568
centroid_euclidean: same=0.280205 different=0.685358
cov_frobenius: same=0.148735 different=0.160356
pca_subspace: same=0.297882 different=0.182030
pairwise_spearman: same=0.001163 different=0.003761
pairwise_pearson: same=0.000584 different=0.003920
retrieval_acc_centroid_cosine: 0.960000
retrieval_acc_centroid_euclidean: 0.960000
retrieval_acc_pca_subspace: 0.735000
Correspondence-free topology metrics
kNN distance Wasserstein (k=3): 0.076312
kNN distance Wasserstein (k=5): 0.074912
kNN mean distance diff (k=3): 0.066392
kNN mean distance diff (k=5): 0.067234
distance histogram JS: 0.194646
distance histogram Wasserstein: 0.075757
neighbor graph mean kNN distance diff (k=3): 0.066392
neighbor graph mean kNN distance diff (k=5): 0.067234
small values -> class geometry preserved
large values -> geometry differs across datasets
Saved JSON to: ./outputs/geometry_vit_b16/imagenet_vs_imagenet_v2_nc200_spc10_knn3_5_seed1.json
[timing] total: 33.62s

---

Cross-dataset geometry evaluation
pairwise distance correlation is not reliable for cross-dataset comparison without sample correspondence
Source dataset: imagenet
Target dataset: imagenet_sketch
Backbone: vit_b16
Selected classes: 200
centroid_cosine: same=0.833795 different=0.657865
centroid_euclidean: same=0.502295 different=0.722163
cov_frobenius: same=0.144624 different=0.147650
pca_subspace: same=0.213843 different=0.163704
pairwise_spearman: same=0.009255 different=-0.002723
pairwise_pearson: same=0.014485 different=-0.002626
retrieval_acc_centroid_cosine: 0.765000
retrieval_acc_centroid_euclidean: 0.705000
retrieval_acc_pca_subspace: 0.365000
Correspondence-free topology metrics
kNN distance Wasserstein (k=3): 0.129201
kNN distance Wasserstein (k=5): 0.122784
kNN mean distance diff (k=3): 0.117333
kNN mean distance diff (k=5): 0.112846
distance histogram JS: 0.253187
distance histogram Wasserstein: 0.120604
neighbor graph mean kNN distance diff (k=3): 0.117333
neighbor graph mean kNN distance diff (k=5): 0.112846
small values -> class geometry preserved
large values -> geometry differs across datasets
Saved JSON to: ./outputs/geometry_vit_b16/imagenet_vs_imagenet_sketch_nc200_spc10_knn3_5_seed1.json
[timing] total: 33.49s

---
Same-dataset upper bound / split-half reliability estimate
pairwise distance correlation is not reliable for cross-dataset comparison without sample correspondence
Source dataset: imagenet_split_A
Target dataset: imagenet_split_B
Backbone: vit_b16
Selected classes: 200
centroid_cosine: same=0.960916+/-0.000276 different=0.674137+/-0.000826
centroid_euclidean: same=0.233271+/-0.001134 different=0.692173+/-0.001300
cov_frobenius: same=0.139606+/-0.000795 different=0.154239+/-0.000836
pca_subspace: same=0.320848+/-0.001778 different=0.181591+/-0.000345
pairwise_spearman: same=-0.006477+/-0.010866 different=0.000051+/-0.000890
pairwise_pearson: same=-0.004617+/-0.009354 different=0.000016+/-0.000721
retrieval_acc_centroid_cosine: 0.989000+/-0.006633
retrieval_acc_centroid_euclidean: 0.990000+/-0.006325
retrieval_acc_pca_subspace: 0.823000+/-0.028740
Correspondence-free topology metrics
kNN distance Wasserstein (k=3): 0.057741
kNN distance Wasserstein (k=5): 0.055693
kNN mean distance diff (k=3): 0.046120
kNN mean distance diff (k=5): 0.046184
distance histogram JS: 0.167141
distance histogram Wasserstein: 0.056469
neighbor graph mean kNN distance diff (k=3): 0.046120
neighbor graph mean kNN distance diff (k=5): 0.046184
small values -> class geometry preserved
large values -> geometry differs across datasets
Saved JSON to: ./outputs/geometry_vit_b16/imagenet_split_half_upper_bound_nc200_spc10_rep5_knn3_5_seed1.json
[timing] total: 158.57s

---
Same-dataset upper bound / split-half reliability estimate
pairwise distance correlation is not reliable for cross-dataset comparison without sample correspondence
Source dataset: imagenet_sketch_split_A
Target dataset: imagenet_sketch_split_B
Backbone: vit_b16
Selected classes: 200
centroid_cosine: same=0.969918+/-0.000707 different=0.762616+/-0.000986
centroid_euclidean: same=0.212455+/-0.002324 different=0.607343+/-0.001373
cov_frobenius: same=0.115672+/-0.000802 different=0.135787+/-0.000475
pca_subspace: same=0.409338+/-0.002827 different=0.219122+/-0.000667
pairwise_spearman: same=0.003448+/-0.012357 different=-0.000287+/-0.000488
pairwise_pearson: same=0.000662+/-0.012725 different=0.000074+/-0.000755
retrieval_acc_centroid_cosine: 0.980000+/-0.004472
retrieval_acc_centroid_euclidean: 0.983000+/-0.005099
retrieval_acc_pca_subspace: 0.886000+/-0.009165
Correspondence-free topology metrics
kNN distance Wasserstein (k=3): 0.067790
kNN distance Wasserstein (k=5): 0.062330
kNN mean distance diff (k=3): 0.051134
kNN mean distance diff (k=5): 0.049012
distance histogram JS: 0.159328
distance histogram Wasserstein: 0.061517
neighbor graph mean kNN distance diff (k=3): 0.051134
neighbor graph mean kNN distance diff (k=5): 0.049012
small values -> class geometry preserved
large values -> geometry differs across datasets
Saved JSON to: ./outputs/geometry_vit_b16/imagenet_sketch_split_half_upper_bound_nc200_spc10_rep5_knn3_5_seed1.json
[timing] total: 159.17s

