# Debiased Explainable Pairwise Ranking

Pairwise ranking model BPR not only outperforms pointwise counterparts but also able to handle implicit feedback. But it is a black-box model and vulnerable to exposure bias. This exposure bias usually translates into an unfairness against the least popular items because they risk being under-exposed by the recommender system. **One approach to address this problem is to use EBPR (Explainable BPR) loss function.**

## Project structure
```
.
├── code
│   ├── EBPR_model.py
│   ├── engine_EBPR.py
│   ├── hyperparameter_tuning.py
│   ├── __init__.py
│   ├── metrics.py
│   ├── nbs
│   │   ├── reco-tut-elf-ml100k-01-data-preparation.py
│   │   └── reco-tut-elf-notebook.py
│   ├── preprocess.py
│   ├── train_EBPR.py
│   └── utils.py
├── data
│   └── bronze
│       ├── lastfm-2k
│       ├── ml-100k
│       └── ml-1m
├── docs
├── LICENSE
├── models
│   ├── EBPR_ml-100k_batchsize_100_opt_adam_lr_0.001_latent_100_l2reg_0_Epoch31_NDCG@10_0.4173_HR@10_0.6946_MEP@10_0.9274_WMEP@10_0.3581_Avg_Pop@10_0.4685_EFD@10_1.2144_Avg_Pair_Sim@10_0.2616.model
│   ├── EBPR_ml-100k_batchsize_100_opt_adam_lr_0.001_latent_100_l2reg_0_Epoch35_NDCG@10_0.4222_HR@10_0.6946_MEP@10_0.9244_WMEP@10_0.3534_Avg_Pop@10_0.4667_EFD@10_1.2195_Avg_Pair_Sim@10_0.2609.model
│   ├── EBPR_ml-100k_batchsize_100_opt_adam_lr_0.001_latent_100_l2reg_0_Epoch48_NDCG@10_0.4130_HR@10_0.6925_MEP@10_0.9176_WMEP@10_0.3472_Avg_Pop@10_0.4662_EFD@10_1.2205_Avg_Pair_Sim@10_0.2588.model
│   ├── EBPR_ml-100k_batchsize_100_opt_adam_lr_0.001_latent_20_l2reg_0_Epoch42_NDCG@10_0.3980_HR@10_0.6713_MEP@10_0.9238_WMEP@10_0.3579_Avg_Pop@10_0.4765_EFD@10_1.1796_Avg_Pair_Sim@10_0.2717.model
│   ├── EBPR_ml-100k_batchsize_100_opt_adam_lr_0.001_latent_20_l2reg_0_Epoch48_NDCG@10_0.3986_HR@10_0.6649_MEP@10_0.9255_WMEP@10_0.3589_Avg_Pop@10_0.4745_EFD@10_1.1934_Avg_Pair_Sim@10_0.2696.model
│   ├── EBPR_ml-100k_batchsize_100_opt_adam_lr_0.001_latent_20_l2reg_0_Epoch49_NDCG@10_0.4022_HR@10_0.6776_MEP@10_0.9211_WMEP@10_0.3576_Avg_Pop@10_0.4723_EFD@10_1.1993_Avg_Pair_Sim@10_0.2701.model
│   ├── EBPR_ml-100k_batchsize_100_opt_adam_lr_0.001_latent_50_l2reg_0.0_Epoch49_NDCG@10_0.3838_HR@10_0.6628_MEP@10_0.9282_WMEP@10_0.3593_Avg_Pop@10_0.4658_EFD@10_1.2251_Avg_Pair_Sim@10_0.2607.model
│   ├── EBPR_ml-100k_batchsize_500_opt_adam_lr_0.001_latent_10_l2reg_0_Epoch47_NDCG@10_0.3482_HR@10_0.5949_MEP@10_0.9060_WMEP@10_0.3486_Avg_Pop@10_0.4998_EFD@10_1.0814_Avg_Pair_Sim@10_0.2918.model
│   ├── EBPR_ml-100k_batchsize_500_opt_adam_lr_0.001_latent_10_l2reg_0_Epoch48_NDCG@10_0.3429_HR@10_0.6045_MEP@10_0.9030_WMEP@10_0.3479_Avg_Pop@10_0.5051_EFD@10_1.0634_Avg_Pair_Sim@10_0.2960.model
│   ├── EBPR_ml-100k_batchsize_500_opt_adam_lr_0.001_latent_10_l2reg_0_Epoch49_NDCG@10_0.3384_HR@10_0.5885_MEP@10_0.9035_WMEP@10_0.3497_Avg_Pop@10_0.5100_EFD@10_1.0507_Avg_Pair_Sim@10_0.2977.model
│   ├── EBPR_ml-100k_batchsize_500_opt_adam_lr_0.001_latent_20_l2reg_0.001_Epoch48_NDCG@10_0.3709_HR@10_0.6405_MEP@10_0.9205_WMEP@10_0.3583_Avg_Pop@10_0.4934_EFD@10_1.1147_Avg_Pair_Sim@10_0.2861.model
│   ├── EBPR_ml-100k_batchsize_500_opt_adam_lr_0.001_latent_20_l2reg_0.001_Epoch49_NDCG@10_0.3733_HR@10_0.6416_MEP@10_0.9175_WMEP@10_0.3582_Avg_Pop@10_0.4913_EFD@10_1.1205_Avg_Pair_Sim@10_0.2877.model
│   ├── EBPR_ml-100k_batchsize_500_opt_adam_lr_0.001_latent_20_l2reg_0.001_Epoch49_NDCG@10_0.3822_HR@10_0.6437_MEP@10_0.9137_WMEP@10_0.3551_Avg_Pop@10_0.4894_EFD@10_1.1286_Avg_Pair_Sim@10_0.2858.model
│   ├── EBPR_ml-100k_batchsize_50_opt_adam_lr_0.001_latent_100_l2reg_0.001_Epoch22_NDCG@10_0.4080_HR@10_0.6914_MEP@10_0.9295_WMEP@10_0.3576_Avg_Pop@10_0.4741_EFD@10_1.1954_Avg_Pair_Sim@10_0.2659.model
│   ├── EBPR_ml-100k_batchsize_50_opt_adam_lr_0.001_latent_100_l2reg_0.001_Epoch32_NDCG@10_0.4043_HR@10_0.6808_MEP@10_0.9273_WMEP@10_0.3537_Avg_Pop@10_0.4700_EFD@10_1.2071_Avg_Pair_Sim@10_0.2609.model
│   ├── EBPR_ml-100k_batchsize_50_opt_adam_lr_0.001_latent_100_l2reg_0.001_Epoch34_NDCG@10_0.4192_HR@10_0.7010_MEP@10_0.9275_WMEP@10_0.3507_Avg_Pop@10_0.4679_EFD@10_1.2125_Avg_Pair_Sim@10_0.2592.model
│   ├── EBPR_ml-100k_batchsize_50_opt_adam_lr_0.001_latent_20_l2reg_0.001_Epoch46_NDCG@10_0.3952_HR@10_0.6681_MEP@10_0.9316_WMEP@10_0.3612_Avg_Pop@10_0.4728_EFD@10_1.1976_Avg_Pair_Sim@10_0.2690.model
│   ├── EBPR_ml-100k_batchsize_50_opt_adam_lr_0.001_latent_20_l2reg_0.001_Epoch46_NDCG@10_0.4058_HR@10_0.6819_MEP@10_0.9234_WMEP@10_0.3559_Avg_Pop@10_0.4736_EFD@10_1.1930_Avg_Pair_Sim@10_0.2691.model
│   ├── EBPR_ml-100k_batchsize_50_opt_adam_lr_0.001_latent_20_l2reg_0.001_Epoch48_NDCG@10_0.4040_HR@10_0.6776_MEP@10_0.9285_WMEP@10_0.3585_Avg_Pop@10_0.4702_EFD@10_1.2034_Avg_Pair_Sim@10_0.2659.model
│   ├── EBPR_ml-100k_batchsize_50_opt_adam_lr_0.001_latent_20_l2reg_1e-05_Epoch45_NDCG@10_0.4015_HR@10_0.6808_MEP@10_0.9211_WMEP@10_0.3546_Avg_Pop@10_0.4748_EFD@10_1.1922_Avg_Pair_Sim@10_0.2700.model
│   ├── EBPR_ml-100k_batchsize_50_opt_adam_lr_0.001_latent_20_l2reg_1e-05_Epoch46_NDCG@10_0.4003_HR@10_0.6755_MEP@10_0.9221_WMEP@10_0.3572_Avg_Pop@10_0.4717_EFD@10_1.2022_Avg_Pair_Sim@10_0.2683.model
│   └── EBPR_ml-100k_batchsize_50_opt_adam_lr_0.001_latent_20_l2reg_1e-05_Epoch46_NDCG@10_0.4042_HR@10_0.6734_MEP@10_0.9231_WMEP@10_0.3548_Avg_Pop@10_0.4729_EFD@10_1.1968_Avg_Pair_Sim@10_0.2673.model
├── notebooks
│   ├── reco-tut-elf-ml100k-01-data-preparation.ipynb
│   └── reco-tut-elf-notebook.ipynb
├── outputs
│   └── Hyperparameter_tuning_EBPR_ml-100k.csv
└── README.md  
```