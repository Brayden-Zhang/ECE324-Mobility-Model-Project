# NeurIPS Experiment Checklist

This is the full experiment suite to support a NeurIPS-level submission for TrajectoryFM-HMT + Macro Mobility head.

## Workflow
1. Run the eval scripts to produce JSON artifacts in `cache/` (use the SLURM scripts for full-scale runs).
2. Refresh the paper and eval tables with:

```bash
PYTHONPATH=src python -m route_rangers.cli.collect_results
```

## Core Evidence
1. Recovery and prediction (UniTraj-style)
2. Next-location and destination probes
3. Robustness to noise and dropouts
4. Data efficiency (fractional training centroids)
5. Transfer across datasets (zero-shot)
6. Compute efficiency (params + throughput)
7. Length sensitivity evaluation (short/medium/long buckets)
8. Invariance suite: prefix prediction, time-shift robustness, downsample robustness
9. Embedding retrieval (kNN) for destination similarity
10. Reverse-order stress test
11. Change-detection analog (prefix vs suffix embedding drift)

## Macro Mobility Evidence
1. Movement Distribution monthly KL/JS/L1
2. Macro head ablations (with/without macro loss)
3. Temporal generalization: train on months M1..M10, test M11..M12

## Region/Zone Evidence
1. Commuting zone destination probe (classification)
2. Zone-level robustness to OOD regions (train on subset of zones)

## Mobility Foundation Evidence (MoveGPT-style)
1. Next-POI prediction: Acc@1/5/10, Recall@K, NDCG@10, MRR
2. User identification probe
3. Cross-city leave-one-city-out transfer (zero-shot)

## Ablations
1. No graph vs graph (layers/knn/temporal window)
2. No space-time encoder vs enabled
3. Tokenizer: hashed H3 vs fixed H3 vocab vs VQ
4. Context: OSM/Overture vs none
5. Region masking on/off
6. Flow head on/off (token-only baseline)
7. Trip features on/off
8. Length-weighted loss vs token-weighted loss

## Scaling
1. Depth/width scaling: (D=6/8/12, H=256/512)
2. Graph scaling: (layers=2/4/6, knn=8/16/32)
3. Macro mix prob sweep: 0.2/0.5/0.8

## Baselines
1. UniTraj external checkpoint on WorldTrace sample
2. Internal ablations as baselines
3. External foundation-model baselines listed in `docs/baselines.md`
4. Fallbacks: classic sequence models (LSTM/GRU/Transformer) on H3 tokens

## Downstream Tasks (Product-Relevant)
1. Destination inference (early prediction)
2. OD flow forecasting (macro movement distribution)
3. Trip volume forecasting by zone/time
4. Travel-time estimation with uncertainty
5. Anomaly detection in mobility shocks

## Task-to-Metric Alignment
1. Destination inference → accuracy@k, top-1
2. OD flow forecasting → KL/JS/L1
3. Trip volume forecasting → MAE/MAPE/RMSE
4. Travel-time estimation → MAE/RMSE + calibration (NLL)
5. Anomaly detection → AUROC, PR-AUC

## Scripted Runs (SLURM)
1. Stage-1 training: `scripts/slurm_train_hmt_nohash_full.sh`
2. Stage-2 training: `scripts/slurm_train_hmt_stage2.sh`
3. Full eval: `scripts/slurm_eval_full.sh`
4. External UniTraj eval: `scripts/slurm_unitraj_eval_external.sh`
5. Length sensitivity: `scripts/slurm_length_sensitivity.sh`
6. Invariance suite: `scripts/slurm_invariance_suite.sh`
7. Embedding retrieval: `scripts/slurm_embedding_retrieval.sh`
8. Reverse-order stress: `scripts/slurm_reverse_order.sh`
9. Change detection: `scripts/slurm_change_detection.sh`
10. Next POI eval: `python -m route_rangers.cli.run_next_poi_eval ...`
11. Cross-city transfer: `python -m route_rangers.cli.run_cross_city_transfer ...`

## Notes
- Keep evals on the same dataset split for fair comparisons.
- Log configs, random seeds, and hardware for reproducibility.
