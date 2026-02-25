# Unrestrained CB Unlearning with Low Rank Updates (retain_coef=0)

This mostly uses orthogonal circuit breakers with a batch size set to 0 to nullify the orthogonal turn (needlessly complicated).

## Unrestrained Unlearning LoRA Rank Sweep

Goal: Determine minimum LoRA rank needed for tamper-resistant unlearning with orth circuit breakers.

All runs: remove=23, orth=5, retain=0, 32 steps, 1024 examples, 1ep.

Trained with pdbs=1 so no orth loss (pairwise loss requires batch_size >= 2).

| rank | WMDP Robust | MMLU | cb_loss | orth_loss | Tamper WMDP (step 105) | Tamper Cloze (step 105) | Notes |
|------|-------------|------|---------|-----------|------------------------|-------------------------|-------|
| -    | 42.97%      | 45.10% | -     | -         | -                      | -                       | Baseline |
| 8    | 26.73%      | 22.95% | 0.0007 | 0.9999   | 26.73%                 | 0.141                   | |
| 16   | 26.73%      | 22.95% | 0.0007 | 0.9999   | 26.73%                 | 0.143                   | |
| 32   | 26.73%      | 22.95% | 0.0008 | 0.9999   | 26.73%                 | 0.143                   | |
| 64   | 26.73%      | 22.95% | 0.0009 | 0.9999   | 26.73%                 | 0.142                   | |
| 128  | 26.73%      | 22.95% | 0.0009 | 0.9999   | 26.73%                 | 0.144                   | |
| 256  | 26.73%      | 22.95% | 0.0009 | 0.9999   | 26.73%                 | 0.144                   | |
| 512  | 26.73%      | 22.95% | 0.0010 | 0.9999   | 26.73%                 | 0.144                   | |

### Unrestrained Tamper Resistance: LoRA Rank Sweep

### all-module LoRA (AdamW tamper, ~30ep)

| rank | Metric | Step 0 | Step 500 | Step 1000 | Step 2000 | Step 3300 | Notes |
|------|--------|--------|----------|-----------|-----------|-----------|-------|
| - | WMDP | 42.97 | - | - | - | - | Baseline |
| | MMLU | 45.10 | - | - | - | - | |
| 8 | WMDP | 26.2 | 29.8 | 31.8 | 29.1 | 29.3 | Peaks step 800, settles ~29 |
| | MMLU | 25.5 | 38.1 | 40.6 | 39.3 | 39.2 | |
| 16 | WMDP | 26.7 | 27.1 | 29.0 | 29.1 | 29.1 | Gradual climb to ~29 |
| | MMLU | 23.0 | 25.3 | 24.9 | 24.8 | 24.6 | |
| 32 | WMDP | 26.7 | 26.0 | 25.7 | 25.5 | 25.8 | Flat through 3300 |
| | MMLU | 23.0 | 25.9 | 25.6 | 25.1 | 25.2 | |
| 64 | WMDP | 26.7 | 26.5 | 26.7 | 26.8 | 27.8 | Flat through 3300 |
| | MMLU | 23.0 | 23.7 | 26.3 | 26.3 | 26.5 | |
| 128 | WMDP | 25.8 | 26.7 | 26.4 | 24.1 | 23.6 | Flat/declining through 3300 |
| | MMLU | 26.8 | 23.0 | 24.4 | 24.4 | 24.7 | |
| 256 | WMDP | 24.1 | 26.7 | - | - | - | Cancelled step 500, resubmitted |
| | MMLU | 26.9 | 23.0 | - | - | - | |
| 512 | WMDP | 24.1 | 26.7 | - | - | - | Cancelled step 500, resubmitted |
| | MMLU | 26.9 | 23.0 | - | - | - | |

### all-module LoRA (Muon tamper)

[Results removed for using many epochs of the same data.]

### r12 LoRA with Orthogonal Aux Loss (Muon tamper, ~30ep)

r12 models unlearned with pdbs=4 (orth loss active). Not directly comparable to r8/r16 rank sweep models above which used pdbs=1 (orth loss always 0).

| target | unlearn opt | Metric | Step 0 | Step 500 | Step 1000 | Step 3300 | Notes |
|--------|-------------|--------|--------|----------|-----------|-----------|-------|
| attn | Muon | WMDP | 32.7 | 38.0 | 39.6 | - | Completed step 1000 |
| | | MMLU | 38.9 | 43.7 | 44.5 | - | |
| mlp | Muon | WMDP | 34.2 | 40.7 | 40.7 | - | Completed step 1000 |
| | | MMLU | 41.1 | 45.3 | 45.1 | - | |
| all | Muon | WMDP | 28.9 | 35.0 | 34.4 | 31.9 | |
| | | MMLU | 29.9 | 40.7 | 42.4 | 39.7 | |
| all | AdamW | WMDP | 28.5 | 38.7 | 37.8 | 34.9 | |
| | | MMLU | 34.9 | 44.6 | 45.8 | 43.8 | |

### SFT (full-rank) unlearn tamper (AdamW, 3000 steps, 1ep, 48000 chunks, evaluated every 500, batch_size=32)

Unlearned model: SFT (full-rank, not LoRA), rm23, orth5, ret0, pdbs=1, 32 steps, 1024 examples. Training job 2476679.

| Config | Step 0 | Step 500 | Step 1000 | Step 1500 | Step 2000 | Step 2500 | Step 3000 | Notes |
|--------|--------|----------|-----------|-----------|-----------|-----------|-----------|-------|
| cos/wr01/lr2e-5 | | | | | | | | job 2476682 |
| cos/wr01/lr2e-4 | | | | | | | | job 2476683 |
| cos/wr01/lr1e-3 | | | | | | | | job 2476684 |
| cos/ws30/lr2e-4 | | | | | | | | job 2476685 |
| cos/ws100/lr2e-4 | | | | | | | | job 2476686 |
| const/wr01/lr2e-4 | | | | | | | | job 2476687 |

### Layer exclusion ablation (AdamW tamper, all-module r=16, ~18ep)

To the extent these experiments are valid, excluding one layer from unlearning is enough to break tamper resistance.

| Excluded layer | Metric | Step 0 | Step 200 | Step 400 | Step 600 | Step 800 | Step 1000 | Step 1200 | Step 1400 | Step 1600 | Step 1800 | Step 2000 | Notes |
|----------------|--------|--------|----------|----------|----------|----------|-----------|-----------|-----------|-----------|-----------|-----------|-------|
| - | WMDP | 42.97 | - | - | - | - | - | - | - | - | - | - | Baseline |
| | MMLU | 45.10 | - | - | - | - | - | - | - | - | - | - | |
| L0 | WMDP | 27.65 | 41.36 | 43.09 | 43.66 | 44.01 | 43.43 | 42.63 | 41.59 | 40.55 | 41.36 | 41.24 | Running |
| | MMLU | 32.37 | 45.72 | 46.80 | 46.89 | 47.14 | 46.82 | 46.61 | 46.33 | 46.19 | 46.20 | 46.06 | |
| L16 | WMDP | 26.96 | 41.13 | 42.97 | 43.78 | 42.86 | 41.59 | 42.17 | 41.13 | 39.40 | 39.29 | 39.75 | Running |
| | MMLU | 32.07 | 45.73 | 46.83 | 46.62 | 46.94 | 46.70 | 46.45 | 46.18 | 45.83 | 45.78 | 45.58 | |
| L31 | WMDP | 25.92 | 40.90 | 42.63 | 43.09 | 43.32 | 42.86 | 41.13 | 41.01 | 40.09 | 39.75 | 39.52 | Running |
| | MMLU | 34.61 | 45.69 | 46.69 | 46.87 | 46.98 | 46.60 | 46.45 | 46.47 | 46.08 | 46.00 | 45.98 | |

### Repair-then-attack CB on r=8 all-module

Phase 1: WikiText repair (200 steps, lr=2e-5). Phase 2: Bio attack (500 steps, lr=2e-5).

[Results invalidated by using many epochs of the same data.]

| Phase | Step | WMDP | MMLU |
|-------|------|------|------|
| repair | 0 | 26.2% | 25.5% |
| repair | 100 | 24.5% | 23.4% |
| repair | 200 | 24.8% | 26.3% |
| attack | 0 | 24.8% | 26.3% |

### Empirical rank of weight deltas (99% energy threshold)

| Model | Total Frob | QKV Mean/Max Rank | Dense Mean/Max | MLP Up Mean/Max | MLP Down Mean/Max |
|-------|-----------|-------------------|----------------|-----------------|-------------------|
| r=8 (AdamW, pdbs=1) | 16.9 | 4.1 / 6 | 2.0 / 4 | 2.5 / 4 | 1.0 / 2 |
| r=12 (Muon, pdbs=4) | 11.5 | 12.0 / 12 | 12.0 / 12 | 12.0 / 12 | 12.0 / 12 |
| r=12 (AdamW, pdbs=4) | 14.4 | 11.0 / 12 | 10.6 / 12 | 10.5 / 12 | 7.1 / 12 |
| r=16 (AdamW, pdbs=1) | 26.2 | 5.1 / 8 | 2.6 / 7 | 3.3 / 7 | 1.0 / 1 |
| r=32 (AdamW, pdbs=1) | 38.7 | 6.6 / 13 | 3.4 / 10 | 5.1 / 10 | 1.0 / 1 |

### all-module CB LoRA lr=2e-4 tamper (AdamW, 32 epochs, eval every 500 steps)

[Results invalidated by using many epochs of the same data.]

| rank | Metric | Step 0 | Step 500 | Step 1000 | Step 2000 | Step 3520 | Notes |
|------|--------|--------|----------|-----------|-----------|-----------|-------|
| - | WMDP | 42.97 | - | - | - | - | Baseline |
| | MMLU | 45.10 | - | - | - | - | |

### deep-ignorance-e2e-strong-filter tamper (AdamW, 512 examples, 5 epochs, eval every 100 steps)

| lr | Metric | Step 0 | Step 100 | Step 200 | Step 300 | Step 400 | Step 500 | Step 550 |
|----|--------|--------|----------|----------|----------|----------|----------|----------|
| 2e-5 | WMDP | 34.56 | 33.29 | 33.64 | 34.22 | 33.87 | 33.87 | 33.29 |
| | MMLU | 46.00 | 45.74 | 46.18 | 46.23 | **46.47** | 46.34 | 46.31 |
| 5e-5 | WMDP | 34.56 | 32.37 | 33.64 | 34.79 | 34.79 | **35.25** | 34.33 |
| | MMLU | 46.00 | 46.14 | 45.75 | 46.35 | 46.27 | 46.33 | **46.45** |
| 1e-4 | WMDP | 34.56 | 34.33 | 34.45 | 34.33 | 35.02 | **35.14** | 35.02 |
| | MMLU | 46.00 | 44.50 | 44.27 | 44.00 | 44.17 | 44.11 | **44.25** |
| 2e-4 | WMDP | 34.56 | 30.76 | **31.68** | 30.18 | 29.61 | 30.53 | 31.11 |
| | MMLU | 46.00 | 37.23 | 35.97 | 37.82 | **38.04** | 37.86 | 37.95 |

### r=8 all-module tamper variants (AdamW, 3500 steps, ~32ep, eval every 500)

Unlearned model: r=8, all-module, pdbs=1 (orth loss always 0). Baseline: 42.97 / 45.10.

| Attack config | Metric | Step 0 | Step 500 | Step 1000 | Step 1500 | Step 2000 | Step 2500 | Step 3000 | Step 3500 |
|---------------|--------|--------|----------|-----------|-----------|-----------|-----------|-----------|-----------|
| cosine, lr=2e-5 | WMDP | 26.96 | **31.91** | 29.49 | 28.00 | 28.11 | 28.23 | 28.34 | 28.34 |
| | MMLU | 26.18 | **38.20** | 37.52 | 37.12 | 36.83 | 36.82 | 36.80 | 36.80 |
| warmup 10%, lr=2e-5 | WMDP | 26.96 | **32.60** | 28.23 | 29.03 | 28.00 | 28.11 | 28.46 | 28.69 |
| | MMLU | 26.18 | **38.01** | 37.07 | 37.12 | 36.60 | 36.68 | 36.48 | 36.31 |
| cosine + warmup 10%, lr=2e-5 | WMDP | 26.96 | 27.42 | **27.88** | 26.50 | 27.76 | 27.53 | 27.19 | 27.19 |
| | MMLU | 26.18 | 25.79 | **30.67** | 29.84 | 29.55 | 29.43 | 29.43 | 29.43 |
| cosine, lr=2e-4 | WMDP | 26.96 | 28.34 | 27.42 | 27.53 | 27.76 | 27.76 | **27.76** | **27.76** |
| | MMLU | 26.18 | 26.66 | 28.32 | 28.34 | 28.40 | 28.46 | **28.48** | **28.48** |
| warmup 10%, lr=2e-4 | WMDP | 26.96 | 26.61 | **26.73** | 26.61 | 26.61 | 26.61 | 26.61 | 26.50 |
| | MMLU | 26.18 | 26.63 | 26.81 | 27.22 | **27.30** | 26.74 | 26.91 | 26.36 |
| cosine + warmup 10%, lr=2e-4 | WMDP | 26.96 | 33.53 | 35.02 | 34.91 | 35.02 | 35.25 | **35.37** | **35.37** |
| | MMLU | 26.18 | 34.29 | 36.74 | 36.93 | **36.95** | 36.94 | **36.95** | 36.94 |

### WMDP Bio Robust over Tampering Configs for LoRA ranks (AdamW, step 3000, 1ep, evaluated every 500)

Unlearned models: all-module, pdbs=1, rm23, orth5, ret0, all 32 layers. Filtered model: deep-ignorance-e2e-strong-filter. Baseline: 42.97 / 45.10. Filtered model uses LRs 2e-5/5e-5/1e-4 (vs rank sweep: 2e-5/2e-4/1e-3).

| Config | r=2 | r=4 | r=8 | r=16 | r=32 | r=64 | r=128 | r=256 | r=512 | r=1024 | filter |
|--------|-----|-----|-----|------|------|------|-------|-------|-------|--------|--------|
| cos/wr01/lr2e-5 | **45.7** | **46.8** | 36.4 | 28.1 | 28.0 | 26.6 | 22.8 | 25.2 | 26.7 | | 33.6 |
| cos/wr01/lr2e-4 | 36.2 | 40.2 | **38.8** | **38.2** | **37.2** | 34.0 | 28.1 | **32.5** | 26.4 | | 34.3 |
| cos/wr01/lr1e-3 | 26.8 | 26.7 | 26.7 | 26.7 | 26.8 | 26.7 | 26.8 | 26.8 | **28.3** | | 34.1 |
| cos/ws30/lr2e-4 | 37.2 | 36.6 | 38.5 | 36.9 | 29.5 | 34.6 | 29.0 | 31.4 | 26.7 | | **35.4** |
| cos/ws100/lr2e-4 | 41.5 | 35.8 | 34.8 | 32.6 | 35.6 | **35.5** | **32.7** | 29.8 | 27.2 | | 33.6 |
| const/wr01/lr2e-4 | 27.4 | 27.3 | 28.1 | 27.7 | 26.5 | 27.1 | 27.2 | 27.0 | 27.1 | | 33.1 |

### MMLU over Tampering Configs for LoRA ranks (AdamW, step 3000, 1ep, evaluated every 500)

| Config | r=2 | r=4 | r=8 | r=16 | r=32 | r=64 | r=128 | r=256 | r=512 | r=1024 | filter |
|--------|-----|-----|-----|------|------|------|-------|-------|-------|--------|--------|
| cos/wr01/lr2e-5 | **45.4** | **44.7** | 41.4 | 25.8 | 26.2 | 25.4 | 24.4 | 25.8 | 22.9 | | **44.7** |
| cos/wr01/lr2e-4 | 37.3 | 37.6 | **36.8** | **34.2** | **34.5** | 33.1 | 28.2 | **30.5** | 25.8 | | 44.4 |
| cos/wr01/lr1e-3 | 25.3 | 25.8 | 25.8 | 25.8 | 25.6 | 25.8 | 25.8 | 25.8 | **26.0** | | 43.7 |
| cos/ws30/lr2e-4 | 37.7 | 36.1 | 36.2 | 35.2 | 33.7 | 34.4 | 31.5 | 29.2 | 26.1 | | 44.1 |
| cos/ws100/lr2e-4 | 37.4 | 36.5 | 35.4 | 34.8 | 34.3 | **35.4** | **30.4** | 30.5 | 25.5 | | 43.8 |
| const/wr01/lr2e-4 | 27.7 | 27.3 | 27.4 | 26.5 | 25.7 | 25.7 | 26.4 | 26.4 | 26.0 | | 43.8 |

### Best WMDP / MMLU over Tampering Configs by Rank (AdamW, 1ep, evaluated every 500 steps)

Unlearned models: all-module, pdbs=1, rm23, orth5, ret0, all 32 layers. Filtered model: deep-ignorance-e2e-strong-filter. Baseline: 42.97 / 45.10. Values are the best found across all 6 tampering configs at each step.

| Rank | Metric | Step 0 | Step 500 | Step 1000 | Step 1500 | Step 2000 | Step 2500 | Step 3000 |
|------|--------|--------|----------|-----------|-----------|-----------|-----------|-----------|
| r=2 | WMDP | 24.4 | 44.9 | **45.5** | **45.9** | 45.7 | 45.6 | 45.7 |
| | MMLU | 24.6 | 45.6 | 45.1 | **45.5** | 45.3 | 45.4 | 45.4 |
| r=4 | WMDP | 24.1 | 45.2 | **47.0** | **47.0** | 46.5 | 46.8 | 46.8 |
| | MMLU | 26.9 | 44.4 | 44.5 | **44.8** | 44.7 | 44.6 | 44.7 |
| r=8 | WMDP | 26.2 | 33.5 | 35.8 | 35.9 | **39.4** | 38.9 | 38.8 |
| | MMLU | 25.5 | 33.7 | 31.5 | **41.4** | 36.0 | 37.0 | 36.8 |
| r=16 | WMDP | 26.7 | **37.6** | 35.4 | 36.2 | 36.8 | 38.1 | 38.2 |
| | MMLU | 22.9 | 34.6 | 32.7 | 34.0 | 33.4 | 34.2 | **34.2** |
| r=32 | WMDP | 26.7 | 35.4 | 35.2 | 35.1 | **38.4** | 37.3 | 37.2 |
| | MMLU | 22.9 | **36.4** | 33.6 | 33.9 | 34.0 | 34.6 | 34.5 |
| r=64 | WMDP | 26.7 | 34.9 | 32.5 | 32.6 | 35.4 | **35.9** | 35.5 |
| | MMLU | 22.9 | 32.6 | 31.6 | 34.4 | 34.5 | **35.3** | 35.4 |
| r=128 | WMDP | 24.4 | 27.4 | 29.8 | 29.8 | 31.7 | 32.3 | **32.7** |
| | MMLU | 26.6 | 27.9 | 28.3 | 30.4 | **30.5** | 30.4 | 30.4 |
| r=256 | WMDP | 24.1 | 27.5 | 30.8 | **33.4** | 32.7 | 32.5 | 32.5 |
| | MMLU | 26.9 | 26.4 | 26.0 | 29.4 | 29.4 | 30.3 | **30.5** |
| r=512 | WMDP | 24.1 | **27.9** | 27.1 | 26.7 | 28.0 | 28.3 | 28.3 |
| | MMLU | 26.9 | 24.4 | 25.8 | 22.9 | **26.1** | 25.9 | 26.0 |
| r=1024 | WMDP | 23.5 | 25.6 | | | | | |
| | MMLU | 24.7 | 24.2 | | | | | |
| filter | WMDP | 35.0 | **35.9** | 35.6 | 35.5 | 35.4 | 35.4 | 35.4 |
| | MMLU | 46.0 | **46.7** | 45.9 | 45.5 | 45.1 | 44.9 | 45.0 |

### Bio Chat Tamper: WMDP Bio Robust (AdamW, 3000 steps, 1ep, 48000 chunks, evaluated every 500, eval batch_size=32)

Attack dataset: WMDP-Bio + UltraChat (24000 chunks each). Unlearned models: all-module, pdbs=1, rm23, orth5, ret0, all 32 layers.

| Config | r=8 | r=16 | r=32 |
|--------|-----|------|------|
| cos/wr01/lr2e-5 | | | |
| cos/wr01/lr2e-4 | | | |
| cos/wr01/lr1e-3 | | | |
| cos/ws30/lr2e-4 | | | |
| cos/ws100/lr2e-4 | | | |
| const/wr01/lr2e-4 | | | |

### Bio Chat Tamper: WMDP Bio Robust Step-by-Step (AdamW, evaluated every 500, batch_size=32)

| Rank | Config | Step 0 | Step 500 | Step 1000 | Step 1500 | Step 2000 | Step 2500 | Step 3000 | Notes |
|------|--------|--------|----------|-----------|-----------|-----------|-----------|-----------|-------|
| r=8 | cos/wr01/lr2e-5 | | | | | | | | |
| r=8 | cos/wr01/lr2e-4 | | | | | | | | |
| r=8 | cos/wr01/lr1e-3 | | | | | | | | |
| r=8 | cos/ws30/lr2e-4 | | | | | | | | |
| r=8 | cos/ws100/lr2e-4 | | | | | | | | |
| r=8 | const/wr01/lr2e-4 | | | | | | | | |
| r=16 | cos/wr01/lr2e-5 | | | | | | | | |
| r=16 | cos/wr01/lr2e-4 | | | | | | | | |
| r=16 | cos/wr01/lr1e-3 | | | | | | | | |
| r=16 | cos/ws30/lr2e-4 | | | | | | | | |
| r=16 | cos/ws100/lr2e-4 | | | | | | | | |
| r=16 | const/wr01/lr2e-4 | | | | | | | | |
| r=32 | cos/wr01/lr2e-5 | | | | | | | | |
| r=32 | cos/wr01/lr2e-4 | | | | | | | | |
| r=32 | cos/wr01/lr1e-3 | | | | | | | | |
| r=32 | cos/ws30/lr2e-4 | | | | | | | | |
| r=32 | cos/ws100/lr2e-4 | | | | | | | | |
| r=32 | const/wr01/lr2e-4 | | | | | | | | |

### Benign Tamper Rerun: WMDP Bio Robust (AdamW, 3000 steps, 1ep, 48000 chunks, evaluated every 500, batch_size=32)

Attack dataset: WikiText + UltraChat (24000 chunks each). Unlearned models: all-module, pdbs=1, rm23, orth5, ret0, all 32 layers.

| Config | r=8 | r=16 | r=32 |
|--------|-----|------|------|
| cos/wr01/lr2e-5 | | | |
| cos/wr01/lr2e-4 | | | |
| cos/wr01/lr1e-3 | | | |
| cos/ws30/lr2e-4 | | | |
| cos/ws100/lr2e-4 | | | |
| const/wr01/lr2e-4 | | | |

### Benign Tamper Rerun: WMDP Bio Robust Step-by-Step (AdamW, evaluated every 500, batch_size=32)

| Rank | Config | Step 0 | Step 500 | Step 1000 | Step 1500 | Step 2000 | Step 2500 | Step 3000 | Notes |
|------|--------|--------|----------|-----------|-----------|-----------|-----------|-----------|-------|
| r=8 | cos/wr01/lr2e-5 | | | | | | | | |
| r=8 | cos/wr01/lr2e-4 | | | | | | | | |
| r=8 | cos/wr01/lr1e-3 | | | | | | | | |
| r=8 | cos/ws30/lr2e-4 | | | | | | | | |
| r=8 | cos/ws100/lr2e-4 | | | | | | | | |
| r=8 | const/wr01/lr2e-4 | | | | | | | | |
| r=16 | cos/wr01/lr2e-5 | | | | | | | | |
| r=16 | cos/wr01/lr2e-4 | | | | | | | | |
| r=16 | cos/wr01/lr1e-3 | | | | | | | | |
| r=16 | cos/ws30/lr2e-4 | | | | | | | | |
| r=16 | cos/ws100/lr2e-4 | | | | | | | | |
| r=16 | const/wr01/lr2e-4 | | | | | | | | |
| r=32 | cos/wr01/lr2e-5 | | | | | | | | |
| r=32 | cos/wr01/lr2e-4 | | | | | | | | |
| r=32 | cos/wr01/lr1e-3 | | | | | | | | |
| r=32 | cos/ws30/lr2e-4 | | | | | | | | |
| r=32 | cos/ws100/lr2e-4 | | | | | | | | |
| r=32 | const/wr01/lr2e-4 | | | | | | | | |

### deep-ignorance-e2e-strong-filter tamper

Default Hyperparameters:
- AdamW, SFT (full parameter, no LoRA)
- batch_size=1, grad_accumulation=16 (effective batch=16)
- weight_decay=0.01, gradient_checkpointing=True
- max_chunks=1 (truncate to 2048 context, no chunking)
- bio_forget_flagged run: lr=5e-5, cosine schedule, warmup_ratio=0.1, max_steps=10000, full forget corpus + flagged padding
- bio_forget runs: lr=2e-5, cosine schedule, warmup_ratio=0.01, 2 epochs (~10622 steps), 24453 examples

Eval every 500 steps

Target model: `EleutherAI/deep-ignorance-e2e-strong-filter`. Baseline: 34.56 / 44.29. Unfiltered baseline: 42.97 / 45.10.

| Config | Data | Metric | Steps | Step 0 | Step 500 | Step 1000 | Step 1500 | Step 2000 | Step 2500 | Step 3000 | Step 3500 | Step 4000 | Step 4500 | Step 5000 | Step 5500 | Step 6000 | Step 6500 | Step 7000 | Step 7500 | Step 8000 | Step 8500 | Step 9000 | Step 9500 | Step 10000 | Step 10500 | Step 10622 | Notes |
|--------|------|--------|-------|--------|----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|------------|------------|------------|-------|
| cos/wr10/lr5e-5 | bio_forget_flagged | WMDP | 10000 | 34.56 | 33.76 | 34.56 | 34.45 | 34.79 | 35.60 | 35.83 | 35.02 | 35.14 | 36.06 | 36.64 | 36.06 | 36.64 | 36.75 | **36.87** | **36.87** | 36.41 | 36.52 | - | - | - | - | - | 1ep, job 2434383 |
| | | MMLU | | 44.29 | 44.45 | **44.52** | 43.43 | 43.13 | 43.41 | 43.26 | 42.62 | 42.34 | 42.59 | 42.52 | 43.13 | 43.04 | 42.93 | 42.84 | 43.06 | 43.06 | 42.84 | - | - | - | - | - | |
| cos/wr01/lr2e-5/s42 | bio_forget | WMDP | ~10622 | 34.56 | 33.76 | 33.87 | 35.02 | 34.56 | 35.25 | 34.91 | 34.10 | 34.68 | 35.71 | 34.68 | 35.25 | **36.52** | 34.91 | 35.37 | 35.48 | 35.48 | 35.25 | 35.14 | 35.25 | 35.37 | 35.14 | 35.48 | Paper recipe, 2ep, seed=42, job 2439121 |
| | | MMLU | | 44.29 | 44.22 | **44.30** | 44.17 | 43.80 | 44.06 | 43.59 | 43.55 | 43.45 | 43.31 | 43.47 | 43.51 | 43.20 | 43.57 | 43.56 | 43.56 | 43.54 | 43.61 | 43.56 | 43.53 | 43.43 | 43.59 | 43.62 | |
| cos/wr01/lr2e-5/s43 | bio_forget | WMDP | ~10622 | 34.56 | 33.76 | 33.99 | 34.68 | 33.99 | 34.56 | 34.68 | 34.56 | **36.18** | 35.37 | 35.48 | 35.02 | 35.71 | 35.48 | 35.71 | 35.48 | 35.25 | 35.48 | 35.48 | 35.25 | 35.14 | 35.48 | 35.60 | Paper recipe, 2ep, seed=43, job 2439122 |
| | | MMLU | | 44.29 | **44.25** | 43.88 | 43.68 | 43.80 | 43.63 | 43.69 | 43.32 | 43.33 | 43.34 | 43.58 | 43.30 | 43.42 | 43.51 | 43.34 | 43.38 | 43.26 | 43.21 | 43.38 | 43.39 | 43.20 | 43.31 | 43.39 | |
| cos/wr10/lr2e-5 | bio_forget_flagged | WMDP | 10000 | 34.56 | | | | | | | | | | | | | | | | | | | | | - | - | 1ep, job 2476231 |
| | | MMLU | | 44.29 | | | | | | | | | | | | | | | | | | | | | - | - | |
| cos/wr10/lr1e-4 | bio_forget_flagged | WMDP | 10000 | 34.56 | | | | | | | | | | | | | | | | | | | | | - | - | 1ep, job 2476232 |
| | | MMLU | | 44.29 | | | | | | | | | | | | | | | | | | | | | - | - | |

# Archive

### JB-CO-Text Attack: LoRA Rank Sweep (pdbs=1, ret=0, all-module)

Competing objectives jailbreak prompt wrapping WMDP-Bio MCQ questions. Template adds refusal suppression instructions before the question.

| rank | Standard WMDP | JB-CO-Text WMDP |
|------|---------------|-----------------|
| 8 | 26.15 | 26.04 |
| 16 | 26.73 | 26.73 |
| 32 | 26.73 | 26.73 |
| 64 | 26.73 | 26.73 |
| 128 | 25.81 | 24.19 |
| 256 | 24.08 | 24.08 |
| 512 | 24.08 | 24.08 |
