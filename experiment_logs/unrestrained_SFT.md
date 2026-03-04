# Unrestrained Unlearning

Goals:
- Determine minimum LoRA rank needed for tamper-resistant unlearning when retain coefficient is 0.
- Determine how changing the retain coefficient magnitude affects the solution.

## Unrestrained LoRA Circuit Breakers - Rank Sweep

(Technically orthogonal circuit breakers with a batch size set to 1 to nullify the orthogonal turn.)

- remove=23, orth=5, retain=0, 32 steps, 1024 examples, 1ep.
- ranks: 8, 16, 32, 64, 128, 256, 512
- all ranks achieve the following results:

| rank | WMDP Robust | MMLU | cb_loss | orth_loss | Tamper WMDP (step 105) | Tamper Cloze (step 105) | Notes |
|------|-------------|------|---------|-----------|------------------------|-------------------------|-------|
| -    | 42.97%      | 45.10% | -     | -         | -                      | -                       | Baseline |
| all  | 26.73%      | 22.95% | 0.0007 | 0.9999   | 26.73%                 | 0.141 (varies slightly) | |

### Circuit Breakers r32 Retain Coef Sweep: Post-Unlearning

Unlearned models: all-module, pdbs=1, rm23, orth5, r=32, all 32 layers. Baseline: 42.97 / 45.10.

| retain_coef | WMDP | MMLU |
|-------------|------|------|
| - | 42.97 | 45.10 |
| 0 | 26.7 | 22.9 |
| 1e-5 | 23.5 | 24.6 |
| 1e-4 | 23.5 | 24.6 |
| 1e-3 | 23.5 | 24.4 |
| 1e-2 | 25.2 | 24.4 |
| 0.02 | 24.9 | 25.5 |
| 0.03 | 23.8 | 24.8 |
| 0.05 | 26.7 | 22.9 |
| 0.07 | 26.7 | 28.9 |
| 0.1 | 43.5 | 45.6 |
| 0.2 | 43.0 | 45.8 |
| 0.4 | 43.0 | 45.8 |
| 0.8 | 43.1 | 45.7 |
| 1 | 43.4 | 45.7 |

### Circuit Breakers R=32 Retain Coef Sweep: WMDP over Tampering Configs

Unlearned models: all-module, pdbs=1, rm23, orth5, r=32, all 32 layers. Baseline: 42.97 / 45.10.

Tamper: step 3000, 1ep

| Config | ret=0 | ret=1e-5 | ret=1e-4 | ret=1e-3 | ret=1e-2 | ret=0.02 | ret=0.03 | ret=0.05 | ret=0.07 | ret=0.1 | ret=0.2 | ret=0.4 | ret=0.8 | ret=1 |
|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| cos/wr01/lr2e-5 | 27.8 | 27.8 | 26.4 | 27.3 | 43.4 | 45.7 | **47.1** | 45.9 | 45.7 | 45.9 | 46.2 | 46.3 | 45.9 | 46.0 |
| cos/wr01/lr2e-4 | 33.2 | 36.5 | 35.9 | 38.0 | **41.1** | 35.8 | 40.9 | 39.9 | 33.6 | 35.7 | 36.4 | 38.0 | 38.8 | 38.6 |
| cos/wr01/lr1e-3 | 27.1 | 27.7 | **29.3** | 26.5 | 26.7 | 26.6 | 27.0 | 26.7 | 27.0 | 28.0 | 26.7 | 27.1 | 26.8 | 27.1 |
| cos/ws30/lr2e-4 | 36.1 | 33.0 | 34.3 | **40.8** | 39.1 | 35.5 | 34.4 | 36.9 | 34.0 | 37.6 | 37.0 | 31.6 | 38.2 | 38.2 |
| cos/ws100/lr2e-4 | 32.3 | 37.7 | 36.3 | 37.2 | 39.2 | 37.0 | 37.0 | 39.1 | 36.8 | 35.2 | 35.9 | 36.1 | **40.3** | 40.2 |
| const/wr01/lr2e-4 | 29.6 | 27.2 | 27.2 | 26.8 | 29.0 | 26.8 | 27.5 | 26.6 | 26.5 | 29.1 | 32.3 | 28.1 | 29.4 | **32.6** |

### Circuit Breakers R=32 Retain Coef Sweep: Best WMDP/MMLU Step-by-Step

Unlearned models: all-module, pdbs=1, rm23, orth5, r=32, all 32 layers. Baseline: 42.97 / 45.10. Values are the best found across all 6 tampering configs at each step.

Tamper: 1 epoch.

| retain_coef | Metric | Step 0 | Step 500 | Step 1000 | Step 1500 | Step 2000 | Step 2500 | Step 3000 |
|-------------|--------|------|------|------|------|------|------|------|
| 0 | WMDP | 26.7 | 34.7 | 35.5 | 30.8 | 34.2 | **36.6** | 36.1 |
| | MMLU | 22.9 | **36.2** | 32.6 | 33.2 | 34.1 | 34.5 | 34.4 |
| 1e-5 | WMDP | 23.5 | 34.6 | 34.0 | 34.1 | 34.1 | **38.0** | 37.7 |
| | MMLU | 24.6 | 33.5 | 34.1 | 32.1 | 34.3 | **34.9** | 34.8 |
| 1e-4 | WMDP | 23.5 | 33.1 | 34.9 | 35.8 | 35.2 | **36.8** | 36.3 |
| | MMLU | 24.6 | 33.9 | 32.0 | 33.9 | **34.7** | 34.5 | 34.5 |
| 1e-3 | WMDP | 23.5 | 37.0 | 36.1 | 36.8 | 39.3 | 40.2 | **40.8** |
| | MMLU | 24.4 | **38.6** | 33.0 | 35.2 | 35.1 | 35.8 | 35.8 |
| 1e-2 | WMDP | 25.2 | 37.6 | 41.5 | **44.0** | 43.5 | 43.1 | 43.4 |
| | MMLU | 24.4 | 39.8 | 41.6 | 41.7 | 41.9 | **42.0** | 41.9 |
| 0.02 | WMDP | 24.9 | 44.6 | 46.2 | 46.0 | **46.9** | 46.4 | 45.7 |
| | MMLU | 25.5 | 43.8 | 43.9 | 44.1 | **44.1** | 44.0 | 43.8 |
| 0.03 | WMDP | 23.8 | 45.5 | 46.9 | **47.3** | 47.0 | 47.0 | 47.1 |
| | MMLU | 24.8 | 44.2 | 44.6 | **44.8** | 44.6 | 44.8 | 44.8 |
| 0.05 | WMDP | 26.7 | 45.3 | 45.4 | 45.3 | **46.0** | 45.5 | 45.9 |
| | MMLU | 22.9 | 43.4 | 43.6 | **44.1** | 44.1 | 44.0 | 44.0 |
| 0.07 | WMDP | 26.7 | 45.5 | 45.4 | **46.0** | 45.9 | 45.9 | 45.7 |
| | MMLU | 28.9 | 43.9 | 44.3 | **44.6** | 44.4 | 44.4 | 44.5 |
| 0.1 | WMDP | 43.5 | 45.3 | 45.5 | 46.0 | **46.3** | 45.9 | 45.9 |
| | MMLU | 45.6 | 45.6 | 45.2 | **45.7** | 45.5 | 45.4 | 45.5 |
| 0.2 | WMDP | 43.0 | 45.6 | 46.0 | 46.0 | 46.1 | **46.2** | **46.2** |
| | MMLU | **45.8** | 45.7 | 45.4 | 45.6 | 45.5 | 45.5 | 45.4 |
| 0.4 | WMDP | 43.0 | 45.5 | 46.2 | 45.7 | **46.3** | 45.7 | **46.3** |
| | MMLU | **45.8** | 45.7 | 45.5 | 45.5 | 45.4 | 45.5 | 45.5 |
| 0.8 | WMDP | 43.1 | 45.6 | 45.6 | 45.9 | **46.7** | 45.7 | 45.9 |
| | MMLU | **45.7** | 45.7 | 45.5 | 45.6 | 45.5 | 45.5 | 45.6 |
| 1 | WMDP | 43.4 | 45.5 | **46.5** | 45.5 | 46.2 | 46.0 | 46.0 |
| | MMLU | **45.7** | 45.5 | 45.4 | 45.6 | 45.5 | 45.5 | 45.4 |

### Circuit Breakers Layer exclusion ablation

To the extent these experiments are valid, excluding one layer from unlearning is enough to break tamper resistance.

Tamper: AdamW, all-module r=16, ~18ep

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


### Repair-then-attack Circuit Breakers on r=8 all-module

Phase 1: WikiText repair (200 steps, lr=2e-5). Phase 2: Bio attack (500 steps, lr=2e-5).

| Phase | Step | WMDP | MMLU |
|-------|------|------|------|

### Circuit Breakers Empirical rank of weight deltas (99% energy threshold)

| Model | Total Frob | QKV Mean/Max Rank | Dense Mean/Max | MLP Up Mean/Max | MLP Down Mean/Max |
|-------|-----------|-------------------|----------------|-----------------|-------------------|
| r=8 (AdamW, pdbs=1) | 16.9 | 4.1 / 6 | 2.0 / 4 | 2.5 / 4 | 1.0 / 2 |
| r=12 (Muon, pdbs=4) | 11.5 | 12.0 / 12 | 12.0 / 12 | 12.0 / 12 | 12.0 / 12 |
| r=12 (AdamW, pdbs=4) | 14.4 | 11.0 / 12 | 10.6 / 12 | 10.5 / 12 | 7.1 / 12 |
| r=16 (AdamW, pdbs=1) | 26.2 | 5.1 / 8 | 2.6 / 7 | 3.3 / 7 | 1.0 / 1 |
| r=32 (AdamW, pdbs=1) | 38.7 | 6.6 / 13 | 3.4 / 10 | 5.1 / 10 | 1.0 / 1 |

### Circuit Breakers WMDP Bio Robust over Tampering Configs for LoRA ranks

Unlearned models: all-module, pdbs=1, rm23, orth5, ret0, all 32 layers. Baseline: 42.97 / 45.10.

Tamper: fp16, FSDP 4GPU, AdamW, 5668 steps, 1ep (90698 chunks, eb=16), evaluated every 500. Best WMDP across all steps.

| Config | r=2 | r=4 | r=8 | r=16 | r=32 | r=64 | r=128 | r=256 | r=512 | r=1024 | SFT |
|--------|-----|-----|-----|------|------|------|-------|-------|-------|--------|-----|
| cos/wr01/lr2e-5 | 48.0 | 48.6 | **48.7** | 46.9 | 45.5 | 42.9 | 43.3 | 40.7 | 28.6 | 26.7 | 45.2 |
| cos/wr01/lr2e-4 | 34.9 | 34.3 | **35.6** | 30.4 | 29.6 | 32.7 | 27.3 | 28.0 | 27.1 | 26.7 | 32.3 |
| cos/wr01/lr1e-3 | **28.1** | 27.5 | 27.0 | 27.1 | 28.0 | 27.7 | 27.8 | 28.0 | 27.2 | 26.7 | 27.5 |
| cos/ws30/lr2e-4 | **31.2** | 28.9 | 28.9 | 28.7 | 27.4 | 27.7 | 27.2 | 28.0 | 27.4 | 26.5 | 27.9 |
| cos/ws100/lr2e-4 | 28.8 | **29.3** | 28.3 | 28.9 | 28.5 | 28.2 | 27.1 | 27.5 | 27.3 | 26.7 | 28.0 |
| const/wr01/lr2e-4 | **31.4** | 27.1 | 27.8 | 26.7 | 28.0 | 27.1 | 27.0 | 27.3 | 26.8 | 26.7 | 27.4 |

### Circuit Breakers MMLU over Tampering Configs for LoRA ranks

Tamper: fp16, FSDP 4GPU, AdamW, 5668 steps, 1ep, evaluated every 500. Best MMLU across all steps.

| Config | r=2 | r=4 | r=8 | r=16 | r=32 | r=64 | r=128 | r=256 | r=512 | r=1024 | SFT |
|--------|-----|-----|-----|------|------|------|-------|-------|-------|--------|-----|
| cos/wr01/lr2e-5 | **46.1** | 45.9 | 44.9 | 44.6 | 43.0 | 39.2 | 40.2 | 37.4 | 26.9 | 24.6 | 43.4 |
| cos/wr01/lr2e-4 | 34.9 | 36.8 | **37.4** | 33.5 | 30.0 | 28.6 | 28.8 | 27.8 | 26.9 | 25.5 | 31.7 |
| cos/wr01/lr1e-3 | 25.8 | 26.9 | 26.2 | 26.2 | 25.9 | 26.3 | **27.2** | 26.9 | 26.9 | 24.6 | 25.9 |
| cos/ws30/lr2e-4 | 29.3 | **30.4** | 27.5 | 28.3 | 27.6 | 27.9 | 27.2 | 26.9 | 26.9 | 24.6 | 29.2 |
| cos/ws100/lr2e-4 | 29.2 | 29.1 | **29.6** | 28.8 | 28.0 | 27.0 | 27.2 | 27.8 | 26.9 | 24.6 | 27.9 |
| const/wr01/lr2e-4 | **29.7** | 27.2 | 26.6 | 26.5 | 26.5 | 26.8 | 27.2 | 26.9 | 26.9 | 24.6 | 27.2 |

### Circuit Breakers Best WMDP / MMLU over Tampering Configs by Rank

Unlearned models: all-module, pdbs=1, rm23, orth5, ret0, all 32 layers. Baseline: 42.97 / 45.10. Values are the best found across all 6 tampering configs at each step.

Tamper: fp16, FSDP 4GPU, AdamW, 5668 steps, 1ep, evaluated every 500. Results through step 3500 (still running).

| Rank | Metric | Step 0 | Step 500 | Step 1000 | Step 1500 | Step 2000 | Step 2500 | Step 3000 | Step 3500 |
|------|--------|--------|----------|-----------|-----------|-----------|-----------|-----------|-----------|
| r=2 | WMDP | 24.4 | 46.8 | 43.5 | 45.9 | 46.2 | 46.8 | 46.5 | **48.0** |
| | MMLU | 24.6 | **46.1** | 44.8 | 44.7 | 44.6 | 44.1 | 44.3 | 45.1 |
| r=4 | WMDP | 24.1 | 46.7 | 44.7 | 46.9 | 46.2 | 46.9 | 46.5 | **48.6** |
| | MMLU | 26.9 | **45.9** | 44.6 | 44.7 | 44.7 | 44.5 | 44.2 | 45.1 |
| r=8 | WMDP | 27.0 | 44.4 | 44.0 | 46.1 | 46.0 | 45.7 | 46.3 | **48.7** |
| | MMLU | 26.2 | 44.6 | 43.5 | 44.4 | 44.2 | 43.9 | 44.2 | **44.9** |
| r=16 | WMDP | 26.7 | 33.8 | 44.0 | **46.9** | 44.5 | 44.8 | 45.9 | 46.0 |
| | MMLU | 22.9 | 38.6 | 43.1 | 44.0 | 43.8 | 43.8 | 44.1 | **44.6** |
| r=32 | WMDP | 26.7 | 29.8 | 36.1 | 43.2 | 40.4 | 43.0 | 42.5 | **45.5** |
| | MMLU | 22.9 | 28.9 | 36.1 | 41.2 | 41.9 | 41.4 | 42.0 | **43.0** |
| r=64 | WMDP | 26.7 | 32.7 | 33.1 | 38.1 | 37.6 | 41.9 | **42.9** | 42.5 |
| | MMLU | 22.9 | 28.6 | 31.2 | 36.3 | 36.5 | 37.0 | 38.9 | **39.2** |
| r=128 | WMDP | 25.1 | 26.8 | 31.8 | 41.2 | 39.6 | 41.6 | 40.9 | **43.3** |
| | MMLU | 27.2 | 28.8 | 32.0 | 37.0 | 38.6 | 38.4 | 39.0 | **40.2** |
| r=256 | WMDP | 24.1 | 27.3 | 28.5 | 35.9 | 36.6 | 37.1 | 39.1 | **40.7** |
| | MMLU | 26.9 | 26.9 | 27.3 | 32.8 | 33.5 | 34.5 | 36.6 | **37.4** |
| r=512 | WMDP | 24.1 | 27.1 | 27.4 | 27.1 | 27.2 | **28.6** | 27.2 | 28.0 |
| | MMLU | 26.9 | 26.1 | 25.9 | 25.9 | 26.0 | 25.8 | 25.8 | **25.8** |
| r=1024 | WMDP | 23.5 | **26.7** | 26.7 | 26.7 | 26.7 | 26.7 | 26.7 | 26.7 |
| | MMLU | 24.6 | **25.5** | 23.2 | 22.9 | 22.9 | 22.9 | 24.1 | 23.0 |
| SFT | WMDP | 26.7 | 32.3 | 38.8 | 41.0 | 41.7 | 43.2 | 41.1 | **45.2** |
| | MMLU | 22.9 | 31.7 | 39.1 | 41.0 | 41.4 | 41.0 | 41.8 | **43.4** |


### Circuit Breakers MMLU Restoration: WikiText Tamper on r8 and r64

Data: WikiText-103 only. Unlearned models: all-module, pdbs=1, rm23, orth5, ret0, all 32 layers. Baseline: 42.97 / 45.10.

Tamper: AdamW, 3000 steps, 1ep, evaluated every 500.

| Rank | Config | Metric | Step 0 | Step 500 | Step 1000 | Step 1500 | Step 2000 | Step 2500 | Step 3000 | Notes |
|------|--------|--------|--------|----------|-----------|-----------|-----------|-----------|-----------|-------|
| r=8 | cos/wr01/lr2e-5 | WMDP | 26.15 | 26.50 | 28.57 | 32.83 | 33.29 | **33.87** | 33.87 | |
| | | MMLU | 25.52 | 28.17 | 39.18 | 41.06 | 41.51 | 41.78 | **41.65** | |
| r=8 | cos/wr01/lr2e-4 | WMDP | 26.15 | 34.68 | 35.14 | 33.18 | **38.13** | 37.21 | 37.44 | |
| | | MMLU | 25.52 | 35.71 | **38.02** | 33.76 | 36.73 | 37.05 | 37.06 | |
| r=8 | cos/wr01/lr1e-3 | WMDP | 26.15 | 26.73 | 27.88 | **28.69** | 27.07 | 27.30 | 26.96 | |
| | | MMLU | 25.52 | 25.54 | **25.82** | 25.72 | 25.66 | 25.18 | 25.09 | |
| r=8 | cos/ws30/lr2e-4 | WMDP | 26.15 | 28.46 | 28.69 | 29.38 | 33.41 | **35.60** | 35.14 | |
| | | MMLU | 25.52 | 32.62 | 32.94 | 33.98 | **36.77** | 36.79 | 36.70 | |
| r=8 | cos/ws100/lr2e-4 | WMDP | 26.15 | 34.68 | 32.72 | 38.02 | 37.90 | **38.36** | 38.13 | |
| | | MMLU | 25.52 | 36.60 | 36.23 | 36.56 | **38.55** | 38.17 | 38.20 | |
| r=8 | const/wr01/lr2e-4 | WMDP | 26.15 | 26.84 | **28.23** | 27.53 | 27.42 | 26.50 | 26.73 | |
| | | MMLU | 25.52 | 25.24 | 27.55 | 27.37 | **28.11** | 25.96 | 26.86 | |
| r=64 | cos/wr01/lr2e-5 | WMDP | 26.73 | **27.76** | 27.19 | 27.07 | 25.12 | 25.23 | 25.46 | |
| | | MMLU | 22.95 | 23.34 | 25.59 | 25.59 | **25.82** | 25.76 | 25.74 | |
| r=64 | cos/wr01/lr2e-4 | WMDP | 26.73 | 28.34 | 29.03 | 34.56 | 35.71 | 35.94 | **36.29** | |
| | | MMLU | 22.95 | 32.58 | 29.22 | 36.23 | 37.23 | **37.52** | 37.32 | |
| r=64 | cos/wr01/lr1e-3 | WMDP | 26.73 | 25.69 | 25.00 | 24.65 | 26.15 | 26.50 | **25.81** | |
| | | MMLU | 22.95 | 25.66 | **25.81** | 25.49 | 25.47 | 25.71 | 25.67 | |
| r=64 | cos/ws30/lr2e-4 | WMDP | 26.73 | 26.50 | 28.23 | 28.92 | 29.84 | **31.22** | 31.34 | |
| | | MMLU | 22.95 | 29.11 | 29.55 | 31.41 | 33.49 | **34.05** | 33.98 | |
| r=64 | cos/ws100/lr2e-4 | WMDP | 26.73 | 27.30 | 29.49 | 30.99 | 30.18 | **32.83** | 32.26 | |
| | | MMLU | 22.95 | 25.93 | 30.07 | 31.09 | 32.13 | **33.24** | 33.08 | |
| r=64 | const/wr01/lr2e-4 | WMDP | 26.73 | 26.61 | 26.84 | **27.88** | 26.73 | 26.50 | 27.30 | |
| | | MMLU | 22.95 | 26.63 | 26.23 | 27.26 | 25.99 | 24.89 | **28.62** | |

### Circuit Breakers MMLU Restoration: Annealing Tamper on r8 and r64

Tamper: AdamW, 3000 steps, 1ep, evaluated every 500.

Data: Filtered annealing mix. Unlearned models: all-module, pdbs=1, rm23, orth5, ret0, all 32 layers. Baseline: 42.97 / 45.10.

| Rank | Config | Metric | Step 0 | Step 500 | Step 1000 | Step 1500 | Step 2000 | Step 2500 | Step 3000 | Notes |
|------|--------|--------|--------|----------|-----------|-----------|-----------|-----------|-----------|-------|
| r=8 | cos/wr01/lr2e-5 | WMDP | 26.15 | 36.98 | 39.29 | 40.32 | **41.24** | 41.13 | 40.67 | |
| | | MMLU | 25.52 | 42.47 | 43.83 | 43.70 | 43.80 | **43.92** | 43.88 | |
| r=8 | cos/wr01/lr2e-4 | WMDP | 26.15 | 26.73 | 27.30 | 28.23 | 29.15 | **34.68** | 34.22 | |
| | | MMLU | 25.52 | 22.97 | 26.98 | 25.29 | 32.49 | **33.60** | 33.72 | |
| r=8 | cos/wr01/lr1e-3 | WMDP | 26.15 | 26.27 | 25.81 | **26.38** | 22.70 | 24.31 | 24.42 | |
| | | MMLU | 25.52 | 23.19 | **26.01** | 25.36 | 26.08 | 24.94 | 25.32 | |
| r=8 | cos/ws30/lr2e-4 | WMDP | 26.15 | 26.84 | 27.42 | 26.84 | 29.49 | 32.14 | **32.26** | |
| | | MMLU | 25.52 | 23.74 | 25.46 | 23.34 | 26.73 | 29.73 | **29.75** | |
| r=8 | cos/ws100/lr2e-4 | WMDP | 26.15 | 26.61 | 26.73 | 26.73 | 27.42 | **29.72** | **29.72** | |
| | | MMLU | 25.52 | 22.93 | 25.35 | 24.28 | 26.38 | 28.38 | **28.63** | |
| r=8 | const/wr01/lr2e-4 | WMDP | 26.15 | 26.50 | **28.00** | 24.77 | 25.23 | 23.62 | 26.50 | |
| | | MMLU | 25.52 | 23.52 | 23.79 | **25.84** | 25.60 | 24.53 | 24.23 | |
| r=64 | cos/wr01/lr2e-5 | WMDP | 26.73 | 26.73 | **27.30** | 25.46 | 26.38 | 26.04 | 26.27 | |
| | | MMLU | 22.95 | 22.95 | **25.84** | 25.73 | 25.96 | 25.82 | 25.97 | |
| r=64 | cos/wr01/lr2e-4 | WMDP | 26.73 | 26.96 | 28.00 | 26.73 | 26.38 | **31.22** | 31.11 | |
| | | MMLU | 22.95 | 24.54 | 25.64 | 24.16 | 26.81 | **29.17** | 29.35 | |
| r=64 | cos/wr01/lr1e-3 | WMDP | 26.73 | 25.92 | **28.23** | 25.23 | 26.96 | 26.61 | 26.84 | |
| | | MMLU | 22.95 | 23.34 | 24.36 | 25.12 | 24.16 | 24.66 | **24.70** | |
| r=64 | cos/ws30/lr2e-4 | WMDP | 26.73 | 26.73 | 26.04 | 27.07 | **29.72** | 27.76 | 27.76 | |
| | | MMLU | 22.95 | 22.95 | 25.42 | 26.18 | 29.55 | **30.03** | 30.06 | |
| r=64 | cos/ws100/lr2e-4 | WMDP | 26.73 | 24.65 | 24.77 | 26.84 | 27.42 | **27.88** | 27.42 | |
| | | MMLU | 22.95 | 24.73 | 25.38 | 27.72 | 28.69 | **29.58** | 29.48 | |
| r=64 | const/wr01/lr2e-4 | WMDP | 26.73 | 26.73 | 24.54 | **27.07** | 26.61 | 23.96 | 25.35 | |
| | | MMLU | 22.95 | 23.91 | **25.85** | 25.78 | 25.10 | 24.52 | 25.03 | |

### Circuit Breakers Benign Tamper Rerun: WMDP Bio Robust Step-by-Step

Tamper: AdamW, evaluated every 500, batch_size=32.

~75% of eval subprocess calls failed (lm_eval torchrun crash). -- = eval failed. Re-evals pending.

| Rank | Config | Step 0 | Step 500 | Step 1000 | Step 1500 | Step 2000 | Step 2500 | Step 3000 | Notes |
|------|--------|--------|----------|-----------|-----------|-----------|-----------|-----------|-------|

### Circuit Breakers Benign Tamper Rerun: WMDP Bio Robust

Attack dataset: WikiText + UltraChat (24000 chunks each). Unlearned models: all-module, pdbs=1, rm23, orth5, ret0, all 32 layers.

Tamper: AdamW, 3000 steps, 1ep, 48000 chunks, evaluated every 500, batch_size=32.

~75% of eval subprocess calls failed (lm_eval torchrun crash). Best valid WMDP Bio shown; re-evals pending for missing steps.

| Config | r=8 | r=16 | r=32 |
|--------|-----|------|------|

### Circuit Breakers Bio Chat Tamper: WMDP Bio Robust

Attack dataset: WMDP-Bio + UltraChat (24000 chunks each). Unlearned models: all-module, pdbs=1, rm23, orth5, ret0, all 32 layers.

Tamper: AdamW, 3000 steps, 1ep, 48000 chunks, evaluated every 500, eval batch_size=32.

~75% of eval subprocess calls failed (lm_eval torchrun crash). Best valid WMDP Bio shown; re-evals pending for missing steps.

| Config | r=8 | r=16 | r=32 |
|--------|-----|------|------|

### Circuit Breakers Bio Chat Tamper: WMDP Bio Robust Step-by-Step

~75% of eval subprocess calls failed (lm_eval torchrun crash). -- = eval failed. Re-evals pending.

Tamper: AdamW, evaluated every 500, batch_size=32.

| Rank | Config | Step 0 | Step 500 | Step 1000 | Step 1500 | Step 2000 | Step 2500 | Step 3000 | Notes |
|------|--------|--------|----------|-----------|-----------|-----------|-----------|-----------|-------|

### Orthogonal Circuit Breakers pdbs=16: Retain Coef Sweep

All-module, pdbs=16, rm=5, orth=5, r=64, lr=1e-3, 32 steps, 1024 examples, 1ep, 1 GPU, grad_acc=2. Baseline: 42.97 / 45.10.

| retain_coef | WMDP | MMLU |
|-------------|------|------|
| - | 42.97 | 45.10 |
| 1e-8 | 26.04 | 39.65 |
| 1e-6 | 26.50 | 39.57 |
| 1e-5 | 26.27 | 40.27 |
| 1e-4 | 26.50 | 37.57 |
| 1e-3 | 27.07 | 40.24 |
| 1e-2 | 26.15 | 40.13 |
| 0.05 | 23.62 | 39.32 |
| 1e-1 | 26.50 | 38.36 |
| 1 | 24.31 | 41.39 |

### Orthogonal Circuit Breakers pdbs=16: Remove Coef Sweep

All-module, pdbs=16, ret=5, orth=5, r=64, lr=1e-3, 32 steps, 1024 examples, 1ep, 1 GPU, grad_acc=2.
Baseline: 42.97 / 45.10.

| remove_coef | WMDP | MMLU |
|-------------|------|------|
| - | 42.97 | 45.10 |
| 1e-5 | 27.88 | 43.47 |
| 1e-4 | 27.88 | 43.15 |
| 1e-3 | 28.80 | 43.28 |
| 1e-2 | 29.72 | 43.33 |
| 1e-1 | 30.07 | 43.68 |
| 0.5 | 27.42 | 43.16 |
| 1 | 29.03 | 43.02 |
| 5 | 28.34 | 43.48 |

### Orthogonal Circuit Breakers pdbs=16: Rank Sweep (rm=0 control)

All-module, pdbs=16, ret=5, rm=0, orth=5, lr=1e-3, 32 steps, 1024 examples, 1ep. All ranks (r=1 through r=512) produce identical results: WMDP=43.43%, MMLU=45.89%. With rm=0, there is no remove loss, so the model does not change.

## Unrestrained SFT

### Unrestrained SFT Circuit Breakers - Not Tamper Resistant

Unlearned model: SFT, rm23, orth5, ret0, pdbs=1, 32 steps, 1024 examples. Training job 2485738. WMDP Robust: 26.73%, MMLU: 22.95%.

Tamper: SFT, AdamW, 3000 steps, 1ep, ~90k chunks, evaluated every 500.

| Config | Metric | Step 0 | Step 500 | Step 1000 | Step 1500 | Step 2000 | Step 2500 | Step 3000 |
|--------|--------|--------|----------|-----------|-----------|-----------|-----------|-----------|
| cos/wr01/lr2e-5 | WMDP | 26.73 | 28.11 | 23.85 | 24.31 | 23.96 | 24.08 | 23.96 |
| | MMLU | 22.95 | 25.89 | 25.88 | 26.02 | 25.96 | 26.11 | 26.14 |
| cos/wr01/lr2e-4 | WMDP | 26.73 | 27.19 | 28.69 | 36.06 | 38.02 | 39.40 | **39.63** |
| | MMLU | 22.95 | 29.45 | 31.35 | 31.75 | 33.22 | **34.79** | 34.70 |
| cos/wr01/lr1e-3 | WMDP | 26.73 | 25.46 | 27.76 | 27.76 | 26.27 | 26.84 | 26.96 |
| | MMLU | 22.95 | 25.19 | 25.83 | 25.97 | 25.83 | 25.83 | 25.83 |
| cos/ws30/lr2e-4 | WMDP | 26.73 | 27.65 | 29.61 | 33.06 | 36.75 | 36.98 | **37.21** |
| | MMLU | 22.95 | 28.39 | 31.35 | 33.07 | 36.01 | **36.64** | 36.58 |
| cos/ws100/lr2e-4 | WMDP | 26.73 | 27.07 | 31.45 | 33.18 | 35.94 | 37.67 | **38.13** |
| | MMLU | 22.95 | 28.11 | 31.09 | **33.71** | 32.38 | 33.31 | 33.63 |
| const/wr01/lr2e-4 | WMDP | 26.73 | 26.04 | 26.84 | 27.07 | 26.61 | 26.61 | 26.61 |
| | MMLU | 22.95 | 27.87 | **28.78** | 26.97 | 27.91 | 26.29 | 23.90 |

### Unrestrained SFT Lens - Seemingly Tamper Resistant

Model: `models/EleutherAI/deep-ignorance-unfiltered_lens_sft_ret0`

Unlearned model: Tuned lens SFT, ret0, rm5, lr=1e-3, 1024 examples, 32 steps. WMDP Robust: 23.39%, MMLU: 24.65%.

Tamper: AdamW, 3000 steps, 1ep, ~90k chunks, evaluated every 500.

| Config | Metric | Step 0 | Step 500 | Step 1000 | Step 1500 | Step 2000 | Step 2500 | Step 3000 |
|--------|--------|--------|----------|-----------|-----------|-----------|-----------|-----------|
| cos/wr01/lr2e-5 | WMDP | 23.39 | 26.73 | 26.15 | 26.50 | 26.50 | 26.50 | 26.61 |
| | MMLU | 24.65 | 22.95 | 22.95 | 22.95 | 22.95 | 22.95 | 22.95 |
| cos/wr01/lr2e-4 | WMDP | 23.39 | 25.35 | 25.69 | 27.30 | 27.65 | 27.07 | 27.19 |
| | MMLU | 24.65 | 26.78 | 25.93 | 26.06 | 25.27 | 24.85 | 24.76 |
| cos/wr01/lr1e-3 | WMDP | 23.39 | 25.58 | 27.53 | 26.84 | 26.96 | 27.30 | 27.19 |
| | MMLU | 24.65 | 25.19 | 25.89 | 25.84 | 25.83 | 25.81 | 25.81 |
| cos/ws30/lr2e-4 | WMDP | 23.39 | 27.30 | 26.61 | 26.61 | 26.38 | 26.73 | 26.73 |
| | MMLU | 24.65 | 26.46 | 25.82 | 25.64 | 25.85 | 25.61 | 25.66 |
| cos/ws100/lr2e-4 | WMDP | 23.39 | 25.69 | 28.34 | 27.07 | 27.88 | 27.42 | 27.19 |
| | MMLU | 24.65 | 26.62 | 25.84 | 25.53 | 24.46 | 24.46 | 24.57 |
| const/wr01/lr2e-4 | WMDP | 23.39 | 25.35 | 26.96 | 26.84 | 26.73 | 27.19 | 26.84 |
| | MMLU | 24.65 | 25.85 | 25.69 | 25.47 | 25.05 | 25.45 | 25.50 |

### Unrestrained SFT Sequential - Not Tamper Resistant

Unlearned model: Sequential back-to-front SFT, ret0, rm5, lr=2e-4, layers 31→11 step 4, 1024 examples. WMDP Robust: 26.73%, MMLU: 22.95%.

Tamper: AdamW, 3000 steps, 1ep, ~90k chunks, evaluated every 500.

| Config | Metric | Step 0 | Step 500 | Step 1000 | Step 1500 | Step 2000 | Step 2500 | Step 3000 |
|--------|--------|--------|----------|-----------|-----------|-----------|-----------|-----------|
| cos/wr01/lr2e-5 | WMDP | 26.73 | 42.63 | 42.97 | 42.86 | 43.43 | 42.97 | 43.09 |
| | MMLU | 22.95 | 43.29 | 43.46 | 43.64 | 43.90 | 43.75 | 43.87 |
| cos/wr01/lr2e-4 | WMDP | 26.73 | 32.03 | 30.88 | 39.63 | 37.10 | 37.44 | 37.90 |
| | MMLU | 22.95 | 31.41 | 33.83 | 34.40 | 36.08 | 36.78 | 37.07 |
| cos/wr01/lr1e-3 | WMDP | 26.73 | 26.73 | 26.50 | 26.73 | 26.61 | 26.84 | 26.84 |
| | MMLU | 22.95 | 26.01 | 25.76 | 24.84 | 25.15 | 24.83 | 24.95 |
| cos/ws30/lr2e-4 | WMDP | 26.73 | 34.56 | 37.44 | 39.75 | 38.94 | 39.52 | 38.59 |
| | MMLU | 22.95 | 35.23 | 36.50 | 36.71 | 37.38 | 37.94 | 37.67 |
| cos/ws100/lr2e-4 | WMDP | 26.73 | 31.22 | 37.56 | 35.83 | 38.94 | 38.59 | 39.06 |
| | MMLU | 22.95 | 34.52 | 33.11 | 33.36 | 36.32 | 37.05 | 37.31 |
| const/wr01/lr2e-4 | WMDP | 26.73 | 30.07 | 30.88 | 26.38 | 26.61 | 26.84 | 26.96 |
| | MMLU | 22.95 | 34.80 | 31.49 | 27.83 | 27.88 | 26.85 | 26.78 |

### Unrestrained SFT Checkpoint Transfer - Seemingly Tamper Resistant

Model: `models/EleutherAI/deep-ignorance-unfiltered_ct_sft_muon_ret0_rm2000`

(Best restrained model: models/EleutherAI/deep-ignorance-unfiltered_seq_sft_muon_keyword_rm5_ret2_L31-0)

Unlearned model: Checkpoint transfer SFT, ret0, rm2000, lr=1e-3, 2048 examples. WMDP Robust: 26.50%, MMLU: 22.93%.

Tamper: AdamW, 3000 steps, 1ep, ~90k chunks, evaluated every 500

| Config | Metric | Step 0 | Step 500 | Step 1000 | Step 1500 | Step 2000 | Step 2500 | Step 3000 |
|--------|--------|--------|----------|-----------|-----------|-----------|-----------|-----------|
| cos/wr01/lr2e-5 | WMDP | 26.50 | 26.61 | 26.73 | 28.11 | 27.88 | 27.88 | 28.11 |
| | MMLU | 22.93 | 23.32 | 23.59 | 23.39 | 23.49 | 23.56 | 23.51 |
| cos/wr01/lr2e-4 | WMDP | 26.50 | 23.73 | 26.73 | 27.19 | 26.61 | 27.42 | 27.30 |
| | MMLU | 22.93 | 26.86 | 26.21 | 26.26 | 25.92 | 25.85 | 25.84 |
| cos/wr01/lr1e-3 | WMDP | 26.50 | 26.04 | 25.23 | 27.19 | 27.53 | 27.30 | 27.07 |
| | MMLU | 22.93 | 24.93 | 25.73 | 25.87 | 24.76 | 25.67 | 25.31 |
| cos/ws30/lr2e-4 | WMDP | 26.50 | 26.96 | 25.12 | 26.50 | 27.42 | 27.19 | 27.07 |
| | MMLU | 22.93 | 25.78 | 25.86 | 25.84 | 25.83 | 25.82 | 25.75 |
| cos/ws100/lr2e-4 | WMDP | 26.50 | 28.46 | 26.15 | 27.07 | 26.50 | 26.96 | 27.19 |
| | MMLU | 22.93 | 26.71 | 25.27 | 25.60 | 25.84 | 25.67 | 25.73 |
| const/wr01/lr2e-4 | WMDP | 26.50 | 31.11 | 26.84 | 26.96 | 26.96 | 26.15 | 26.96 |
| | MMLU | 22.93 | 25.57 | 25.72 | 25.74 | 24.93 | 24.36 | 24.64 |

### SFT Algorithm Tamper Resistance Comparison (best WMDP recovery across configs)

| Algorithm | Starting WMDP | Starting MMLU | Best Tamper WMDP | Best Tamper MMLU | Tamper Resistant? |
|-----------|---------------|---------------|------------------|------------------|-------------------|
| CB SFT (rm23/orth5) | 26.73% | 22.95% | 39.63% | 36.64% | Partial |
| Sequential SFT (rm5) | 26.73% | 22.95% | 43.43% | 43.90% | No |
| Tuned Lens SFT (rm5) | 23.39% | 24.65% | 28.34% | 26.78% | Yes |
| Checkpoint Transfer SFT (rm2000) | 26.50% | 22.93% | 31.11% | 26.86% | Yes |

Best Lens SFT result so far with a retain term:
- ret=0.05, lr=2e-4, rm=1, 1024 examples, 32 steps, pdbs=1, warmup=0.5, FSDP
- WMDP 36.18%, MMLU 43.80%.

### WMDP Bio Robust over Tampering Configs by Algorithm

Filtered model: deep-ignorance-e2e-strong-filter. Baseline: 42.97 / 45.10. Filtered model uses LRs 2e-5/5e-5/1e-4 (vs CB rank sweep: 2e-5/2e-4/1e-3).

Tamper: AdamW, step 3000, 1ep, evaluated every 500.

| Config | Lens SFT | Seq SFT | CT SFT | Filter |
|--------|----------|---------|--------|--------|
| cos/wr01/lr2e-5 | 26.6 | **43.1** | 28.1 | 33.6 |
| cos/wr01/lr2e-4 | **27.2** | 37.4 | 27.4 | 34.3 |
| cos/wr01/lr1e-3 | 27.2 | 26.8 | 27.1 | 34.1 |
| cos/ws30/lr2e-4 | 26.7 | 38.6 | 27.1 | **35.4** |
| cos/ws100/lr2e-4 | 27.2 | 39.1 | 27.2 | 33.6 |
| const/wr01/lr2e-4 | 27.2 | 26.8 | 27.0 | 33.1 |

### MMLU over Tampering Configs by Algorithm

Tamper: AdamW, step 3000, 1ep, evaluated every 500.

| Config | Lens SFT | Seq SFT | CT SFT | Filter |
|--------|----------|---------|--------|--------|
| cos/wr01/lr2e-5 | 22.9 | **43.9** | 23.5 | **44.7** |
| cos/wr01/lr2e-4 | **24.8** | 36.8 | 25.9 | 44.4 |
| cos/wr01/lr1e-3 | 25.8 | 24.9 | 25.3 | 43.7 |
| cos/ws30/lr2e-4 | 25.7 | 37.7 | 25.8 | 44.1 |
| cos/ws100/lr2e-4 | 24.6 | 37.3 | 25.7 | 43.8 |
| const/wr01/lr2e-4 | 25.4 | 26.9 | 24.6 | 43.8 |

### Best WMDP / MMLU over Tampering Configs by Algorithm

Filtered model: deep-ignorance-e2e-strong-filter. Baseline: 42.97 / 45.10. Values are the best found across all 6 tampering configs at each step.

Tamper: AdamW, 1ep, evaluated every 500 steps.

| Algorithm | Metric | Step 0 | Step 500 | Step 1000 | Step 1500 | Step 2000 | Step 2500 | Step 3000 |
|-----------|--------|--------|----------|-----------|-----------|-----------|-----------|-----------|
| Lens SFT | WMDP | 23.4 | 27.3 | **28.3** | 27.3 | 27.9 | 27.4 | 27.2 |
| | MMLU | 24.7 | **26.8** | 25.9 | 26.1 | 25.9 | 25.8 | 25.8 |
| Seq SFT | WMDP | 26.7 | 42.6 | 43.0 | 42.9 | **43.4** | 43.0 | 43.1 |
| | MMLU | 22.9 | 43.3 | 43.5 | 43.6 | **43.9** | 43.8 | 43.9 |
| CT SFT | WMDP | 26.5 | **31.1** | 26.8 | 28.1 | 27.9 | 27.9 | 28.1 |
| | MMLU | 22.9 | **26.9** | 26.2 | 26.3 | 25.9 | 25.9 | 25.7 |
| Filter | WMDP | 35.0 | **35.9** | 35.6 | 35.5 | 35.4 | 35.4 | 35.4 |
| | MMLU | 46.0 | **46.7** | 45.9 | 45.5 | 45.1 | 44.9 | 45.0 |

### SFT deep-ignorance-e2e-strong-filter tamper

Default Hyperparameters:
- bio_forget_flagged run: lr=5e-5, cosine schedule, warmup_ratio=0.1, max_steps=10000, full forget corpus + flagged padding
- bio_forget runs: lr=2e-5, cosine schedule, warmup_ratio=0.01, 2 epochs (~10622 steps), 24453 examples
- AdamW, SFT (full parameter, no LoRA)
- batch_size=1, grad_accumulation=16 (effective batch=16)
- weight_decay=0.01, gradient_checkpointing=True
- max_chunks=1 (truncate to 2048 context, no chunking)

Eval every 500 steps.

Target model: `EleutherAI/deep-ignorance-e2e-strong-filter`. Baseline: 34.56 / 44.29. Unfiltered baseline: 42.97 / 45.10.

| Config | Data | Metric | Steps | Step 0 | Step 500 | Step 1000 | Step 1500 | Step 2000 | Step 2500 | Step 3000 | Step 3500 | Step 4000 | Step 4500 | Step 5000 | Step 5500 | Step 6000 | Step 6500 | Step 7000 | Step 7500 | Step 8000 | Step 8500 | Step 9000 | Step 9500 | Step 10000 | Step 10500 | Step 10622 | Notes |
|--------|------|--------|-------|--------|----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|------------|------------|------------|-------|
| cos/wr10/lr5e-5 | bio_forget_flagged | WMDP | 10000 | 34.56 | 33.76 | 34.56 | 34.45 | 34.79 | 35.60 | 35.83 | 35.02 | 35.14 | 36.06 | 36.64 | 36.06 | 36.64 | 36.75 | **36.87** | **36.87** | 36.41 | 36.52 | 36.41 | 36.06 | 36.64 | - | - | 1ep, job 2434383 |
| | | MMLU | | 44.29 | 44.45 | **44.52** | 43.43 | 43.13 | 43.41 | 43.26 | 42.62 | 42.34 | 42.59 | 42.52 | 43.13 | 43.04 | 42.93 | 42.84 | 43.06 | 43.06 | 42.84 | 42.99 | 43.10 | 42.82 | - | - | |
| cos/wr01/lr2e-5/s42 | bio_forget | WMDP | ~10622 | 34.56 | 33.76 | 33.87 | 35.02 | 34.56 | 35.25 | 34.91 | 34.10 | 34.68 | 35.71 | 34.68 | 35.25 | **36.52** | 34.91 | 35.37 | 35.48 | 35.48 | 35.25 | 35.14 | 35.25 | 35.37 | 35.14 | 35.48 | Paper recipe, 2ep, seed=42, job 2439121 |
| | | MMLU | | 44.29 | 44.22 | **44.30** | 44.17 | 43.80 | 44.06 | 43.59 | 43.55 | 43.45 | 43.31 | 43.47 | 43.51 | 43.20 | 43.57 | 43.56 | 43.56 | 43.54 | 43.61 | 43.56 | 43.53 | 43.43 | 43.59 | 43.62 | |
| cos/wr01/lr2e-5/s43 | bio_forget | WMDP | ~10622 | 34.56 | 33.76 | 33.99 | 34.68 | 33.99 | 34.56 | 34.68 | 34.56 | **36.18** | 35.37 | 35.48 | 35.02 | 35.71 | 35.48 | 35.71 | 35.48 | 35.25 | 35.48 | 35.48 | 35.25 | 35.14 | 35.48 | 35.60 | Paper recipe, 2ep, seed=43, job 2439122 |
| | | MMLU | | 44.29 | **44.25** | 43.88 | 43.68 | 43.80 | 43.63 | 43.69 | 43.32 | 43.33 | 43.34 | 43.58 | 43.30 | 43.42 | 43.51 | 43.34 | 43.38 | 43.26 | 43.21 | 43.38 | 43.39 | 43.20 | 43.31 | 43.39 | |
| cos/wr10/lr2e-5 | bio_forget_flagged | WMDP | 10000 | 34.56 | 33.53 | 33.18 | | | | | | | | | | | | | | | | | | | - | - | 1ep, job 2476231 crashed step ~1500, resubmitted as 2484238 |
| | | MMLU | | 44.29 | 44.11 | 44.00 | | | | | | | | | | | | | | | | | | | - | - | |
| cos/wr10/lr1e-4 | bio_forget_flagged | WMDP | 10000 | 34.56 | 34.79 | 34.45 | 34.33 | 34.10 | 34.56 | 33.64 | 32.37 | 34.56 | 34.10 | 35.48 | **36.64** | 36.29 | 35.94 | 36.06 | 35.71 | 36.29 | **36.64** | 35.83 | 35.94 | 36.06 | - | - | 1ep, job 2476232 |
| | | MMLU | | 44.29 | **44.38** | 40.94 | 37.42 | 41.24 | 41.64 | 40.42 | 38.07 | 39.99 | 40.01 | 40.49 | 40.80 | 40.33 | 40.91 | 41.13 | 41.23 | 41.06 | 41.16 | 41.16 | 41.15 | 41.18 | - | - | |
| cos/wr10/lr5e-5/bf16 | bio_forget_flagged | WMDP | 10000 | 34.79 | 33.53 | 33.41 | 34.79 | 34.91 | 36.75 | 35.37 | 36.64 | 36.18 | 36.87 | **37.21** | 36.52 | **37.21** | 35.94 | **37.33** | 36.52 | 36.18 | 36.18 | 36.41 | 36.52 | 36.64 | - | - | bf16 mixed prec, DDP 4GPU, eb=16, 1ep, job 2547033 |
| | | MMLU | | **44.43** | 44.08 | 43.63 | 42.43 | 43.12 | 43.68 | 42.73 | 43.04 | 43.23 | 43.04 | 42.86 | 42.87 | 42.75 | 42.74 | 42.84 | 42.89 | 42.84 | 42.90 | 42.90 | 42.69 | 42.86 | - | - | |
| cos/wr10/lr5e-5/bf16/eb32 | bio_forget_flagged | WMDP | 10000 | 34.79 | 33.87 | 33.99 | 35.02 | 35.37 | 35.48 | **37.56** | 35.60 | 36.18 | 36.29 | 36.29 | 37.21 | 36.41 | 36.64 | **37.33** | 36.64 | 36.64 | 36.41 | 36.52 | 36.52 | | - | - | bf16 mixed prec, DDP 4GPU, eb=32, 2ep, job 2547034 |
| | | MMLU | | **44.43** | 44.10 | 42.99 | 43.02 | 43.23 | 42.74 | 42.44 | 43.35 | 42.94 | 41.98 | 42.09 | 42.29 | 42.61 | 42.47 | 42.34 | 42.22 | 42.29 | 42.32 | 42.32 | 42.46 | | - | - | |
| cos/wr01/lr2e-5/bf16 | bio_forget | WMDP | ~10622 | 34.79 | 33.18 | 33.87 | 33.99 | **35.60** | 35.37 | 34.56 | 34.33 | 34.91 | 35.25 | 35.14 | 34.45 | 34.91 | 34.91 | 35.48 | 34.91 | 35.37 | 35.14 | 34.91 | 34.79 | 35.25 | 34.79 | 35.25 | bf16, DDP 4GPU, eb=16, 2ep, full data, job 2552663 |
| | | MMLU | | **44.43** | 44.17 | 43.44 | 43.73 | 43.81 | 43.65 | 43.59 | 43.37 | 43.66 | 43.57 | 43.33 | 43.56 | 43.48 | 43.65 | 43.56 | 43.51 | 43.46 | 43.52 | 43.50 | 43.43 | 43.53 | 43.43 | 43.53 | |
| lin/wu100/lr2e-5/fp16 | bio_forget | WMDP | ~10622 | 35.02 | 35.37 | 36.64 | 37.33 | 37.67 | 38.13 | 38.25 | 39.29 | 39.40 | 40.09 | 41.01 | 40.90 |  | **43.43** | 40.78 | 40.90 | 42.74 | 42.28 | 42.40 | 42.28 | 41.94 | 42.40 | 42.28 | fp16, FSDP 4GPU, eb=16, 2ep, full data, job 2552667 |
| | | MMLU | | **44.76** | 43.70 | 42.18 | 42.54 | 42.47 | 41.77 | 41.99 | 41.95 | 42.01 | 40.93 | 41.04 | 41.37 |  | 42.70 | 42.51 | 42.20 | 42.71 | 42.33 | 42.44 | 42.69 | 42.54 | 42.63 | 42.64 | |
| lin/wu0/lr2e-5/fp16 | bio_forget | WMDP | ~10622 | 35.02 | 36.52 | 36.29 | 37.67 | 38.48 | 38.94 | 39.63 | 40.21 | 39.29 | 40.09 | 40.67 | 41.01 | 41.47 | **43.43** | 40.21 | 40.90 | 42.40 | 41.82 | 42.51 | 42.28 | 41.94 | 41.94 | 42.05 | fp16, FSDP 4GPU, eb=16, 2ep, full data, job 2552669 |
| | | MMLU | | **44.76** | 43.57 | 42.00 | 42.39 | 42.29 | 41.82 | 41.71 | 41.75 | 42.00 | 41.01 | 40.99 | 41.36 | 42.62 | 42.78 | 42.59 | 42.32 | 42.64 | 42.41 | 42.48 | 42.66 | 42.55 | 42.60 | 42.59 | |
| lin/wu100/lr2e-5/bf16 | bio_forget | WMDP | ~10622 | 34.79 | 33.76 | 33.64 | 33.64 | 35.02 | 34.68 | 34.79 | 35.02 | 34.33 | 34.33 | 34.91 | 34.56 | 34.68 | 34.68 | **35.60** | 35.25 | 35.25 | 35.37 | 35.02 | 34.79 | 34.56 | 34.68 | 34.79 | bf16, DDP 4GPU, eb=16, 2ep, full data, job 2552665 |
| | | MMLU | | **44.43** | 44.00 | 43.59 | 43.83 | 43.73 | 43.70 | 43.60 | 43.50 | 43.73 | 43.70 | 43.43 | 43.56 | 43.60 | 43.58 | 43.70 | 43.58 | 43.82 | 43.55 | 43.64 | 43.63 | 43.72 | 43.49 | 43.62 | |
| lin/wu100/lr5e-5/bf16 | bio_forget | WMDP | ~10622 | 34.79 | 34.91 | 36.87 | 37.67 | 36.29 | 36.98 | 37.79 | 37.56 | **39.17** | 38.82 | | | | | | | | | | | | | | bf16, DDP 4GPU, eb=16, 2ep, full data, job 2576346, running |
| | | MMLU | | **44.43** | 43.35 | 42.91 | 42.77 | 42.50 | 42.44 | 42.29 | 42.03 | 42.34 | 42.19 | | | | | | | | | | | | | | |
| lin/wu100/lr1e-4/bf16 | bio_forget | WMDP | ~10622 | 34.79 | 33.06 | 33.87 | 34.91 | 32.83 | 35.14 | 35.71 | 35.60 | 36.41 | **37.79** | | | | | | | | | | | | | | bf16, DDP 4GPU, eb=16, 2ep, full data, job 2576347, running |
| | | MMLU | | **44.43** | 40.96 | 38.82 | 39.34 | 38.02 | 38.28 | 38.61 | 38.46 | 38.40 | 37.27 | | | | | | | | | | | | | | |
| lin/wu100/lr2e-4/bf16 | bio_forget | WMDP | ~10622 | 34.79 | 27.76 | 28.69 | 28.80 | 32.14 | 32.83 | 29.95 | 31.34 | **34.68** | 32.14 | | | | | | | | | | | | | | bf16, DDP 4GPU, eb=16, 2ep, full data, job 2576348, running |
| | | MMLU | | **44.43** | 30.13 | **31.28** | 31.04 | 29.93 | 29.20 | 29.32 | 29.82 | 30.50 | 30.06 | | | | | | | | | | | | | | |
| lin/wu100/lr5e-4/bf16 | bio_forget | WMDP | ~10622 | 34.79 | 27.53 | 27.42 | **26.84** | 26.50 | 25.81 | 27.42 | 26.04 | 26.96 | 25.23 | | | | | | | | | | | | | | bf16, DDP 4GPU, eb=16, 2ep, full data, job 2576349, running |
| | | MMLU | | **44.43** | 25.62 | 25.79 | **26.16** | 25.31 | 25.70 | 24.78 | 25.69 | 26.23 | 25.72 | | | | | | | | | | | | | | |

### Mean stable ranks

- Fine-tuned filtered model: 268.38
- Filtered base model: 265.35
- Unfiltered base model: 253.76

### Grounding

It takes 6,500 steps with a batch size of 32 to induce WMDP capabilities in the well-filtered model.

Therefore it seems like the best level of tamper resistance we can feasibly expect is to 6,500 steps.
