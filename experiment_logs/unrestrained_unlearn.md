# Unrestrained CB Unlearning with Low Rank Updates (retain_coef=0)

This mostly uses orthogonal circuit breakers with a batch size set to 0 to nullify the orthogonal turn (needlessly complicated).

## Unrestrained Unlearning LoRA Rank Sweep

Goal: Determine minimum LoRA rank needed for tamper-resistant unlearning with orth circuit breakers.

All runs: remove=23, orth=5, retain=0, 32 steps, 1024 examples.

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

### all-module LoRA (AdamW tamper) — WMDP / MMLU

| rank | Step 0 | Step 500 | Step 1000 | Step 2000 | Step 3300 | Notes |
|------|--------|----------|-----------|-----------|-----------|-------|
| - | 42.97 / 45.10 | - | - | - | - | Baseline |
| 8 | 26.2 / 25.5 | 29.8 / 38.1 | 31.8 / 40.6 | 29.1 / 39.3 | 29.3 / 39.2 | Peaks step 800, settles ~29 |
| 16 | 26.7 / 23.0 | 27.1 / 25.3 | 29.0 / 24.9 | 29.1 / 24.8 | 29.1 / 24.6 | Gradual climb to ~29 |
| 32 | 26.7 / 23.0 | 26.0 / 25.9 | 25.7 / 25.6 | 25.5 / 25.1 | 25.8 / 25.2 | Flat through 3300 |
| 64 | 26.7 / 23.0 | 26.5 / 23.7 | 26.7 / 26.3 | 26.8 / 26.3 | 27.8 / 26.5 | Flat through 3300 |
| 128 | 25.8 / 26.8 | 26.7 / 23.0 | 26.4 / 24.4 | 24.1 / 24.4 | 23.6 / 24.7 | Flat/declining through 3300 |
| 256 | 24.1 / 26.9 | 26.7 / 23.0 | - | - | - | Cancelled step 500, resubmitted |
| 512 | 24.1 / 26.9 | 26.7 / 23.0 | - | - | - | Cancelled step 500, resubmitted |

### all-module LoRA (Muon tamper) — WMDP / MMLU

[Results removed for using many epochs of the same data.]

### r12 LoRA with Orthogonal Aux Loss (Muon tamper) — WMDP / MMLU

r12 models unlearned with pdbs=4 (orth loss active). Not directly comparable to r8/r16 rank sweep models above which used pdbs=1 (orth loss always 0).

| target | unlearn opt | Step 0 | Step 500 | Step 1000 | Step 3300 | Notes |
|--------|-------------|--------|----------|-----------|-----------|-------|
| attn | Muon | 32.7 / 38.9 | 38.0 / 43.7 | 39.6 / 44.5 | - | Completed step 1000 |
| mlp | Muon | 34.2 / 41.1 | 40.7 / 45.3 | 40.7 / 45.1 | - | Completed step 1000 |
| all | Muon | 28.9 / 29.9 | 35.0 / 40.7 | 34.4 / 42.4 | 31.9 / 39.7 | |
| all | AdamW | 28.5 / 34.9 | 38.7 / 44.6 | 37.8 / 45.8 | 34.9 / 43.8 | |

### SFT (full-rank) unlearn tamper

[Results removed for using many epochs of the same data.]

### Layer exclusion ablation (AdamW tamper, all-module r=16) — WMDP / MMLU

To the extent these experiments are valid, excluding one layer from unlearning is enough to break tamper resistance.

| Excluded layer | Step 0 | Step 200 | Step 400 | Step 600 | Step 800 | Step 1000 | Step 1200 | Step 1400 | Step 1600 | Step 1800 | Step 2000 | Notes |
|----------------|--------|----------|----------|----------|----------|-----------|-----------|-----------|-----------|-----------|-----------|-------|
| - | 42.97 / 45.10 | - | - | - | - | - | - | - | - | - | - | Baseline |
| L0 | 27.65 / 32.37 | 41.36 / 45.72 | 43.09 / 46.80 | 43.66 / 46.89 | 44.01 / 47.14 | 43.43 / 46.82 | 42.63 / 46.61 | 41.59 / 46.33 | 40.55 / 46.19 | 41.36 / 46.20 | 41.24 / 46.06 | Running |
| L16 | 26.96 / 32.07 | 41.13 / 45.73 | 42.97 / 46.83 | 43.78 / 46.62 | 42.86 / 46.94 | 41.59 / 46.70 | 42.17 / 46.45 | 41.13 / 46.18 | 39.40 / 45.83 | 39.29 / 45.78 | 39.75 / 45.58 | Running |
| L31 | 25.92 / 34.61 | 40.90 / 45.69 | 42.63 / 46.69 | 43.09 / 46.87 | 43.32 / 46.98 | 42.86 / 46.60 | 41.13 / 46.45 | 41.01 / 46.47 | 40.09 / 46.08 | 39.75 / 46.00 | 39.52 / 45.98 | Running |

### 500-step CB tamper sweep on r=8 all-module — WMDP / MMLU

[Results removed for using many epochs of the same data.]

### Repair-then-attack CB on r=8 all-module — WMDP / MMLU

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

### all-module CB LoRA lr=2e-4 tamper (AdamW, 32 epochs, eval every 500 steps) — WMDP / MMLU

[Results invalidated by using many epochs of the same data.]

| rank | Step 0 | Step 500 | Step 1000 | Step 2000 | Step 3520 | Notes |
|------|--------|----------|-----------|-----------|-----------|-------|
| - | 42.97 / 45.10 | - | - | - | - | Baseline |

### deep-ignorance-e2e-strong-filter tamper (AdamW, 512 examples, 5 epochs, eval every 100 steps) — WMDP / MMLU

| lr | Step 0 | Step 100 | Step 200 | Step 300 | Step 400 | Step 500 | Step 550 |
|----|--------|----------|----------|----------|----------|----------|----------|
| 2e-5 | 34.56 / 46.00 | 33.29 / 45.74 | 33.64 / 46.18 | 34.22 / 46.23 | 33.87 / 46.47 | 33.87 / 46.34 | 33.29 / 46.31 |
| 5e-5 | 34.56 / 46.00 | 32.37 / 46.14 | 33.64 / 45.75 | 34.79 / 46.35 | 34.79 / 46.27 | 35.25 / 46.33 | 34.33 / 46.45 |
| 1e-4 | 34.56 / 46.00 | 34.33 / 44.50 | 34.45 / 44.27 | 34.33 / 44.00 | 35.02 / 44.17 | 35.14 / 44.11 | 35.02 / 44.25 |
| 2e-4 | 34.56 / 46.00 | 30.76 / 37.23 | 31.68 / 35.97 | 30.18 / 37.82 | 29.61 / 38.04 | 30.53 / 37.86 | 31.11 / 37.95 |

### r=8 all-module tamper variants (AdamW, 3500 steps, eval every 500) — WMDP / MMLU

Unlearned model: r=8, all-module, pdbs=1 (orth loss always 0). Baseline: 42.97 / 45.10.

| Attack config | Step 0 | Step 500 | Step 1000 | Step 1500 | Step 2000 | Step 2500 | Step 3000 | Step 3500 |
|---------------|--------|----------|-----------|-----------|-----------|-----------|-----------|-----------|
| cosine, lr=2e-5 | 26.96 / 26.18 | 31.91 / 38.20 | 29.49 / 37.52 | 28.00 / 37.12 | 28.11 / 36.83 | 28.23 / 36.82 | 28.34 / 36.80 | 28.34 / 36.80 |
| warmup 10%, lr=2e-5 | 26.96 / 26.18 | 32.60 / 38.01 | 28.23 / 37.07 | 29.03 / 37.12 | 28.00 / 36.60 | 28.11 / 36.68 | 28.46 / 36.48 | 28.69 / 36.31 |
| cosine + warmup 10%, lr=2e-5 | 26.96 / 26.18 | 27.42 / 25.79 | 27.88 / 30.67 | 26.50 / 29.84 | 27.76 / 29.55 | 27.53 / 29.43 | 27.19 / 29.43 | 27.19 / 29.43 |
| cosine, lr=2e-4 | 26.96 / 26.18 | 28.34 / 26.66 | 27.42 / 28.32 | 27.53 / 28.34 | 27.76 / 28.40 | 27.76 / 28.46 | 27.76 / 28.48 | 27.76 / 28.48 |
| warmup 10%, lr=2e-4 | 26.96 / 26.18 | 26.61 / 26.63 | 26.73 / 26.81 | 26.61 / 27.22 | 26.61 / 27.30 | 26.61 / 26.74 | 26.61 / 26.91 | 26.50 / 26.36 |
| cosine + warmup 10%, lr=2e-4 | 26.96 / 26.18 | 33.53 / 34.29 | 35.02 / 36.74 | 34.91 / 36.93 | 35.02 / 36.95 | 35.25 / 36.94 | 35.37 / 36.95 | 35.37 / 36.94 |

### WMDP Bio Robust over Tampering Configs for LoRA ranks (AdamW, step 3000, evaluated every 500)

Unlearned models: all-module, pdbs=1, rm23, orth5, ret0, all 32 layers. Filtered model: deep-ignorance-e2e-strong-filter. Baseline: 42.97 / 45.10. Filtered model uses LRs 2e-5/5e-5/1e-4 (vs rank sweep: 2e-5/2e-4/1e-3).

| Config | r=2 | r=4 | r=8 | r=16 | r=32 | r=64 | r=128 | r=256 | r=512 | filter |
|--------|-----|-----|-----|------|------|------|-------|-------|-------|--------|
| cos/wr01/lr2e-5 | **45.7** | **46.8** | 36.4 | 28.1 | 28.0 | 26.6 | 22.8 | 25.2 | 26.7 | 33.6 |
| cos/wr01/lr2e-4 | 36.2 | 40.2 | **38.8** | **38.2** | **37.2** | 34.0 | 28.1 | **32.5** | 26.4 | 34.3 |
| cos/wr01/lr1e-3 | 26.8 | 26.7 | 26.7 | 26.7 | 26.8 | 26.7 | 26.8 | 26.8 | **28.3** | 34.1 |
| cos/ws30/lr2e-4 | 37.2 | 36.6 | 38.5 | 36.9 | 29.5 | 34.6 | 29.0 | 31.4 | 26.7 | **35.4** |
| cos/ws100/lr2e-4 | 41.5 | 35.8 | 34.8 | 32.6 | 35.6 | **35.5** | **32.7** | 29.8 | 27.2 | 33.6 |
| const/wr01/lr2e-4 | 27.4 | 27.3 | 28.1 | 27.7 | 26.5 | 27.1 | 27.2 | 27.0 | 27.1 | 33.1 |

### MMLU over Tampering Configs for LoRA ranks (AdamW, step 3000, evaluated every 500)

| Config | r=2 | r=4 | r=8 | r=16 | r=32 | r=64 | r=128 | r=256 | r=512 | filter |
|--------|-----|-----|-----|------|------|------|-------|-------|-------|--------|
| cos/wr01/lr2e-5 | **45.4** | **44.7** | 41.4 | 25.8 | 26.2 | 25.4 | 24.4 | 25.8 | 22.9 | **44.7** |
| cos/wr01/lr2e-4 | 37.3 | 37.6 | **36.8** | **34.2** | **34.5** | 33.1 | 28.2 | **30.5** | 25.8 | 44.4 |
| cos/wr01/lr1e-3 | 25.3 | 25.8 | 25.8 | 25.8 | 25.6 | 25.8 | 25.8 | 25.8 | **26.0** | 43.7 |
| cos/ws30/lr2e-4 | 37.7 | 36.1 | 36.2 | 35.2 | 33.7 | 34.4 | 31.5 | 29.2 | 26.1 | 44.1 |
| cos/ws100/lr2e-4 | 37.4 | 36.5 | 35.4 | 34.8 | 34.3 | **35.4** | **30.4** | 30.5 | 25.5 | 43.8 |
| const/wr01/lr2e-4 | 27.7 | 27.3 | 27.4 | 26.5 | 25.7 | 25.7 | 26.4 | 26.4 | 26.0 | 43.8 |

### Best WMDP / MMLU over Tampering Configs by Rank (AdamW, evaluated every 500 steps)

Unlearned models: all-module, pdbs=1, rm23, orth5, ret0, all 32 layers. Filtered model: deep-ignorance-e2e-strong-filter. Baseline: 42.97 / 45.10. Values are the best found across all 6 tampering configs at each step.

| Rank | Step 0 | Step 500 | Step 1000 | Step 1500 | Step 2000 | Step 2500 | Step 3000 |
|------|--------|----------|-----------|-----------|-----------|-----------|-----------|
| r=2 | 24.4 / 24.6 | 44.9 / 45.6 | **45.5** / 45.1 | **45.9** / **45.5** | 45.7 / 45.3 | 45.6 / 45.4 | 45.7 / 45.4 |
| r=4 | 24.1 / 26.9 | 45.2 / 44.4 | **47.0** / 44.5 | **47.0** / **44.8** | 46.5 / 44.7 | 46.8 / 44.6 | 46.8 / 44.7 |
| r=8 | 26.2 / 25.5 | 33.5 / 33.7 | 35.8 / 31.5 | 35.9 / **41.4** | **39.4** / 36.0 | 38.9 / 37.0 | 38.8 / 36.8 |
| r=16 | 26.7 / 22.9 | **37.6** / 34.6 | 35.4 / 32.7 | 36.2 / 34.0 | 36.8 / 33.4 | 38.1 / 34.2 | 38.2 / **34.2** |
| r=32 | 26.7 / 22.9 | 35.4 / **36.4** | 35.2 / 33.6 | 35.1 / 33.9 | **38.4** / 34.0 | 37.3 / 34.6 | 37.2 / 34.5 |
| r=64 | 26.7 / 22.9 | 34.9 / 32.6 | 32.5 / 31.6 | 32.6 / 34.4 | 35.4 / 34.5 | **35.9** / **35.3** | 35.5 / 35.4 |
| r=128 | 24.4 / 26.6 | 27.4 / 27.9 | 29.8 / 28.3 | 29.8 / 30.4 | 31.7 / **30.5** | 32.3 / 30.4 | **32.7** / 30.4 |
| r=256 | 24.1 / 26.9 | 27.5 / 26.4 | 30.8 / 26.0 | **33.4** / 29.4 | 32.7 / 29.4 | 32.5 / 30.3 | 32.5 / **30.5** |
| r=512 | 24.1 / 26.9 | **27.9** / 24.4 | 27.1 / 25.8 | 26.7 / 22.9 | 28.0 / **26.1** | 28.3 / 25.9 | 28.3 / 26.0 |
| filter | 35.0 / 46.0 | **35.9** / **46.7** | 35.6 / 45.9 | 35.5 / 45.5 | 35.4 / 45.1 | 35.4 / 44.9 | 35.4 / 45.0 |

### Bio Chat Tamper: WMDP Bio Robust (AdamW, step 3000, evaluated every 500, eval batch_size=32)

Attack dataset: WMDP-Bio + UltraChat. Unlearned models: all-module, pdbs=1, rm23, orth5, ret0, all 32 layers.

| Config | r=8 | r=16 | r=32 |
|--------|-----|------|------|
| cos/wr01/lr2e-5 | | | |
| cos/wr01/lr2e-4 | | | |
| cos/wr01/lr1e-3 | | | |
| cos/ws30/lr2e-4 | | | |
| cos/ws100/lr2e-4 | | | |
| const/wr01/lr2e-4 | | | |

### Bio Chat Tamper: WMDP Bio Robust Step-by-Step (AdamW, evaluated every 500, eval batch_size=32)

| Rank | Config | Step 0 | Step 500 | Step 1000 | Step 1500 | Step 2000 | Step 2500 | Step 3000 |
|------|--------|--------|----------|-----------|-----------|-----------|-----------|-----------|
| r=8 | cos/wr01/lr2e-5 | | | | | | | |
| r=8 | cos/wr01/lr2e-4 | | | | | | | |
| r=8 | cos/wr01/lr1e-3 | | | | | | | |
| r=8 | cos/ws30/lr2e-4 | | | | | | | |
| r=8 | cos/ws100/lr2e-4 | | | | | | | |
| r=8 | const/wr01/lr2e-4 | | | | | | | |
| r=16 | cos/wr01/lr2e-5 | | | | | | | |
| r=16 | cos/wr01/lr2e-4 | | | | | | | |
| r=16 | cos/wr01/lr1e-3 | | | | | | | |
| r=16 | cos/ws30/lr2e-4 | | | | | | | |
| r=16 | cos/ws100/lr2e-4 | | | | | | | |
| r=16 | const/wr01/lr2e-4 | | | | | | | |
| r=32 | cos/wr01/lr2e-5 | | | | | | | |
| r=32 | cos/wr01/lr2e-4 | | | | | | | |
| r=32 | cos/wr01/lr1e-3 | | | | | | | |
| r=32 | cos/ws30/lr2e-4 | | | | | | | |
| r=32 | cos/ws100/lr2e-4 | | | | | | | |
| r=32 | const/wr01/lr2e-4 | | | | | | | |

### Benign Tamper Rerun: WMDP Bio Robust (AdamW, step 3000, evaluated every 500, eval batch_size=32)

Attack dataset: WikiText + UltraChat. Unlearned models: all-module, pdbs=1, rm23, orth5, ret0, all 32 layers.

| Config | r=8 | r=16 | r=32 |
|--------|-----|------|------|
| cos/wr01/lr2e-5 | | | |
| cos/wr01/lr2e-4 | | | |
| cos/wr01/lr1e-3 | | | |
| cos/ws30/lr2e-4 | | | |
| cos/ws100/lr2e-4 | | | |
| const/wr01/lr2e-4 | | | |

### Benign Tamper Rerun: WMDP Bio Robust Step-by-Step (AdamW, evaluated every 500, eval batch_size=32)

| Rank | Config | Step 0 | Step 500 | Step 1000 | Step 1500 | Step 2000 | Step 2500 | Step 3000 |
|------|--------|--------|----------|-----------|-----------|-----------|-----------|-----------|
| r=8 | cos/wr01/lr2e-5 | | | | | | | |
| r=8 | cos/wr01/lr2e-4 | | | | | | | |
| r=8 | cos/wr01/lr1e-3 | | | | | | | |
| r=8 | cos/ws30/lr2e-4 | | | | | | | |
| r=8 | cos/ws100/lr2e-4 | | | | | | | |
| r=8 | const/wr01/lr2e-4 | | | | | | | |
| r=16 | cos/wr01/lr2e-5 | | | | | | | |
| r=16 | cos/wr01/lr2e-4 | | | | | | | |
| r=16 | cos/wr01/lr1e-3 | | | | | | | |
| r=16 | cos/ws30/lr2e-4 | | | | | | | |
| r=16 | cos/ws100/lr2e-4 | | | | | | | |
| r=16 | const/wr01/lr2e-4 | | | | | | | |
| r=32 | cos/wr01/lr2e-5 | | | | | | | |
| r=32 | cos/wr01/lr2e-4 | | | | | | | |
| r=32 | cos/wr01/lr1e-3 | | | | | | | |
| r=32 | cos/ws30/lr2e-4 | | | | | | | |
| r=32 | cos/ws100/lr2e-4 | | | | | | | |
| r=32 | const/wr01/lr2e-4 | | | | | | | |

### deep-ignorance-e2e-strong-filter tamper

- AdamW, SFT (full parameter, no LoRA)
- batch_size=1, grad_accumulation=16 (effective batch=16)
- weight_decay=0.01, gradient_checkpointing=True
- max_chunks=1 (truncate to 2048 context, no chunking)
- bio_flagged run: lr=5e-5, cosine schedule, warmup_ratio=0.1, max_steps=10000, 24453 examples
- bio_forget runs: lr=2e-5, cosine schedule, warmup_ratio=0.01, 2 epochs (~10622 steps), 24453 examples

WMDP Bio / MMLU eval every 500

Target model: `EleutherAI/deep-ignorance-e2e-strong-filter`. Baseline: 34.56 / 44.29. Unfiltered baseline: 42.97 / 45.10.

| Config | Data | Steps | Step 0 | Step 500 | Step 1000 | Step 1500 | Step 2000 | Step 2500 | Step 3000 | Step 3500 | Step 4000 | Step 4500 | Step 5000 | Step 5500 | Step 6000 | Step 6500 | Step 7000 | Step 7500 | Step 8000 | Step 8500 | Step 9000 | Step 9500 | Step 10000 | Step 10500 | Step 10622 | Notes |
|--------|------|-------|--------|----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|------------|------------|------------|-------|
| cos/wr10/lr5e-5 | bio_flagged | 10000 | 34.56 / 44.29 | 33.76 / 44.45 | 34.56 / 44.52 | 34.45 / 43.43 | 34.79 / 43.13 | 35.60 / 43.41 | 35.83 / 43.26 | 35.02 / 42.62 | 35.14 / 42.34 | 36.06 / 42.59 | 36.64 / 42.52 | 36.06 / 43.13 | 36.64 / 43.04 | 36.75 / 42.93 | **36.87** / 42.84 | **36.87** / 43.06 | 36.41 / 43.06 | 36.52 / 42.84 | - | - | - | - | - | Job 2434383 |
| cos/wr01/lr2e-5/s42 | bio_forget | ~10622 | 34.56 / 44.29 | 33.76 / 44.22 | 33.87 / 44.30 | 35.02 / 44.17 | 34.56 / 43.80 | 35.25 / 44.06 | 34.91 / 43.59 | 34.10 / 43.55 | 34.68 / 43.45 | 35.71 / 43.31 | 34.68 / 43.47 | 35.25 / 43.51 | **36.52** / 43.20 | 34.91 / 43.57 | 35.37 / 43.56 | 35.48 / 43.56 | 35.48 / 43.54 | 35.25 / 43.61 | 35.14 / 43.56 | 35.25 / 43.53 | 35.37 / 43.43 | 35.14 / 43.59 | 35.48 / 43.62 | Paper recipe, 2ep, seed=42, job 2439121 |
| cos/wr01/lr2e-5/s43 | bio_forget | ~10622 | 34.56 / 44.29 | 33.76 / 44.25 | 33.99 / 43.88 | 34.68 / 43.68 | 33.99 / 43.80 | 34.56 / 43.63 | 34.68 / 43.69 | 34.56 / 43.32 | **36.18** / 43.33 | 35.37 / 43.34 | 35.48 / 43.58 | 35.02 / 43.30 | 35.71 / 43.42 | 35.48 / 43.51 | 35.71 / 43.34 | 35.48 / 43.38 | 35.25 / 43.26 | 35.48 / 43.21 | 35.48 / 43.38 | 35.25 / 43.39 | 35.14 / 43.20 | 35.48 / 43.31 | 35.60 / 43.39 | Paper recipe, 2ep, seed=43, job 2439122 |

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
