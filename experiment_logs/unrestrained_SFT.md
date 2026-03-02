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

### SFT (full-rank) unlearn tamper (AdamW, 3000 steps, 1ep, ~90k chunks, evaluated every 500)

Unlearned model: SFT (full-rank, not LoRA), rm23, orth5, ret0, pdbs=1, 32 steps, 1024 examples. Training job 2485738. WMDP Robust: 26.73%, MMLU: 22.95%.

Previous job 2476679 had a bug where the SFT reference pass used current weights instead of original weights, so unlearning had no effect (WMDP stayed at baseline 42.86%). Fixed by caching original params and swapping them in for the reference pass.

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

### Lens SFT ret0 tamper (AdamW, 3000 steps, 1ep, ~90k chunks, evaluated every 500)

Unlearned model: Tuned lens SFT, ret0, rm5, lr=1e-3, 1024 examples, 32 steps. WMDP Robust: 23.39%, MMLU: 24.65%.

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

### Sequential SFT ret0 tamper (AdamW, 3000 steps, 1ep, ~90k chunks, evaluated every 500)

Unlearned model: Sequential back-to-front SFT, ret0, rm5, lr=2e-4, layers 31→11 step 4, 1024 examples. WMDP Robust: 26.73%, MMLU: 22.95%.

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

### Checkpoint Transfer SFT ret0 tamper (AdamW, 3000 steps, 1ep, ~90k chunks, evaluated every 500)

Unlearned model: Checkpoint transfer SFT, ret0, rm2000, lr=1e-3, 2048 examples. WMDP Robust: 26.50%, MMLU: 22.93%.

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

### SFT Tamper Resistance Comparison (best WMDP recovery across configs)

| Algorithm | Starting WMDP | Starting MMLU | Best Tamper WMDP | Best Tamper MMLU | Tamper Resistant? |
|-----------|---------------|---------------|------------------|------------------|-------------------|
| Orth CB SFT (rm23/orth5) | 26.73% | 22.95% | 39.63% | 34.70% | No |
| Sequential SFT (rm5) | 26.73% | 22.95% | 43.09% | 43.87% | No |
| Tuned Lens SFT (rm5) | 23.39% | 24.65% | 28.34% | 26.78% | Yes |
| Checkpoint Transfer SFT (rm2000) | 26.50% | 22.93% | 28.11% | 26.86% | Yes |

\* = eval failed at step 3000 (MMLU=0.00); last valid step used.

Best lens SFT result so far: ret=0.05, lr=2e-4, rm=1, 1024 examples, 32 steps, pdbs=1, warmup=0.5, FSDP → WMDP 36.18%, MMLU 43.80%.

### Orth CB r32 Retain Coef Sweep: Post-Unlearning

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

### Orth CB r32 Retain Coef Sweep: WMDP over Tampering Configs (step 3000, 1ep)

Unlearned models: all-module, pdbs=1, rm23, orth5, r=32, all 32 layers. Baseline: 42.97 / 45.10.

| Config | ret=0 | ret=1e-5 | ret=1e-4 | ret=1e-3 | ret=1e-2 | ret=0.02 | ret=0.03 | ret=0.05 | ret=0.07 | ret=0.1 | ret=0.2 | ret=0.4 | ret=0.8 | ret=1 |
|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| cos/wr01/lr2e-5 | 27.8 | 27.8 | 26.4 | 27.3 | 43.4 | 45.7 | **47.1** | 45.9 | 45.7 | 45.9 | 46.2 | 46.3 | 45.9 | 46.0 |
| cos/wr01/lr2e-4 | 33.2 | 36.5 | 35.9 | 38.0 | **41.1** | 35.8 | 40.9 | 39.9 | 33.6 | 35.7 | 36.4 | 38.0 | 38.8 | 38.6 |
| cos/wr01/lr1e-3 | 27.1 | 27.7 | **29.3** | 26.5 | 26.7 | 26.6 | 27.0 | 26.7 | 27.0 | 28.0 | 26.7 | 27.1 | 26.8 | 27.1 |
| cos/ws30/lr2e-4 | 36.1 | 33.0 | 34.3 | **40.8** | 39.1 | 35.5 | 34.4 | 36.9 | 34.0 | 37.6 | 37.0 | 31.6 | 38.2 | 38.2 |
| cos/ws100/lr2e-4 | 32.3 | 37.7 | 36.3 | 37.2 | 39.2 | 37.0 | 37.0 | 39.1 | 36.8 | 35.2 | 35.9 | 36.1 | **40.3** | 40.2 |
| const/wr01/lr2e-4 | 29.6 | 27.2 | 27.2 | 26.8 | 29.0 | 26.8 | 27.5 | 26.6 | 26.5 | 29.1 | 32.3 | 28.1 | 29.4 | **32.6** |

### Orth CB r32 Retain Coef Sweep: Best WMDP/MMLU Step-by-Step (1ep)

Unlearned models: all-module, pdbs=1, rm23, orth5, r=32, all 32 layers. Baseline: 42.97 / 45.10. Values are the best found across all 6 tampering configs at each step.

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

| Config | r=2 | r=4 | r=8 | r=16 | r=32 | r=64 | r=128 | r=256 | r=512 | r=1024 | SFT | lens | seq | CT | filter |
|--------|-----|-----|-----|------|------|------|-------|-------|-------|--------|-----|------|-----|-----|--------|
| cos/wr01/lr2e-5 | **45.7** | **46.8** | 36.4 | 28.1 | 28.0 | 26.6 | 22.8 | 25.2 | 26.7 | 25.8 | 24.0 | 26.6 | **43.1** | 28.1 | 33.6 |
| cos/wr01/lr2e-4 | 36.2 | 40.2 | **38.8** | **38.2** | **37.2** | 34.0 | 28.1 | **32.5** | 26.4 | **26.6** | **39.6** | **27.2** | 37.4 | 27.4 | 34.3 |
| cos/wr01/lr1e-3 | 26.8 | 26.7 | 26.7 | 26.7 | 26.8 | 26.7 | 26.8 | 26.8 | **28.3** | 24.7 | 27.0 | 27.2 | 26.8 | 27.1 | 34.1 |
| cos/ws30/lr2e-4 | 37.2 | 36.6 | 38.5 | 36.9 | 29.5 | 34.6 | 29.0 | 31.4 | 26.7 | 24.2 | 37.2 | 26.7 | 38.6 | 27.1 | **35.4** |
| cos/ws100/lr2e-4 | 41.5 | 35.8 | 34.8 | 32.6 | 35.6 | **35.5** | **32.7** | 29.8 | 27.2 | 24.7 | 38.1 | 27.2 | 39.1 | 27.2 | 33.6 |
| const/wr01/lr2e-4 | 27.4 | 27.3 | 28.1 | 27.7 | 26.5 | 27.1 | 27.2 | 27.0 | 27.1 | 24.8 | 26.6 | 27.2 | 26.8 | 27.0 | 33.1 |

### MMLU over Tampering Configs for LoRA ranks (AdamW, step 3000, 1ep, evaluated every 500)

| Config | r=2 | r=4 | r=8 | r=16 | r=32 | r=64 | r=128 | r=256 | r=512 | r=1024 | SFT | lens | seq | CT | filter |
|--------|-----|-----|-----|------|------|------|-------|-------|-------|--------|-----|------|-----|-----|--------|
| cos/wr01/lr2e-5 | **45.4** | **44.7** | 41.4 | 25.8 | 26.2 | 25.4 | 24.4 | 25.8 | 22.9 | 23.3 | 26.1 | 22.9 | **43.9** | 23.5 | **44.7** |
| cos/wr01/lr2e-4 | 37.3 | 37.6 | **36.8** | **34.2** | **34.5** | 33.1 | 28.2 | **30.5** | 25.8 | **22.9** | **34.7** | **24.8** | 36.8 | 25.9 | 44.4 |
| cos/wr01/lr1e-3 | 25.3 | 25.8 | 25.8 | 25.8 | 25.6 | 25.8 | 25.8 | 25.8 | **26.0** | 23.0 | 25.8 | 25.8 | 24.9 | 25.3 | 43.7 |
| cos/ws30/lr2e-4 | 37.7 | 36.1 | 36.2 | 35.2 | 33.7 | 34.4 | 31.5 | 29.2 | 26.1 | 24.2 | **36.6** | 25.7 | 37.7 | 25.8 | 44.1 |
| cos/ws100/lr2e-4 | 37.4 | 36.5 | 35.4 | 34.8 | 34.3 | **35.4** | **30.4** | 30.5 | 25.5 | 23.7 | 33.6 | 24.6 | 37.3 | 25.7 | 43.8 |
| const/wr01/lr2e-4 | 27.7 | 27.3 | 27.4 | 26.5 | 25.7 | 25.7 | 26.4 | 26.4 | 26.0 | 25.3 | 23.9 | 25.4 | 26.9 | 24.6 | 43.8 |

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
| r=1024 | WMDP | 23.5 | 26.7 | 26.8 | **27.0** | 26.5 | **27.1** | 26.6 |
| | MMLU | 24.6 | 24.9 | **25.3** | 24.8 | 24.5 | 24.9 | 25.2 |
| SFT | WMDP | 26.7 | 28.1 | 31.5 | 36.1 | 38.0 | 39.4 | **39.6** |
| | MMLU | 22.9 | 29.5 | 31.4 | 33.7 | 36.0 | **36.6** | 36.6 |
| lens | WMDP | 23.4 | 27.3 | **28.3** | 27.3 | 27.9 | 27.4 | 27.2 |
| | MMLU | 24.7 | **26.8** | 25.9 | 26.1 | 25.9 | 25.8 | 25.8 |
| seq | WMDP | 26.7 | 42.6 | 43.0 | 42.9 | **43.4** | 43.0 | 43.1 |
| | MMLU | 22.9 | 43.3 | 43.5 | 43.6 | **43.9** | 43.8 | 43.9 |
| CT | WMDP | 26.5 | **31.1** | 26.8 | 28.1 | 27.9 | 27.9 | 28.1 |
| | MMLU | 22.9 | **26.9** | 26.2 | 26.3 | 25.9 | 25.9 | 25.7 |
| filter | WMDP | 35.0 | **35.9** | 35.6 | 35.5 | 35.4 | 35.4 | 35.4 |
| | MMLU | 46.0 | **46.7** | 45.9 | 45.5 | 45.1 | 44.9 | 45.0 |

### Bio Chat Tamper: WMDP Bio Robust (AdamW, 3000 steps, 1ep, 48000 chunks, evaluated every 500, eval batch_size=32)

Attack dataset: WMDP-Bio + UltraChat (24000 chunks each). Unlearned models: all-module, pdbs=1, rm23, orth5, ret0, all 32 layers.

~75% of eval subprocess calls failed (lm_eval torchrun crash). Best valid WMDP Bio shown; re-evals pending for missing steps.

| Config | r=8 | r=16 | r=32 |
|--------|-----|------|------|
| cos/wr01/lr2e-5 | 26.2 | 26.7 | 28.5 |
| cos/wr01/lr2e-4 | 37.3 | **39.8** | 38.5 |
| cos/wr01/lr1e-3 | 26.2 | 27.1 | 27.3 |
| cos/ws30/lr2e-4 | 26.2 | 35.0 | 26.7 |
| cos/ws100/lr2e-4 | **41.6** | 26.7 | **42.1** |
| const/wr01/lr2e-4 | 31.1 | 31.2 | 31.8 |

### Bio Chat Tamper: WMDP Bio Robust Step-by-Step (AdamW, evaluated every 500, batch_size=32)

~75% of eval subprocess calls failed (lm_eval torchrun crash). -- = eval failed. Re-evals pending.

| Rank | Config | Step 0 | Step 500 | Step 1000 | Step 1500 | Step 2000 | Step 2500 | Step 3000 | Notes |
|------|--------|--------|----------|-----------|-----------|-----------|-----------|-----------|-------|
| r=8 | cos/wr01/lr2e-5 | 26.2 / 25.5 | -- | -- | -- | -- | -- | -- | |
| r=8 | cos/wr01/lr2e-4 | 26.2 / 25.5 | 34.5 / 37.5 | 32.8 / 30.6 | 37.3 / 34.3 | -- | -- | -- | |
| r=8 | cos/wr01/lr1e-3 | 26.2 / 25.5 | -- | -- | -- | -- | -- | -- | |
| r=8 | cos/ws30/lr2e-4 | 26.2 / 25.5 | -- | -- | -- | -- | -- | -- | |
| r=8 | cos/ws100/lr2e-4 | 26.2 / 25.5 | -- | -- | 36.8 / 30.0 | -- | 41.6 / 33.2 | -- | |
| r=8 | const/wr01/lr2e-4 | 26.2 / 25.5 | -- | -- | -- | -- | 31.1 / 28.7 | -- | |
| r=16 | cos/wr01/lr2e-5 | 26.7 / 23.0 | -- | -- | -- | -- | -- | -- | |
| r=16 | cos/wr01/lr2e-4 | 26.7 / 23.0 | -- | -- | -- | 39.8 / 37.6 | 38.7 / 37.4 | -- | |
| r=16 | cos/wr01/lr1e-3 | 26.7 / 23.0 | -- | -- | 27.1 / 25.9 | -- | -- | -- | |
| r=16 | cos/ws30/lr2e-4 | 26.7 / 23.0 | -- | -- | -- | 35.0 / 35.3 | -- | -- | |
| r=16 | cos/ws100/lr2e-4 | 26.7 / 23.0 | -- | -- | -- | -- | -- | -- | |
| r=16 | const/wr01/lr2e-4 | 26.7 / 23.0 | -- | 27.2 / 25.7 | 28.0 / 28.8 | -- | 31.2 / 28.1 | -- | |
| r=32 | cos/wr01/lr2e-5 | 26.7 / 23.0 | -- | -- | -- | 28.5 / 27.9 | -- | -- | |
| r=32 | cos/wr01/lr2e-4 | 26.7 / 23.0 | -- | -- | 38.5 / 35.9 | -- | -- | -- | |
| r=32 | cos/wr01/lr1e-3 | 26.7 / 23.0 | -- | -- | 27.0 / 25.8 | 27.3 / 25.8 | -- | -- | |
| r=32 | cos/ws30/lr2e-4 | 26.7 / 23.0 | -- | -- | -- | -- | -- | -- | |
| r=32 | cos/ws100/lr2e-4 | 26.7 / 23.0 | -- | -- | -- | -- | -- | 42.1 / 35.1 | |
| r=32 | const/wr01/lr2e-4 | 26.7 / 23.0 | -- | -- | 31.8 / 30.0 | -- | -- | -- | |

### Benign Tamper Rerun: WMDP Bio Robust (AdamW, 3000 steps, 1ep, 48000 chunks, evaluated every 500, batch_size=32)

Attack dataset: WikiText + UltraChat (24000 chunks each). Unlearned models: all-module, pdbs=1, rm23, orth5, ret0, all 32 layers.

~75% of eval subprocess calls failed (lm_eval torchrun crash). Best valid WMDP Bio shown; re-evals pending for missing steps.

| Config | r=8 | r=16 | r=32 |
|--------|-----|------|------|
| cos/wr01/lr2e-5 | **40.4** | 31.3 | 26.6 |
| cos/wr01/lr2e-4 | 35.4 | **35.8** | **38.3** |
| cos/wr01/lr1e-3 | 27.1 | 27.0 | 27.4 |
| cos/ws30/lr2e-4 | 36.8 | 34.2 | 38.0 |
| cos/ws100/lr2e-4 | 38.0 | 32.4 | 29.5 |
| const/wr01/lr2e-4 | 28.5 | 28.0 | 28.2 |

### Benign Tamper Rerun: WMDP Bio Robust Step-by-Step (AdamW, evaluated every 500, batch_size=32)

~75% of eval subprocess calls failed (lm_eval torchrun crash). -- = eval failed. Re-evals pending.

| Rank | Config | Step 0 | Step 500 | Step 1000 | Step 1500 | Step 2000 | Step 2500 | Step 3000 | Notes |
|------|--------|--------|----------|-----------|-----------|-----------|-----------|-----------|-------|
| r=8 | cos/wr01/lr2e-5 | -- | 32.5 / 40.3 | -- | -- | -- | 40.4 / 44.3 | -- | |
| r=8 | cos/wr01/lr2e-4 | -- | 35.4 / 37.6 | -- | -- | -- | -- | -- | |
| r=8 | cos/wr01/lr1e-3 | -- | 26.7 / 23.2 | 25.8 / 25.8 | 26.7 / 25.8 | -- | 27.1 / 25.7 | -- | |
| r=8 | cos/ws30/lr2e-4 | -- | 36.8 / 33.6 | -- | 34.0 / 34.8 | -- | -- | -- | |
| r=8 | cos/ws100/lr2e-4 | -- | 36.8 / 36.7 | 33.9 / 33.6 | 35.3 / 33.9 | -- | 38.0 / 35.5 | -- | |
| r=8 | const/wr01/lr2e-4 | 26.2 / 25.5 | 28.5 / 27.8 | -- | 28.5 / 29.0 | 27.5 / 27.6 | -- | -- | |
| r=16 | cos/wr01/lr2e-5 | -- | -- | 29.5 / 33.5 | 31.3 / 37.6 | -- | -- | -- | |
| r=16 | cos/wr01/lr2e-4 | -- | 33.4 / 32.6 | 28.3 / 36.0 | -- | 35.8 / 34.2 | -- | -- | |
| r=16 | cos/wr01/lr1e-3 | -- | 26.7 / 23.0 | -- | 27.0 / 25.8 | 26.5 / 25.8 | 27.0 / 25.9 | -- | |
| r=16 | cos/ws30/lr2e-4 | -- | 29.5 / 34.3 | -- | -- | -- | 34.2 / 35.9 | -- | |
| r=16 | cos/ws100/lr2e-4 | -- | 32.4 / 33.1 | -- | 31.5 / 33.9 | 31.9 / 36.1 | 32.4 / 36.5 | -- | |
| r=16 | const/wr01/lr2e-4 | -- | -- | -- | 26.2 / 28.9 | 28.0 / 28.3 | -- | -- | |
| r=32 | cos/wr01/lr2e-5 | -- | 26.4 / 25.8 | -- | 26.6 / 26.7 | -- | -- | -- | |
| r=32 | cos/wr01/lr2e-4 | -- | 32.0 / 33.4 | 28.2 / 32.9 | 35.0 / 34.3 | 38.3 / 35.8 | -- | -- | |
| r=32 | cos/wr01/lr1e-3 | -- | 26.7 / 23.0 | 27.4 / 26.0 | 26.2 / 25.9 | -- | -- | -- | |
| r=32 | cos/ws30/lr2e-4 | -- | 34.6 / 34.5 | -- | -- | -- | -- | 38.0 / 36.1 | |
| r=32 | cos/ws100/lr2e-4 | -- | 29.5 / 35.2 | -- | -- | -- | -- | -- | |
| r=32 | const/wr01/lr2e-4 | -- | 26.6 / 26.1 | -- | 24.5 / 28.3 | -- | -- | 28.2 / 27.9 | |

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
| cos/wr10/lr2e-5 | bio_forget_flagged | WMDP | 10000 | 34.56 | 33.53 | 33.18 | | | | | | | | | | | | | | | | | | | - | - | 1ep, job 2476231 crashed step ~1500, resubmitted as 2484238 |
| | | MMLU | | 44.29 | 44.11 | 44.00 | | | | | | | | | | | | | | | | | | | - | - | |
| cos/wr10/lr1e-4 | bio_forget_flagged | WMDP | 10000 | 34.56 | 34.79 | 34.45 | 34.33 | 34.10 | 34.56 | 33.64 | 32.37 | 34.56 | 34.10 | 35.48 | **36.64** | 36.29 | 35.94 | 36.06 | 35.71 | 36.29 | **36.64** | 35.83 | | | - | - | 1ep, job 2476232 (running, step ~9500/10000) |
| | | MMLU | | 44.29 | **44.38** | 40.94 | 37.42 | 41.24 | 41.64 | 40.42 | 38.07 | 39.99 | 40.01 | 40.49 | 40.80 | 40.33 | 40.91 | 41.13 | 41.23 | 41.06 | 41.16 | 41.16 | | | - | - | |

### MMLU Restoration: WikiText Tamper on r8 and r64 (AdamW, 3000 steps, 1ep, evaluated every 500)

Data: WikiText-103 only. Unlearned models: all-module, pdbs=1, rm23, orth5, ret0, all 32 layers. Baseline: 42.97 / 45.10.

| Rank | Config | Metric | Step 0 | Step 500 | Step 1000 | Step 1500 | Step 2000 | Step 2500 | Step 3000 | Notes |
|------|--------|--------|--------|----------|-----------|-----------|-----------|-----------|-----------|-------|
| r=8 | cos/wr01/lr2e-5 | WMDP | | | | | | | | |
| | | MMLU | | | | | | | | |
| r=8 | cos/wr01/lr2e-4 | WMDP | | | | | | | | |
| | | MMLU | | | | | | | | |
| r=8 | cos/wr01/lr1e-3 | WMDP | | | | | | | | |
| | | MMLU | | | | | | | | |
| r=8 | cos/ws30/lr2e-4 | WMDP | | | | | | | | |
| | | MMLU | | | | | | | | |
| r=8 | cos/ws100/lr2e-4 | WMDP | | | | | | | | |
| | | MMLU | | | | | | | | |
| r=8 | const/wr01/lr2e-4 | WMDP | | | | | | | | |
| | | MMLU | | | | | | | | |
| r=64 | cos/wr01/lr2e-5 | WMDP | | | | | | | | |
| | | MMLU | | | | | | | | |
| r=64 | cos/wr01/lr2e-4 | WMDP | | | | | | | | |
| | | MMLU | | | | | | | | |
| r=64 | cos/wr01/lr1e-3 | WMDP | | | | | | | | |
| | | MMLU | | | | | | | | |
| r=64 | cos/ws30/lr2e-4 | WMDP | | | | | | | | |
| | | MMLU | | | | | | | | |
| r=64 | cos/ws100/lr2e-4 | WMDP | | | | | | | | |
| | | MMLU | | | | | | | | |
| r=64 | const/wr01/lr2e-4 | WMDP | | | | | | | | |
| | | MMLU | | | | | | | | |

### MMLU Restoration: Annealing Tamper on r8 and r64 (AdamW, 3000 steps, 1ep, evaluated every 500)

Data: Filtered annealing mix. Unlearned models: all-module, pdbs=1, rm23, orth5, ret0, all 32 layers. Baseline: 42.97 / 45.10.

| Rank | Config | Metric | Step 0 | Step 500 | Step 1000 | Step 1500 | Step 2000 | Step 2500 | Step 3000 | Notes |
|------|--------|--------|--------|----------|-----------|-----------|-----------|-----------|-----------|-------|
| r=8 | cos/wr01/lr2e-5 | WMDP | | | | | | | | |
| | | MMLU | | | | | | | | |
| r=8 | cos/wr01/lr2e-4 | WMDP | | | | | | | | |
| | | MMLU | | | | | | | | |
| r=8 | cos/wr01/lr1e-3 | WMDP | | | | | | | | |
| | | MMLU | | | | | | | | |
| r=8 | cos/ws30/lr2e-4 | WMDP | | | | | | | | |
| | | MMLU | | | | | | | | |
| r=8 | cos/ws100/lr2e-4 | WMDP | | | | | | | | |
| | | MMLU | | | | | | | | |
| r=8 | const/wr01/lr2e-4 | WMDP | | | | | | | | |
| | | MMLU | | | | | | | | |
| r=64 | cos/wr01/lr2e-5 | WMDP | | | | | | | | |
| | | MMLU | | | | | | | | |
| r=64 | cos/wr01/lr2e-4 | WMDP | | | | | | | | |
| | | MMLU | | | | | | | | |
| r=64 | cos/wr01/lr1e-3 | WMDP | | | | | | | | |
| | | MMLU | | | | | | | | |
| r=64 | cos/ws30/lr2e-4 | WMDP | | | | | | | | |
| | | MMLU | | | | | | | | |
| r=64 | cos/ws100/lr2e-4 | WMDP | | | | | | | | |
| | | MMLU | | | | | | | | |
| r=64 | const/wr01/lr2e-4 | WMDP | | | | | | | | |
| | | MMLU | | | | | | | | |

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
