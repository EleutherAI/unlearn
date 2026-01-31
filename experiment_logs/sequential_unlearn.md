# Sequential Unlearning Experiments

Sequential back-to-front unlearning: for each layer L (from last to first), use the original model's layers L→N as a "probe" to compute forget loss while retaining performance on retain data by taking a retain loss through the updated model.

Note that hyperparameters don't transfer between number of training steps.

## Method

**Training**: LoRA fine-tuning (not full SFT)
- LoRA rank: 16
- LoRA alpha: 16
- Target modules: default (all linear layers)

**Losses**:
- Forget loss: Cross-entropy with random targets through original model's remaining layers (maximizes entropy)
- Retain loss: L2 norm between current and original hidden states across all layers

**Loss scheduling**:
- retain_coeff ramps from 0 → retain_coef over training
- forget_coeff ramps from remove_coef → 0.75*remove_coef over training

## Experiments

### L2 Norm Retain Loss (LoRA)

| Run | Layers | remove_coef | retain_coef | Steps | WMDP Bio Robust | MMLU | Notes |
|-----|--------|-------------|-------------|-------|-----------------|------|-------|
| - | - | - | - | - | 0.4297 | 0.4510 | Baseline |
| 1 | 28→16 (step 4) | 10 | 5 | 128 | 0.3134 | 0.4479 | |
| 2 | 28→16 (step 4) | 20 | 5 | 128 | 0.3076 | 0.4454 | |
| 3 | 28→16 (step 4) | 30 | 5 | 128 | 0.2995 | 0.4458 | |
| 4 | 28→16 (step 4) | 40 | 5 | 128 | 0.2961 | 0.4410 | |
| 5 | 28→16 (step 4) | 50 | 5 | 128 | **0.2684** | **0.4437** | Below random (0.25) |
| 6 | 28→16 (step 4) | 50 | 5 | 1280 | 0.2535 | 0.2540 | 10x examples, MMLU collapsed |

### KL Retain Loss (LoRA)

| Run | Layers | remove_coef | retain_coef | Steps | WMDP Bio Robust | MMLU | Notes |
|-----|--------|-------------|-------------|-------|-----------------|------|-------|
| - | - | - | - | - | 0.4297 | 0.4510 | Baseline |
| 1 | 28→16 (step 4) | 50 | 5 | 128 | 0.2673 | 0.2295 | Collapsed |
| 2 | 28→16 (step 4) | 20 | 10 | 128 | 0.2673 | - | Collapsed |
| 3 | 28→16 (step 4) | 30 | 10 | 128 | 0.2673 | - | Collapsed |
| 4 | 28→16 (step 4) | 20 | 20 | 128 | 0.2673 | 0.2295 | Collapsed |
| 5 | 28→16 (step 4) | 10 | 20 | 128 | 0.2673 | - | Collapsed |
| 6 | 28→16 (step 4) | 10 | 50 | 128 | 0.2673 | 0.2295 | Collapsed |
| 7 | 28→16 (step 4) | 5 | 50 | 128 | 0.2730 | 0.4034 | Works |
| 8 | 28→16 (step 4) | 5 | 100 | 128 | **0.2788** | **0.4195** | Best KL |
| 9 | 28→16 (step 4) | 5 | 100 | 1280 | 0.2673 | 0.2295 | 10x examples |

### Max Entropy KL Forget Loss, L2 Retain Loss (LoRA)

1024 examples, layers 31→8 (step 4), 4 GPUs, pdbs=4, grad_accum=2, global batch 32, 32 steps/layer, 192 total steps.

| Run | Layers | remove_coef | retain_coef | Steps | retain_loss | forget_loss | WMDP Bio Robust | MMLU | Notes |
|-----|--------|-------------|-------------|-------|-------------|-------------|-----------------|------|-------|
| - | - | - | - | - | - | - | 0.4297 | 0.4510 | Baseline |
| 1 | 31→11 (step 4) | 10 | 5 | 192 | 2.55 | 1.05 | 0.2857 | 0.4067 | |

### Max Entropy KL Forget Loss, KL Retain Loss (SFT, FSDP)

1024 examples, layers 31→8 (step 4), 2 nodes / 8 GPUs, pdbs=1, grad_accum=4, 128 steps/phase, 768 total steps.

| Run | Layers | remove_coef | retain_coef | lr | Steps | retain_loss | forget_loss | WMDP Bio Robust | MMLU | Notes |
|-----|--------|-------------|-------------|-----|-------|-------------|-------------|-----------------|------|-------|
| - | - | - | - | - | - | - | - | 0.4297 | 0.4510 | Baseline |
| 1 | 31→11 (step 4) | 5 | 20 | 1e-3 | 768 | ~7300 | 1.22 | 0.2419 | 0.2453 | Collapsed |
| 2 | 31→11 (step 4) | 5 | 80 | 1e-5 | 768 | | | | | |
| 3 | 31→11 (step 4) | 5 | 80 | 4e-4 | 768 | | | | | |
| 4 | 31→11 (step 4) | 5 | 200 | 1e-5 | 768 | | | | | |
| 5 | 31→11 (step 4) | 5 | 200 | 4e-4 | 768 | | | | | |
| 6 | 31→11 (step 4) | 5 | 400 | 1e-5 | 768 | | | | | |
| 7 | 31→11 (step 4) | 5 | 400 | 4e-4 | 768 | | | | | |
| 8 | 31→11 (step 4) | 5 | 2000 | 1e-5 | 768 | | | | | |
| 9 | 31→11 (step 4) | 5 | 2000 | 4e-4 | 768 | | | | | |

## Hyperparameters (all runs)

| Parameter | Value |
|-----------|-------|
| per_device_batch_size | 4 |
| gradient_accumulation | 2 |
| global_batch_size | 32 |
| learning_rate | 1e-3 |
| lora_r | 16 |
| epochs_per_layer | 1 |
| layers_unlearned | 28, 24, 20, 16 |
| optimizer | AdamW |
| bf16 | True |
| gradient_checkpointing | True |

## Tampering Attack Results

Fine-tuning unlearned models on WMDP-Bio-Remove dataset to test recovery:

| Model | Step 0 | Step 10 | Step 100 |
|-------|--------|---------|----------|
| Baseline (target) | 0.4297 | - | - |
| L2 (rm50/ret5) | 0.2696 | 0.4055 | 0.4182 |
| KL (rm5/ret100) | 0.2788 | 0.4078 | 0.4217 |

Both models show rapid recovery (~10 steps).
Plots saved in `runs/tamper_attack/` and `runs/tamper_kl_rm5_ret100/`.
