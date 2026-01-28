# Sequential Unlearning Experiments

Sequential back-to-front unlearning: for each layer L (from last to first), use the original model's layers L→N as a "probe" to compute forget loss while retaining activations close to original.

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

## Baseline
- Model: EleutherAI/deep-ignorance-unfiltered
- WMDP Bio Robust: ~0.44
- MMLU: ~0.45

## Experiments

| Run | Layers | remove_coef | retain_coef | WMDP Bio Robust | MMLU | Notes |
|-----|--------|-------------|-------------|-----------------|------|-------|
| 1 | 28→16 (step 4) | 10 | 5 | 0.3134 | 0.4479 | |
| 2 | 28→16 (step 4) | 20 | 5 | 0.3076 | 0.4454 | |
| 3 | 28→16 (step 4) | 30 | 5 | 0.2995 | 0.4458 | |
| 4 | 28→16 (step 4) | 40 | 5 | 0.2961 | 0.4410 | |
| 5 | 28→16 (step 4) | 50 | 5 | 0.2684 | 0.4437 | Below random (0.25) |

## Hyperparameters (all runs)

| Parameter | Value |
|-----------|-------|
| num_train_examples | 1024 |
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

## Key Findings

1. Higher remove_coef pushes WMDP closer to (and below) random chance
2. MMLU degradation remains minimal even at rm50 (~1% drop from baseline)
3. Best result: rm50 achieves 0.2684 WMDP with 0.4437 MMLU
4. Random chance for 4-way MCQ is 0.25
