# KL Divergence Retain Loss Hyperparameter Tuning

## Goal
Find the boundary between:
1. Minimal capability impact on both retain and forget
2. Noticeable damage to both capabilities

## Configuration
- **Model**: EleutherAI/deep-ignorance-unfiltered
- **Checkpoint**: EleutherAI/deep-ignorance-pretraining-stage-unfiltered @ global_step38144
- **Loss type**: `--retain_kl_loss` (KL divergence on outputs)
- **Training examples**: 1024
- **GPUs**: 4x GH200
- **Batch size**: 4 per device, grad accumulation 2 (effective batch 32)
- **Learning rate**: 0.001
- **LoRA rank**: 16
- **Target layers**: [5, 10, 15, 20, 25, 30]

## Training Metrics

### 1 Epoch (32 steps)

| retain_coef | remove_coef | steps | retain_kl_loss | cb_loss | retain_acc | forget_acc |
|-------------|-------------|-------|----------------|---------|------------|------------|
| 1           | 3           | 32    | 0.247          | 6.30    | 76.1%      | 71.4%      |
| 3           | 5           | 32    | 0.203          | 6.30    | 78.6%      | 71.5%      |
| 5           | 5           | 32    | 0.156          | 6.31    | 81.1%      | 71.5%      |
| 10          | 5           | 32    | 0.101          | 6.33    | 84.6%      | 71.6%      |
| 5           | 10          | 32    | 0.229          | 6.35    | 77.3%      | 71.1%      |
| 10          | 10          | 32    | 0.169          | 6.36    | 80.4%      | 71.1%      |
| 5           | 15          | 32    | 0.243          | 6.35    | 76.6%      | 71.2%      |
| 5           | 20          | 32    | 0.259          | 6.36    | 75.7%      | 71.0%      |
| 5           | 30          | 32    | 0.272          | 6.51    | 75.5%      | 70.6%      |

### 4 Epochs (128 steps)

| retain_coef | remove_coef | steps | retain_kl_loss | cb_loss | retain_acc | forget_acc |
|-------------|-------------|-------|----------------|---------|------------|------------|
| 1           | 3           | 128   | 0.085          | 4.09    | 86.8%      | 74.9%      |
| 5           | 5           | 128   | 0.049          | 4.06    | 89.9%      | 74.6%      |
| 10          | 5           | 128   | 0.036          | 4.07    | 91.3%      | 74.0%      |

### 10 Epochs (320 steps)

| retain_coef | remove_coef | steps | retain_kl_loss | cb_loss | retain_acc | forget_acc |
|-------------|-------------|-------|----------------|---------|------------|------------|
| 5           | 8           | 320   | pending        | pending | pending    | pending    |

### Training Observations
- **Longer training (4 epochs) significantly reduces losses**: cb_loss drops from ~6.3 to ~4.1
- **retain_kl_loss decreases with more training**: 0.15-0.25 → 0.04-0.09
- **forget_acc improves with more training**: 71% → 74-75% (closer to checkpoint)
- **retain_acc improves with more training**: 76-85% → 87-91%

## Evaluation Results

### 1 Epoch (32 steps)

| retain_coef | remove_coef | steps | WMDP Bio (↓) | MMLU (↑) | Status |
|-------------|-------------|-------|--------------|----------|--------|
| 1           | 3           | 32    | 29.84% ± 1.55% | 33.61% ± 0.40% | ✓ |
| 5           | 5           | 32    | 31.68% ± 1.58% | 35.73% ± 0.40% | ✓ |
| 10          | 5           | 32    | 30.76% ± 1.57% | 34.98% ± 0.40% | ✓ |

### Aggressive Unlearning (High remove_coef, 1 epoch)

| retain_coef | remove_coef | steps | WMDP Bio (↓) | MMLU (↑) | Status |
|-------------|-------------|-------|--------------|----------|--------|
| 5           | 10          | 32    | 31.11% ± 1.57% | 34.55% ± 0.40% | ✓ |
| 5           | 15          | 32    | 30.53% ± 1.56% | 33.81% ± 0.40% | ✓ |
| 5           | 20          | 32    | 30.65% ± 1.56% | 33.68% ± 0.40% | ✓ |
| 5           | 30          | 32    | 30.07% ± 1.55% | 32.40% ± 0.39% | ✓ |

### 4 Epochs (128 steps)

| retain_coef | remove_coef | steps | WMDP Bio (↓) | MMLU (↑) | Status |
|-------------|-------------|-------|--------------|----------|--------|
| 1           | 3           | 128   | 30.76% ± 1.57% | 35.33% ± 0.40% | ✓ |
| 5           | 5           | 128   | 30.88% ± 1.57% | 34.82% ± 0.40% | ✓ |
| 10          | 5           | 128   | 31.22% ± 1.58% | 35.55% ± 0.40% | ✓ |

## Key Findings

### Boundary Analysis (1 epoch)
- **WMDP Bio random chance**: ~25%
- **Achieved WMDP Bio**: 29.84% - 31.68% (incomplete unlearning, ~5-7% above random)
- **MMLU retention**: 33.61% - 35.73%

### Effect of Training Length
- 4 epochs reduces circuit breaker loss by ~35% (6.3 → 4.1)
- Retain accuracy improves by ~10-15% with longer training
- Forget accuracy increases (models more closely match checkpoint)
- **However, 4 epochs does NOT improve unlearning** - WMDP Bio scores are similar or slightly worse

### Comparison: 1 Epoch vs 4 Epochs

| Setting | 1 Epoch WMDP Bio | 4 Epoch WMDP Bio | 1 Epoch MMLU | 4 Epoch MMLU |
|---------|------------------|------------------|--------------|--------------|
| ret=1, rm=3 | 29.84% | 30.76% (+0.9%) | 33.61% | 35.33% (+1.7%) |
| ret=5, rm=5 | 31.68% | 30.88% (-0.8%) | 35.73% | 34.82% (-0.9%) |
| ret=10, rm=5 | 30.76% | 31.22% (+0.5%) | 34.98% | 35.55% (+0.6%) |

### Trade-offs
| Setting | Steps | WMDP Bio | MMLU | Notes |
|---------|-------|----------|------|-------|
| Low retain (1,3) | 32 | Best (29.84%) | Worst (33.61%) | Aggressive unlearning |
| Default (5,5) | 32 | Middle (31.68%) | Best (35.73%) | Good balance |
| High retain (10,5) | 32 | Middle (30.76%) | Good (34.98%) | Strong retain pressure |
| Very aggressive (5,30) | 32 | 30.07% | 32.40% | MMLU degraded, WMDP Bio not improved |

### Conclusions
1. **KL divergence retain loss works** but unlearning incomplete at all settings
2. **1 epoch is sufficient** - longer training does not improve unlearning
3. **retain_coef=1, remove_coef=3 achieves best unlearning** (29.84% WMDP Bio)
4. **Aggressive remove_coef (15-30) degrades MMLU without improving WMDP Bio**
   - remove_coef=30: WMDP Bio 30.07% (worse than ret=1,rm=3), MMLU 32.40% (2-3% degraded)
5. **WMDP Bio plateau at ~30%** - cannot push below with this approach
6. **Boundary identified**: remove_coef > 15 causes MMLU degradation without unlearning benefit

### Recommended Settings
- **For maximum unlearning**: retain_coef=1, remove_coef=3, 1 epoch (WMDP Bio: 29.84%)
- **For balanced performance**: retain_coef=5, remove_coef=5, 1 epoch (MMLU: 35.73%)
- **Avoid**: remove_coef > 15 (damages MMLU without improving unlearning)
- **Avoid longer training** - does not improve unlearning, wastes compute

## Notes
- All models saved to `models/EleutherAI/deep-ignorance-unfiltered_kl_ret{X}_rm{Y}[_ep4]`
- Training time: ~6 min/epoch with 4x GH200
- Evaluation time: ~15 minutes per model
