# Lens Unlearning with Muon Optimizer - Tuning

## Goal
Tune Muon optimizer hyperparameters for lens-based unlearning until achieving:
- WMDP Bio: ≤25% (random chance)
- MMLU STEM: ≥34%

## Configuration
- **Model**: EleutherAI/deep-ignorance-unfiltered
- **Lens**: EleutherAI/deep-ignorance-unfiltered-lens
- **Optimizer**: MuonAdamW (Muon for 2D matrices, AdamW for embeddings/biases)
- **GPUs**: 4x GH200
- **Target layers**: [5, 10, 15, 20, 25, 30]

## Baseline

| Model | WMDP Bio | MMLU STEM |
|-------|----------|-----------|
| Target model (to unlearn) | 42.97% | 36.85% |

## Hyperparameter Grid

| Parameter | Default | Values to try |
|-----------|---------|---------------|
| muon_lr | 0.02 | 1e-4, 1e-3, 0.02 |
| adam_lr | 3e-4 | 1e-4, 3e-4, 1e-3 |
| remove_coef | 5.0 | 2.0, 3.0, 5.0 |
| retain_coef | 0.0 | 0.0, 0.5, 1.0 |
| num_train_examples | 1024 | 512, 1024 |

## Tuning Results (Pre-Fix)

Runs 1-6 had a bug where retain_loss was always 0 for SFT mode.

| Run | muon_lr | adam_lr | remove_coef | retain_coef | examples | steps | WMDP Bio | MMLU STEM | Notes |
|-----|---------|---------|-------------|-------------|----------|-------|----------|-----------|-------|
| 1 | 0.02 | 3e-4 | 5.0 | 0.0 | 1024 | 128 | 24.74% | 23.88% | Over-unlearned |
| 2 | 0.02 | 3e-4 | 2.0 | 0.0 | 1024 | 128 | 24.74% | 23.88% | Same results |
| 3 | 0.02 | 3e-4 | 2.0 | 0.0 | 512 | 64 | 24.74% | 23.88% | Same results |
| 4 | 0.005 | 3e-4 | 2.0 | 0.0 | 1024 | 128 | 24.74% | 23.88% | Same results |
| 5 | 1e-3 | 3e-4 | 2.0 | 1.0 | 1024 | 32 | 24.74% | 23.88% | retain_loss=0 (bug) |
| 6 | 1e-4 | 3e-4 | 2.0 | 1.0 | 1024 | - | - | - | OOM |

**BUG FIXED**: Added frozen reference model for SFT retain loss computation.

## Tuning Results (Post-Fix)

| Run | muon_lr | adam_lr | remove_coef | retain_coef | pdbs | steps | WMDP Bio | MMLU STEM | retain_loss | forget_loss | Notes |
|-----|---------|---------|-------------|-------------|------|-------|----------|-----------|-------------|-------------|-------|
| 7 | 1e-3 | 3e-4 | 2.0 | 1.0 | 1 | 32 | 32.91% | 34.51% | 31-60 | ~1.03 | MMLU good, need more unlearning |
| 8 | 1e-3 | 3e-4 | 3.0 | 1.0 | 1 | 32 | 25.14% | 27.15% | 33-53 | ~1.04 | WMDP good, MMLU too low |
| 9 | 1e-3 | 3e-4 | 4.0 | 0.5 | 1 | 32 | 25.69% | 23.88% | 31-79 | ~1.03 | Both below target |
| 10 | 1e-3 | 3e-4 | 2.5 | 1.0 | 1 | 32 | - | - | - | - | Between runs 7 and 8 |
| 11 | 1e-3 | 3e-4 | 2.0 | 1.5 | 1 | 32 | - | - | - | - | Higher retain to preserve MMLU |

## Best Configuration
(to be filled)

## Tampering Results
(to be filled after successful unlearning)
