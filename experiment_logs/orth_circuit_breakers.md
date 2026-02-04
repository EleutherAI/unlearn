# Orth Circuit Breaker HP Tuning

Goal: Find boundary between minimal capability impact and noticeable damage.

Defaults: `remove_coef=23`, `orth_coef=10`, `retain_coef=2`

## Metric Interpretation

**orth_loss**: Average pairwise cosine similarity between forget item representations in each batch. Measures whether forget items collapse to the same direction or remain diverse.
- **~1.0** = bad (all forget items aligned to same direction, vulnerable to single-direction attacks)
- **~0.5** = moderate diversity
- **~0** = good (orthogonal forget directions)

## Results

| remove | orth | retain | steps | WMDP Robust | MMLU | retain_loss | cb_loss | orth_loss | Notes |
|--------|------|--------|-------|-------------|------|-------------|---------|-----------|-------|
| -      | -    | -      | -     | 42.97%    | 45.10% | -           | -       | -         | Baseline |
| 23     | 5    | 0      | 32    | -         | -     | -           | -       | -         | LoRA planned |
| 23     | 10   | 2      | 32    | -         | -     | -           | -       | -         | LoRA planned |
| 23     | 20   | 2      | 32    | -         | -     | -           | -       | -         | LoRA planned |
| 23     | 0    | 2      | 32    | -         | -     | -           | -       | -         | LoRA planned, orth=0 control |
| 23     | 10   | 2      | 32    | -         | -     | -           | -       | -         | LoRA planned, keyword mask (type: regex) |

## SFT

| remove | orth | retain | steps | WMDP | WMDP Robust | MMLU | retain_loss | cb_loss | orth_loss | Notes |
|--------|------|--------|-------|------|-------------|------|-------------|---------|-----------|-------|
| 23     | 10   | 2      | 32    | -      | -         | -           | -       | -         | SFT planned, keyword mask (type: regex) |
| 30     | 15   | 2      | 32    | 26.55% | -         | -    | 4.30        | 0.18    | 0.99      | |
| 30     | 15   | 2      | 512   | 24.12% | 25.81%    | 23.76% | 1.08        | 0.015   | 0.05      | Severe capability damage |
| 40     | 20   | 2      | 32    | 26.63% | -         | -    | -           | -       | -         | Log truncated |
| 15     | 15   | 2      | 512   | 23.88% | 25.35%    | 29.90% | 1.01        | 0.03    | 0.04      | Capability damage |
| 30     | 100  | 2      | 32    | 39.2%  | -         | -    | 3.52        | 0.35    | 0.33      | Preserved capability, worse unlearning |
| 30     | 100  | 2      | 256   | 29.85% | 28.34%    | 42.18% | 2.95        | 0.10    | 0.09      | Good balance |
| 30     | 15   | 2      | 64    | 27.73% | -         | -    | 3.46        | 0.06    | 0.52      | Good balance, near-baseline MMLU |
| 30     | 15   | 2      | 128   | 24.35% | -         | -    | 2.15        | 0.02    | 0.25      | Better orth, some capability loss |
| 15     | 15   | 15     | 512   | 28.28% | 28.11%    | 43.53% | 0.40        | 0.07    | 0.13      | Good unlearning + MMLU preserved |
| 15     | 15   | 50     | 512   | 35.90% | **33.64%**    | **44.81%** | 0.26        | 0.09    | 0.25      | Better MMLU, weaker unlearning |
