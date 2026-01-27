# Orth Circuit Breaker HP Tuning

Goal: Find boundary between minimal capability impact and noticeable damage.

Defaults: `remove_coef=23`, `orth_coef=10`, `retain_coef=2`

## Baselines

| Model | WMDP Bio | MMLU STEM |
|-------|----------|-----------|
| Target model (to unlearn) | 42.97% | 36.85% |

## Metric Interpretation

**orth_loss**: Average pairwise cosine similarity between forget item representations in each batch. Measures whether forget items collapse to the same direction or remain diverse.
- **~1.0** = bad (all forget items aligned to same direction, vulnerable to single-direction attacks)
- **~0.5** = moderate diversity
- **~0** = good (orthogonal forget directions)

## Results

| remove_coef | orth_coef | steps | WMDP | MMLU | retain_loss | cb_loss | orth_loss | Notes |
|-------------|-----------|-------|------|------|-------------|---------|-----------|-------|
| 15          | 5         | 32    | -    | -    | -           | -       | -         | Not trained |
| 23          | 10        | 32    | -    | -    | -           | -       | -         | Model incomplete |
| 30          | 15        | 32    | 26.55% | 35.14% | 4.30      | 0.18    | 0.99      | |
| 30          | 15        | 512   | 24.12% | 22.14% | 1.08      | 0.015   | 0.05      | Severe capability damage |
| 40          | 20        | 32    | 26.63% | 35.39% | -         | -       | -         | Log truncated |
| 15          | 15        | 512   | 23.88% | 25.98% | 1.01      | 0.03    | 0.04      | Capability damage |
| 30          | 100       | 32    | 39.2% | 36.25% | 3.52      | 0.35    | 0.33      | Preserved capability, worse unlearning |
| 30          | 100       | 256   | pending | pending | -       | -       | -         | Job 2038086 |
