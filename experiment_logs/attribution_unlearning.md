# Attribution-Based Unlearning Experiments

Base model: `EleutherAI/deep-ignorance-unfiltered` (6.86B, Pythia architecture)
Attribution scores: MAGIC per-token backprop through SGD training steps (LR=1e-4, max_seq_len=1024)
Baseline: WMDP Bio Robust=0.4297, MMLU=0.4510

## Available MAGIC Attribution Scores

All scores are per-token tensors of shape [N, 1024] stored as `per_token_scores.pt`.

| Training Data | N | Steps | Eval Task | Output Dir | Status |
|---|---|---|---|---|---|
| WikiText-103 | 1000 | 250 | WMDP Bio Robust | `magic_wmdp_msl1024_output` | Complete |
| WikiText-103 | 1000 | 250 | MMLU | `magic_mmlu_msl1024_output` | Complete |
| bio-retain | 1000 | 250 | WMDP Bio Robust | `magic_wmdp_retain_msl1024_output` | Complete |
| UltraChat | 1000 | 250 | WMDP Bio Robust | `magic_ultrachat_msl1024_output` | Complete |
| bio-forget | 1000 | 250 | WMDP Bio Robust | `magic_wmdp_forget_msl1024_output` | Empty (backward not run) |
| wmdp-lie-o | 1000 | 250 | WMDP Bio Robust | `magic_wmdp_lie_o_msl1024_output` | Empty (backward not run) |
| bio-forget | 10000 | 1250 | WMDP Bio Robust | `magic_wmdp_forget_10k_wmdp_msl1024_output` | Running (job 2452372) |
| bio-forget | 10000 | 1250 | MMLU | `magic_wmdp_forget_10k_mmlu_msl1024_output` | Running (job 2452373) |

Checkpoints stored at `/projects/a6a/public/lucia/magic_{tag}_msl1024_ckpts/`.
Output dirs under `bergson3/runs/`.

Attribution scores indicate the effect on loss, so a negative score indicates that training on the token will reduce loss. This is what we want!

All attribution scores are taken on the same dataset as the unlearning dataset, using a model trained on the unfiltered/weighted unlearning dataset, unless otherwise stated.

## Experiment 1: Attribution-weighted unlearning (all tokens, value-weighted)

Uses WMDP attribution scores as per-token loss weights (normalized by mean absolute value so typical weight ~ ±1). Negative-score tokens get gradient ascent (unlearn), positive-score tokens get gradient descent (reinforce).

- SFT
- bio-retain corpus
- 1k training examples
- Adam, 1 epoch, bs=4, 250 steps.

Script: `bergson3/runs/attribution_unlearn.py`

| LR   | WMDP Bio Robust | MMLU   |
|------|-----------------|--------|
| 5e-6 | 0.4332          |        |
| 1e-5 | 0.4378          |        |
| 2e-5 | 0.4332          |        |
| 5e-5 | 0.3975          | 0.4106 |
| 8e-5 | 0.2523          | 0.2530 |
| 1e-4 | 0.2408          |        |
| 2e-4 | 0.2673          |        |

## Experiment 2: Attribution sign-only unlearning (all tokens, sign-weighted)

Same as experiment 1 but weights are `sign(scores)` ({-1, 0, +1}) instead of normalized raw values.

Script: `bergson3/runs/attribution_unlearn.py --sign_only`

| LR   | WMDP Bio Robust | MMLU   |
|------|-----------------|--------|
| 1e-5 | 0.2673          | 0.2295 |
| 5e-5 | 0.2350          | 0.2465 |
| 8e-5 | 0.2350          | 0.2465 |
| 1e-4 | 0.2350          | 0.2465 |
| 2e-4 | 0.2350          | 0.2465 |

## Experiment 3: Positive-only token training (WikiText, WMDP scores)

Positively-attributed tokens only (WMDP score > 0). Negative/zero tokens masked with -100.

- SFT
- WikiText-103
- 1k examples
- AdamW, cosine schedule, warmup=0.1, bs=32, 31 steps, max_seq_len=256.

Script: `bergson3/runs/validate_attribution.py --experiment positive_only`

| LR   | WMDP Bio Robust | MMLU   |
|------|-----------------|--------|
| 1e-6 | 0.4297          | 0.4494 |
| 1e-5 | 0.4274          | 0.4484 |
| 5e-5 | 0.3894          | 0.3823 |
| 2e-4 | 0.2385          | 0.2502 |

## Experiment 4: Negative-only token training (WikiText, WMDP scores)

Negative-attributed/proponent tokens only (WMDP score < 0). Positive/zero tokens masked with -100.

- SFT
- WikiText-103
- 1k examples
- AdamW, cosine schedule, warmup=0.1, bs=8, 125 steps, max_seq_len=1024.

Script: `bergson3/runs/validate_attribution.py --experiment negative_only`

| LR   | WMDP Bio Robust | MMLU   |
|------|-----------------|--------|
| 1e-5 | 0.4343          | 0.4479 |
| 5e-5 | 0.4666          | 0.4575 |
| 1e-4 | 0.4240          | 0.4476 |
| 2e-4 | 0.3548          | 0.3714 |

## Experiment 5: Selective token training (WMDP+ AND MMLU≤0)

Tokens that detract from WMDP (positive attribution score) and are proponents of MMLU (negative attribution score).

These tokens make up 14.8% of the 37,452 in the dataset.

- SFT
- WikiText-103
- 1k examples
- AdamW, cosine schedule, warmup=0.1, bs=8, 125 steps, max_seq_len=1024.

Script: `bergson3/runs/validate_attribution.py --experiment selective`

| LR   | WMDP Bio Robust | MMLU   |
|------|-----------------|--------|
| 1e-6 | 0.4320          | 0.4484 |
| 1e-5 | 0.4240          | 0.4511 |
| 5e-5 | 0.2811          | 0.3027 |
| 2e-4 | 0.2350          | 0.2467 |

## Experiment 6: Compatible token training (WMDP+ AND MMLU-)

Tokens with positive WMDP score (training hurts WMDP) AND negative MMLU score (training helps MMLU).

- SFT
- WikiText-103
- 1k examples
- AdamW, cosine schedule, warmup=0.1, bs=8, 125 steps, max_seq_len=1024

Script: `bergson3/runs/validate_attribution.py --experiment compatible`

| LR   | WMDP Bio Robust | MMLU |
|------|-----------------|------|
| 1e-5 |                 |      |
| 5e-5 |                 |      |
| 1e-4 |                 |      |
| 2e-4 |                 |      |

## Experiment 7: All tokens baseline (WikiText, no attribution masking)

All tokens (no masking). Baseline to compare against attribution-guided experiments 3-6.

- SFT
- WikiText-103
- 1k examples
- AdamW, cosine schedule, warmup=0.1, bs=8, 125 steps, max_seq_len=1024

Script: `bergson3/runs/validate_attribution.py --experiment all_tokens`

| LR   | WMDP Bio Robust | MMLU   |
|------|-----------------|--------|
| 1e-5 |                 |        |
| 5e-5 |                 |        |
| 1e-4 |                 |        |
| 2e-4 |                 |        |
