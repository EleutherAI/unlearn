# Attribution-Based Unlearning Experiments

Base model: `EleutherAI/deep-ignorance-unfiltered` (6.86B, Pythia architecture)
Training data: WikiText-103 (1000 examples, median length quartile)
Attribution scores: MAGIC per-token backprop through 250 SGD training steps
Baseline: WMDP-bio-robust=0.4297, MMLU=0.4510

## Experiment 1: Attribution-weighted unlearning (all tokens, value-weighted)

Fine-tune on all retain tokens using attribution scores as per-token loss weights (normalized by mean absolute value so typical weight ~ ±1). Negative-score tokens get gradient ascent (unlearn), positive-score tokens get gradient descent (reinforce). Uses bergson functional Trainer with torchopt Adam, 1 epoch, bs=4, 250 steps. WMDP attribution scores from bio-retain corpus.

Script: `bergson3/runs/attribution_unlearn.py`

| LR   | WMDP-bio-robust | MMLU   |
|------|-----------------|--------|
| 5e-6 | 0.4332          |        |
| 1e-5 | 0.4378          |        |
| 2e-5 | 0.4332          |        |
| 5e-5 | 0.3975          | 0.4106 |
| 8e-5 | 0.2523          | 0.2530 |
| 1e-4 | 0.2408          |        |
| 2e-4 | 0.2673          |        |

## Experiment 2: Attribution sign-only unlearning (all tokens, sign-weighted)

Same as experiment 1 but weights are `sign(scores)` ({-1, 0, +1}) instead of normalized raw values. WMDP attribution scores from bio-retain corpus.

Script: `bergson3/runs/attribution_unlearn.py --sign_only`

| LR   | WMDP-bio-robust | MMLU   |
|------|-----------------|--------|
| 1e-5 | 0.2673          | 0.2295 |
| 5e-5 | 0.2350          | 0.2465 |
| 8e-5 | 0.2350          | 0.2465 |
| 1e-4 | 0.2350          | 0.2465 |
| 2e-4 | 0.2350          | 0.2465 |

## Experiment 3: Positive-only token training (WikiText, WMDP scores)

Standard SFT on positively-attributed tokens only (WMDP score > 0). Negative/zero tokens masked with -100. AdamW, cosine schedule, warmup=0.1, bs=32, 31 steps, max_seq_len=256. WMDP attribution scores from WikiText training.

Script: `bergson3/runs/validate_attribution.py --experiment positive_only`

| LR   | WMDP-bio-robust | MMLU   |
|------|-----------------|--------|
| 1e-6 | 0.4297          | 0.4494 |
| 1e-5 | 0.4274          | 0.4484 |
| 5e-5 | 0.3894          | 0.3823 |
| 2e-4 | 0.2385          | 0.2502 |

## Experiment 4: Negative-only token training (WikiText, WMDP scores)

Standard SFT on negatively-attributed tokens only (WMDP score < 0). Positive/zero tokens masked with -100. AdamW, cosine schedule, warmup=0.1, bs=8, 125 steps, max_seq_len=1024. WMDP attribution scores from WikiText training.

Script: `bergson3/runs/validate_attribution.py --experiment negative_only`

| LR   | WMDP-bio-robust | MMLU   |
|------|-----------------|--------|
| 1e-5 | 0.4343          | 0.4479 |
| 5e-5 | 0.4666          | 0.4575 |
| 1e-4 | 0.4240          | 0.4476 |
| 2e-4 | 0.3548          | 0.3714 |

## Experiment 5: Selective token training (WMDP+ AND MMLU≤0)

Standard SFT on tokens with positive WMDP score AND non-positive MMLU score — tokens that contribute to WMDP bio knowledge but don't help MMLU. 5,538 selective tokens out of 37,452 WMDP-positive (14.8%). AdamW, cosine schedule, warmup=0.1, bs=8, 125 steps, max_seq_len=1024.

WMDP scores: `bergson3/runs/magic_wmdp_msl1024_output/per_token_scores.pt`
MMLU scores: `bergson3/runs/magic_mmlu_msl1024_output/per_token_scores.pt`

Script: `bergson3/runs/validate_attribution.py --experiment selective`

| LR   | WMDP-bio-robust | MMLU   |
|------|-----------------|--------|
| 1e-6 | 0.4320          | 0.4484 |
| 1e-5 | 0.4240          | 0.4511 |
| 5e-5 | 0.2811          | 0.3027 |
| 2e-4 | 0.2350          | 0.2467 |

## Experiment 6: Compatible token training (WMDP+ AND MMLU-)

Standard SFT on tokens with positive WMDP score (training hurts WMDP) AND negative MMLU score (training helps MMLU). AdamW, cosine schedule, warmup=0.1, bs=8, 125 steps, max_seq_len=1024.

WMDP scores: `bergson3/runs/magic_wmdp_msl1024_output/per_token_scores.pt`
MMLU scores: `bergson3/runs/magic_mmlu_msl1024_output/per_token_scores.pt`

Script: `bergson3/runs/validate_attribution.py --experiment compatible`

| LR   | WMDP-bio-robust | MMLU |
|------|-----------------|------|
| 1e-5 |                 |      |
| 5e-5 |                 |      |
| 1e-4 |                 |      |
| 2e-4 |                 |      |
