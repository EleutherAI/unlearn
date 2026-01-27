# Tuned Lens Unlearning - Long Run (16384 examples, 2 epochs)

## Configuration
- **Model**: EleutherAI/deep-ignorance-unfiltered
- **Lens**: EleutherAI/deep-ignorance-unfiltered-lens (HuggingFace)
- **Training examples**: 16384 (2x original)
- **Epochs**: 2
- **Total steps**: ~1024 (vs 256 in original)
- **GPUs**: 4x GH200 (distributed training via torchrun)
- **Batch size**: 4 per device, grad accumulation (effective batch 32)
- **Learning rate**: 0.001
- **LoRA rank**: 16
- **Target layers**: [5, 10, 15, 20, 25, 30]
- **Coefficients**: retain_coef=5.0, remove_coef=5.0
- **Job ID**: 2020296

## Baselines

| Model | WMDP Bio | MMLU STEM |
|-------|----------|-----------|
| Target model (to unlearn) | 42.97% | 36.85% |

## Status
Running...

## Purpose
Test if longer training improves tamper resistance. Will run finetune_attack with eval_every=10 after completion.

## Results
(pending)
