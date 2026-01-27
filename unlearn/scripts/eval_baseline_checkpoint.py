#!/usr/bin/env python3
"""Evaluate the baseline checkpoint model on WMDP Bio Robust and MMLU STEM."""

import argparse
import os

import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="EleutherAI/deep-ignorance-pretraining-stage-unfiltered",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="global_step38144",
        help="Model revision/commit",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--include_path",
        type=str,
        default=None,
        help="Path to lm_eval_tasks for custom tasks",
    )
    args = parser.parse_args()

    # Auto-detect include_path if not provided
    if args.include_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        potential_path = os.path.join(script_dir, "..", "lm_eval_tasks")
        if os.path.exists(potential_path):
            args.include_path = potential_path

    print(f"Loading model: {args.model_name} @ {args.revision}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, revision=args.revision)

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
    )

    # Setup task manager with custom tasks
    tm = TaskManager(verbosity="INFO", include_path=args.include_path)

    # Run WMDP Bio Robust
    print("\n" + "=" * 60)
    print("Running WMDP Bio Robust evaluation...")
    print("=" * 60)
    wmdp_results = simple_evaluate(
        model=lm,
        tasks=["wmdp_bio_robust"],
        task_manager=tm,
    )

    if "results" in wmdp_results and "wmdp_bio_robust" in wmdp_results["results"]:
        task_results = wmdp_results["results"]["wmdp_bio_robust"]
        print("\nWMDP Bio Robust Results:")
        for metric, value in task_results.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

    # Run MMLU STEM
    print("\n" + "=" * 60)
    print("Running MMLU STEM evaluation...")
    print("=" * 60)
    mmlu_results = simple_evaluate(
        model=lm,
        tasks=["mmlu_stem"],
    )

    if "results" in mmlu_results and "mmlu_stem" in mmlu_results["results"]:
        task_results = mmlu_results["results"]["mmlu_stem"]
        print("\nMMLU STEM Results:")
        for metric, value in task_results.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model: {args.model_name} @ {args.revision}")

    wmdp_acc = None
    mmlu_acc = None

    if "results" in wmdp_results and "wmdp_bio_robust" in wmdp_results["results"]:
        wmdp_acc = wmdp_results["results"]["wmdp_bio_robust"].get("acc,none")
        wmdp_stderr = wmdp_results["results"]["wmdp_bio_robust"].get("acc_stderr,none")
        if wmdp_acc is not None:
            print(f"WMDP Bio Robust: {wmdp_acc*100:.2f}% ± {wmdp_stderr*100:.2f}%")

    if "results" in mmlu_results and "mmlu_stem" in mmlu_results["results"]:
        mmlu_acc = mmlu_results["results"]["mmlu_stem"].get("acc,none")
        mmlu_stderr = mmlu_results["results"]["mmlu_stem"].get("acc_stderr,none")
        if mmlu_acc is not None:
            print(f"MMLU STEM: {mmlu_acc*100:.2f}% ± {mmlu_stderr*100:.2f}%")


if __name__ == "__main__":
    main()
