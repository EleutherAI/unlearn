"""LoRA checkpoint transfer unlearning.

Trains a LoRA adapter using checkpoint transfer and saves the adapter
separately from the base model. The adapter can be loaded later with
PeftModel.from_pretrained() on top of the base model.

A merged copy is also saved for evaluation with the existing eval scripts.

Best config from checkpoint_transfer_unlearn.md:
  retain_coef=5, remove_coef=5, 512 steps (num_train_examples=2048, pdbs=4)
  WMDP Robust: 37.79%, MMLU: 43.02% (-0.73% from baseline 45.10%)
"""

import os
import tempfile
from dataclasses import dataclass, field
from typing import Literal, cast

import torch
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model
from simple_parsing import ArgumentParser
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    TrainingArguments,
)

from unlearn.algorithm.checkpoint_transfer_unlearn import RRTrainer
from unlearn.algorithm.online_affine_fitter import load_affine_transforms
from unlearn.utils.keyword_masks import apply_keyword_masks
from unlearn.utils.muon import MuonAdamW
from unlearn.utils.unlearning_dataset import get_unlearning_dataset
from unlearn.utils.worker_utils import get_model_and_tokenizer


class MuonRRTrainer(RRTrainer):
    def create_optimizer(self):
        self.optimizer = MuonAdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        return self.optimizer


@dataclass
class LoraCheckpointTransferConfig:
    num_train_examples: int = 2048
    retain_coef: float = 5.0
    remove_coef: float = 5.0
    retain_kl_loss: bool = True
    retain_ce_loss: bool = False

    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    lr: float = 1e-3
    pdbs: int = 4
    epochs: int = 1
    global_batch_size: int = 32
    lr_warmup: bool = False

    layers: list[int] = field(default_factory=lambda: list(range(32)))

    model_name: str = "EleutherAI/deep-ignorance-unfiltered"
    revision: str = "main"
    checkpoint_name: str = "EleutherAI/deep-ignorance-pretraining-stage-unfiltered"
    checkpoint_revision: str = "global_step38144"

    load_affine_from_hub: str | None = "EleutherAI/affine-checkpoint-transfer"

    optimizer: Literal["adamw", "muon"] = "adamw"

    save_path: str = ""
    save_merged: bool = True

    unlearn_corrupt: bool = False
    corrupt_ratio: float = 0.5
    corrupt_ds: Literal["rewritten", "shuffled"] = "rewritten"
    hidden_dim: int = 4096


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"

    NUM_PROC = os.cpu_count() // 2

    parser = ArgumentParser()
    parser.add_arguments(LoraCheckpointTransferConfig, dest="run_cfg")
    run_cfg = parser.parse_args().run_cfg

    print("Parsed arguments:")
    for arg, value in vars(run_cfg).items():
        print(f"  {arg}: {value}")
    print()

    model, tokenizer = get_model_and_tokenizer(
        run_cfg.model_name, revision=run_cfg.revision
    )
    train_dataset = get_unlearning_dataset(run_cfg, tokenizer, NUM_PROC)
    train_dataset.tokenized_bio_remove_dataset = apply_keyword_masks(
        train_dataset.tokenized_bio_remove_dataset, run_cfg, tokenizer
    )

    lora_layers_to_transform = list(range(max(run_cfg.layers) + 1))

    if "OLMo" in run_cfg.model_name:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    else:
        target_modules = [
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ]

    lora_config = LoraConfig(
        r=run_cfg.lora_r,
        lora_alpha=run_cfg.lora_alpha,
        target_modules=target_modules,
        lora_dropout=run_cfg.lora_dropout,
        bias="none",
        layers_to_transform=lora_layers_to_transform,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model = cast(PreTrainedModel, model)
    model.enable_input_require_grads()

    print(f"LoRA trainable parameters: {model.print_trainable_parameters()}")

    # Load checkpoint model
    print(
        f"Loading checkpoint model: {run_cfg.checkpoint_name} @ "
        f"{run_cfg.checkpoint_revision}"
    )
    checkpoint_model = AutoModelForCausalLM.from_pretrained(
        run_cfg.checkpoint_name,
        revision=run_cfg.checkpoint_revision,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    checkpoint_model.eval()
    for param in checkpoint_model.parameters():
        param.requires_grad = False

    # Load affine transforms
    affine_transforms = {}
    if run_cfg.load_affine_from_hub:
        print(f"Loading affine transforms from {run_cfg.load_affine_from_hub}...")
        with tempfile.TemporaryDirectory() as tmpdir:
            hf_hub_download(
                repo_id=run_cfg.load_affine_from_hub,
                filename="affine_transforms.safetensors",
                local_dir=tmpdir,
            )
            hf_hub_download(
                repo_id=run_cfg.load_affine_from_hub,
                filename="metadata.json",
                local_dir=tmpdir,
            )
            affine_transforms = load_affine_transforms(tmpdir, device="cuda")

        for idx, transform in affine_transforms.items():
            affine_transforms[idx] = transform.to(
                device=model.device if hasattr(model, "device") else "cuda",
                dtype=torch.float16,
            )
            affine_transforms[idx].requires_grad_(False)
        print(f"Loaded affine transforms for layers: {list(affine_transforms.keys())}")

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    grad_acc_steps = max(
        1, run_cfg.global_batch_size // (run_cfg.pdbs * world_size)
    )

    print(
        f"Running with {world_size} GPUs. Per device batch: {run_cfg.pdbs}. "
        f"Grad Acc steps: {grad_acc_steps}."
    )

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=run_cfg.lr,
        gradient_accumulation_steps=grad_acc_steps,
        per_device_train_batch_size=run_cfg.pdbs,
        per_device_eval_batch_size=run_cfg.pdbs,
        num_train_epochs=run_cfg.epochs,
        weight_decay=0.01,
        gradient_checkpointing=True,
        bf16=True,
        save_strategy="no",
        warmup_steps=10 if run_cfg.lr_warmup else 0,
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "fsdp_version": 2,
            "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "activation_checkpointing": True,
            "state_dict_type": "FULL_STATE_DICT",
        },
    )

    TrainerClass = MuonRRTrainer if run_cfg.optimizer == "muon" else RRTrainer
    trainer_kwargs = dict(
        run_args=run_cfg,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        lora_target_layers=run_cfg.layers,
        checkpoint_model=checkpoint_model,
        affine_transforms=affine_transforms,
    )
    if run_cfg.optimizer == "muon":
        print(f"Using Muon optimizer (lr={run_cfg.lr})")
    trainer = TrainerClass(**trainer_kwargs)

    model.train()
    trainer.train()

    if run_cfg.save_path:
        adapter_path = os.path.join(run_cfg.save_path, "adapter")
        os.makedirs(adapter_path, exist_ok=True)

        trainer.accelerator.wait_for_everyone()
        if trainer.accelerator.is_main_process:
            unwrapped = trainer.accelerator.unwrap_model(trainer.model)
            unwrapped.save_pretrained(adapter_path, safe_serialization=True)
            tokenizer.save_pretrained(adapter_path)
            print(f"Saved LoRA adapter to {adapter_path}")

            if run_cfg.save_merged:
                merged_path = os.path.join(run_cfg.save_path, "merged")
                os.makedirs(merged_path, exist_ok=True)
                merged_model = unwrapped.merge_and_unload()
                merged_model.save_pretrained(
                    merged_path, safe_serialization=True
                )
                tokenizer.save_pretrained(merged_path)
                print(f"Saved merged model to {merged_path}")

        trainer.accelerator.wait_for_everyone()

    print("Done")
