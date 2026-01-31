"""Sequential back-to-front unlearning (SFT - full parameter training with FSDP)."""

import os
from dataclasses import dataclass
from typing import Literal, cast

import torch
import torch.nn.functional as F
from simple_parsing import ArgumentParser
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import seed_worker

from unlearn.algorithm.sequential_unlearn import forward_from_layer
from unlearn.utils.math import max_entropy_kl_loss
from unlearn.utils.unlearning_dataset import get_unlearning_dataset
from unlearn.utils.worker_utils import get_model_and_tokenizer


class SequentialSftTrainer(Trainer):

    def __init__(
        self,
        run_args,
        model,
        args,
        train_dataset,
        tokenizer,
        layers_to_unlearn,
        frozen_model,
        steps_per_phase,
        **kwargs,
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
        )
        self.run_args = run_args
        self.tokenizer = tokenizer
        self.layers_to_unlearn = layers_to_unlearn
        self.frozen_model = frozen_model
        self.steps_per_phase = steps_per_phase
        self.current_training_step = 0

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(
            DataLoader(self.train_dataset, **dataloader_params)
        )  # type: ignore

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        target_device = inputs["input_ids"].device

        retain_input_ids = inputs["input_ids"].to(target_device)
        forget_input_ids = inputs["bio_remove_input_ids"].to(target_device)
        forget_attention_mask = inputs["bio_remove_attention_mask"].to(target_device)

        # Determine current target layer from training progress
        phase_idx = min(
            self.current_training_step // self.steps_per_phase,
            len(self.layers_to_unlearn) - 1,
        )
        target_layer = self.layers_to_unlearn[phase_idx]

        # Per-phase coefficient scheduling
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        phase_step = self.current_training_step - phase_idx * self.steps_per_phase
        scheduled_coeff = min(
            1.0,
            phase_step
            / (self.run_args.num_train_examples / (self.run_args.pdbs * world_size)),
        )
        retain_coeff = self.run_args.retain_coef * (0.25 + 0.75 * scheduled_coeff)
        forget_coeff = self.run_args.remove_coef * (1 - 0.25 * scheduled_coeff)

        model.train()

        # Retain loss - KL divergence against frozen reference
        if retain_coeff > 0:
            with torch.no_grad():
                ref_outputs = self.frozen_model(retain_input_ids, attention_mask=None)
                ref_logits = ref_outputs.logits.to(target_device)

            current_logits = model(
                input_ids=retain_input_ids,
                attention_mask=None,
            ).logits

            retain_loss = F.kl_div(
                input=F.log_softmax(current_logits, dim=-1),
                target=F.softmax(ref_logits, dim=-1),
                reduction="batchmean",
            )
        else:
            retain_loss = torch.tensor(0.0, device=target_device)

        # Forget loss - entropy maximization via frozen model's later layers
        if forget_coeff > 0:
            outputs = model(
                input_ids=forget_input_ids,
                attention_mask=forget_attention_mask,
                output_hidden_states=True,
            )

            # Connect loss to full model output for FSDP backward bookkeeping
            if hasattr(outputs, "logits"):
                dummy_loss = outputs.logits.mean() * 0.0
            elif isinstance(outputs, tuple):
                dummy_loss = outputs[0].mean() * 0.0
            else:
                dummy_loss = outputs.mean() * 0.0

            hidden_at_layer = outputs.hidden_states[target_layer]

            orig_device = next(self.frozen_model.parameters()).device
            orig_dtype = next(self.frozen_model.parameters()).dtype

            logits = forward_from_layer(
                self.frozen_model,
                hidden_at_layer.to(device=orig_device, dtype=orig_dtype),
                target_layer,
            )

            mask = forget_attention_mask.bool().to(logits.device)
            logits_masked = logits[mask]

            if logits_masked.numel() > 0:
                if self.run_args.use_max_entropy_kl:
                    vocab_size = logits.shape[-1]
                    log_vocab = torch.log(
                        torch.tensor(float(vocab_size), device=target_device)
                    )
                    forget_loss = (
                        max_entropy_kl_loss(logits_masked).to(target_device) / log_vocab
                        + dummy_loss
                    )
                else:
                    vocab_size = logits.shape[-1]
                    batch_size, seq_len = logits.shape[:2]
                    random_targets = torch.randint(
                        0, vocab_size, (batch_size, seq_len), device=logits.device
                    )
                    targets_flat = random_targets[mask]
                    ce_loss = F.cross_entropy(
                        logits_masked.float(), targets_flat, reduction="mean"
                    )
                    log_vocab = torch.log(
                        torch.tensor(float(vocab_size), device=target_device)
                    )
                    forget_loss = (ce_loss / log_vocab).to(target_device) + dummy_loss
            else:
                forget_loss = dummy_loss
        else:
            forget_loss = torch.tensor(0.0, device=target_device)

        loss = retain_coeff * retain_loss + forget_coeff * forget_loss

        if (
            self.current_training_step % 8 == 0
            and int(os.environ.get("LOCAL_RANK", 0)) == 0
        ):
            print(
                f"step {self.current_training_step} | "
                f"layer {target_layer} "
                f"(phase {phase_idx + 1}/{len(self.layers_to_unlearn)}) | "
                f"retain_coeff: {retain_coeff:.4f} | "
                f"forget_coeff: {forget_coeff:.4f} | "
                f"retain_loss: {retain_loss.item():.4f} | "
                f"forget_loss: {forget_loss.item():.4f}"
            )

        self.current_training_step += 1
        return (loss,) if return_outputs else loss


@dataclass
class SequentialSftUnlearnConfig:
    num_train_examples: int = 1024
    unlearn_corrupt: bool = False
    corrupt_ratio: float = 0.5
    corrupt_ds: Literal["rewritten", "shuffled"] = "rewritten"
    lr: float = 1e-5
    pdbs: int = 1
    retain_coef: float = 20.0
    remove_coef: float = 5.0
    start_layer: int | None = None
    end_layer: int = 8
    layer_step: int = 4
    model_name: str = "EleutherAI/deep-ignorance-unfiltered"
    save_name: str = ""
    revision: str = "main"
    epochs_per_layer: int = 1
    warmup_ratio: float = 0.0
    use_ultrachat: bool = False
    use_max_entropy_kl: bool = False


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"

    NUM_PROC = (os.cpu_count() or 32) // 2
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    parser = ArgumentParser()
    parser.add_arguments(SequentialSftUnlearnConfig, dest="run_cfg")
    run_cfg = parser.parse_args().run_cfg

    print("Parsed arguments:")
    for arg, value in vars(run_cfg).items():
        print(f"  {arg}: {value}")

    model, tokenizer = get_model_and_tokenizer(
        run_cfg.model_name, revision=run_cfg.revision
    )
    model = cast(PreTrainedModel, model)

    for param in model.parameters():
        param.requires_grad = True
    model.enable_input_require_grads()

    train_dataset = get_unlearning_dataset(run_cfg, tokenizer, NUM_PROC)

    frozen_model = AutoModelForCausalLM.from_pretrained(
        run_cfg.model_name,
        revision=run_cfg.revision,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
    )
    frozen_model.eval()
    for param in frozen_model.parameters():
        param.requires_grad = False
    print("Loaded frozen reference model.")

    num_layers = model.config.num_hidden_layers
    if run_cfg.start_layer is None:
        run_cfg.start_layer = num_layers - 1

    layers_to_unlearn = list(
        range(run_cfg.start_layer, run_cfg.end_layer - 1, -run_cfg.layer_step)
    )
    print(f"Layers to unlearn (back-to-front): {layers_to_unlearn}")

    global_batch_size = 32
    grad_acc_steps = max(1, global_batch_size // (run_cfg.pdbs * world_size))
    steps_per_phase = max(
        1,
        run_cfg.epochs_per_layer * len(train_dataset) // (world_size * run_cfg.pdbs),
    )
    total_epochs = run_cfg.epochs_per_layer * len(layers_to_unlearn)

    print(
        f"Running with {world_size} GPUs. Per device batch: {run_cfg.pdbs}. "
        f"Grad Acc steps: {grad_acc_steps}. "
        f"Steps per phase: {steps_per_phase}. Total epochs: {total_epochs}."
    )

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=run_cfg.lr,
        warmup_ratio=run_cfg.warmup_ratio,
        gradient_accumulation_steps=grad_acc_steps,
        per_device_train_batch_size=run_cfg.pdbs,
        per_device_eval_batch_size=run_cfg.pdbs,
        num_train_epochs=total_epochs,
        weight_decay=0.01,
        gradient_checkpointing=False,
        bf16=True,
        max_grad_norm=1.0,
        save_strategy="no",
    )

    trainer = SequentialSftTrainer(
        run_args=run_cfg,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        layers_to_unlearn=layers_to_unlearn,
        frozen_model=frozen_model,
        steps_per_phase=steps_per_phase,
    )

    model.train()
    trainer.train()

    if run_cfg.save_name:
        if "models/" in run_cfg.model_name:
            run_cfg.model_name = run_cfg.model_name.replace("models/", "")
        save_path = f"./models/{run_cfg.model_name}_{run_cfg.save_name}"

        trainer.accelerator.wait_for_everyone()
        state_dict = trainer.accelerator.get_state_dict(trainer.model, unwrap=False)
        if trainer.accelerator.is_main_process:
            unwrapped = trainer.accelerator.unwrap_model(trainer.model)
            unwrapped.save_pretrained(
                save_path, state_dict=state_dict, safe_serialization=True
            )
            tokenizer.save_pretrained(save_path)
        trainer.accelerator.wait_for_everyone()

    print("Done!")
