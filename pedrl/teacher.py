"""Stage 1 — train the pedagogical teacher with GRPO.

The policy is base model + LoRA, prompted WITH privileged information.
Reward: r_ped = correctness * G_spike, where G_spike is computed by the same
network with the LoRA adapter disabled (i.e. the frozen initial student) under
the student's un-privileged prompt.

The same routine also implements optional stage 3 (plain GRPO on the student,
no privileged info, correctness-only reward) via pedagogical=False.
"""

import os

from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from .config import PedRLConfig, filter_kwargs_for_dataclass
from .data import load_gsm8k
from .modeling import dtype_key, load_model, load_tokenizer, make_lora_config, pick_dtype, set_seed
from .rewards import PedagogicalReward


class AdapterCheckpointCallback(TrainerCallback):
    """Save the LoRA adapter every `every` optimizer steps, so learning curves
    (eval accuracy vs rollouts) can be reconstructed after training."""

    def __init__(self, every: int, out_dir: str):
        self.every = every
        self.out_dir = out_dir

    def on_step_end(self, args, state, control, **kwargs):
        if self.every > 0 and state.global_step % self.every == 0:
            path = os.path.join(self.out_dir, f"step_{state.global_step:04d}")
            kwargs["model"].save_pretrained(path)
            print(f"[checkpoint] saved adapter at step {state.global_step} -> {path}")


def train_teacher(cfg: PedRLConfig, pedagogical: bool = True,
                  init_adapter: str = None, save_dir: str = None,
                  max_steps: int = None, log_name: str = None) -> str:
    import torch

    set_seed(cfg.seed)
    tokenizer = load_tokenizer(cfg.model_name)
    dataset = load_gsm8k(cfg, tokenizer, split="train", n=cfg.n_train,
                         filter_path=cfg.train_filter)
    if not pedagogical:
        # student RL / baseline RL: prompt the model with the student's own context
        dataset = dataset.map(lambda ex: {"prompt": ex["student_prompt"]})

    save_dir = save_dir or cfg.teacher_adapter_dir
    max_steps = max_steps or (cfg.teacher_steps if pedagogical else cfg.student_rl_steps)
    log_name = log_name or ("teacher" if pedagogical else "student_rl")

    reward = PedagogicalReward(
        tokenizer, cfg, pedagogical=pedagogical,
        log_path=os.path.join(cfg.output_dir, f"reward_log_{log_name}.jsonl"),
    )

    dtype = pick_dtype()
    args = GRPOConfig(**filter_kwargs_for_dataclass(GRPOConfig, dict(
        output_dir=os.path.join(cfg.output_dir, "grpo_teacher" if pedagogical else "grpo_student"),
        max_steps=max_steps,
        learning_rate=cfg.teacher_lr,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_generations=cfg.num_generations,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        temperature=cfg.grpo_temperature,
        beta=cfg.grpo_kl_beta,
        logging_steps=cfg.logging_steps,
        save_strategy="no",
        report_to="none",
        seed=cfg.seed,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        model_init_kwargs={dtype_key(): dtype},
        gradient_checkpointing=True,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=min(10, max(1, max_steps // 10)),
    ), label="GRPOConfig"))

    callbacks = [AdapterCheckpointCallback(cfg.checkpoint_every, save_dir + "_checkpoints")]

    if init_adapter is not None:
        # continue training an existing adapter (stage 3)
        if hasattr(args, "model_init_kwargs"):
            args.model_init_kwargs = None  # only valid when model is passed as a string
        model = load_model(cfg.model_name, adapter_dir=init_adapter, trainable_adapter=True)
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward,
            args=args,
            train_dataset=dataset,
            callbacks=callbacks,
        )
    else:
        trainer = GRPOTrainer(
            model=cfg.model_name,
            reward_funcs=reward,
            args=args,
            train_dataset=dataset,
            peft_config=make_lora_config(cfg),
            callbacks=callbacks,
        )

    reward.attach(trainer.accelerator.unwrap_model(trainer.model))
    trainer.train()

    os.makedirs(save_dir, exist_ok=True)
    trainer.accelerator.unwrap_model(trainer.model).save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"[teacher] saved adapter to {save_dir}")
    return save_dir
