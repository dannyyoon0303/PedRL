"""Stage 1 — GRPO on the privileged debugging teacher (and baselines).

Identical training loop to pedrl/teacher.py; only the dataset (verified
DebugBench problems) and the reward (executable tests x G_spike) differ.
pedagogical=False gives vanilla GRPO on the blind student prompt — used for
both baseline-rl and the optional student-rl stage 3.
"""

import os

from trl import GRPOConfig, GRPOTrainer

from pedrl.config import filter_kwargs_for_dataclass
from pedrl.modeling import dtype_key, load_model, load_tokenizer, make_lora_config, pick_dtype, set_seed
from pedrl.teacher import AdapterCheckpointCallback

from .config import DebugPedRLConfig
from .data import load_debugbench
from .rewards import DebugPedagogicalReward


def train_teacher(cfg: DebugPedRLConfig, pedagogical: bool = True,
                  init_adapter: str = None, save_dir: str = None,
                  max_steps: int = None, log_name: str = None) -> str:
    import torch

    set_seed(cfg.seed)
    tokenizer = load_tokenizer(cfg.model_name)
    dataset = load_debugbench(cfg, tokenizer, split="train", n=cfg.n_train,
                              filter_path=cfg.train_filter)
    if not pedagogical:
        # baseline / student RL: prompt the model with the student's own context
        dataset = dataset.map(lambda ex: {"prompt": ex["student_prompt"]})

    save_dir = save_dir or cfg.teacher_adapter_dir
    max_steps = max_steps or (cfg.teacher_steps if pedagogical else cfg.student_rl_steps)
    log_name = log_name or ("teacher" if pedagogical else "student_rl")

    reward = DebugPedagogicalReward(
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
