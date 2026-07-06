"""Stage 1 — train the pedagogical teacher with GRPO.

The policy is base model + LoRA, prompted WITH privileged information.
Reward: r_ped = correctness * G_spike, where G_spike is computed by the same
network with the LoRA adapter disabled (i.e. the frozen initial student) under
the student's un-privileged prompt.

The same routine also implements optional stage 3 (plain GRPO on the student,
no privileged info, correctness-only reward) via pedagogical=False.
"""

import os

from trl import GRPOConfig, GRPOTrainer

from .config import PedRLConfig
from .data import load_gsm8k
from .modeling import load_model, load_tokenizer, make_lora_config, pick_dtype, set_seed
from .rewards import PedagogicalReward


def train_teacher(cfg: PedRLConfig, pedagogical: bool = True,
                  init_adapter: str = None, save_dir: str = None,
                  max_steps: int = None) -> str:
    import torch

    set_seed(cfg.seed)
    tokenizer = load_tokenizer(cfg.model_name)
    dataset = load_gsm8k(cfg, tokenizer, split="train", n=cfg.n_train)
    if not pedagogical:
        # student RL: prompt the model with the student's own context
        dataset = dataset.map(lambda ex: {"prompt": ex["student_prompt"]})

    save_dir = save_dir or cfg.teacher_adapter_dir
    max_steps = max_steps or (cfg.teacher_steps if pedagogical else cfg.student_rl_steps)

    reward = PedagogicalReward(tokenizer, cfg, pedagogical=pedagogical)

    dtype = pick_dtype()
    args = GRPOConfig(
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
        model_init_kwargs={"torch_dtype": dtype},
        gradient_checkpointing=True,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=min(10, max(1, max_steps // 10)),
    )

    if init_adapter is not None:
        # continue training an existing adapter (stage 3)
        args.model_init_kwargs = None  # only valid when model is passed as a string
        model = load_model(cfg.model_name, adapter_dir=init_adapter, trainable_adapter=True)
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward,
            args=args,
            train_dataset=dataset,
        )
    else:
        trainer = GRPOTrainer(
            model=cfg.model_name,
            reward_funcs=reward,
            args=args,
            train_dataset=dataset,
            peft_config=make_lora_config(cfg),
        )

    reward.attach(trainer.accelerator.unwrap_model(trainer.model))
    trainer.train()

    os.makedirs(save_dir, exist_ok=True)
    trainer.accelerator.unwrap_model(trainer.model).save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"[teacher] saved adapter to {save_dir}")
    return save_dir
