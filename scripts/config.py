from dataclasses import dataclass
from typing import Literal
@dataclass
class Config:
    gradient_accumulation_steps: int = 32
    max_grad_norm: float | None = 1.0
    group_size: int = 8
    baseline: Literal["mean", "none"] = "mean"
    advantage_eps: float = 1e-6
    advantage_normalizer: Literal["std", "none", "mean"] = "std"
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none"
    cliprange: float | None = None
    loss_normalization: Literal["sequence", "constant"] = "sequence"
    normalization_constant: int | None = None
    num_rollout_steps: int = 200
    rollout_batch_size: int = 256
    n_train_examples: int = 6400
    n_val_examples: int = 1024


    learning_rate:float = 1e-5
    adam_betas:tuple[float,float] = (0.9,0.95)
    weight_decay: float = 0.0

    sampling_temperature: float = 1.0
    sampling_top_p: float = 1.0
    sampling_max_tokens: int = 512
    stop_strings: tuple[str,...] = ("</answer>",)


    model_name_or_path: str = "allenai/OLMo-2-0425-1B"
    prompt_path: str = "cs336_alignment/prompts/r1_zero.prompt"
    train_path: str = "data/gsm8k/train.jsonl"
    val_path: str = "data/gsm8k/test.jsonl"
    seed: int = 42

    eval_every_n_steps: int = 10
    log_rollouts_every_n_steps: int = 40
    


