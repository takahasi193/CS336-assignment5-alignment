import torch
from transformers import PreTrainedTokenizerBase
from typing import Callable,Literal

from cs336_alignment.tokenize import tokenize_prompt_and_output
from cs336_alignment.log_probs import get_response_log_probs
from cs336_alignment.rewards import compute_rollout_rewards,compute_group_normalized_rewards
from cs336_alignment.loss import compute_policy_gradient_loss,aggregate_loss_across_microbatch
from torch.nn.utils import clip_grad_norm_

def grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
):  

    device=next(model.parameters()).device

    # shape (rollout_size,)
    raw_rewards,raw_rewards_metadata=compute_rollout_rewards(reward_fn,rollout_responses,repeated_ground_truths)
    # shape (rollout_size,)
    advantages,advantages_metadata=compute_group_normalized_rewards(raw_rewards,group_size,baseline,advantage_eps,advantage_normalizer)
    non_zero_mask=(advantages!=0)
    train_batch=tokenize_prompt_and_output(repeated_prompts,rollout_responses,tokenizer)
    # shape (rollout_size,seq_len)
    input_ids=train_batch['input_ids']
    # shape (rollout_size,seq_len)
    labels=train_batch['labels']
    # shape (rollout_size,seq_len)
    mask=train_batch['response_mask']
    rollout_size=len(input_ids)
    micro_size=rollout_size//gradient_accumulation_steps
    active_input_ids = input_ids[non_zero_mask]
    active_labels = labels[non_zero_mask]
    active_mask = mask[non_zero_mask]
    active_advantages = advantages[non_zero_mask]


    # 列表中 每个元素 shape (micro_size,seq_len)
    splited_input_ids=list(active_input_ids.split(micro_size))
    splited_advantages=list(active_advantages.split(micro_size))
    splited_labels=list(active_labels.split(micro_size))
    splited_mask=list(active_mask.split(micro_size))
    if old_log_probs is not None:
        active_old_log_probs = old_log_probs[non_zero_mask]
        splited_old_log_probs=list(active_old_log_probs.split(micro_size))
    else:
        splited_old_log_probs=None
    total_loss = torch.tensor(0.0, device=device)
    metadata=raw_rewards_metadata|advantages_metadata
    if len(active_input_ids)==0:
        return total_loss,metadata

    optimizer.zero_grad()
    for i in range(len(splited_input_ids)):
        cur_input_ids = splited_input_ids[i].to(device)
        cur_labels = splited_labels[i].to(device)
        cur_mask = splited_mask[i].to(device)
        cur_advantages = splited_advantages[i].to(device)
        cur_old_log_probs = splited_old_log_probs[i].to(device) if splited_old_log_probs is not None else None
        probs_result=get_response_log_probs(model,cur_input_ids,cur_labels)
        log_probs=probs_result['log_probs']
        micro_per_token_loss,gradient_loss_metadata=compute_policy_gradient_loss(cur_advantages,log_probs,importance_reweighting_method,cur_old_log_probs if splited_old_log_probs is not None else None,cliprange,cur_mask)
        loss=aggregate_loss_across_microbatch(micro_per_token_loss,cur_mask,loss_normalization,normalization_constant)
        if loss_normalization=="sequence":
            scaled_loss=loss*(len(splited_input_ids[i])/rollout_size)
        else:
            scaled_loss=loss
        total_loss+=scaled_loss.detach()
        scaled_loss.backward()
    norm=None
    if max_grad_norm is not None:
        norm=clip_grad_norm_(model.parameters(),max_norm=max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()
    metadata=gradient_loss_metadata|raw_rewards_metadata|advantages_metadata
    if norm is not None:
        metadata['norm']=norm
    return total_loss,metadata