import torch
from typing import Literal

def compute_policy_gradient_loss(
 raw_rewards_or_advantages: torch.Tensor,
 policy_log_probs: torch.Tensor,
 importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
 old_log_probs: torch.Tensor | None = None,
 cliprange: float | None = None,
 response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # 'raw_rewards_or_advantages' shape (micro_size,)
    # 'policy_log_probs' shape (micro_size,seq_len)
    if raw_rewards_or_advantages.dim()==1:
        raw_rewards_or_advantages=raw_rewards_or_advantages.unsqueeze(-1)
    loss=None
    metadata={}
    if importance_reweighting_method=="gspo":
        assert old_log_probs is not None
        assert cliprange is not None
        assert response_mask is not None
        # shape (micro_size,1)
        seq_len=response_mask.sum(dim=-1,keepdim=True)
        delta_log_probs=(policy_log_probs-old_log_probs)*response_mask
        s=torch.exp(delta_log_probs.sum(dim=-1,keepdim=True)/seq_len.clamp(min=0))
        clipped_s=torch.min(s*raw_rewards_or_advantages,raw_rewards_or_advantages*torch.clamp(s,1-cliprange,1+cliprange))
        loss=-1.0*clipped_s
        loss=loss.expand_as(policy_log_probs)
    elif importance_reweighting_method=="noclip":
        assert old_log_probs is not None
        importance_weight=torch.exp(policy_log_probs-old_log_probs)
        loss=-1.0*raw_rewards_or_advantages*importance_weight

    elif importance_reweighting_method=="grpo":
        assert old_log_probs is not None
        assert cliprange is not None
        importance_weight=torch.exp(policy_log_probs-old_log_probs)
        clipped_importance_weight=torch.min(importance_weight*raw_rewards_or_advantages,raw_rewards_or_advantages*torch.clamp(importance_weight,1-cliprange,1+cliprange))
        loss=-1.0*clipped_importance_weight
    else:
        # shape (micro_size,seq)
        loss=-1.0*raw_rewards_or_advantages*policy_log_probs
    return loss,metadata


def aggregate_loss_across_microbatch(
 per_token_policy_gradient_loss: torch.Tensor,
 mask: torch.Tensor,
 loss_normalization: Literal["sequence", "constant"] = "sequence",
 normalization_constant: int | None = None,
) -> torch.Tensor:
    # shape (micro_batch_size,seq_len)
    per_token_policy_gradient_res_loss=per_token_policy_gradient_loss*mask
    if loss_normalization=="sequence":
        # shape (micro_batch_size,1)
        response_len=mask.sum(dim=-1,keepdim=True).clamp(min=1)  
        aggregate_loss=per_token_policy_gradient_res_loss.sum(dim=-1,keepdim=True)/response_len
        aggregate_loss=aggregate_loss.squeeze()
        aggregate_loss=aggregate_loss.mean(dim=0)
    else:
        assert normalization_constant is not None and normalization_constant!=0
        aggregate_loss=per_token_policy_gradient_res_loss.sum()
        aggregate_loss/=normalization_constant

    return aggregate_loss 