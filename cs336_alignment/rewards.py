import torch
from typing import Literal
from typing import Callable


# 得到单个问答对的奖励
def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    assert len(rollout_responses)==len(repeated_ground_truths)

    raw_rewards=[]
    total_format_reward=0

    for rollout_response,repeated_ground_truth in zip(rollout_responses,repeated_ground_truths):
        reward_data=reward_fn(rollout_response,repeated_ground_truth)
        raw_rewards.append(reward_data['reward'])
        total_format_reward+=reward_data['format_reward']

    mean_total_reward=0
    mean_format_reward=0
    if len(raw_rewards)!=0:
        mean_total_reward=sum(raw_rewards)/len(raw_rewards)
        mean_format_reward=total_format_reward/len(raw_rewards)
    # shape (rollout_size,)
    raw_rewards=torch.tensor(raw_rewards,dtype=torch.float32)

    metadata={}
    metadata['mean_total_reward']=mean_total_reward
    metadata['mean_format_reward']=mean_format_reward

    return raw_rewards,metadata



# 输入单个问答对得到的分数，标准化每一组问答的奖励(得到Advantage score)
def compute_group_normalized_rewards(
 raw_rewards: torch.Tensor,
 group_size: int,
 baseline: Literal["mean", "none"] = "mean",
 advantage_eps: float = 1e-6,
 advantage_normalizer: Literal["std", "none", "mean"] = "std",
):  
    assert len(raw_rewards)%group_size==0
    batch_size=len(raw_rewards)//group_size
    raw_rewards=raw_rewards.reshape(batch_size,group_size)
    if baseline=="mean":
        raw_rewards_baseline=torch.mean(raw_rewards,dim=-1,keepdim=True)

    else:
        raw_rewards_baseline=torch.zeros((batch_size,1),dtype=raw_rewards.dtype)

    if advantage_normalizer=="std":
        raw_rewards_std=torch.std(raw_rewards,dim=-1,keepdim=True)

    elif advantage_normalizer=="none":
        raw_rewards_std=torch.ones((batch_size,1),dtype=raw_rewards.dtype)
        advantage_eps=0

    else:
        raw_rewards_std=torch.mean(raw_rewards,dim=-1,keepdim=True)
    
    group_normalized_rewards=(raw_rewards-raw_rewards_baseline)/(raw_rewards_std+advantage_eps)
    # shape (rollout_size,)
    group_normalized_rewards=group_normalized_rewards.reshape(batch_size*group_size)

    metadata={}
    metadata['baseline']=raw_rewards_baseline.mean()
    metadata['std']=raw_rewards_std.mean()
    return group_normalized_rewards,metadata




