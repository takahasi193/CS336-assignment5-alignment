from transformers import PreTrainedModel
import torch
import torch.nn.functional as F


# 输入拼接好的rollout材料，得到模型的对数probility

def get_response_log_probs(
 model: PreTrainedModel,
 input_ids: torch.Tensor,
 labels: torch.Tensor,
 return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    
    model_result=model(input_ids)
    # shape (rollout_size,seq_len,vocab_size)    
    logits=model_result.logits
    log_logits=torch.log_softmax(logits,dim=-1)
    logits=F.softmax(logits,dim=-1)
    # 原shape (rollout_size,seq_len)
    # shape (rollout_size,seq_len,1)
    labels=labels.unsqueeze(-1)
    # shape (rollout_size,seq_len)
    log_probs=torch.gather(input=log_logits,dim=-1,index=labels).squeeze()
    token_entropy=None
    if return_token_entropy:
        token_entropy=-1.0*torch.sum(logits*log_logits,dim=-1)
    result={}
    result['log_probs']=log_probs
    if token_entropy is not None:
        result['token_entropy']=token_entropy
    return result




