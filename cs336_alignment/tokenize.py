from transformers import PreTrainedTokenizer
import torch


# 拼接rollout的提示词与输出，进行encoding，得到input_ids和labels，以及response的对应掩码
def tokenize_prompt_and_output(
 prompt_strs: list[str],
 output_strs: list[str],
 tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
    pad_token_id=tokenizer.pad_token_id
    max_len=0
    prompt_ids=[]
    output_ids=[]
    input_ids=[]
    labels=[]
    response_masks=[]
    concat_tensors=[]
    train_batch={}
    for prompt_str,output_str in zip(prompt_strs,output_strs):
        prompt_id=torch.tensor(tokenizer.encode(prompt_str,add_special_tokens=False))
        output_id=torch.tensor(tokenizer.encode(output_str,add_special_tokens=False))
        concat_tensor=torch.cat([prompt_id,output_id],dim=-1)
        prompt_ids.append(prompt_id)
        output_ids.append(output_id)
        concat_tensors.append(concat_tensor)
        max_len=max(max_len,len(concat_tensor)-1)

    for index,concat_tensor in enumerate(concat_tensors):
        concat_tensor_len=len(concat_tensor)-1
        if concat_tensor_len<max_len:
            pad_num=max_len-concat_tensor_len
            pad_tensor=torch.full((pad_num,),pad_token_id,dtype=concat_tensor.dtype)
            concat_tensors[index]=torch.cat([concat_tensor,pad_tensor],dim=-1)
            concat_tensor=concat_tensors[index]
        input_id=concat_tensor[:-1]
        label=concat_tensor[1:]
        response_mask=torch.zeros_like(label)
        response_mask[len(prompt_ids[index])-1:len(prompt_ids[index])-1+len(output_ids[index])]=1
        input_ids.append(input_id)
        labels.append(label)
        response_masks.append(response_mask)

    # shape (rollout_size,seq_len)
    train_batch['input_ids']=torch.stack(input_ids,dim=0)
    train_batch['labels']=torch.stack(labels,dim=0)
    train_batch['response_mask']=torch.stack(response_masks,dim=0)

    return train_batch       


        









