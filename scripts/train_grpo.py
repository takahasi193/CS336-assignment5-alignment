from cs336_alignment.train import grpo_train_step
from cs336_alignment.vllm_utils import VLLMServer
from config import Config
from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import torch
import random
import json
import os


config=Config()

def eval_model(vllm:VLLMServer,clip_eval_prompt:list[str],clip_eval_truth:list[str]):
    total_format_reward=0
    total_answer_reward=0
    sampling_params={
    "temperature":config.sampling_temperature,
    "max_tokens":config.sampling_max_tokens,
    "n":1,
    "seed":config.seed,
    "stop":list(config.stop_strings),
    "include_stop_str_in_output":True
    }

    completions=vllm.generate_completions(prompts=clip_eval_prompt,sampling_params=sampling_params,batch_size=config.rollout_batch_size)
    for completion,truth in zip(completions,clip_eval_truth):
        reward_result=r1_zero_reward_fn(completion.text,truth)
        total_format_reward+=reward_result["format_reward"]
        total_answer_reward+=reward_result["answer_reward"]
    aver_format_reward=total_format_reward/len(clip_eval_prompt)
    aver_answer_reward=total_answer_reward/len(clip_eval_prompt)
    eval_result={}
    eval_result['aver_format_reward']=aver_format_reward
    eval_result['aver_answer_reward']=aver_answer_reward
    return eval_result



def train_grpo():
    train_repeated_prompts=[]
    eval_repeated_prompts=[]
    train_questions=[]
    eval_questions=[]
    train_repeated_ground_truths=[]
    eval_repeated_ground_truths=[]
    torch.manual_seed(config.seed)
    model,tokenizer=get_model_and_tokenizer(model_id_or_dir=config.model_name_or_path,device="cuda:0")
    gsm8k_train_data_path=r"data/gsm8k/train.jsonl"
    gsm8k_eval_data_path=r"data/gsm8k/test.jsonl"

    with open(gsm8k_train_data_path,"r",encoding="utf-8") as f:
        train_text=f.read()
        docs=train_text.split("\n")
    for doc in docs:
        if not doc.strip():
            continue
        qanda=json.loads(doc)
        train_questions.append(qanda["question"])
        for _ in range(config.group_size):
            train_repeated_ground_truths.append(qanda["answer"].split("####")[-1].strip())

    with open(gsm8k_eval_data_path,"r",encoding="utf-8") as f:
        eval_text=f.read()
        docs=eval_text.split("\n")
    for doc in docs:
        if not doc.strip():
            continue
        qanda=json.loads(doc)
        eval_questions.append(qanda["question"])
        eval_repeated_ground_truths.append(qanda["answer"].split("####")[-1].strip())


    with open("cs336_alignment/prompts/r1_zero.prompt", "r", encoding="utf-8") as f:
        prompt_template = f.read()

    for question in train_questions:
        formatted_prompt = prompt_template.format(question=question)
        for _ in range(config.group_size):
            train_repeated_prompts.append(formatted_prompt)

    for question in eval_questions:
        formatted_prompt = prompt_template.format(question=question)
        eval_repeated_prompts.append(formatted_prompt)

    

    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,betas=config.adam_betas,weight_decay=config.weight_decay)
    vllm=VLLMServer(model_id="allenai/OLMo-2-0425-1B",gpu=1)
    vllm.start()
    vllm.init_weight_sync("cuda:0")

    sampling_params={
    "temperature":config.sampling_temperature,
    "max_tokens":config.sampling_max_tokens,
    "n":1,
    "seed":config.seed,
    "stop":list(config.stop_strings),
    "include_stop_str_in_output":True
    }

    save_dir="checkpoints/grpo_standard"

    os.makedirs(save_dir,exist_ok=True)

    model.train()

    for step in range(config.num_rollout_steps):
        vllm.sync_policy_weights(model)
        train_index=random.randrange(0,len(train_repeated_prompts)-config.rollout_batch_size,config.group_size)
        if not step%config.eval_every_n_steps:
            clip_prompts=eval_repeated_prompts[:1024]
            clip_ground_truths=eval_repeated_ground_truths[:1024]
            eval_result=eval_model(vllm,clip_prompts,clip_ground_truths)
            print(f"aver format reward:{eval_result["aver_format_reward"]}")
            print(f"aver answer reward:{eval_result["aver_answer_reward"]}")
        
        clip_prompts=train_repeated_prompts[train_index:train_index+config.rollout_batch_size]
        clip_ground_truths=train_repeated_ground_truths[train_index:train_index+config.rollout_batch_size]
        completions=vllm.generate_completions(prompts=clip_prompts,sampling_params=sampling_params,batch_size=config.rollout_batch_size)    
        rollout_responses=[]
        for completion in completions:
            rollout_responses.append(completion.text)

        loss,metadata=grpo_train_step(
         model=model,
         tokenizer=tokenizer,
         optimizer=optimizer,
         gradient_accumulation_steps=config.gradient_accumulation_steps,
         max_grad_norm=config.max_grad_norm,
         reward_fn=r1_zero_reward_fn,
         repeated_prompts=clip_prompts,
         rollout_responses=rollout_responses,
         repeated_ground_truths=clip_ground_truths,
         group_size=config.group_size,
         baseline=config.baseline,
         advantage_eps=config.advantage_eps,
         advantage_normalizer=config.advantage_normalizer,
         importance_reweighting_method=config.importance_reweighting_method,
         old_log_probs=None,
         cliprange=config.cliprange,
         loss_normalization=config.loss_normalization,
         normalization_constant=config.normalization_constant
        )

        if not step%config.log_rollouts_every_n_steps:
            print(f"step {step}:train loss: {loss}")
            print(metadata)


    vllm.stop()
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":
    train_grpo()


