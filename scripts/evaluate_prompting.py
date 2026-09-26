import json
import os

from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.drgrpo_grader import question_only_reward_fn,r1_zero_reward_fn
test_data_path="data/gsm8k/test.jsonl"
question_only_log_output_dir="log/only_question/"
zero_shot_log_output_dir="log/zero_shot/"
few_shot_log_output_dir="log/few_shot/"
os.makedirs(question_only_log_output_dir,exist_ok=True)
os.makedirs(zero_shot_log_output_dir,exist_ok=True)
os.makedirs(few_shot_log_output_dir,exist_ok=True)
question_only_log_output_path=os.path.join(question_only_log_output_dir,"evaluate_log.jsonl")
zero_shot_log_output_path=os.path.join(zero_shot_log_output_dir,"evaluate_log.jsonl")
few_shot_log_output_path=os.path.join(few_shot_log_output_dir,"evaluate_log.jsonl")
type_log_output_path=r"log/evaluate_sum.jsonl"


questions=[]
anwsers=[]
q_only_prompts=[]
zero_shot_prompts=[]
few_shot_prompts=[]


with open(test_data_path,"r",encoding="utf-8") as f:
    text=f.read()
    docs=text.split("\n")
for doc in docs:
    if not doc.strip():
        continue
    qanda=json.loads(doc)
    questions.append(qanda["question"])
    anwsers.append(qanda["answer"].split("####")[-1].strip())

q_only_prompt=r"{question} Please put your final answer within \\boxed{{}}."
zero_shot_prompt=r"""A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""
few_shot_prompt=r"""A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Assistant: <think> There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is 6. </think> <answer> 6 </answer>
User: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Assistant: <think> There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is 5. </think> <answer> 5 </answer>
User: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Assistant: <think> Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is 39. </think> <answer> 39 </answer>
User: {question}
Assistant: <think>"""


for question in questions:
    q1=q_only_prompt.format(question=question)
    q2=zero_shot_prompt.format(question=question)
    q3=few_shot_prompt.format(question=question)
    q_only_prompts.append(q1)
    zero_shot_prompts.append(q2)
    few_shot_prompts.append(q3)

vllm=VLLMServer(model_id="allenai/OLMo-2-0425-1B",gpu=0)
vllm.start()
sampling_params={
    "temperature":1.0,
    "max_tokens":512,
    "n":1,
    "seed":42
}

only_q_completions=vllm.generate_completions(prompts=q_only_prompts,sampling_params=sampling_params,batch_size=16)
sampling_params["stop"]=["</answer>"]
sampling_params["include_stop_str_in_output"]=True
zero_shot_completions=vllm.generate_completions(prompts=zero_shot_prompts,sampling_params=sampling_params,batch_size=16)
few_shot_completions=vllm.generate_completions(prompts=few_shot_prompts,sampling_params=sampling_params,batch_size=16)

reward_types={}

with open(question_only_log_output_path,"w",encoding="utf-8") as f:
    reward_type={}
    for vllm_com,anwser,question in zip(only_q_completions,anwsers,questions):
        question_only_result=question_only_reward_fn(vllm_com.text,anwser)
        format_reward=question_only_result["format_reward"]
        answer_reward=question_only_result["answer_reward"]
        res_type="unknown"
        if format_reward==1 and answer_reward==1:
            res_type="type1"
            reward_type["type1"]=reward_type.get("type1",0)+1
        elif format_reward==1 and answer_reward==0:
            res_type="type2"
            reward_type["type2"]=reward_type.get("type2",0)+1
        elif format_reward==0 and answer_reward==0:
            res_type="type3"
            reward_type["type3"]=reward_type.get("type3",0)+1
        log={
            "question":question,
            "response":vllm_com.text,
            "answer":anwser,
            "type":res_type
        }
        f.write(json.dumps(log)+"\n")
    reward_types["only_question"]=reward_type
    print(reward_type)


with open(zero_shot_log_output_path,"w",encoding="utf-8") as f:
    reward_type={}
    for vllm_com,anwser,question in zip(zero_shot_completions,anwsers,questions):
        zero_shot_result=r1_zero_reward_fn(vllm_com.text,anwser)
        format_reward=zero_shot_result["format_reward"]
        answer_reward=zero_shot_result["answer_reward"]
        res_type="unknown"
        if format_reward==1 and answer_reward==1:
            res_type="type1"
            reward_type["type1"]=reward_type.get("type1",0)+1
        elif format_reward==1 and answer_reward==0:
            res_type="type2"
            reward_type["type2"]=reward_type.get("type2",0)+1
        elif format_reward==0 and answer_reward==0:
            res_type="type3"
            reward_type["type3"]=reward_type.get("type3",0)+1
        log={
            "question":question,
            "response":vllm_com.text,
            "answer":anwser,
            "type":res_type
        }
        f.write(json.dumps(log)+"\n")
    reward_types["zero_shot"]=reward_type
    print(reward_type)


with open(few_shot_log_output_path,"w",encoding="utf-8") as f:
    reward_type={}
    for vllm_com,anwser,question in zip(few_shot_completions,anwsers,questions):
        few_shot_result=r1_zero_reward_fn(vllm_com.text,anwser)
        format_reward=few_shot_result["format_reward"]
        answer_reward=few_shot_result["answer_reward"]
        res_type="unknown"
        if format_reward==1 and answer_reward==1:
            res_type="type1"
            reward_type["type1"]=reward_type.get("type1",0)+1
        elif format_reward==1 and answer_reward==0:
            res_type="type2"
            reward_type["type2"]=reward_type.get("type2",0)+1
        elif format_reward==0 and answer_reward==0:
            res_type="type3"
            reward_type["type3"]=reward_type.get("type3",0)+1
        log={
            "question":question,
            "response":vllm_com.text,
            "answer":anwser,
            "type":res_type
        }
        f.write(json.dumps(log)+"\n")
    reward_types["few_shot"]=reward_type
    print(reward_type)

with open(type_log_output_path,"w",encoding="utf-8") as f:
    f.writelines(json.dumps(f"{key}: {value}")+"\n" for key,value in reward_types.items())
