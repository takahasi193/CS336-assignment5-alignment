import json
import os

from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.drgrpo_grader import question_only_reward_fn
test_data_path="data/gsm8k/test.jsonl"
log_output_dir="log/only_question/"
os.makedirs(log_output_dir,exist_ok=True)
log_output_path=os.path.join(log_output_dir,"test_log.jsonl")
questions=[]
anwsers=[]
q_only_prompts=[]
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
for question in questions:
    q=q_only_prompt.format(question=question)
    q_only_prompts.append(q)

vllm=VLLMServer(model_id="allenai/OLMo-2-0425-1B",gpu=0)
vllm.start()
sampling_params={
    "temperature":1.0,
    "max_tokens":512,
    "n":1,
    "seed":42
}
vllm_coms=vllm.generate_completions(prompts=q_only_prompts,sampling_params=sampling_params,batch_size=16)
reward_type={}
with open(log_output_path,"w",encoding="utf-8") as f:
    for vllm_com,anwser,question in zip(vllm_coms,anwsers,questions):
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
print(reward_type)

