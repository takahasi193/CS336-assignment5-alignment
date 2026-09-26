import json
import regex
with open("test_data/test_log.jsonl","r",encoding="utf-8") as f:
    docs=f.readlines()
    for doc in docs:
       dic=json.loads(doc)
       if dic["type"]=="type1":
           print("question: "+dic["question"]+"\n")
           print("response: "+dic["response"]+"\n")