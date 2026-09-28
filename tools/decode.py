import argparse
import json
from transformers import AutoTokenizer
p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--result', required=True)
a = p.parse_args()
with open(a.result, encoding='utf-8') as f:
    result = json.load(f)
tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True, trust_remote_code=False)
print(tokenizer.decode(result['generated_ids'], skip_special_tokens=True))