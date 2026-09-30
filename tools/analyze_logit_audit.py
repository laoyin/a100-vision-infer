"""Locate the first native/vLLM divergence in diagnostic MTP token scores."""
import argparse,json
from pathlib import Path

def analyze(native,upstream,events):
    actual=native['results'][0]['generated_ids']
    expected=upstream['results'][0]['generated_ids']
    prompt=native['prompt_token_ids']
    upstream_prompt=upstream['results'][0]['prompt_token_ids']
    first=next((i for i,(a,b) in enumerate(zip(actual,expected)) if a!=b),None)
    if first is None and len(actual)!=len(expected):first=min(len(actual),len(expected))
    latest=max((e['session'] for e in events),default=None)
    matches=[]
    for event in events:
        if event['session']!=latest:continue
        for row in event['rows']:
            index=event['consumed']-len(prompt)+row['row']+1
            if index==first:
                values=dict(zip(row['top_ids'],row['logits']))
                a=actual[first] if first<len(actual) else None
                b=expected[first] if first<len(expected) else None
                matches.append(dict(generated_index=index,selected_id=row.get('selected_id'),
                    native_id=a,vllm_id=b,top_ids=row['top_ids'],logits=row['logits'],
                    native_minus_vllm_logit=values[a]-values[b] if a in values and b in values else None))
    return dict(prompt_equal=prompt==upstream_prompt,first_difference=first,
        native_length=len(actual),vllm_length=len(expected),diagnostic_session=latest,
        matching_scores=matches,
        note='Native scores only; these do not establish the upstream numerical root cause. Audit timing is excluded from speed ranking.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('native-report','native-log','vllm-report','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    events=[]
    for line in a.native_log.read_text(encoding='utf-8',errors='replace').splitlines():
        if '"logit_audit"' not in line:continue
        try:event=json.loads(line[line.index('{'):])
        except (ValueError,json.JSONDecodeError):continue
        if event.get('event')=='logit_audit':events.append(event)
    result=analyze(json.loads(a.native_report.read_text(encoding='utf-8')),
        json.loads(a.vllm_report.read_text(encoding='utf-8')),events)
    a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
