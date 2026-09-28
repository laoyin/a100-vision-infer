"""Summarize measured resident-worker runs without claiming output equivalence."""
import argparse,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
 base=json.loads((a.directory/'baseline-c1.json').read_text());reference=base['results'][0]['generated_ids']
 rows=[]
 for name in ['baseline-c1','optimized-c1','graph-c1','optimized-c2']:
  report=json.loads((a.directory/(name+'.json')).read_text())
  rows.append({'run':name,'output_tokens_per_second':report['output_tokens_per_second'],'ttft_p50_seconds':report['ttft_seconds']['p50'] if report['ttft_seconds'] else None,
               'latency_p50_seconds':report['latency_seconds']['p50'] if report['latency_seconds'] else None,
               'successful':report['successful'],'tokens_equal_to_baseline':all(r['generated_ids']==reference for r in report['results'])})
 with (a.directory/'summary.json').open('x') as f:json.dump(rows,f,indent=2)
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
