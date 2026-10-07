"""Pair first-token proxy and complete long-JSON latency for the same profile."""
import argparse
import json
from pathlib import Path


def matched_ratios(summary):
    grouped={}
    passed={p['name'] for p in summary.get('profiles',[]) if p.get('status')=='passed'}
    for row in summary.get('comparison',[]):
        ratio=row.get('latency_ratio_vllm_over_native')
        if row['native'] in passed and ratio is not None:
            grouped.setdefault(row['native'],{})[row['vllm']]=ratio
    return {name:min(r['mtp2'],r['mtp3']) for name,r in grouped.items() if 'mtp2' in r and 'mtp3' in r}


def combine(first,long):
    a,b=matched_ratios(first),matched_ratios(long)
    rows=[dict(profile=name,first_token_proxy_speedup_vs_fastest_vllm=a[name],
               complete_json_speedup_vs_fastest_vllm=b[name],both_faster=a[name]>1 and b[name]>1)
          for name in sorted(a.keys() & b.keys())]
    return dict(profiles=rows,both_paths_faster=any(r['both_faster'] for r in rows),
                note='Ratios require matching tokens against both vLLM MTP2/3. First-token probe is one-token completion latency, not streaming TTFT. JSON syntax does not prove field accuracy.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args()
    summaries=[]
    for name in ('first-token','long-json'):
        path=a.run/f'matrix-{name}'/'summary.json'
        summaries.append(json.loads(path.read_text(encoding='utf-8')) if path.exists() else {})
    report=combine(*summaries)
    report['missing_reports']=[name for name,s in zip(('first-token','long-json'),summaries) if not s]
    (a.run/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
