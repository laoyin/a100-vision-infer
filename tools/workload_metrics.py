"""Separate length/completeness and post-first-token timing from throughput."""
import json
import math
import statistics


def workload_metrics(results,min_tokens=0):
    rows=[]
    for r in results:
        n=len(r.get('generated_ids',[]))
        valid=False
        try:
            value=json.loads(r.get('text',''),parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
            valid=isinstance(value,(dict,list))
        except (ValueError,TypeError):
            pass
        completed=r.get('finish_reason') in ('eos','stop')
        total=r.get('total_seconds',r.get('latency_seconds'))
        first=r.get('ttft_seconds')
        remaining=None
        if isinstance(total,(float,int)) and isinstance(first,(float,int)) and math.isfinite(total) and math.isfinite(first) and 0<=first<total and n>1:
            remaining=total-first
        rows.append(dict(output_tokens=n,json_valid=valid,natural_stop=completed,
                         complete_json=valid and completed,long_enough=n>=min_tokens,
                         post_first_token_seconds=remaining,
                         post_first_token_tokens_per_second=(n-1)/remaining if remaining else None))
    speeds=[r['post_first_token_tokens_per_second'] for r in rows if r['post_first_token_tokens_per_second'] is not None]
    return dict(requests=rows,min_required_tokens=min_tokens,
                min_output_tokens=min((r['output_tokens'] for r in rows),default=0),
                all_complete_json=bool(rows) and all(r['complete_json'] for r in rows),
                all_long_enough=bool(rows) and all(r['long_enough'] for r in rows),
                post_first_token_tps_p50=statistics.median(speeds) if speeds else None,
                note='Post-first-token rate includes token delivery and completion overhead; speculative token bursts are not per-token GPU timings. JSON syntax is not field accuracy.')
