"""End-to-end HTTP benchmark. Counts errors and measures client TTFT/latency, not just decode speed."""
import argparse,concurrent.futures,json,time,urllib.request,statistics

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--url',default='http://127.0.0.1:8000/v1/chat/completions')
    p.add_argument('--body',required=True);p.add_argument('--requests',type=int,default=20);p.add_argument('--concurrency',type=int,default=2)
    p.add_argument('--api-key');p.add_argument('--out',required=True);p.add_argument('--timeout',type=float,default=600);a=p.parse_args()
    with open(a.body,encoding='utf-8') as f:body=json.load(f)
    body['stream']=True;payload=json.dumps(body).encode();headers={'Content-Type':'application/json'}
    if a.api_key:headers['Authorization']='Bearer '+a.api_key
    def run(i):
        start=time.perf_counter();first=None;usage={};content='';reason=None
        try:
            with urllib.request.urlopen(urllib.request.Request(a.url,data=payload,headers=headers),timeout=a.timeout) as response:
                for line in response:
                    if not line.startswith(b'data: '):continue
                    data=line[6:].strip()
                    if data==b'[DONE]':break
                    event=json.loads(data)
                    if 'error' in event:raise RuntimeError(event['error'])
                    delta=event.get('choices',[{}])[0].get('delta',{}).get('content','')
                    if delta:
                        if first is None:first=time.perf_counter()-start
                        content+=delta
                    usage=event.get('usage',usage);reason=event.get('choices',[{}])[0].get('finish_reason') or reason
            json_valid=None
            if body.get('response_format',{}).get('type')=='json_object':
                try:json_valid=isinstance(json.loads(content),dict)
                except ValueError:json_valid=False
            return {'index':i,'ok':reason in ('stop','length'),'ttft':first,'latency':time.perf_counter()-start,
                    'tokens':usage.get('completion_tokens',0),'finish_reason':reason,'json_valid':json_valid}
        except Exception as exc:return {'index':i,'ok':False,'latency':time.perf_counter()-start,'error':str(exc)}
    start=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:results=list(pool.map(run,range(a.requests)))
    duration=time.perf_counter()-start;success=[r for r in results if r['ok']]
    def stats(values):
        values=sorted(values)
        if not values:return None
        return {'p50':statistics.median(values),'p95':values[min(len(values)-1,int((len(values)-1)*0.95+0.5))]}
    report={'concurrency':a.concurrency,'requests':a.requests,'successful':len(success),'wall_seconds':duration,
            'successful_requests_per_second':len(success)/duration,'output_tokens_per_second':sum(r.get('tokens',0) for r in success)/duration,
            'ttft_seconds':stats([r['ttft'] for r in success if r['ttft'] is not None]),'latency_seconds':stats([r['latency'] for r in success]),'results':results}
    with open(a.out,'x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))

if __name__=='__main__':main()