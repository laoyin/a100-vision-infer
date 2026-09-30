"""HTTP/tokenizer frontend to the native MPI worker. No Python model forward/generate."""
import argparse
import asyncio
import base64
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
import uvicorn
from request_builder import build_request
from validation import validate_body
from preflight import inspect as inspect_artifact


def create_app(args):
    jobs={};state={'ready':False,'fatal':None,'completed':0,'submitted':0};process=None;processor=None;config=None;generation=None
    write_lock=threading.Lock();preprocess_lock=threading.Lock();loop=None;admission=None
    spool=Path(args.spool).resolve();spool.mkdir(parents=True,exist_ok=True)
    def send(command):
        if process is None or process.poll() is not None:raise RuntimeError('Native worker unavailable')
        with write_lock:process.stdin.write(json.dumps(command)+'\n');process.stdin.flush()
    def cleanup(job):
        path=job['path'].resolve()
        if path.parent!=spool:raise RuntimeError('Unexpected spool path')
        if path.exists():shutil.rmtree(path)
    def deliver(event):
        if event.get('event')=='ready':state['ready']=True;state['resources']={k:v for k,v in event.items() if k!='event'};return
        if event.get('event')=='fatal':
            state['fatal']=event.get('message','Native worker failed');state['ready']=False
            for job_id,job in list(jobs.items()):
                job['queue'].put_nowait(event);job['finished']=True;cleanup(job)
                if job.get('detached'):jobs.pop(job_id,None);admission.release()
            return
        job=jobs.get(event.get('id'))
        if job:
            job['queue'].put_nowait(event)
            if event.get('event') in ('done','error'):
                state['completed']+=1;job['finished']=True
                # Keep job until HTTP coroutine consumes final result; the request directory can now be removed.
                cleanup(job)
                if job.get('detached'):
                    jobs.pop(event['id'],None);admission.release()
    def reader():
        try:
            for line in process.stdout:
                try:event=json.loads(line)
                except json.JSONDecodeError:continue
                loop.call_soon_threadsafe(deliver,event)
        finally:loop.call_soon_threadsafe(deliver,{'event':'fatal','message':'Native worker exited; see stderr'})
    @asynccontextmanager
    async def lifespan(app):
        nonlocal process,processor,config,generation,loop,admission
        from transformers import AutoProcessor
        loop=asyncio.get_running_loop();admission=asyncio.Semaphore(args.max_queue)
        await asyncio.to_thread(inspect_artifact,Path(args.model),args.tp)
        processor=AutoProcessor.from_pretrained(args.hf_model,local_files_only=True,trust_remote_code=False)
        config=json.loads((Path(args.hf_model)/'config.json').read_text())
        manifest=json.loads((Path(args.model)/'manifest.json').read_text())
        if any(config.get(k)!=manifest['config'].get(k) for k in ('text_config','vision_config','image_token_id','video_token_id')):
            raise ValueError('HF processor model configuration differs from native artifact')
        gp=Path(args.hf_model)/'generation_config.json';generation=json.loads(gp.read_text()) if gp.exists() else {}
        command=['mpirun','-np',str(args.tp),str(Path(args.worker).resolve()),'--model',str(Path(args.model).resolve()),
                 '--max-context',str(args.max_context),'--max-concurrency',str(args.max_concurrency),'--prefill-chunk',str(args.prefill_chunk),
                 '--workspace-mib',str(args.workspace_mib),'--host-prefix-cache-mib',str(args.host_prefix_cache_mib),
                 '--image-cache-mib',str(args.image_cache_mib),'--prefix-cache-mib',str(args.prefix_cache_mib)]
        if args.cuda_graph:command.append('--cuda-graph')
        if args.baseline:command.append('--baseline')
        for name in ('extra_fusions','cublas_prefill','tp_lm_head','vector_gemv','mtp_draft_graph','gdn_chunk','flash_prefill','cache_vision_weights','gdn_cooperative','bf16_tp_reduce','fused_gdn_conv'):
            if getattr(args,name,False):command.append('--'+name.replace('_','-'))
        command+=['--mtp-tokens',str(getattr(args,'mtp_tokens',0)),'--weight-cache-mib',str(getattr(args,'weight_cache_mib',0))]
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1,start_new_session=True)
        threading.Thread(target=reader,daemon=True).start()
        try:yield
        finally:
            if process.poll() is None:
                try:send({'op':'shutdown'});process.stdin.close();await asyncio.to_thread(process.wait,30)
                except (Exception,asyncio.CancelledError):
                    import signal
                    os.killpg(process.pid,signal.SIGTERM)
                    try:await asyncio.to_thread(process.wait,10)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);await asyncio.to_thread(process.wait)
            for job in list(jobs.values()):cleanup(job)
    app=FastAPI(title='A100 Vision Infer',lifespan=lifespan)
    @app.middleware('http')
    async def authentication(request,call_next):
        if args.api_key and request.url.path!='/health' and request.headers.get('authorization')!='Bearer '+args.api_key:
            from fastapi.responses import JSONResponse
            return JSONResponse({'error':'Invalid API key'},status_code=401)
        return await call_next(request)
    @app.get('/health')
    async def health():return state
    @app.get('/metrics')
    async def metrics():return {**state,'inflight':len(jobs)}
    @app.get('/v1/models')
    async def models():return {'object':'list','data':[{'id':args.model_name,'object':'model','owned_by':'local'}]}
    @app.post('/v1/cancel/{job_id}')
    async def cancel(job_id:str):
        if job_id not in jobs:raise HTTPException(404,'Unknown request')
        send({'op':'cancel','id':job_id});return {'id':job_id,'cancel_requested':True}
    def prepare(body,path):
        from PIL import Image,ImageOps
        images=[];messages=[]
        for message in body.get('messages',[]):
            role=message.get('role');content=message.get('content')
            if role not in ('system','user','assistant'):raise ValueError('Supported roles: system/user/assistant')
            if isinstance(content,str):messages.append({'role':role,'content':content});continue
            if not isinstance(content,list):raise ValueError('Invalid message content')
            parts=[]
            for part in content:
                if part.get('type')=='text':parts.append({'type':'text','text':str(part['text'])});continue
                if part.get('type')!='image_url':raise ValueError('Only text and image_url content supported')
                url=part['image_url'];url=url['url'] if isinstance(url,dict) else url
                if not isinstance(url,str) or not url.startswith('data:image/') or ';base64,' not in url:
                    raise ValueError('Use a base64 data:image URL; remote URLs and arbitrary server paths are not fetched')
                binary=base64.b64decode(url.split(';base64,',1)[1],validate=True)
                if len(binary)>args.max_image_bytes:raise ValueError('Image byte limit exceeded')
                with Image.open(io.BytesIO(binary)) as image:images.append(ImageOps.exif_transpose(image).convert('RGB'))
                parts.append({'type':'image'})
            messages.append({'role':role,'content':parts})
        if not messages:raise ValueError('messages is required')
        with preprocess_lock:
            return build_request(processor,config,generation,messages,images,path,max_pixels=args.max_pixels,
                   max_context=args.max_context,max_new_tokens=int(body.get('max_tokens',256)),thinking=bool(body.get('enable_thinking',False)),
                   content_format=getattr(args,'frontend_format','hf'),bf16_patches=getattr(args,'bf16_patches',False))
    async def events(job_id,body):
        job=jobs[job_id];tokens=[];emitted='';done=None;stop=body.get('stop',[]);stop=[stop] if isinstance(stop,str) else stop
        try:
            while True:
                event=await job['queue'].get();kind=event.get('event')
                if kind in ('fatal','error'):raise RuntimeError(event.get('message','Native error'))
                if kind=='token':
                    tokens.append(event['token']);text=processor.tokenizer.decode(tokens,skip_special_tokens=True).rstrip('\ufffd')
                    boundary=min((text.find(s) for s in stop if s and s in text),default=-1)
                    if boundary>=0:send({'op':'cancel','id':job_id});text=text[:boundary];job['stop_text']=text
                    if 'stop_text' in job:text=job['stop_text']
                    # Withhold suffixes that could become a multi-token stop string.
                    hold=max((k for s in stop for k in range(1,len(s)) if text.endswith(s[:k])),default=0) if 'stop_text' not in job else 0
                    stable=text[:-hold] if hold else text
                    if stable.startswith(emitted):
                        delta=stable[len(emitted):];emitted=stable
                        if delta:yield {'delta':delta}
                elif kind=='done':
                    done=event;full=job.get('stop_text',processor.tokenizer.decode(event['generated_ids'],skip_special_tokens=True))
                    if full.startswith(emitted) and len(full)>len(emitted):yield {'delta':full[len(emitted):]}
                    reason='stop' if event['finish_reason']=='eos' or 'stop_text' in job else event['finish_reason']
                    yield {'done':True,'text':full,'reason':reason,'metrics':event,'usage':{'prompt_tokens':event['input_tokens'],
                          'completion_tokens':len(event['generated_ids']),'total_tokens':event['input_tokens']+len(event['generated_ids'])}}
                    break
        finally:
            if not job['finished']:
                job['detached']=True
                try:send({'op':'cancel','id':job_id})
                except RuntimeError:cleanup(job);jobs.pop(job_id,None);admission.release()
            else:jobs.pop(job_id,None);admission.release()
    @app.post('/v1/chat/completions')
    async def chat(request:Request):
        if args.api_key and request.headers.get('authorization')!='Bearer '+args.api_key:raise HTTPException(401,'Invalid API key')
        if not state['ready']:raise HTTPException(503,state['fatal'] or 'Model loading')
        raw=bytearray()
        async for part in request.stream():
            raw.extend(part)
            if len(raw)>args.max_body_bytes:raise HTTPException(413,'Request body limit exceeded')
        try:body=json.loads(raw)
        except ValueError:raise HTTPException(400,'Invalid JSON')
        try:validate_body(body,args.max_context)
        except ValueError as exc:raise HTTPException(400,str(exc))
        if body.get('model',args.model_name)!=args.model_name:raise HTTPException(404,'Unknown model')
        response_format=body.get('response_format') or {'type':'text'}
        if not isinstance(response_format,dict) or response_format.get('type') not in ('text','json_object') or body.get('n',1)!=1 or body.get('tools'):
            raise HTTPException(400,'Only n=1 and text/json_object response formats are supported; no tool execution')
        if response_format['type']=='json_object' and (body.get('enable_thinking',False) or body.get('stop')):raise HTTPException(400,'Disable thinking and text stop strings for JSON grammar mode')
        if not isinstance(body.get('stop',[]),(str,list)) or isinstance(body.get('stop'),list) and not all(isinstance(x,str) for x in body['stop']):raise HTTPException(400,'Invalid stop strings')
        try:await asyncio.wait_for(admission.acquire(),timeout=0.05)
        except asyncio.TimeoutError:raise HTTPException(429,'Queue full')
        job_id='chatcmpl-'+uuid.uuid4().hex;path=spool/job_id
        try:
            await asyncio.to_thread(prepare,body,path)
            job={'queue':asyncio.Queue(),'path':path,'finished':False};jobs[job_id]=job
            sampling={k:body[k] for k in ['temperature','top_p','top_k','seed','repetition_penalty','timeout_seconds'] if k in body}
            sampling['json_object']=response_format['type']=='json_object'
            send({'op':'submit','id':job_id,'request':str(path),'sampling':sampling});state['submitted']+=1
        except Exception as exc:
            jobs.pop(job_id,None)
            if path.exists():cleanup({'path':path})
            admission.release();raise HTTPException(400,str(exc))
        if body.get('stream',False):
            async def stream():
                try:
                    async for event in events(job_id,body):
                        payload={'id':job_id,'object':'chat.completion.chunk','created':int(time.time()),'model':args.model_name,
                            'choices':[{'index':0,'delta':{'content':event['delta']} if 'delta' in event else {},'finish_reason':event.get('reason')}]}
                        if event.get('done'):payload['usage']=event['usage']
                        yield 'data: '+json.dumps(payload,ensure_ascii=False)+'\n\n'
                    yield 'data: [DONE]\n\n'
                except Exception as exc:yield 'data: '+json.dumps({'error':str(exc)})+'\n\n'
            return StreamingResponse(stream(),media_type='text/event-stream',headers={'X-Request-ID':job_id})
        try:
            async for event in events(job_id,body):
                if event.get('done'):result=event
        except Exception as exc:raise HTTPException(500,str(exc))
        return {'id':job_id,'object':'chat.completion','created':int(time.time()),'model':args.model_name,
                'choices':[{'index':0,'message':{'role':'assistant','content':result['text']},'finish_reason':result['reason']}],
                'usage':result['usage'],'avi_metrics':result['metrics']}
    return app


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--hf-model',required=True);p.add_argument('--model-name',default='a100-vision')
    p.add_argument('--worker',default='build/avi-worker');p.add_argument('--tp',type=int,choices=[1,2,4],default=2)
    p.add_argument('--max-context',type=int,default=20480);p.add_argument('--max-concurrency',type=int,default=2)
    p.add_argument('--prefill-chunk',type=int,default=128);p.add_argument('--max-pixels',type=int,default=4000000)
    p.add_argument('--frontend-format',choices=['hf','vllm-string'],default='hf')
    p.add_argument('--bf16-patches',action='store_true')
    p.add_argument('--image-cache-mib',type=int,default=256);p.add_argument('--prefix-cache-mib',type=int,default=512)
    p.add_argument('--workspace-mib',type=int,default=8192);p.add_argument('--host-prefix-cache-mib',type=int,default=0)
    p.add_argument('--max-queue',type=int,default=64);p.add_argument('--max-image-bytes',type=int,default=20000000)
    p.add_argument('--max-body-bytes',type=int,default=50000000);p.add_argument('--spool',default='./requests')
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8000)
    p.add_argument('--api-key',default=os.environ.get('AVI_API_KEY'));p.add_argument('--cuda-graph',action='store_true');p.add_argument('--baseline',action='store_true')
    p.add_argument('--mtp-tokens',type=int,choices=range(6),default=0)
    p.add_argument('--weight-cache-mib',type=int,default=0)
    for name in ('extra-fusions','cublas-prefill','tp-lm-head','vector-gemv','mtp-draft-graph','gdn-chunk','flash-prefill','cache-vision-weights','gdn-cooperative','bf16-tp-reduce','fused-gdn-conv'):
        p.add_argument('--'+name,action='store_true')
    args=p.parse_args();uvicorn.run(create_app(args),host=args.host,port=args.port)

if __name__=='__main__':main()
