"""HTTP contract test with fake tokenizer/native transport; no GPU/model validation."""
import json,queue,sys,tempfile,time,types,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from fastapi.testclient import TestClient
import serve

class FakeWorker:
 def __init__(self,*args,**kwargs):
  self.events=queue.Queue();self.events.put(json.dumps({'event':'ready'}));self.stdout=self;self.stdin=self;self.returncode=None
 def __iter__(self):return self
 def __next__(self):
  event=self.events.get(timeout=5)
  if event is None:raise StopIteration
  return event
 def write(self,line):
  command=json.loads(line)
  if command['op']=='shutdown':self.returncode=0;self.events.put(None)
  if command['op']=='submit':
   for event in [{'event':'token','token':1},{'event':'done','finish_reason':'eos','generated_ids':[1],'input_tokens':3}]:
    self.events.put(json.dumps({**event,'id':command['id']}))
 def flush(self):pass
 def close(self):pass
 def poll(self):return self.returncode
 def wait(self,*args):return self.returncode

class HttpTests(unittest.TestCase):
 def test_completion_stream_validation_and_cleanup(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);model=root/'model';model.mkdir();(model/'config.json').write_text('{}');(model/'manifest.json').write_text('{"config":{}}')
   args=types.SimpleNamespace(spool=str(root/'spool'),hf_model=str(model),model=str(model),tp=2,max_queue=4,worker='unused',max_context=64,max_concurrency=2,prefill_chunk=4,image_cache_mib=0,prefix_cache_mib=0,host_prefix_cache_mib=0,workspace_mib=8,cuda_graph=False,baseline=False,api_key='secret',model_name='test',max_body_bytes=10000,max_image_bytes=1000,max_pixels=1000)
   processor=types.SimpleNamespace(tokenizer=types.SimpleNamespace(decode=lambda *args,**kwargs:'ok'))
   module=types.ModuleType('transformers');module.AutoProcessor=types.SimpleNamespace(from_pretrained=lambda *args,**kwargs:processor)
   def build(*args,**kwargs):Path(args[5]).mkdir()
   with patch.dict(sys.modules,{'transformers':module}),patch.object(serve.subprocess,'Popen',FakeWorker),patch.object(serve,'build_request',build),patch.object(serve,'inspect_artifact',lambda *args:None):
    with TestClient(serve.create_app(args)) as client:
     for _ in range(100):
      if client.get('/health').json()['ready']:break
      time.sleep(.01)
     headers={'Authorization':'Bearer secret'};body={'model':'test','messages':[{'role':'user','content':'hello'}]}
     self.assertEqual(client.post('/v1/chat/completions',json=body).status_code,401)
     self.assertEqual(client.post('/v1/chat/completions',headers=headers,json=[]).status_code,400)
     self.assertEqual(client.post('/v1/chat/completions',headers=headers,json={**body,'temperature':'bad'}).status_code,400)
     response=client.post('/v1/chat/completions',headers=headers,json=body)
     self.assertEqual(response.status_code,200,response.text)
     self.assertEqual(response.json()['choices'][0]['message']['content'],'ok')
     response=client.post('/v1/chat/completions',headers=headers,json={**body,'stream':True})
     self.assertIn('data: [DONE]',response.text)
     self.assertIn('"content": "ok"',response.text)
     self.assertEqual(client.get('/metrics',headers=headers).json()['inflight'],0)
     self.assertEqual(list((root/'spool').iterdir()),[])
