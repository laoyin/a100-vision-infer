"""Create a JSON request with embedded local images for HTTP testing."""
import argparse,base64,json,mimetypes
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',action='append',type=Path,required=True)
    p.add_argument('--prompt',required=True)
    p.add_argument('--model',default='a100-vision')
    p.add_argument('--max-tokens',type=int,default=256)
    p.add_argument('--json-object',action='store_true')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();content=[]
    for path in a.image:
        mime=mimetypes.guess_type(path.name)[0] or 'image/png'
        if not mime.startswith('image/'):raise ValueError('Expected an image')
        content.append({'type':'image_url','image_url':{'url':'data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode()}})
    content.append({'type':'text','text':a.prompt})
    body={'model':a.model,'messages':[{'role':'user','content':content}],'max_tokens':a.max_tokens,'temperature':0,'stream':True}
    if a.json_object:body['response_format']={'type':'json_object'}
    with a.out.open('x',encoding='utf-8') as f:json.dump(body,f,ensure_ascii=False)

if __name__=='__main__':main()
