"""Resident CPU preprocessing for end-to-end native/vLLM comparison."""
import base64
import io
import json
from pathlib import Path

class Frontend:
    def __init__(self,model,body):
        from transformers import AutoProcessor
        self.model=Path(model)
        self.processor=AutoProcessor.from_pretrained(model,local_files_only=True,trust_remote_code=False)
        self.config=json.loads((self.model/'config.json').read_text(encoding='utf-8'))
        generation=self.model/'generation_config.json'
        self.generation=json.loads(generation.read_text()) if generation.exists() else {}
        self.body=json.loads(Path(body).read_text(encoding='utf-8'))

    def prepare(self,out,max_pixels,max_context,max_tokens):
        from PIL import Image,ImageOps
        from request_builder import build_request
        messages,images=[],[]
        for message in self.body['messages']:
            content=message['content']
            if isinstance(content,str):
                messages.append(dict(message));continue
            parts=[]
            for part in content:
                if part['type']=='text':parts.append(dict(part));continue
                if part['type']!='image_url':raise ValueError('Only text and image_url are supported')
                url=part['image_url']
                if isinstance(url,dict):url=url['url']
                if not url.startswith('data:image/') or ';base64,' not in url:raise ValueError('Only embedded images supported')
                binary=base64.b64decode(url.split(',',1)[1],validate=True)
                with Image.open(io.BytesIO(binary)) as image:
                    images.append(ImageOps.exif_transpose(image).convert('RGB'))
                parts.append({'type':'image'})
            messages.append({'role':message['role'],'content':parts})
        return build_request(self.processor,self.config,self.generation,messages,images,out,
                             max_pixels=max_pixels,max_context=max_context,max_new_tokens=max_tokens,thinking=False)
