"""Export exact GPT/ByteLevel token bytes for native JSON grammar masking."""
import json
from pathlib import Path

def export_vocabulary(model,out,vocab_size):
    path=Path(model)/'tokenizer.json'
    if not path.exists():return False
    tokenizer=json.loads(path.read_text(encoding='utf-8'))
    def byte_level(decoder):
        if not isinstance(decoder,dict):return False
        if decoder.get('type')=='ByteLevel':return True
        return decoder.get('type')=='Sequence' and len(decoder.get('decoders',[]))==1 and byte_level(decoder['decoders'][0])
    if tokenizer.get('model',{}).get('type')!='BPE' or not byte_level(tokenizer.get('decoder')):return False
    base=list(range(33,127))+list(range(161,173))+list(range(174,256))
    codepoints=base.copy();extra=0
    for byte in range(256):
        if byte not in base:base.append(byte);codepoints.append(256+extra);extra+=1
    inverse={chr(code):byte for byte,code in zip(base,codepoints)}
    table=['']*vocab_size
    for token,index in tokenizer['model']['vocab'].items():
        if index>=vocab_size:raise ValueError('Tokenizer exceeds model vocabulary')
        try:table[index]=bytes(inverse[ch] for ch in token).hex()
        except KeyError:return False
    for item in tokenizer.get('added_tokens',[]):
        if item['id']>=vocab_size:raise ValueError('Added token exceeds model vocabulary')
        table[item['id']]='' if item.get('special',False) else item['content'].encode('utf-8').hex()
    (Path(out)/'token_bytes.json').write_text(json.dumps(table),encoding='utf-8')
    return True