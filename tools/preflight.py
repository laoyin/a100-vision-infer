"""CPU artifact integrity check; does not load CUDA or certify numerical accuracy."""
import argparse,json,math
from pathlib import Path

def descriptor(root, desc):
    root=Path(root).resolve()
    shape=desc.get('shape')
    if not isinstance(shape,list) or not shape or any(type(x) is not int or x<=0 for x in shape):raise ValueError('Invalid tensor shape')
    sizes={'BF16':2,'F32':4,'I64':8,'U8':1}
    if desc.get('dtype') not in sizes:raise ValueError('Invalid dtype')
    relative=Path(desc['file'])
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid tensor path')
    path=(root/relative).resolve()
    if not path.is_relative_to(root):raise ValueError('Tensor escapes artifact')
    expected=math.prod(shape)*sizes[desc['dtype']]
    if not path.is_file() or path.stat().st_size!=expected:raise ValueError(f'Missing/truncated tensor: {relative}')
    if 'scale' in desc:
        if desc['dtype']!='U8' or len(shape)!=2 or desc['scale']['shape'] not in ([shape[0]],[shape[0],(shape[1]+127)//128]) or desc['scale']['dtype']!='F32':raise ValueError('Invalid quantization scale')
        expected+=descriptor(root,desc['scale'])
    return expected

def inspect(root,tp=None):
    root=Path(root)
    if (root/'INCOMPLETE').exists():raise ValueError('Incomplete export')
    m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if m.get('format')!='avi-v1' or m.get('tp') not in (1,2,4) or len(m['ranks'])!=m['tp']:raise ValueError('Invalid format/TP')
    if tp is not None and m['tp']!=tp:raise ValueError('TP mismatch')
    from format_utils import expected_shapes
    names=set(expected_shapes(m['config']))
    sizes=[]
    for rank in m['ranks']:
        if set(rank['tensors'])!=names:raise ValueError('Weight names mismatch')
        sizes.append(sum(descriptor(root,d) for d in rank['tensors'].values()))
    if m.get('json_grammar_supported'):
        table=json.loads((root/'token_bytes.json').read_text())
        if len(table)!=m['config']['text_config']['vocab_size']:raise ValueError('Vocabulary size mismatch')
        for entry in table:bytes.fromhex(entry)
    return {'status':'passed','tp':m['tp'],'tensors_per_rank':len(names),'disk_bytes_per_rank':sizes,'note':'CPU metadata/file-size validation only'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--tp',type=int)
    a=p.parse_args();print(json.dumps(inspect(a.model,a.tp),indent=2))
if __name__=='__main__':main()
