"""Offline merge of a local BF16 base and PEFT adapter, including visual adapters."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists(): p.error('Output must not exist')
    import torch
    from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
    from peft import PeftModel
    adapter_config=json.loads((a.adapter/'adapter_config.json').read_text())
    base_config=json.loads((a.base/'config.json').read_text())
    if base_config.get('quantization_config'): p.error('Merge into the original unquantized base checkpoint')
    model=Qwen3_5ForConditionalGeneration.from_pretrained(a.base, dtype=torch.bfloat16,
          device_map='cpu', local_files_only=True, trust_remote_code=False)
    model=PeftModel.from_pretrained(model,a.adapter,is_trainable=False,local_files_only=True)
    adapted=[name for name,_ in model.named_modules() if 'lora_A' in name or 'modules_to_save' in name]
    if not adapted: raise RuntimeError('No adapter modules loaded')
    visual=[name for name in adapted if '.visual.' in name]
    print(f'Adapter modules: {len(adapted)}, visual-related: {len(visual)}')
    print('Review adapter coverage against training args; freeze flags alone do not prove tensor coverage.')
    model=model.merge_and_unload(safe_merge=True)
    a.out.mkdir(parents=True,exist_ok=False)
    model.save_pretrained(a.out,safe_serialization=True,max_shard_size='4GB')
    # Prefer a processor saved alongside training output; otherwise preserve the base processor.
    processor_source=a.adapter if (a.adapter/'preprocessor_config.json').exists() else a.base
    AutoProcessor.from_pretrained(processor_source,local_files_only=True,trust_remote_code=False).save_pretrained(a.out)
    audit={'base':str(a.base.resolve()),'adapter':str(a.adapter.resolve()),'adapter_config':adapter_config,
           'adapted_modules':adapted,'visual_modules':visual,'status':'merged; numerical equivalence not yet tested'}
    (a.out/'merge_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__': main()