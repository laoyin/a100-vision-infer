"""Strict small-model generation regression; does not relax token mismatches."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--tp',type=int,required=True)
    a=p.parse_args()
    reference=json.loads((a.root/f'tp{a.tp}-mtp0.json').read_text())['generated_ids']
    for window in (1,2,3,5):
        result=json.loads((a.root/f'tp{a.tp}-mtp{window}.json').read_text())
        if result['generated_ids']!=reference:
            raise SystemExit(f'TP{a.tp} MTP{window}: token mismatch; retain outputs for diagnosis')
        if len(reference)>1 and result['mtp']['rounds']<1:
            raise SystemExit('MTP execution was not exercised')
    for name in ('graph','cached'):
        path=a.root/f'tp{a.tp}-mtp-{name}.json'
        if path.exists() and json.loads(path.read_text())['generated_ids']!=reference:
            raise SystemExit(f'TP{a.tp} MTP {name}: token mismatch')
    print(f'TP{a.tp}: native MTP windows 1/2/3/5 match target greedy output')
if __name__=='__main__':main()
