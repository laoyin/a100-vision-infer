"""Compare two native paths for the same artifact; not an independent model accuracy check."""
import argparse,json
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);a=p.parse_args()
 b=json.loads(a.baseline.read_text());c=json.loads(a.candidate.read_text());reports={}
 def check(suffix):
  x=np.fromfile(str(a.baseline)+suffix,dtype='<f4').astype('float64');y=np.fromfile(str(a.candidate)+suffix,dtype='<f4').astype('float64')
  if x.shape!=y.shape or not x.size or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Invalid trace '+suffix)
  cosine=float(x@y/max(np.linalg.norm(x)*np.linalg.norm(y),1e-30));rmse=float(np.linalg.norm(x-y)/max(np.linalg.norm(x),1e-30))
  reports[suffix]={'cosine':cosine,'relative_rmse':rmse}
  if cosine<.99 or rmse>.15:raise ValueError('Native path discrepancy: '+suffix+' '+str(reports[suffix]))
 check('.prefill_logits.f32')
 if Path(str(a.baseline)+'.vision.f32').exists():check('.vision.f32')
 for step in range(1,min(len(b['generated_ids']),len(c['generated_ids']))):
  if b['generated_ids'][:step]!=c['generated_ids'][:step]:break
  check(f'.decode_{step}.f32')
 print(json.dumps({'checks':reports,'tokens_equal':b['generated_ids']==c['generated_ids'],'note':'Native path consistency only; independently validate business quality and original runtime outputs.'},indent=2))
if __name__=='__main__':main()
