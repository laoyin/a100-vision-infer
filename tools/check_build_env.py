"""Fail before compilation when installed nvcc and PyTorch CUDA disagree."""
import os,re,subprocess

def cuda_release(text):
    match=re.search(r'release (\d+\.\d+)',text)
    if not match:raise ValueError('Cannot identify nvcc CUDA version')
    return match.group(1)

def main():
    import torch,torchvision
    compiler=os.environ.get('CUDACXX','nvcc')
    toolkit=cuda_release(subprocess.check_output([compiler,'--version'],text=True))
    if torch.version.cuda!=toolkit:
        raise SystemExit(f'CUDA mismatch: nvcc={toolkit}, torch={torch.version.cuda}. Select an already-installed matching Python environment or CUDA compiler. No packages or drivers were changed.')
    if torch.cuda.device_count()<2:raise SystemExit('Two visible GPUs required for TP2 acceptance')
    for i in range(torch.cuda.device_count()):
        if torch.cuda.get_device_capability(i)!=(8,0):raise SystemExit('Expected SM80 GPUs')
    print('Build environment:',torch.__version__,torchvision.__version__,'nvcc',toolkit,'visible GPUs',torch.cuda.device_count())
if __name__=='__main__':main()
