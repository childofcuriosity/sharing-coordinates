"""Numerical check of the batch-mean gradient assumption; no selection changes."""
from types import SimpleNamespace
import torch
from common import ROOT,read,write
from run_study import setup,document_blocks,losses

def main():
    results=[]
    for model_name in ['pythia160m','qwen06b','smol360m']:
        args=SimpleNamespace(model=model_name,seed=0,device='cuda:3')
        model,tok,bank,modules,a,b,theta,base=setup(args)
        blocks,data=document_blocks(tok,'valid',16)
        expected=read(ROOT/'results'/f'{model_name}-seed0'/'selection.json')['data']
        assert data['token_sha256']==expected['token_sha256']
        raw=torch.load(ROOT/'results'/f'{model_name}-seed0'/'selection_gradients.pt',map_location='cpu',weights_only=True)
        theta=theta.detach().requires_grad_();loss=losses(model,bank,blocks[:4].to(args.device),theta).mean()
        observed,=torch.autograd.grad(loss,theta)
        reference=raw['gradients'][:4].mean(0).to(args.device)
        relative=float((observed-reference).double().norm()/reference.double().norm())
        row=dict(model=model_name,relative_batch_gradient_difference=relative,
            batch_loss=float(loss.detach()),mean_recorded_document_loss=sum(raw['losses'][:4])/4)
        assert relative<1e-4,row
        results.append(row);print(row,flush=True)
        del model,bank,theta,observed,reference,loss;torch.cuda.empty_cache()
    write(ROOT/'research/BATCH_GRADIENT_CHECK.json',{'scope':'Selected first four validation documents, seed0 in each model; no performance-based choices','results':results})

if __name__=='__main__':main()
