"""Diagnose batch-size-dependent task gradients without inspecting task outcomes."""
from types import SimpleNamespace,MethodType
import contextlib
import torch
from torch.nn import functional as F
from common import ROOT,write
from run_study import setup,document_blocks,losses

def inject_dtype(self,index,hidden,output):
    delta=F.linear(hidden.to(self.theta.dtype),self.theta[index].reshape(self.channels,self.inputs))
    return output.index_add(-1,self.output_ids,delta.to(output.dtype))

def main():
    results=[]
    for mode in ['fp32_math','fp32_eager','fp64_math']:
        args=SimpleNamespace(model='pythia160m',seed=0,device='cuda:3')
        model,tok,bank,_,a,b,theta,_=setup(args)
        if mode=='fp32_eager':model.config._attn_implementation='eager'
        if mode=='fp64_math':
            model.double();bank.inject=MethodType(inject_dtype,bank);theta=a@b
        blocks,_=document_blocks(tok,'valid',16);x=theta.detach().requires_grad_()
        context=torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH) if mode!='fp32_eager' else contextlib.nullcontext()
        with context:
            batchloss=losses(model,bank,blocks[:4].to(args.device),x).mean();g,=torch.autograd.grad(batchloss,x)
            singles=[];l=[]
            for block in blocks[:4]:
                loss=losses(model,bank,block[None].to(args.device),x).mean();gg,=torch.autograd.grad(loss,x);singles.append(gg);l.append(float(loss.detach()))
        ref=torch.stack(singles).mean(0)
        row=dict(mode=mode,relative_gradient_difference=float((g-ref).double().norm()/ref.double().norm()),
                 batch_loss=float(batchloss.detach()),mean_single_loss=sum(l)/4)
        results.append(row);print(row,flush=True)
        del model,bank,theta,x,g,gg,ref,batchloss,loss;torch.cuda.empty_cache()
    write(ROOT/'research/PYTHIA_ARITHMETIC_CHECK.json',dict(scope='Selected validation documents, seed0; backend/precision diagnosis, no performance comparison',results=results))

if __name__=='__main__':main()
