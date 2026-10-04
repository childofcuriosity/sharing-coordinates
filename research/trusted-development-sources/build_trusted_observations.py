"""Create compact observations; calibration and evaluation truth is segregated."""
import argparse
import hashlib
from pathlib import Path
import time
import torch
from torch.nn import functional as F

from experiments.train_enhancement import source_hashes
from experiments.run_autograd_tomography import autodiff_response
from src.trusted_recovery import compress_observations
from src.language import load_language_checkpoint,load_byte_stream,sample_batch
from src.response_identifiability import pack_basis_matrix
from src.enhancement import effective_forward


def cpu_dict(obs):
    return {k:v.detach().cpu() if isinstance(v,torch.Tensor) else v
            for k,v in obs.tensor_dict().items()}


def synthetic(seed,regime):
    rng=torch.Generator().manual_seed(3000000+seed)
    a=torch.randn(6,3,generator=rng,dtype=torch.double).softmax(-1)
    b=torch.randn(3,16,generator=rng,dtype=torch.double)
    if regime=='rank_01': a=.1*a+.9/3
    elif regime=='rank_001': a=.01*a+.99/3
    elif regime=='boundary_001':
        a=.01*a+.99*F.one_hot(torch.arange(6)%3,3).double()
    elif regime!='interior': raise ValueError(regime)
    g=torch.randn(16,6,16,generator=rng,dtype=torch.double)
    g=g/g.flatten(1).norm(dim=1)[:,None,None]
    return a,b,g,{}


def language(seed,checkpoint,device):
    model,config,meta=load_language_checkpoint(str(checkpoint),device=device)
    model=model.double().eval()
    if config.tau_end!=1: raise ValueError('Expected tau=1')
    factors=pack_basis_matrix(model)
    a=model.router.probabilities().detach();b=factors.bases.detach()
    theta=a@b
    path=Path('data/wikitext-2/train.txt')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=meta['corpus_sha256']['train']: raise ValueError('Corpus mismatch')
    stream=load_byte_stream(str(path),torch.device(device))
    rng=torch.Generator().manual_seed(3100000+seed)
    gradients=[]
    for _ in range(16):
        x,y=sample_batch(stream,2,32,rng)
        variable=theta.detach().requires_grad_()
        output=effective_forward(model,variable,factors.layout,x)
        loss=F.cross_entropy(output.flatten(0,1),y.flatten())
        g,=torch.autograd.grad(loss,variable)
        gradients.append(g.detach()/g.norm())
    return a,b,torch.stack(gradients),dict(checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        checkpoint_path=str(checkpoint),corpus_sha256=digest,gradient_batch_seed=3100000+seed)


def build(seed,kind,split,device,checkpoint=None,pilot=False):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    if device.startswith('cuda'): torch.cuda.set_device(device)
    started=time.monotonic()
    regimes=['language'] if kind=='language' else ['interior','rank_01','rank_001','boundary_001']
    if pilot: regimes=regimes[:1]
    records=[]
    for ri,regime in enumerate(regimes):
        if kind=='language':
            a,b,gradients,metadata=language(seed,checkpoint,device)
        else:
            a,b,gradients,metadata=synthetic(seed,regime)
            a=a.to(device);b=b.to(device);gradients=gradients.to(device)
        theta=a@b
        responses=torch.stack([autodiff_response(a,b,g) for g in gradients])
        rng=torch.Generator(device=device).manual_seed(3300000+seed*10+ri)
        tn=torch.randn(theta.shape,generator=rng,dtype=theta.dtype,device=device)
        tn=tn/tn.norm()*theta.norm()
        rn=torch.randn(responses.shape,generator=rng,dtype=responses.dtype,device=device)
        rn=rn/rn.flatten(1).norm(dim=1)[:,None,None]*responses.flatten(1).norm(dim=1)[:,None,None]
        for noise in ([0.,1e-6,1e-3] if pilot else [0.,1e-8,1e-6,1e-4,1e-3,1e-2]):
            observed_theta=theta+noise*tn
            observed_responses=responses+noise*rn
            for q in ([4] if pilot else [4,12]):
                key=dict(kind=kind,split=split,seed=seed,regime=regime,noise=noise,q=q)
                try:
                    fit,u=compress_observations(observed_theta,gradients[:q],observed_responses[:q],a.shape[1],noise)
                    hold,hu=compress_observations(observed_theta,gradients[12:],observed_responses[12:],a.shape[1],noise)
                    if float((u-hu).norm())>1e-10:
                        raise RuntimeError('Independent compression row bases differ')
                    true_c=b@u.T
                    truth=dict(a=a.cpu(),c=true_c.cpu(),a_norm=float(a.norm()),b_norm=float(b.norm()),
                               b_outside_squared=float((b-true_c@u).square().sum()))
                    records.append(dict(key=key,status='ok',fit=cpu_dict(fit),holdout=cpu_dict(hold),truth=truth))
                except (RuntimeError,ValueError) as error:
                    records.append(dict(key=key,status='failed',error=str(error)))
        del a,b,gradients,responses,rn,tn,theta
    sources=source_hashes(['experiments/build_trusted_observations.py','src/trusted_recovery.py',
        'experiments/train_enhancement.py','experiments/run_autograd_tomography.py',
        'src/response_factor_recovery.py','src/response_identifiability.py','src/enhancement.py',
        'src/language.py','src/models.py','src/patterns.py','research/TRUSTED_RECOVERY_PROTOCOL.md'])
    return dict(format='trusted-observations-v1',kind=kind,split=split,seed=seed,pilot=pilot,records=records,
                metadata=metadata,source_sha256=sources,torch_version=torch.__version__,
                device=device,seconds=time.monotonic()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--kind',choices=['synthetic','language'],required=True)
    p.add_argument('--split',choices=['development','calibration','evaluation'],required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--device',default='cpu')
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise SystemExit('Refusing observation overwrite')
    if args.pilot and args.split!='development': raise SystemExit('Pilot only in development')
    result=build(args.seed,args.kind,args.split,args.device,args.checkpoint,args.pilot)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(result,args.output)
    print(dict(seed=args.seed,records=len(result['records']),seconds=result['seconds']))
