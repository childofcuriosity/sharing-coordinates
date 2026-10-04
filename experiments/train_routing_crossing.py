"""Actual forward autograd SGD for the predeclared constructed crossing."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import scipy
from scipy.integrate import solve_ivp
import torch
from src.routing_geometry import boundary_example, flow_rhs, balanced_partitions, labels_from_partition, partition

ROOT=Path(__file__).resolve().parents[1]
PARTS=list(balanced_partitions(6,3))
LABELS=[]
for groups in PARTS:
    base=labels_from_partition(groups)
    for perm in itertools.permutations(range(3)):
        LABELS.append(np.asarray(perm)[base])
LABELS=np.stack(LABELS)
ONEHOT=np.eye(3)[LABELS]
PAIRS=np.array([[tuple(g) for g in groups] for groups in PARTS])
MARGIN=1e-8


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(a,b,loss,step,lr):
    scores=np.einsum('ik,hik->h',a,ONEHOT)
    order=np.argsort(scores);best=int(order[-1])
    theta=a@b
    distances=((theta[:,None]-theta[None,:])**2).sum(-1)
    costs=distances[PAIRS[:,:,0],PAIRS[:,:,1]].sum(1)/2
    idx=np.argsort(costs);winner=int(idx[0])
    _,_,p,q,_=boundary_example()
    return dict(step=step,time=step*lr,loss=loss,
        coefficient_partition=partition(LABELS[best]),weight_partition=PARTS[winner],
        coefficient_gap=float(scores[order[-1]]-scores[order[-2]]),
        weight_gap=float(costs[idx[1]]-costs[winner]),
        p_minus_q=float((a*(np.eye(3)[p]-np.eye(3)[q])).sum()),
        min_a=float(a.min()),sigma_a=float(np.linalg.svd(a,compute_uv=False)[-1]),
        sigma_b=float(np.linalg.svd(b,compute_uv=False)[-1]))


def state(z,b):
    return dict(z=z.detach().double().numpy().tolist(),b=b.detach().double().numpy().tolist(),
        a=z.softmax(-1).detach().double().numpy().tolist())


def run(z0,b0,target,lr,dtype,name):
    z=torch.tensor(z0,dtype=dtype,requires_grad=True)
    b=torch.tensor(b0,dtype=dtype,requires_grad=True)
    t=torch.tensor(target,dtype=dtype)
    opt=torch.optim.SGD([z,b],lr=lr,momentum=0,weight_decay=0)
    initial=state(z,b);trace=[];first=None;streak=0;sustained=None
    steps=round(.4/lr)
    for step in range(steps+1):
        a=z.softmax(-1)
        # Explicit supervised predictions on the canonical input dataset.
        x=torch.eye(9,dtype=dtype)
        prediction=(x@b.T)@a.T
        loss=.5*(prediction-t.T).square().sum()
        row=evaluate(a.detach().double().numpy(),b.detach().double().numpy(),float(loss.detach()),step,lr)
        trace.append(row)
        origin=trace[0]
        eligible=(origin['coefficient_partition']==origin['weight_partition'] and
                  origin['coefficient_gap']>MARGIN and origin['weight_gap']>MARGIN)
        qualifies=(eligible and row['coefficient_partition']!=row['weight_partition'] and
                   row['weight_partition']==origin['weight_partition'] and
                   row['coefficient_gap']>MARGIN and row['weight_gap']>MARGIN)
        streak=streak+1 if qualifies else 0
        if qualifies and first is None:first=dict(step=step,time=step*lr,state=state(z,b))
        if streak>=5 and sustained is None:sustained=step-4
        if step==steps:break
        opt.zero_grad()
        loss.backward()
        opt.step()
    losses=np.array([r['loss'] for r in trace]);tol=1e-12 if dtype==torch.float64 else 1e-7
    return dict(name=name,dtype=str(dtype),lr=lr,steps=steps,target=target.tolist(),initial=initial,
        final=state(z,b),initially_eligible=eligible,first_crossing=first,sustained_crossing_start_step=sustained,
        loss_nonincreasing=bool(np.max(np.diff(losses))<=tol),max_loss_increase=float(np.max(np.diff(losses))),
        min_a=min(r['min_a'] for r in trace),min_sigma_a=min(r['sigma_a'] for r in trace),
        min_sigma_b=min(r['sigma_b'] for r in trace),trace=trace)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'results/routing_training/training.json')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(1)
    a,b,p,q,target=boundary_example();z=np.log(a)
    dz,db=flow_rhs(z,b,target);z0=z-.1*dz;b0=b-.1*db
    # Independent autograd/analytical gradient agreement, before any optimization.
    zz=torch.tensor(z0,requires_grad=True);bb=torch.tensor(b0,requires_grad=True)
    loss=.5*(zz.softmax(-1)@bb-torch.tensor(target)).square().sum();loss.backward()
    vz,vb=flow_rhs(z0,b0,target)
    gradient_error=max(float(np.max(np.abs(zz.grad.numpy()+vz))),float(np.max(np.abs(bb.grad.numpy()+vb))))
    assert gradient_error<1e-12
    y0=np.r_[z0.ravel(),b0.ravel()]
    def rhs(t,y):
        vz,vb=flow_rhs(y[:18].reshape(6,3),y[18:].reshape(3,9),target)
        return np.r_[vz.ravel(),vb.ravel()]
    times=np.linspace(0,.4,161)
    sol=solve_ivp(rhs,[0,.4],y0,t_eval=times,rtol=1e-11,atol=1e-13)
    assert sol.success
    ref=[]
    for step,y in enumerate(sol.y.T):
        zs=y[:18].reshape(6,3);bs=y[18:].reshape(3,9)
        aa=np.exp(zs-zs.max(1,keepdims=True));aa/=aa.sum(1,keepdims=True)
        ref.append(evaluate(aa,bs,float(.5*np.square(aa@bs-target).sum()),step,.0025))
    reference_y=sol.y[:,-1]
    primary=[]
    for dtype in [torch.float64,torch.float32]:
        for lr in [.02,.01,.005,.0025]:
            result=run(z0,b0,target,lr,dtype,f'primary-{str(dtype)}-{lr}')
            end=np.r_[np.asarray(result['final']['z']).ravel(),np.asarray(result['final']['b']).ravel()]
            result['endpoint_parameter_error_vs_flow']=float(np.linalg.norm(end-reference_y))
            primary.append(result)
    perturb=[]
    for scale in [.0001,.001,.01]:
        for seed in range(20):
            rng=np.random.default_rng(seed)
            zp=z0+scale*rng.standard_normal(z0.shape);bp=b0+scale*rng.standard_normal(b0.shape)
            result=run(zp,bp,target,.005,torch.float64,f'perturb-{scale}-{seed}')
            result.update(scale=scale,seed=seed);perturb.append(result)
    aa=np.exp(z0-z0.max(1,keepdims=True));aa/=aa.sum(1,keepdims=True)
    control=run(z0,b0,aa@b0,.005,torch.float64,'stationary-target-control')
    def counts(runs):
        return dict(runs=len(runs),eligible=sum(r['initially_eligible'] for r in runs),
            crossings=sum(r['first_crossing'] is not None for r in runs),
            sustained=sum(r['sustained_crossing_start_step'] is not None for r in runs),
            nonincreasing_loss=sum(r['loss_nonincreasing'] for r in runs))
    report=dict(scope='Constructed fixed-target supervised regression; actual autograd forward SGD; not natural LLM prevalence.',
        protocol_sha256=digest(ROOT/'research/ROUTING_TRAINING_PROTOCOL.md'),
        source_sha256={p:digest(ROOT/p) for p in ['experiments/train_routing_crossing.py','src/routing_geometry.py']},
        versions=dict(torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__),
        initialization='Explicit negative tangent displacement of size 0.1; no backward integration',
        gradient_error=gradient_error,primary=primary,perturbations=perturb,control=control,
        forward_ode_reference=ref,summary=dict(primary=counts(primary),
            perturbations={str(s):counts([r for r in perturb if r['scale']==s]) for s in [.0001,.001,.01]},
            control_crossing=control['first_crossing'] is not None))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(report['summary'],indent=2))
    for r in primary:
        print(r['name'], 'first', None if r['first_crossing'] is None else r['first_crossing']['step'],
              'loss',r['trace'][0]['loss'],r['trace'][-1]['loss'], 'endpoint_error',r['endpoint_parameter_error_vs_flow'])

if __name__=='__main__':main()
