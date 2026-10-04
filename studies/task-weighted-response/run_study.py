"""Two-stage prospective selection/evaluation on preserved trained shared modules."""
import argparse,contextlib,hashlib,json,time,traceback
import numpy as np
import torch
from torch.nn import functional as F
from common import ROOT,PARENT,read,write,sha,check_freeze
from weighted import PARTS,INDICES,labels,clustering_costs,task_costs,batch_moments
from experiments.train_llm_routing_trajectory import attach_twelve
from experiments.llm_protocol import MODELS
from src.llm_shared import load_model,domain_documents,make_blocks

METHODS=['weight','isotropic_response','task_response','task_single','activation','direct_loss','random']

def sync():torch.cuda.synchronize()
def now():sync();return time.perf_counter()

def document_blocks(tok,split,n,exclude=()):
    rows=[]
    for doc_id,text in domain_documents('general',split):
        digest=hashlib.sha256(text.encode()).hexdigest()
        rows.append((digest,doc_id,text))
    blocks=[];meta=[];skipped=[];seen=set(exclude)
    for digest,doc_id,text in sorted(rows):
        if digest in seen:
            skipped.append(doc_id);continue
        tokens=tok.encode(text,add_special_tokens=False)
        if len(tokens)<129:continue
        seen.add(digest);block=tokens[:129];blocks.append(block)
        meta.append(dict(document_id=doc_id,document_sha256=digest,token_sha256=hashlib.sha256(np.asarray(block,dtype=np.int64).tobytes()).hexdigest()))
        if len(blocks)==n:break
    if len(blocks)!=n:raise ValueError(('insufficient distinct documents',split,len(blocks),n))
    return torch.tensor(blocks),dict(split=split,documents=meta,skipped_duplicate_ids=skipped,
        raw_sha256=sha(PARENT/f'data/wikitext-2/{split}.txt'),token_sha256=hashlib.sha256(np.asarray(blocks,dtype=np.int64).tobytes()).hexdigest())

def losses(model,bank,blocks,theta=None):
    bank.prepare(theta)
    pred=model(input_ids=blocks[:,:-1],use_cache=False).logits
    return F.cross_entropy(pred.float().transpose(1,2),blocks[:,1:],reduction='none').mean(1)

def evaluate(model,bank,blocks,device,theta=None):
    with torch.no_grad():return torch.cat([losses(model,bank,x.to(device),theta).cpu() for x in blocks.split(4)]).tolist()

def collect_gradients(model,bank,blocks,theta,a,b,device,out):
    start=now();n=len(blocks);gs=[];ls=[]
    l,d=theta.shape;ss=torch.zeros(l,l,device=device,dtype=torch.float64);tt=ss.clone();cc=0.
    gm=torch.zeros(l,d,device=device,dtype=torch.float64);ym=gm.clone()
    c=torch.diag_embed(a)-a[:,:,None]*a[:,None,:];c2=c@c;gram=a@a.T
    for block in blocks:
        independent=theta.detach().clone().requires_grad_()
        loss=losses(model,bank,block[None].to(device),independent).mean()
        g,=torch.autograd.grad(loss,independent);gd=g.double()
        y=gram@gd+torch.einsum('lij,lj->li',c2,gd@b.T)@b
        ss+=gd@gd.T;tt+=y@gd.T;cc+=float(y.square().sum());gm+=gd;ym+=y
        gs.append(g.detach().cpu());ls.append(float(loss.detach()))
    ss/=n;tt/=n;cc/=n;gm/=n;ym/=n
    raw=dict(s=ss.cpu().numpy(),t=tt.cpu().numpy(),c=cc,sm=(gm@gm.T).cpu().numpy(),tm=(ym@gm.T).cpu().numpy(),cm=float(ym.square().sum()),n=n)
    duration=now()-start
    torch.save({'gradients':torch.stack(gs),'losses':ls,'arithmetic':'FP32 task gradients; FP64 fixed-factor response'},out)
    return raw,ls,duration

def moment_costs(raw,batch):
    return task_costs(*batch_moments(**raw,batch=batch))

def activation_moments(model,bank,blocks,theta,b,modules,device):
    start=now();v=torch.zeros(12,4,4,device=device,dtype=torch.float64);counts=[0]*12;hooks=[]
    for i,name in enumerate(modules):
        def hook(module,args,out,i=i):
            x=args[0].detach().double().reshape(-1,bank.inputs)
            bx=F.linear(x,b.reshape(4*bank.channels,bank.inputs)).reshape(-1,4,bank.channels)
            v[i].add_(torch.einsum('nkc,njc->kj',bx,bx));counts[i]+=len(x)
        hooks.append(model.get_submodule(name).register_forward_hook(hook))
    with torch.no_grad():
        for x in blocks.split(4):losses(model,bank,x.to(device),theta)
    for h in hooks:h.remove()
    assert len(set(counts))==1 and counts[0]==128*len(blocks)
    return (v/counts[0]).cpu().numpy(),now()-start

def fold(bank,theta,index,device):
    lab=torch.tensor(labels(index),device=device)
    bank.hard_routes=F.one_hot(lab,4).to(dtype=bank.bases.dtype)
    with torch.no_grad():bank.bases.copy_(torch.stack([theta[lab==k].mean(0) for k in range(4)]))

def setup(args):
    torch.set_num_threads(2);torch.manual_seed(810000+args.seed);torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    frozen=check_freeze();t=now();model,tok,entry=load_model(MODELS[args.model],args.device)
    bank,modules=attach_twelve(model,args.seed);model.float()
    cp=PARENT/f'results/llm_routing_trajectories/{args.model}-adamw-seed{args.seed}/final.pt'
    state=torch.load(cp,map_location=args.device,weights_only=True)
    with torch.no_grad():bank.logits.copy_(state['logits']);bank.bases.copy_(state['bases'])
    for p in bank.parameters():p.requires_grad_(False)
    a=bank.logits.detach().double().softmax(-1);b=bank.bases.detach().double();theta=(a@b).float()
    return model,tok,bank,modules,a,b,theta,dict(freeze_sha256=frozen,checkpoint_sha256=sha(cp),model_revision=entry['revision'],model_id=MODELS[args.model],setup_seconds=now()-t)

def select(args,out):
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    model,tok,bank,modules,a,b,theta,base=setup(args);t=now()
    blocks,data=document_blocks(tok,'valid',16);data_seconds=now()-t
    torch.cuda.reset_peak_memory_stats();raw,soft_nll,gradient_seconds=collect_gradients(model,bank,blocks,theta,a,b,args.device,out/'selection_gradients.pt')
    v,activation_seconds=activation_moments(model,bank,blocks,theta,b,modules,args.device)
    aa=a.cpu().numpy();bb=b.cpu().numpy();costs={};times={};chosen={}
    operations={'weight':lambda:clustering_costs(aa,bb@bb.T),
        'isotropic_response':lambda:clustering_costs(aa,np.eye(4)),
        'task_response':lambda:moment_costs(raw,4)[0],
        'task_single':lambda:moment_costs(raw,1)[0],
        'activation':lambda:clustering_costs(aa,v)}
    for name,fn in operations.items():
        t=time.perf_counter();costs[name]=fn();chosen[name]=int(np.argmin(costs[name]));times[name]=time.perf_counter()-t
    rng=np.random.default_rng(820000+args.seed)
    pool=list(dict.fromkeys(chosen[k] for k in ['weight','isotropic_response','activation']))
    while len(pool)<64:
        candidate=int(rng.integers(len(PARTS)))
        if candidate not in pool:pool.append(candidate)
    t=now();direct=[]
    for index in pool:
        fold(bank,theta,index,args.device);direct.append(evaluate(model,bank,blocks,args.device))
    direct_seconds=now()-t;chosen['direct_loss']=pool[int(np.argmin(np.mean(direct,axis=1)))];chosen['random']=int(np.random.default_rng(821000+args.seed).integers(len(PARTS)))
    np.savez_compressed(out/'selection_objectives.npz',**costs,activation_moments=v,**raw)
    independent_cost={k:times[k] for k in times}
    for k in ['task_response','task_single']:independent_cost[k]+=gradient_seconds
    independent_cost['activation']+=activation_seconds
    independent_cost['direct_loss']=direct_seconds+activation_seconds+sum(times[k] for k in ['weight','isotropic_response','activation'])
    independent_cost['random']=0.
    result=dict(**base,model=args.model,seed=args.seed,modules=modules,data=data,soft_selection_nll=soft_nll,
        selected={k:dict(index=index,partition=PARTS[index]) for k,index in chosen.items()},
        objective_gaps={k:float(np.partition(val,1)[1]-val.min()) for k,val in costs.items()},
        selection_costs_at_winners={k:{m:float(val[i]) for m,val in costs.items()} for k,i in chosen.items()},
        direct_candidates=pool,direct_document_nll=direct,cost=dict(data_seconds=data_seconds,
        gradient_response_seconds=gradient_seconds,activation_seconds=activation_seconds,cpu_search_seconds=times,
        direct_search_seconds=direct_seconds,standalone_selection_seconds=independent_cost,
        gradient_forward_backward_calls=16,activation_forward_calls=4,direct_forward_calls=256,
        direct_candidate_count=64,analytic_candidate_count=len(PARTS),selection_documents=16,prediction_tokens_per_selection_pass=2048,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30),
        torch_version=torch.__version__,artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    write(out/'selection.json',result);print(args.model,args.seed,'selection locked',chosen,flush=True)

def assess(args,out):
    if (out/'evaluation.json').exists():raise FileExistsError(out/'evaluation.json')
    lock=read(ROOT/'SELECTION_LOCK.json');assert lock['freeze_sha256']==check_freeze()
    for name,digest in lock['selections'].items():assert sha(ROOT/name)==digest
    selection=read(out/'selection.json')
    model,tok,bank,modules,a,b,theta,base=setup(args)
    assert base['checkpoint_sha256']==selection['checkpoint_sha256']
    exclusion=[x['document_sha256'] for x in selection['data']['documents']]
    t=now();blocks,data=document_blocks(tok,'test',32,exclusion)
    train,trainmeta=make_blocks(tok,'general','train','adapt',128)
    data_seconds=now()-t
    train_hashes={hashlib.sha256(text.encode()).hexdigest() for _,text in domain_documents('general','train')}
    assert not train_hashes.intersection(exclusion),'selection/train duplicate article'
    assert not train_hashes.intersection(x['document_sha256'] for x in data['documents']),'evaluation/train duplicate article'
    raw,soft_nll,gradient_seconds=collect_gradients(model,bank,blocks,theta,a,b,args.device,out/'evaluation_gradients.pt')
    task,w=moment_costs(raw,4);single,_=moment_costs(raw,1)
    np.savez_compressed(out/'evaluation_objectives.npz',task_response=task,task_single=single,**raw)
    gen=torch.Generator().manual_seed(830000+args.seed)
    batches=[torch.randperm(len(train),generator=gen)[:4].tolist() for _ in range(64)]
    distinct=sorted({s['index'] for s in selection['selected'].values()})
    order=np.random.default_rng(831000+args.seed).permutation(distinct).tolist()
    actions={};bank.bases.requires_grad_(True)
    for index in order:
        aliases=[name for name,s in selection['selected'].items() if s['index']==index]
        fold(bank,theta,index,args.device);t=now();immediate=evaluate(model,bank,blocks,args.device);evaluation_seconds=now()-t
        recovery={}
        for optimizer in ['sgd','adamw']:
            fold(bank,theta,index,args.device)
            opt=torch.optim.SGD([bank.bases],lr=.01) if optimizer=='sgd' else torch.optim.AdamW([bank.bases],lr=1e-4,weight_decay=0.)
            history=[];torch.cuda.reset_peak_memory_stats();t=now()
            for ids in batches:
                opt.zero_grad(set_to_none=True);loss=losses(model,bank,train[ids].to(args.device)).mean()
                if not torch.isfinite(loss):raise FloatingPointError((args.model,args.seed,index,optimizer,len(history)))
                loss.backward()
                if optimizer=='adamw':torch.nn.utils.clip_grad_norm_([bank.bases],1.)
                opt.step();history.append(float(loss.detach()))
            seconds=now()-t;peak=torch.cuda.max_memory_allocated()/2**30;t=now();after=evaluate(model,bank,blocks,args.device);evalsec=now()-t
            cp=out/f'partition-{index}-{optimizer}.pt';torch.save({'bases':bank.bases.detach().cpu(),'partition':PARTS[index]},cp)
            recovery[optimizer]=dict(document_nll=after,training_nll=history,training_seconds=seconds,
                evaluation_seconds=evalsec,steps=64,forward_backward_calls=64,prediction_tokens=32768,
                peak_allocated_gib=peak,checkpoint_sha256=sha(cp))
        actions[str(index)]=dict(method_aliases=aliases,immediate_document_nll=immediate,immediate_evaluation_seconds=evaluation_seconds,
            heldout_response_mse=float(task[index]),heldout_response_relative_mse=float(task[index]/batch_moments(**raw,batch=4)[2]),
            recovery=recovery)
        write(out/f'action-{index}.json',actions[str(index)])
        print(args.model,args.seed,aliases,'recovery complete',flush=True)
    result=dict(**base,selection_sha256=sha(out/'selection.json'),selection_lock_sha256=sha(ROOT/'SELECTION_LOCK.json'),
        model=args.model,seed=args.seed,data=data,recovery_data=trainmeta,data_seconds=data_seconds,
        soft_document_nll=soft_nll,evaluation_gradient_seconds=gradient_seconds,evaluation_gradient_calls=32,
        recovery_batch_indices=batches,execution_order=order,actions=actions,
        artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='selection.json'})
    write(out/'evaluation.json',result)

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['select','evaluate']);p.add_argument('--model',choices=['pythia160m','qwen06b','smol360m'],required=True);p.add_argument('--seed',type=int,choices=[0,1,2],required=True);p.add_argument('--device',default='cuda:0');args=p.parse_args()
    out=ROOT/'results'/f'{args.model}-seed{args.seed}'
    try:
        (select if args.phase=='select' else assess)(args,out)
    except Exception as exc:
        failure=ROOT/'results'/f'{args.model}-seed{args.seed}-{args.phase}-failure.json'
        if not failure.exists():write(failure,{'error':repr(exc),'traceback':traceback.format_exc()})
        raise

if __name__=='__main__':main()
