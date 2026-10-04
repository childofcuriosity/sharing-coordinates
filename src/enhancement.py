"""Additive controls for prospective research; legacy measurement code is unchanged."""
import math
import torch
from torch.nn import functional as F


def effective_forward(model, theta, layout, tokens):
    """Byte-LM forward with independent layer-effective weights and fixed remainder."""
    tensors = {s.key:theta[:,s.start:s.end].reshape(theta.shape[0],*s.parameter_shape_without_basis)
               for s in layout}
    batch,length=tokens.shape
    positions=torch.arange(length,device=tokens.device)
    x=F.embedding(tokens,model.token_embedding.weight.detach())+F.embedding(
        positions,model.position_embedding.weight.detach())[None]
    causal=torch.ones(length,length,dtype=torch.bool,device=x.device).triu(1)
    heads=model.attention.num_heads
    width=model.dimension//heads
    for i in range(model.depth):
        def linear(value,name):
            bias=tensors.get(name+'.bias')
            return F.linear(value,tensors[name+'.weight'][i],None if bias is None else bias[i])
        h=F.layer_norm(x,(model.dimension,))
        q,k,v=linear(h,'attention.qkv').chunk(3,-1)
        q,k,v=[t.view(batch,length,heads,width).transpose(1,2) for t in (q,k,v)]
        scores=(q@k.transpose(-1,-2))/math.sqrt(width)
        scores=scores.masked_fill(causal[None,None],torch.finfo(scores.dtype).min)
        values=(scores.softmax(-1)@v).transpose(1,2).contiguous().view(batch,length,model.dimension)
        x=x+model.residual_scale*linear(values,'attention.output')
        h=F.layer_norm(x,(model.dimension,))
        x=x+model.residual_scale*linear(F.gelu(linear(h,'up')),'down')
    h=F.layer_norm(x,(model.dimension,),model.final_norm.weight.detach(),model.final_norm.bias.detach(),model.final_norm.eps)
    return F.linear(h,model.token_embedding.weight.detach())


def same_size_gauge(a, seed, family, trials=5000):
    """First feasible candidate, fixed budget, router-only and exact size constraint."""
    a=a.detach().cpu().double()
    l,k=a.shape
    native=a.argmax(-1)
    sizes=sorted(torch.bincount(native,minlength=k).tolist())
    rng=torch.Generator().manual_seed(seed)
    eye=torch.eye(k,dtype=a.dtype)
    counts=dict(attempted=0,condition=0,simplex=0,same_sizes=0,changed=0)
    selected=None
    if family not in ('positive_stochastic','signed_local'):
        raise ValueError('Unknown gauge family')
    for index in range(trials):
        counts['attempted']+=1
        if family=='positive_stochastic':
            raw=torch.rand(k,k,generator=rng,dtype=a.dtype)+.01
            strength=.02+.98*float(torch.rand((),generator=rng))
            m=(1-strength)*eye+strength*raw/raw.sum(-1,keepdim=True)
        else:
            direction=torch.randn(k,k,generator=rng,dtype=a.dtype)
            direction-=direction.mean(-1,keepdim=True)
            direction/=direction.norm()
            scale=math.exp(-3.5+4*float(torch.rand((),generator=rng)))
            m=eye+scale*direction
        condition=float(torch.linalg.cond(m))
        if not math.isfinite(condition) or condition>30:
            continue
        counts['condition']+=1
        transformed=a@m
        if float(transformed.min())<=1e-8:
            continue
        counts['simplex']+=1
        assignment=transformed.argmax(-1)
        if sorted(torch.bincount(assignment,minlength=k).tolist())!=sizes:
            continue
        counts['same_sizes']+=1
        if torch.equal(native[:,None]==native[None,:],assignment[:,None]==assignment[None,:]):
            continue
        counts['changed']+=1
        if selected is None:
            selected=dict(index=index,matrix=m.tolist(),assignment=assignment.tolist(),condition=condition,
                          min_probability=float(transformed.min()))
    return dict(family=family,seed=seed,trials=trials,counts=counts,selected=selected,
                native_assignment=native.tolist(),sorted_sizes=sizes,
                status='found' if selected else 'search_failed')


def balanced_effective_partition(theta, sizes, restarts=16, iterations=50, seed=760000):
    """Fixed-size pair-swap local search in Gram coordinates; not global optimum."""
    theta=theta.detach().cpu().double()
    gram=theta@theta.T
    sizes=[n for n in sizes if n>0]
    labels=torch.repeat_interleave(torch.arange(len(sizes)),torch.tensor(sizes))
    if labels.numel()!=len(theta):
        raise ValueError('Group sizes must sum to depth')
    def objective(labels):
        # Constant total sum of squared norms omitted; maximize centroid fit.
        return sum(float(gram[labels==i][:,labels==i].sum())/n for i,n in enumerate(sizes))
    rng=torch.Generator().manual_seed(seed)
    best=None
    for _ in range(restarts):
        current=labels[torch.randperm(len(labels),generator=rng)].clone()
        value=objective(current)
        for _ in range(iterations):
            winner=None
            for i in range(len(labels)):
                for j in range(i):
                    if current[i]==current[j]:
                        continue
                    candidate=current.clone()
                    candidate[i],candidate[j]=current[j],current[i]
                    score=objective(candidate)
                    if score>value+1e-12 and (winner is None or score>winner[0]):
                        winner=(score,candidate)
            if winner is None:
                break
            value,current=winner
        if best is None or value>best[0]:
            best=(value,current.clone())
    return best[1],dict(sse=float(gram.diagonal().sum())-best[0],restarts=restarts,
                        iterations=iterations,seed=seed,sorted_sizes=sorted(sizes))
