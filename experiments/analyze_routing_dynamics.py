"""Exploratory theory checks and all-checkpoint post hoc geometric audit.

No new training, no changes to the confirmatory LLM protocol or outcomes.
"""
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from scipy.integrate import solve_ivp
from src.llm_shared import sha256
from src.llm_decisions import balanced_effective_partition
from src.routing_geometry import (partition,balanced_partitions,labels_from_partition,
    squared_distances,partition_cost,router_assignment,full_response,
    hard_response_score,boundary_example,flow_rhs)
from experiments.llm_protocol import ROOT,MODELS,CONFIG,unit_path,check_freeze


def integer_certificate(numerator):
    n,k=numerator.shape;parts=list(balanced_partitions(n,k))
    # Integer scores and pair-distance sums avoid floating-point tie claims.
    assignment=[]
    for p in parts:
        for permutation in itertools.permutations(range(k)):
            labels=np.empty(n,dtype=int)
            for group,c in zip(p,permutation):labels[list(group)]=c
            assignment.append((int(numerator[np.arange(n),labels].sum()),tuple(map(int,labels))))
    assignment.sort(reverse=True)
    d=squared_distances(numerator)
    costs=sorted((int(sum(d[i,j] for g in p for i,j in itertools.combinations(g,2))),p) for p in parts)
    return dict(labeled_assignments=len(assignment),unlabeled_partitions=len(parts),
        router_best_score_numerator=assignment[0][0],router_runnerup_score_numerator=assignment[1][0],
        router_best_score_ties=sum(s==assignment[0][0] for s,_ in assignment),
        router_next_distinct_score=max(s for s,_ in assignment if s<assignment[0][0]),
        router_best_labels=assignment[0][1],router_partition=partition(assignment[0][1]),
        cluster_pair_sum_numerator=costs[0][0],cluster_runnerup_pair_sum_numerator=costs[1][0],
        cluster_partition=costs[0][1],pair_sum_to_sse_divisor=n//k,
        router_partition_pair_sum=int(sum(d[i,j] for g in partition(assignment[0][1]) for i,j in itertools.combinations(g,2))))


def main():
    freeze=check_freeze();torch.set_num_threads(1)
    out=ROOT/'results/routing_dynamics';out.mkdir(exist_ok=True)
    n=np.array([[8,11,4],[8,5,10],[10,9,4],[5,11,7],[6,10,7],[11,5,7]])
    a=n/23.;b=np.concatenate([np.eye(3),np.zeros((3,6))],axis=1)
    m1=np.array([[1,4,5],[5,4,1],[5,2,3]])/10.
    m2=np.array([[2,1,7],[4,4,2],[5,2,3]])/10.
    k4=np.full((8,4),23,dtype=int);k4[:6,:3]+=16*n;k4[6:,3]=391
    assertions=dict(k3=integer_certificate(n),k3_denominator=23,
        k4=integer_certificate(k4),k4_denominator=460,
        same_product_router_change=integer_certificate(n@np.rint(m1*10).astype(int)),
        same_product_dynamic_geometry=integer_certificate(n@np.rint(m2*10).astype(int)),
        gauges=dict(router=m1.tolist(),dynamic=m2.tolist()))
    parts=list(balanced_partitions(6,3))
    static=[]
    for name,aa,bb in [('original',a,b),('anisotropic',a,np.diag([4.,1.,1.])@b),
                       ('gauge_router',a@m1,np.linalg.solve(m1,b)),('gauge_dynamic',a@m2,np.linalg.solve(m2,b))]:
        da=squared_distances(aa);dt=squared_distances(aa@bb)
        dynamic=min(parts,key=lambda p:partition_cost(da,p));weight=min(parts,key=lambda p:partition_cost(dt,p))
        r=full_response(aa,bb);constants=[]
        for p in parts:
            h=np.eye(3)[labels_from_partition(p)]
            constants.append(float(np.square(r-np.kron(h@h.T,np.eye(9))).sum())-9*hard_response_score(aa,labels_from_partition(p)))
        assert np.ptp(constants)<1e-10
        static.append(dict(name=name,router_partition=partition(router_assignment(aa)[0]),
            weight_optimum=weight,dynamic_optimum=dynamic,alpha=float(aa.min()),
            sigma_a=float(np.linalg.svd(aa,compute_uv=False)[-1]),sigma_b=float(np.linalg.svd(bb,compute_uv=False)[-1]),
            weight_cost_at_weight_optimum=float(partition_cost(dt,weight)),weight_cost_at_dynamic_optimum=float(partition_cost(dt,dynamic)),
            router_sse_at_weight_optimum=float(partition_cost(da,weight)),router_sse_at_dynamic_optimum=float(partition_cost(da,dynamic)),
            operator_constant_range=float(np.ptp(constants)),product_difference_from_original=float(np.linalg.norm(aa@bb-a@b))))
    aa,bb,p,q,target=boundary_example();z=np.log(aa);y0=np.concatenate([z.ravel(),bb.ravel()])
    difference=np.eye(3)[p]-np.eye(3)[q]
    def rhs(t,y):
        dz,db=flow_rhs(y[:18].reshape(6,3),y[18:].reshape(3,9),target)
        return np.concatenate([dz.ravel(),db.ravel()])
    dz,db=flow_rhs(z,bb,target)
    da=np.array([(np.diag(row)-np.outer(row,row))@d for row,d in zip(aa,dz)])
    flow=[]
    for end in [-.1,-.01,0.,.01,.1]:
        if end:
            sol=solve_ivp(rhs,[0,end],y0,rtol=1e-12,atol=1e-14);assert sol.success
            y=sol.y[:,-1]
        else:y=y0
        zz=y[:18].reshape(6,3);av=np.exp(zz-zz.max(1,keepdims=True));av/=av.sum(1,keepdims=True)
        bv=y[18:].reshape(3,9);theta=av@bv;costs=sorted((partition_cost(squared_distances(theta),g),g) for g in parts)
        labels,score,gap=router_assignment(av)
        flow.append(dict(t=end,loss=float(np.square(theta-target).sum()/2),
            router_partition=partition(labels),router_gap=float(gap),p_minus_q=float((av*difference).sum()),
            cluster_partition=costs[0][1],cluster_gap=float(costs[1][0]-costs[0][0]),
            alpha=float(av.min()),sigma_a=float(np.linalg.svd(av,compute_uv=False)[-1]),sigma_b=float(np.linalg.svd(bv,compute_uv=False)[-1])))
    boundary=dict(integer=integer_certificate(np.rint(aa*92).astype(int)),denominator=92,
        a=aa.tolist(),b=bb.tolist(),p=p.tolist(),q=q.tolist(),target=target.tolist(),
        score_gap_derivative=float((da*difference).sum()),trajectory=flow,
        scope='Constructed squared effective-weight loss and Euclidean factor gradient flow. Negative time selects an ordinary earlier initial state, not a different objective. Numeric integration illustrates, but does not prove, the transverse-crossing theorem.')
    smallparts=list(balanced_partitions(12,4));indices=np.array(smallparts,dtype=int)
    rows=[];inputs={}
    for model in MODELS:
        for seed in CONFIG['seeds']:
            path=unit_path(model,seed,'evaluation')/'checkpoint.pt';inputs[str(path.relative_to(ROOT))]=sha256(path)
            cp=torch.load(path,weights_only=True);av=cp['a'].numpy();bv=cp['b'].numpy()
            gram=bv@bv.T;x=av@np.linalg.cholesky(gram)
            ar,router_score,router_gap=router_assignment(av)
            wt=balanced_effective_partition(torch.from_numpy(x),4).numpy()
            dyn=balanced_effective_partition(torch.from_numpy(av),4).numpy()
            dt=squared_distances(x);dr=squared_distances(av)
            actions={name:partition(lab) for name,lab in [('coefficient_assignment',ar),('weight_lloyd',wt),('router_lloyd',dyn)]}
            same=ar[:,None]==ar[None,:];between=float(dt[~same].min());within=float(dt[same].max())
            # The tangent Gram spectrum measures departure from regular-simplex geometry.
            _,_,v=np.linalg.svd(np.ones((1,4)),full_matrices=True);tangent=v[1:].T
            eig=np.linalg.eigvalsh(tangent.T@gram@tangent)
            row=dict(model=model,seed=seed,primary_decision_unit=model in CONFIG['decision_models'],
                actions=actions,router_score=router_score,router_assignment_gap=router_gap,
                tangent_basis_gram_eigenvalues=eig.tolist(),tangent_gram_condition=float(eig[-1]/eig[0]),
                pairwise_separation_certificate=between>within,between_min_squared=between,within_max_squared=within,
                argmax_counts=np.bincount(av.argmax(1),minlength=4).tolist(),
                weight_cost={key:float(partition_cost(dt,g)) for key,g in actions.items()},
                router_cost={key:float(partition_cost(dr,g)) for key,g in actions.items()},
                response_partition_term={key:hard_response_score(av,labels_from_partition(g)) for key,g in actions.items()},
                coefficient_vs_weight_different=actions['coefficient_assignment']!=actions['weight_lloyd'],
                router_vs_weight_different=actions['router_lloyd']!=actions['weight_lloyd'])
            if model=='pythia160m':
                exact={}
                for name,d in [('weight',dt),('router',dr)]:
                    costs=np.zeros(len(indices))
                    for i,j in itertools.combinations(range(3),2):costs+=d[indices[:,:,i],indices[:,:,j]].sum(1)/3
                    order=np.argsort(costs);best=int(order[0]);second=int(order[1])
                    exact[name]=dict(partitions=len(indices),best_partition=smallparts[best],cost=float(costs[best]),
                        runnerup_gap=float(costs[second]-costs[best]),
                        coefficient_excess=float(partition_cost(d,actions['coefficient_assignment'])-costs[best]))
                row['exhaustive']=exact
            rows.append(row)
    report=dict(format='routing-dynamics-analysis-v1',scope='Post hoc analysis of every existing evaluation checkpoint; no new language-model training or new downstream-loss comparisons. Constructed examples establish possibility, not typicality.',
        original_llm_freeze_sha256=freeze,input_sha256=inputs,source_sha256={p:sha256(ROOT/p) for p in ['src/routing_geometry.py','experiments/analyze_routing_dynamics.py']},
        exact_certificates=assertions,static_examples=static,boundary_flow=boundary,trained_checkpoints=rows,
        counts=dict(checkpoints=len(rows),primary_decision_units=sum(r['primary_decision_unit'] for r in rows),
            coefficient_vs_weight_different=sum(r['coefficient_vs_weight_different'] for r in rows),
            router_vs_weight_different=sum(r['router_vs_weight_different'] for r in rows),
            sufficient_pairwise_separation=sum(r['pairwise_separation_certificate'] for r in rows)))
    with (out/'analysis.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(report['counts'],indent=2))
    for row in rows:
        if 'exhaustive' in row:print(json.dumps(dict(model=row['model'],seed=row['seed'],exact=row['exhaustive'])))
    print('Router-boundary speed',boundary['score_gap_derivative'])
    for row in flow:print(json.dumps(row))


if __name__=='__main__':main()
