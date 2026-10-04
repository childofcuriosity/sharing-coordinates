"""Equal-size router decision audit with paired post-fold recovery training."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F

from experiments.train_enhancement import source_hashes
from experiments.run_router_decisions import _byte_unigram_bpb
from src.language import load_language_checkpoint,load_byte_stream,sample_batch,corpus_sha256
from src.response_identifiability import pack_basis_matrix
from src.enhancement import same_size_gauge,balanced_effective_partition
from src.router_decisions import (effective_basis_tensors,materialize_partition_folding,
    apply_common_router_gauge,compare_effective_tensors,make_fixed_language_batches,
    evaluate_fixed_batches,compare_fixed_batch_logits,effective_theta_ward)


def run(checkpoint,device):
    start=time.monotonic()
    torch.set_num_threads(4)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32=False
    model,config,meta=load_language_checkpoint(str(checkpoint),device=device)
    hashes=corpus_sha256(config)
    if hashes!=meta['corpus_sha256']:
        raise ValueError('Corpus provenance mismatch')
    seed=config.seed
    train=load_byte_stream(config.train_path,torch.device(device))
    test=load_byte_stream(config.test_path,torch.device(device))
    batches,sampling=make_fixed_language_batches(test,8,128,64,740000+seed,'nonoverlap')
    a=model.router.probabilities().detach()
    native=a.argmax(-1)
    sizes=sorted(torch.bincount(native,minlength=config.num_bases).tolist())
    with torch.no_grad():
        common=effective_basis_tensors(model)
        packed=pack_basis_matrix(model)
        theta=a@packed.bases
    balanced,baseline_info=balanced_effective_partition(theta,sizes,seed=760000+seed)
    actions={'native':native,'balanced_effective':balanced.to(device)}
    # Ward has the same number of groups, but not necessarily the same sizes.
    ward,ward_info=effective_theta_ward(common,len([n for n in sizes if n]))
    actions['ward_descriptive']=ward.to(device)
    searches={}
    for family,offset in [('positive_stochastic',720000),('signed_local',730000)]:
        found=same_size_gauge(a,offset+seed,family)
        if found['selected']:
            matrix=torch.tensor(found['selected']['matrix'],device=device,dtype=a.dtype)
            transformed,legal=apply_common_router_gauge(model,matrix)
            with torch.no_grad():
                equivalence=compare_fixed_batch_logits(model,transformed,batches,1.)
                parameters=compare_effective_tensors(common,effective_basis_tensors(transformed))
            passed=(equivalence['max_abs_logit_error']<=1e-4 and
                    equivalence['max_abs_batch_nll_error']<=1e-5 and parameters['relative_l2']<=1e-5)
            found.update(equivalence=equivalence,effective_equivalence=parameters,legal=legal,equivalence_pass=passed)
            if not passed:
                found['status']='invalid_numerical_equivalence'
            else:
                assignment=transformed.router.probabilities().argmax(-1).detach()
                if assignment.cpu().tolist()!=found['selected']['assignment']:
                    raise RuntimeError('FP32 gauge changed selected argmax')
                actions[family]=assignment
            del transformed
        searches[family]=found
    soft=evaluate_fixed_batches(model,batches,1.,False)
    unigram=_byte_unigram_bpb(Path(config.train_path),Path(config.test_path))
    outcomes={}
    for name,assignment in actions.items():
        folded,fit=materialize_partition_folding(model,common,assignment)
        folded.router.logits.requires_grad_(False)
        generator=torch.Generator().manual_seed(770000+seed)
        optimizer=torch.optim.AdamW([p for p in folded.parameters() if p.requires_grad],lr=1e-4,weight_decay=.1)
        stages={'0':evaluate_fixed_batches(folded,batches,1.,True)}
        for step in range(500):
            folded.train()
            x,y=sample_batch(train,16,128,generator)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=folded(x,hard=True)
                loss=F.cross_entropy(logits.reshape(-1,256),y.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite recovery loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(folded.parameters(),1.)
            optimizer.step()
            if step+1 in (100,500):
                stages[str(step+1)]=evaluate_fixed_batches(folded,batches,1.,True)
        outcomes[name]=dict(fit=fit,group_sizes=sorted(torch.bincount(assignment).tolist()),stages=stages)
        print(json.dumps({'seed':seed,'action':name,'bpb':{s:r['instantaneous_bpb'] for s,r in stages.items()}}),flush=True)
        del folded,optimizer
    sources=source_hashes(['experiments/run_enhanced_decisions.py','experiments/train_enhancement.py',
        'experiments/run_router_decisions.py','src/enhancement.py','src/router_decisions.py',
        'src/language.py','src/models.py','src/patterns.py','src/response_identifiability.py',
        'research/ENHANCEMENT_PROTOCOL.md'])
    return dict(format='enhanced-decisions-v1',seed=seed,checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                source_sha256=sources,corpus_sha256=hashes,sampling=sampling,searches=searches,
                router_movement=meta['extra']['router_mean_l1_movement'],
                initialization_dominated=meta['extra']['router_mean_l1_movement']<.01,
                weak_model=soft['instantaneous_bpb']>unigram-.1,soft=soft,unigram_bpb=unigram,
                balanced_baseline=baseline_info,ward_baseline=ward_info,actions=outcomes,
                device=device,torch_version=torch.__version__,seconds=time.monotonic()-start)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--device',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise SystemExit('Refusing to overwrite existing result')
    result=run(a.checkpoint,a.device)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
