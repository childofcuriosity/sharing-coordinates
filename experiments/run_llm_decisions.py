"""Execute predeclared recovered-router decisions in a common frozen backbone."""
import argparse
import json
import time
import torch

from src.llm_shared import load_model,attach_bank,make_blocks,token_loss,sha256
from src.llm_decisions import balanced_assignment,balanced_effective_partition,partition_difference,refit_bases
from experiments.llm_protocol import CONFIG,MODELS,check_freeze,unit_path
from experiments.calibrate_llm_recovery import accepted
from experiments.train_llm_shared import evaluate


def run(args):
    freeze=check_freeze()
    if args.model not in CONFIG['decision_models'] or args.seed not in CONFIG['seeds']:
        raise ValueError('Not a declared decision unit')
    out=unit_path(args.model,args.seed,'evaluation')
    if (out/'decisions.json').exists():raise ValueError('Refusing overwrite')
    policy_path='research/LLM_ACCEPTANCE_POLICY.json'
    policy=json.load(open(policy_path))
    if policy['freeze_sha256']!=freeze:raise ValueError('Policy freeze mismatch')
    recovered=json.loads((out/'recovery.json').read_text())
    training=json.loads((out/'training.json').read_text())
    if recovered['freeze_sha256']!=freeze or training['freeze_sha256']!=freeze:
        raise ValueError('Source freeze mismatch')
    if sha256(out/'checkpoint.pt')!=training['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    torch.set_num_threads(4);torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(args.seed)
    start=time.monotonic()
    model,tokenizer,entry=load_model(MODELS[args.model],args.device)
    bank,modules=attach_bank(model,args.seed,CONFIG['rank'],CONFIG['channels'])
    checkpoint=torch.load(out/'checkpoint.pt',weights_only=True,map_location=args.device)
    with torch.no_grad():
        bank.logits.copy_(checkpoint['logits']);bank.bases.copy_(checkpoint['bases'])
        theta=bank.prepare().detach().clone()
    bank.logits.requires_grad_(False)
    a=checkpoint['a'].cpu()
    labels={'native_router':balanced_assignment(a)}
    metadata={'native_router':dict(source='Native router; reference action, not optimality oracle')}
    coordinates=a@torch.linalg.cholesky(checkpoint['b'].cpu()@checkpoint['b'].cpu().T)
    labels['effective_clustering']=balanced_effective_partition(coordinates,CONFIG['rank'])
    metadata['effective_clustering']=dict(source='Balanced effective-parameter k-means; fixed initial/fixed budget')
    gen=torch.Generator().manual_seed(5200000+args.seed)
    random_labels=torch.arange(len(a))//(len(a)//CONFIG['rank'])
    labels['random_balanced']=random_labels[torch.randperm(len(a),generator=gen)]
    metadata['random_balanced']=dict(source='Fixed-seed balanced random control')
    for noise in CONFIG['decision_noise']:
        key=f'joint_noise_{noise:g}'
        record=next(r for r in recovered['records'] if r['family']=='natural_general'
            and r['q']==CONFIG['decision_query'] and r['noise']==noise)
        result=record.get('methods',{}).get('joint_fit',dict(status='observation_failed'))
        if result.get('status')=='ok':
            labels[key]=balanced_assignment(torch.tensor(result['estimate']['a']))
            metadata[key]=dict(source='Recovered router; no truth used for its selection',
                recovery_error=result['evaluation']['orbit_relative_error'],
                combined_acceptance=accepted(result,'empirical_score',policy['policy']['joint_fit']['empirical_score']),
                residual_acceptance=accepted(result,'residual_only_score',policy['policy']['joint_fit']['residual_only_score']))
        else:
            metadata[key]=dict(status='recovery_failed',recovery_status=result.get('status'))
    train,train_meta=make_blocks(tokenizer,'general','train','adapt',CONFIG['length'])
    evaluation={d:make_blocks(tokenizer,d,'test','loss',CONFIG['length'],256)[0]
                for d in CONFIG['evaluation_domains']}
    rng=torch.Generator().manual_seed(5210000+args.seed)
    batches=[torch.randperm(len(train),generator=rng)[:CONFIG['batch']].tolist()
             for _ in range(CONFIG['decision_steps'])]
    original={d:evaluate(model,bank,x,args.device) for d,x in evaluation.items()}
    actions={}
    for name,partition in labels.items():
        bank.hard_routes=torch.nn.functional.one_hot(partition,CONFIG['rank']).to(args.device,dtype=bank.bases.dtype)
        with torch.no_grad():bank.bases.copy_(refit_bases(theta,partition,CONFIG['rank']))
        immediate={d:evaluate(model,bank,x,args.device) for d,x in evaluation.items()}
        optimizer=torch.optim.AdamW([bank.bases],lr=CONFIG['decision_lr'],weight_decay=0.)
        losses=[]
        for ids in batches:
            optimizer.zero_grad(set_to_none=True)
            loss=token_loss(model,train[ids].to(args.device),bank)
            loss.backward();torch.nn.utils.clip_grad_norm_([bank.bases],CONFIG['clip']);optimizer.step()
            losses.append(float(loss.detach()))
        after={d:evaluate(model,bank,x,args.device) for d,x in evaluation.items()}
        actions[name]=dict(status='ok',labels=partition.tolist(),group_sizes=torch.bincount(partition).tolist(),
            co_membership_difference_from_native=partition_difference(partition,labels['native_router']),
            immediate_nll=immediate,after_recovery_nll=after,training_nll=losses,**metadata[name])
        print(args.model,args.seed,name,'complete',flush=True)
    for key,info in metadata.items():
        if key not in actions:actions[key]=info
    result=dict(model=args.model,seed=args.seed,freeze_sha256=freeze,policy_sha256=sha256(policy_path),
        recovery_sha256=sha256(out/'recovery.json'),original_nll=original,actions=actions,
        recovery_batch_indices=batches,train_data=train_meta,steps=CONFIG['decision_steps'],
        trainable_recovery_parameters=bank.bases.numel(),seconds=time.monotonic()-start,
        scope='Matched K-base hard routing; no realized inference speedup claim. Rejection means retaining native action.')
    with (out/'decisions.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--seed',required=True,type=int)
    p.add_argument('--device',required=True)
    run(p.parse_args())
